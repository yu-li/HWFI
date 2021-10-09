import os
import random
import warnings

import numpy as np
import torch.backends.cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import dataloaders
import networks
import utils
from dataloaders import custom_transform as tr
from eval import evaluate
from networks.losses import LaplacianLoss
from parsers import parser

tqdm.monitor_interval = 0
warnings.filterwarnings('ignore')


def parse_and_set_args(block):
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    block.log("Enabling torch.backends.cudnn.benchmark")

    args.rank = int(os.getenv('RANK', 0))
    args.world_size = int(os.getenv("WORLD_SIZE", 1))

    if args.local_rank:
        args.rank = args.local_rank
    if args.local_rank is not None and args.local_rank != 0:
        utils.block_print()

    block.log("Creating save directory: {}".format(os.path.join(args.save, args.name)))
    args.save_root = os.path.join(args.save, args.name)
    os.makedirs(args.save_root, exist_ok=True)
    assert os.path.exists(args.save_root)

    # temporary directory for torch pre-trained models
    os.makedirs(args.torch_home, exist_ok=True)
    os.environ['TORCH_HOME'] = args.torch_home

    args.network_class = utils.module_to_dict(networks)[args.model]
    args.optimizer_class = utils.module_to_dict(torch.optim)[args.optimizer]
    args.dataset_class = utils.module_to_dict(dataloaders)[args.dataset]
    args.val_dataset_class = utils.module_to_dict(dataloaders)[args.val_dataset]

    return args


def initialize_distributed(args):
    # Manually set the device ids.
    torch.cuda.set_device(args.gpu_ids[args.rank % len(args.gpu_ids)])

    # Call the init process
    if args.world_size > 1:
        init_method = 'env://'
        torch.distributed.init_process_group(backend=args.distributed_backend,
                                             world_size=args.world_size,
                                             rank=args.rank,
                                             init_method=init_method)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_train_and_valid_data_loaders(block, args):
    transform = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomVerticalFlip(),
        tr.RandomReverseTemporalOrder(),
        tr.RandomCrop((args.crop_size[0], args.crop_size[1])),
        tr.Normalize(),
        tr.ToTensor()
    ])

    # training dataloader
    tkwargs = {'batch_size': args.batch_size, 'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
    if args.dataset == 'VIMEO':
        train_dataset = args.dataset_class(args.train_file, split='train', transform=transform)
    else:
        train_dataset = args.dataset_class(args.train_file, transform=transform)

    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=(train_sampler is None), **tkwargs)

    block.log('Number of Training Images: {}:({} mini-batches)'.format(len(train_loader.dataset), len(train_loader)))

    # validation dataloader
    t_transform = transforms.Compose([tr.Normalize(), tr.PadImage(args.stride), tr.ToTensor()])
    vkwargs = {'batch_size': args.val_batch_size, 'num_workers': args.workers, 'pin_memory': True, 'drop_last': False}

    if args.dataset == 'VIMEO':
        val_dataset = args.val_dataset_class(args.val_file, split='eval', transform=t_transform)
    else:
        val_dataset = args.val_dataset_class(args.val_file, transform=t_transform)

    val_sampler = None

    val_loader = DataLoader(val_dataset, sampler=val_sampler, shuffle=False, **vkwargs)

    block.log('Number of Validation Images: {}:({} mini-batches)'.format(len(val_loader.dataset), len(val_loader)))

    return train_loader, train_sampler, val_loader


def load_model(model, optimizer, block, args):
    # trained weights
    checkpoint = torch.load(args.resume, map_location='cpu')

    # used for partial initialization
    if 'state_dict' in checkpoint:
        input_dict = checkpoint['state_dict']
    else:
        input_dict = checkpoint
    curr_dict = model.state_dict()
    state_dict = input_dict.copy()
    for key in input_dict:
        if key not in curr_dict:
            print(key)
            continue
        if curr_dict[key].shape != input_dict[key].shape:
            state_dict.pop(key)
            print("key {} skipped because of size mismatch.".format(key))
    model.load_state_dict(state_dict, strict=False)
    if 'optimizer' in checkpoint and args.start_epoch < 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if args.start_epoch < 0 and 'epoch' in checkpoint:
        args.start_epoch = max(0, checkpoint['epoch'])
    else:
        args.start_epoch = max(0, args.start_epoch)
    block.log("Successfully loaded checkpoint (at epoch {})".format(checkpoint['epoch']))


def build_and_initialize_model_loss_and_optimizer(block, args):
    model = args.network_class(args.finetune)
    block.log('Number of parameters: {val:,}'.format(
        val=sum([p.data.nelement() if p.requires_grad else 0 for p in model.parameters()])))
    for name, module in model.named_children():
        block.log('Number of ' + name + ' parameters: {val:,}'.format(
            val=sum([p.data.nelement() if p.requires_grad else 0 for p in module.parameters()])))

    block.log('Initializing CUDA')
    assert torch.cuda.is_available(), 'only GPUs support at the moment'
    model.cuda(torch.cuda.current_device())
    if args.finetune:
        optimizer = args.optimizer_class([{
            'params': model.flow_net.parameters(),
            'initial_lr': args.lr * args.flow_lr_coe
        }, {
            'params': model.grid_net.parameters(),
            'initial_lr': args.lr
        }, {
            'params': model.feature_extractor.parameters(),
            'initial_lr': args.lr
        }, {
            'params': model.metric.parameters(),
            'initial_lr': args.lr
        }],
                                         lr=args.lr,
                                         weight_decay=args.weight_decay)
    else:
        optimizer = args.optimizer_class([{
            'params': [p for p in model.parameters() if p.requires_grad],
            'initial_lr': args.lr
        }],
                                         lr=args.lr,
                                         weight_decay=args.weight_decay)

    block.log("Attempting to Load checkpoint '{}'".format(args.resume))
    if args.resume and os.path.isfile(args.resume):
        load_model(model, optimizer, block, args)
    elif args.resume:
        block.log("No checkpoint found at '{}'".format(args.resume))
        exit(1)
    else:
        block.log("Random initialization, checkpoint not provided.")
        args.start_epoch = 0
    block.log("Build criterion function")
    criterion = LaplacianLoss()

    # Run multi-process when it is needed.
    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu_ids[args.rank % len(args.gpu_ids)]],
                                                          output_device=args.gpu_ids[args.rank % len(args.gpu_ids)],
                                                          find_unused_parameters=True)
    return model, optimizer, criterion


def get_learning_rate_scheduler(optimizer, block, args):
    block.log('Base leaning rate {}.'.format(args.lr))
    if args.lr_scheduler == 'MultiStepLR':
        block.log('Using multi-step learning rate scheduler with {} gamma '
                  'and {} milestones.'.format(args.lr_gamma, args.lr_milestones))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=args.lr_milestones,
                                                            gamma=args.lr_gamma)
    elif args.lr_scheduler == 'CosWarmLR':
        block.log('Using CosineAnnealingWarmRestarts decay learning rate scheduler')
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=args.lr_T_max,
                                                                            T_mult=args.lr_T_mult,
                                                                            eta_min=args.lr_min)
    else:
        raise NameError('Unknown {} learning rate scheduler'.format(args.lr_scheduler))

    return lr_scheduler


def forward_only(inputs_gpu, model, args, criterion):
    # Forward pass.
    im0 = inputs_gpu['image'][0]
    im1 = inputs_gpu['image'][-1]
    gt = inputs_gpu['image'][1]
    pred = model(im0, im1)
    loss = criterion(pred, gt)

    return loss, pred


def calc_linf_grad_norm(args, parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = max(p.grad.data.abs().max() for p in parameters)
    max_norm_reduced = torch.cuda.FloatTensor([max_norm])
    if args.world_size > 1:
        torch.distributed.all_reduce(max_norm_reduced, op=torch.distributed.ReduceOp.MAX)
    return max_norm_reduced[0].item()


def train_step(batch_cpu, model, optimizer, block, args, criterion, print_linf_grad=False):
    # Move data to GPU.
    inputs = {k: [b.cuda() for b in batch_cpu[k]] for k in batch_cpu if k in ['image']}
    # Forward pass.
    loss, pred = forward_only(inputs, model, args, criterion)

    # Backward and SGP steps.
    optimizer.zero_grad()
    loss.backward()

    # Calculate and print norm infinity of the gradients.
    grad_max = calc_linf_grad_norm(args, model.parameters())
    if print_linf_grad:
        block.log('gradients Linf: {:0.6f}'.format(grad_max))

    # Clip gradients by value.
    if args.clip_gradients > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradients, norm_type=2)
    optimizer.step()
    return loss, pred


def evaluate_epoch(model, val_loader, block, args, criterion, epoch=0):
    # calculate validation loss and metrics
    v_psnr, v_ssim, v_ie, loss_values = evaluate(args, val_loader, model, epoch, block, criterion)

    # Move back the model to train mode.
    model.train()

    return v_psnr, v_ssim, v_ie, loss_values


def train_epoch(epoch, args, model, optimizer, lr_scheduler, train_sampler, train_loader, v_psnr, v_ssim, v_ie, v_loss,
                block, criterion):
    # Average loss calculator.
    loss_values = utils.AverageMeter()

    # This will ensure the data is shuffled each epoch.
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    # Get number of batches in one epoch.
    num_batches = len(train_loader)

    global_index = 0

    for i, batch in enumerate(train_loader):

        # Set global index.
        global_index = epoch * num_batches + i

        # Move one step.
        loss, pred = train_step(batch, model, optimizer, block, args, criterion,
                                ((global_index + 1) % args.print_freq == 0))

        # Update the loss accumulator.
        loss_values.update(loss.data.item(), pred.size(0))
        if (global_index + 1) % args.print_freq == 0:
            # Reduce the loss.
            if args.world_size > 1:
                tmp = torch.Tensor([loss_values.avg]).cuda()
                torch.distributed.all_reduce(tmp)
                t_loss = tmp.item() / args.world_size
            else:
                t_loss = loss_values.avg

            # And reset the loss accumulator.
            loss_values.reset()

            # Print some output.
            dict2print = {
                'iter': global_index,
                'epoch': str(epoch) + '/' + str(args.epochs),
                'batch': str(i + 1) + '/' + str(num_batches)
            }
            str2print = ' '.join(key + " : " + str(dict2print[key]) for key in dict2print)
            str2print += (' trainLoss:' + ' %.5f' % t_loss)
            str2print += ' valLoss' + ' %.5f' % v_loss
            str2print += ' valPSNR' + ' %.3f' % v_psnr
            str2print += ' lr:' + ' %1.5f' % (optimizer.state_dict()['param_groups'][0]['lr'])
            block.log(str2print)

        # Break the training loop if we have reached the maximum number of batches.
        if (i + 1) >= num_batches:
            break
    lr_scheduler.step()
    return global_index


def save_model(model, optimizer, epoch, global_index, psnr, block, args):
    # Write on rank zero only
    if args.rank == 0:
        if args.world_size > 1:
            model_ = model.module
        else:
            model_ = model
        state_dict = model_.state_dict()
        tmp_keys = state_dict.copy()
        for k in state_dict:
            [tmp_keys.pop(k) if (k in tmp_keys and ikey in k) else None for ikey in model_.ignore_keys]
        state_dict = tmp_keys.copy()
        # save checkpoint
        model_optim_state = {
            'epoch': epoch,
            'state_dict': state_dict,
        }
        model_name = os.path.join(args.save_root,
                                  '_ckpt_epoch_%03d_iter_%07d_psnr_%1.2f.pt.tar' % (epoch, global_index, psnr))
        torch.save(model_optim_state, model_name)
        block.log('saved model {}'.format(model_name))

        return model_name


def train(model, optimizer, lr_scheduler, train_loader, train_sampler, val_loader, block, args, criterion):
    # Set the model to train mode.
    model.train()

    # Perform an initial evaluation.
    if args.initial_eval:
        block.log('Initial evaluation.')

        v_psnr, v_ssim, v_ie, v_loss = evaluate_epoch(model, val_loader, block, args, criterion, args.start_epoch)
    else:
        v_psnr, v_ssim, v_ie, v_loss = 20.0, 0.5, 15.0, 0.0

    for epoch in range(args.start_epoch, args.epochs):

        # Train for an epoch.
        global_index = train_epoch(epoch, args, model, optimizer, lr_scheduler, train_sampler, train_loader, v_psnr,
                                   v_ssim, v_ie, v_loss, block, criterion)

        if (epoch + 1) % args.save_freq == 0:
            v_psnr, v_ssim, v_ie, v_loss = evaluate_epoch(model, val_loader, block, args, criterion, epoch + 1)
            save_model(model, optimizer, epoch + 1, global_index, v_psnr, block, args)

    return 0


def main():
    # Parse the args.
    with utils.TimerBlock("\nParsing Arguments") as block:
        args = parse_and_set_args(block)

    # Initialize torch.distributed.
    with utils.TimerBlock("Initializing Distributed"):
        initialize_distributed(args)

    # Set all random seed for reproducibility.
    with utils.TimerBlock("Setting Random Seed"):
        set_random_seed(args.seed)

    # Train and validation data loaders.
    with utils.TimerBlock("Building {} Dataset".format(args.dataset)) as block:
        train_loader, train_sampler, val_loader = get_train_and_valid_data_loaders(block, args)

    # Build the model and optimizer.
    with utils.TimerBlock("Building {} Model, Loss and {} Optimizer".format(args.model,
                                                                            args.optimizer_class.__name__)) as block:
        model, optimizer, criterion = build_and_initialize_model_loss_and_optimizer(block, args)

    # Learning rate scheduler.
    with utils.TimerBlock("Building {} Learning Rate Scheduler".format(args.optimizer)) as block:
        lr_scheduler = get_learning_rate_scheduler(optimizer, block, args)

    with utils.TimerBlock("Training Model") as block:
        train(model, optimizer, lr_scheduler, train_loader, train_sampler, val_loader, block, args, criterion)

    return 0


if __name__ == '__main__':
    main()
