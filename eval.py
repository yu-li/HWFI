import os
import sys
import warnings

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from imageio import imsave
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from torchvision import transforms
from tqdm import tqdm

import dataloaders
import networks
import utils
from dataloaders import custom_transform as tr
from networks.losses import LaplacianLoss
from parsers import parser

tqdm.monitor_interval = 0
warnings.filterwarnings('ignore')


def main():
    with utils.TimerBlock("\nParsing Arguments") as block:
        args = parser.parse_args()

        args.rank = int(os.getenv('RANK', 0))
        args.world_size = int(os.getenv("WORLD_SIZE", 1))

        block.log("Creating save directory: {}".format(args.save))
        args.save_root = os.path.join(args.save, args.name)
        if args.write_images:
            os.makedirs(args.save_root, exist_ok=True)
            assert os.path.exists(args.save_root)
        else:
            os.makedirs(args.save, exist_ok=True)
            assert os.path.exists(args.save)

        os.makedirs(args.torch_home, exist_ok=True)
        os.environ['TORCH_HOME'] = args.torch_home
        ids = ",".join(str(i) for i in args.gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = ids
        args.gpus = len(args.gpu_ids)
        block.log('Number of gpus: {} | {}'.format(args.gpus, list(range(args.gpus))))

        args.network_class = utils.module_to_dict(networks)[args.model]
        args.dataset_class = utils.module_to_dict(dataloaders)[args.val_dataset]
        block.log('save_root: {}'.format(args.save_root))
        block.log('val_file: {}'.format(args.val_file))

    with utils.TimerBlock("Building {} Dataset".format(args.dataset)) as block:
        vkwargs = {
            'batch_size': args.gpus * args.val_batch_size,
            'num_workers': args.gpus * args.workers,
            'pin_memory': True,
            'drop_last': False
        }
        transform = transforms.Compose([tr.Normalize(), tr.PadImage(args.stride), tr.ToTensor()])
        if args.dataset == 'VIMEO':
            val_dataset = args.dataset_class(args.val_file, split='eval', transform=transform)
        else:
            val_dataset = args.dataset_class(args.val_file, transform=transform)
        val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **vkwargs)

        block.log('Number of Validation Images: {}:({} mini-batches)'.format(len(val_loader.dataset), len(val_loader)))

    with utils.TimerBlock("Building {} Model".format(args.model)) as block:
        model = args.network_class()

        block.log('Number of parameters: {val:,}'.format(
            val=sum([p.data.nelement() if p.requires_grad else 0 for p in model.parameters()])))

        block.log('Initializing CUDA')
        assert torch.cuda.is_available(), 'Code supported for GPUs only at the moment'
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(len(args.gpu_ids)))
        criterion = LaplacianLoss()
        torch.manual_seed(args.seed)

        block.log("Attempting to Load checkpoint '{}'".format(args.resume))
        if args.resume and os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)

            # Partial initialization
            if 'state_dict' in checkpoint:
                input_dict = checkpoint['state_dict']
            else:
                input_dict = checkpoint
            curr_dict = model.module.state_dict()
            state_dict = input_dict.copy()
            for key in input_dict:
                print(key)
                if key not in curr_dict:
                    print('error:', key)
                    continue
                if curr_dict[key].shape != input_dict[key].shape:
                    state_dict.pop(key)
                    print("key {} skipped because of size mismatch.".format(key))
            model.module.load_state_dict(state_dict, strict=False)
            if 'epoch' in checkpoint:
                epoch = checkpoint['epoch']
            else:
                epoch = 0
            block.log("Successfully loaded checkpoint (at epoch {})".format(epoch))
        elif args.resume:
            block.log("No checkpoint found at '{}'.\nAborted.".format(args.resume))
            sys.exit(0)
        else:
            block.log("Random initialization, checkpoint not provided.")
            epoch = 0

    with utils.TimerBlock("Inference started ") as block:
        evaluate(args, val_loader, model, epoch, block, criterion)


def evaluate(args, val_loader, model, epoch, block, criterion):
    model.eval()

    loss_values = utils.AverageMeter()
    avg_metrics = np.zeros((0, 3), dtype=float)
    num_batches = len(val_loader)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, total=num_batches)):

            inputs = [b.cuda() for b in batch['image']]
            im0, gt, im1 = inputs[0], inputs[len(inputs) // 2], inputs[-1]
            pred = model(im0, im1)
            loss = criterion(pred, gt)
            loss_values.update(loss.mean().data.item(), pred.size(0))

            batch_size, _, _, _ = pred.shape

            for b in range(batch_size):
                input_filenames = batch['input_files'][1][b]
                in_height, in_width = batch['ishape'][0][b], batch['ishape'][1][b]
                first_target = (im0[b].data.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                first_target = first_target[:in_height, :in_width, :]

                second_target = (im1[b].data.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                second_target = second_target[:in_height, :in_width, :]

                pred_image = np.round(
                    (pred[b].data.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255.0)).astype(np.uint8)
                pred_image = pred_image[:in_height, :in_width, :]

                gt_image = (gt[b].data.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                gt_filename = '/'.join(input_filenames.split(os.sep)[-4:])
                gt_image = gt_image[:in_height, :in_width, :]

                # calculate metrics using skimage
                psnr = compare_psnr(pred_image, gt_image)
                ssim = compare_ssim(pred_image, gt_image, multichannel=True, gaussian_weights=True)
                err = 128.0 + pred_image - gt_image
                ie = np.mean(np.abs(err - 128.0))
                avg_metrics = np.vstack((avg_metrics, np.array([psnr, ssim, ie])))

                # write_images
                if args.write_images:
                    tmp_filename = os.path.join(args.save_root, "%s-%s.png" % (gt_filename[:], 'It_pred'))
                    os.makedirs(os.path.dirname(tmp_filename), exist_ok=True)
                    imsave(tmp_filename, pred_image)

                    tmp_filename = os.path.join(args.save_root, "%s-%s.png" % (gt_filename[:], 'It'))
                    os.makedirs(os.path.dirname(tmp_filename), exist_ok=True)
                    imsave(tmp_filename, gt_image)

                    tmp_filename = os.path.join(args.save_root, "%s-%s.png" % (gt_filename[:], 'im0'))
                    os.makedirs(os.path.dirname(tmp_filename), exist_ok=True)
                    imsave(tmp_filename, first_target)

                    os.makedirs(os.path.dirname(tmp_filename), exist_ok=True)
                    tmp_filename = os.path.join(args.save_root, "%s-%s.png" % (gt_filename[:], "im1"))
                    imsave(tmp_filename, second_target)

            if (i + 1) >= num_batches:
                break
    avg_metrics = np.nanmean(avg_metrics, axis=0)
    v_psnr, v_ssim, v_ie = avg_metrics[0], avg_metrics[1], avg_metrics[2]
    t_loss = loss_values.avg
    result2print = 'Overall PSNR: {:.2f}, SSIM: {:.3f}, IE: {:.2f}'.format(v_psnr, v_ssim, v_ie)
    block.log(result2print)
    # Move back the model to train mode.
    model.train()

    torch.cuda.empty_cache()
    block.log('max memory allocated (GB): {:.3f}: '.format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))

    return v_psnr, v_ssim, v_ie, t_loss


if __name__ == '__main__':
    main()
