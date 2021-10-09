import argparse

import networks

# Collect all available model classes
model_names = sorted(el for el in networks.__dict__ if not el.startswith("__") and callable(networks.__dict__[el]))

parser = argparse.ArgumentParser(description="A PyTorch Implementation of Video Interpolation")

parser.add_argument('--model',
                    metavar='MODEL',
                    default='HWFI',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: HWFI)')
parser.add_argument('-s',
                    '--save',
                    '--save_root',
                    default='./result_folder',
                    type=str,
                    help='Path of the output folder',
                    metavar='SAVE_PATH')
parser.add_argument('--torch_home',
                    default='./pretrained',
                    type=str,
                    metavar='TORCH_HOME',
                    help='Path to save pre-trained models from torchvision')
parser.add_argument('-n',
                    '--name',
                    default='trial_0',
                    type=str,
                    metavar='EXPERIMENT_NAME',
                    help='Name of experiment folder.')
parser.add_argument('--dataset',
                    default='VIMEO',
                    type=str,
                    metavar='TRAINING_DATALOADER_CLASS',
                    help='Specify training dataset class for loading (Default: VIMEO)')
parser.add_argument('--val_dataset',
                    default='VIMEO',
                    type=str,
                    metavar='VAL_DATALOADER_CLASS',
                    help='Specify val dataset class for loading (Default: VIMEO)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='CHECKPOINT_PATH',
                    help='path to checkpoint file (default: none)')

# Resources
parser.add_argument('--distributed_backend',
                    default='nccl',
                    type=str,
                    metavar='DISTRIBUTED_BACKEND',
                    help='backend used for communication between processes.')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loader workers (default: 10)')
parser.add_argument('-ids', '--gpu_ids', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7], help='GPUs to use')

# Learning rate parameters.
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--flow_lr_coe',
                    default=0.5,
                    type=float,
                    metavar='LR',
                    help='relative learning rate w.r.t basic learning rate (default: 0.5)')
parser.add_argument('--lr_scheduler',
                    default='MultiStepLR',
                    type=str,
                    metavar='LR_Scheduler',
                    help='Scheduler for learning' + ' rate (only CosineAnnealingLR and MultiStepLR supported.')
parser.add_argument('--lr_gamma', default=0.1, type=float, help='learning rate will be multiplied by this gamma')
parser.add_argument('--lr_milestones',
                    type=int,
                    nargs='+',
                    default=[40, 80],
                    help="Spatial dimension to " + "crop training samples for training")
parser.add_argument('--lr_T_max', default=20, type=int, help='T_max for CosineAnnealingLR')
parser.add_argument('--lr_T_mult', default=1, type=int, help='T_mult for CosineAnnealingLR')
parser.add_argument('--lr_min', default=1e-5, type=float, help='eta_min for CosineAnnealingLR')

# Gradient.
parser.add_argument('--clip_gradients', default=-1.0, type=float, help='If positive, clip the gradients by this value.')

# Optimization hyper-parameters
parser.add_argument('-b',
                    '--batch_size',
                    default=4,
                    type=int,
                    metavar='BATCH_SIZE',
                    help='mini-batch per gpu size (default : 4)')
parser.add_argument('--weight_decay',
                    default=0.0001,
                    type=float,
                    metavar='WEIGHT_DECAY',
                    help='weight_decay (default = 0.001)')
parser.add_argument('--seed', default=1234, type=int, metavar="SEED", help='seed for initializing training. ')
parser.add_argument('--optimizer',
                    default='Adamax',
                    type=str,
                    metavar='OPTIMIZER',
                    help='Specify optimizer from torch.optim (Default: Adamax)')
parser.add_argument('--finetune', action='store_true', help='finetune flow net or not')

# Training sequence, supports a single sequence for now
parser.add_argument('--train_file', required=False, metavar="TRAINING_FILE", help='training file (default : Required)')
parser.add_argument('--crop_size',
                    type=int,
                    nargs='+',
                    default=[256, 256],
                    metavar="CROP_SIZE",
                    help="Spatial dimension to crop training samples for training (default : [256, 256])")
parser.add_argument('--stride',
                    type=int,
                    default=64,
                    help='the largest factor a model reduces spatial size of inputs during a forward pass.')
parser.add_argument('--print_freq',
                    default=1,
                    type=int,
                    metavar="PRINT_FREQ",
                    help='frequency of printing training status (default: 1)')
parser.add_argument('--save_freq',
                    type=int,
                    default=10,
                    metavar="SAVE_FREQ",
                    help='frequency of saving intermediate models, in epoches (default: 10)')
parser.add_argument('--start_epoch', type=int, default=-1, help="Set epoch number during resuming")
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar="EPOCHES",
                    help='number of total epochs to run (default: 100)')

# Validation sequence, supports a single sequence for now
parser.add_argument('--val_file', metavar="VALIDATION_FILE", help='validation file (default : None)')
parser.add_argument('--val_batch_size', type=int, default=1, help="Batch size to use for validation.")
parser.add_argument('--initial_eval', action='store_true', help='Perform initial evaluation before training.')
parser.add_argument('--write_images',
                    action='store_true',
                    help='write to folder \'args.save/args.name\' prediction and ground-truth images.')

# Required for torch distributed launch
parser.add_argument('--local_rank', default=None, type=int, help='Torch Distributed')

# video test args
parser.add_argument("--extract_dir",
                    type=str,
                    default="/data/algceph/ssd/samuelzhu/frame_interp/tmpNet2",
                    help='path to save extracted frames')
parser.add_argument("--ffmpeg_dir", type=str, default="/usr/bin/", help='path to ffmpeg.exe')
parser.add_argument("--video", nargs='+', type=str, default=["./input_video"], help='path of videos to be converted')
parser.add_argument("--fps", type=float, default=24, help='specify fps of output video. Default: 30.')
parser.add_argument("--sf",
                    type=int,
                    default=4,
                    help='specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames')
parser.add_argument(
    "--test_batch_size",
    type=int,
    default=1,
    help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
parser.add_argument("--output", type=str, default="./output_video", help='Specify output dir')
parser.add_argument("--folder_name", type=str, default="x4", help='Specify tmp folder file name. Default: x4')
