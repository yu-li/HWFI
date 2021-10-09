import os
import os.path
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from imageio import imsave
from tqdm import tqdm

import dataloaders
import networks
import utils
from dataloaders import custom_transform as tr
from parsers import parser


def init_set():
    with utils.TimerBlock("\nParsing Arguments") as block:
        args = parser.parse_args()

        args.rank = int(os.getenv('RANK', 0))

        os.makedirs(args.torch_home, exist_ok=True)
        os.environ['TORCH_HOME'] = args.torch_home

        args.network_class = utils.module_to_dict(networks)[args.model]
        args.dataset_class = utils.module_to_dict(dataloaders)[args.val_dataset]
        block.log('save_video_root: {}'.format(args.output))
        block.log('video_file_path: {}'.format(args.video))
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        ids = ""
        for i in args.gpu_ids:
            ids += (str(i) + ', ')
        os.environ["CUDA_VISIBLE_DEVICES"] = ids[:-2]

    with utils.TimerBlock("Building {} Model".format(args.model)) as block:
        model = args.network_class()

        block.log('Number of parameters: {val:,}'.format(
            val=sum([p.data.nelement() if p.requires_grad else 0 for p in model.parameters()])))

        block.log('Initializing CUDA')
        assert torch.cuda.is_available(), 'Code supported for GPUs only at the moment'
        model = model.cuda()
        torch.manual_seed(args.seed)

        block.log("Attempting to Load checkpoint '{}'".format(args.resume))
        if args.resume and os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')

            # Partial initialization
            if 'state_dict' in checkpoint:
                input_dict = checkpoint['state_dict']
            else:
                input_dict = checkpoint
            curr_dict = model.state_dict()
            state_dict = {}
            for key in input_dict:
                if 'module' in key:
                    key_ = key[7:]
                else:
                    key_ = key
                state_dict[key_] = input_dict[key]
                if key_ not in curr_dict:
                    print(key_)
                    continue
                if curr_dict[key_].shape != input_dict[key].shape:
                    state_dict.pop(key_)
                    print("key {} skipped because of size mismatch.".format(key_))
            model.load_state_dict(state_dict, strict=False)
            if 'epoch' in checkpoint:
                epoch = checkpoint['epoch']
            else:
                epoch = 0
            block.log("Successfully loaded checkpoint (at epoch {})".format(epoch))
        elif args.resume:
            block.log("No checkpoint found at '{}'.\nAborted.".format(args.resume))
            sys.exit(0)
        else:
            epoch = 0
            block.log("Random initialization, checkpoint not provided.")
    return args, model, epoch


def build_dataloader(args, data_root):
    with utils.TimerBlock("Building {} Dataset".format(args.dataset)) as block:
        vkwargs = {
            'batch_size': args.test_batch_size,
            'num_workers': args.workers,
            'pin_memory': True,
            'drop_last': False
        }
        transform = transforms.Compose([tr.Normalize(), tr.PadImage(args.stride), tr.ToTensor()])
        video_dataset = args.dataset_class(root=data_root, transform=transform)

        video_loader = torch.utils.data.DataLoader(video_dataset, shuffle=False, **vkwargs)

        block.log('Number of Validation Images: {}:({} mini-batches)'.format(len(video_loader.dataset),
                                                                             len(video_loader)))
    return video_loader


def extract_frames(video, out_dir, args):

    error = ""
    print('{} -i {} -vsync 0 {}/%06d.png'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), video, out_dir))
    retn = os.system('{} -i "{}" -vsync 0 {}/%06d.png'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), video, out_dir))
    if retn:
        error = "Error converting file:{}. Exiting.".format(video)
    return error


def create_video(dir, video, args):
    error = ""
    print('{} -r {} -i {}/%d.png -pix_fmt yuv420p -vcodec libx264 {}'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"),
                                                                             args.fps, dir,
                                                                             os.path.join(args.output, video)))
    retn = os.system('{} -r {} -i {}/%d.png -pix_fmt yuv420p -vcodec libx264 "{}"'.format(
        os.path.join(args.ffmpeg_dir, "ffmpeg"), args.fps, dir, os.path.join(args.output, video)))
    if retn:
        error = "Error creating output video. Exiting."
    return error


def main():

    # init setting
    args, model, _ = init_set()
    model.eval()
    extraction_dir = os.path.join(args.extract_dir, args.folder_name)
    if not os.path.isdir(extraction_dir):
        os.makedirs(extraction_dir)

    for video in args.video:
        print(video)
        video_name, _ = os.path.splitext(os.path.basename(video))
        # Create extraction folder and extract frames
        with utils.TimerBlock("Convert Video to frames of %s" % video_name) as block:
            extraction_path = os.path.join(extraction_dir, video_name, "input")
            output_path = os.path.join(extraction_dir, video_name, "output")
            if not os.path.isdir(extraction_path):
                os.makedirs(extraction_path)
            if not os.path.isdir(output_path):
                os.makedirs(output_path)

            error = extract_frames(os.path.join(video), extraction_path, args)
            if error:
                block.log(error)
                exit(1)
        # dataloader
        data_loader = build_dataloader(args, extraction_path)
        tr = transforms.Compose([transforms.ToPILImage()])

        # Interpolate frames
        with utils.TimerBlock("Interpolate frames of %s" % video_name) as block:
            num_batches = len(data_loader)
            frame_counter = 1
            with torch.no_grad():
                for i, batch in enumerate(tqdm(data_loader, total=num_batches)):

                    inputs = [b.cuda() for b in batch['image']]

                    im0, im1 = inputs[0], inputs[-1]
                    batch_size, _, _, _ = im0.shape
                    for batch_index in range(batch_size):
                        h, w = batch['ishape'][0][batch_index], batch['ishape'][1][batch_index]
                        (tr(inputs[0][batch_index].cpu().detach()[:, :h, :w])).save(
                            os.path.join(output_path,
                                         str(frame_counter + args.sf * batch_index) + ".png"))
                    frame_counter += 1
                    for intermediate_index in range(1, args.sf):
                        t = intermediate_index / args.sf
                        outputs = model(im0, im1, t) * 255.0
                        for batch_index in range(batch_size):
                            h, w = batch['ishape'][0][batch_index], batch['ishape'][1][batch_index]
                            im = np.round((outputs[batch_index].cpu().numpy()[:, :h, :w].transpose(1, 2, 0)).clip(
                                0, 255.0)).astype(np.uint8)
                            tmp_filename = os.path.join(output_path,
                                                        str(frame_counter + args.sf * batch_index) + ".png")
                            os.makedirs(os.path.dirname(tmp_filename), exist_ok=True)
                            imsave(tmp_filename, im)
                        frame_counter += 1

                    # Set counter accounting for batching of frames
                    frame_counter += args.sf * (batch_size - 1)
                    if i + 1 == num_batches:
                        h, w = batch['ishape'][0][batch_size - 1], batch['ishape'][1][batch_size - 1]
                        (tr(inputs[-1][batch_size - 1].cpu().detach()[:, :h, :w] / 255.0)).save(
                            os.path.join(output_path,
                                         str(frame_counter + 1) + ".png"))
                        frame_counter += 1
            block.log("FrameCount = %d" % frame_counter)
        with utils.TimerBlock("Create slow motion video of %s" % video_name) as block:
            create_video(output_path, os.path.basename(video), args)


if __name__ == '__main__':
    main()
