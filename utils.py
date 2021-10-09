import os
import subprocess
import sys
import time
from inspect import isclass

import numpy as np


class TimerBlock:
    def __init__(self, title):
        print(("{}".format(title)))

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.clock()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")

    def log(self, string):
        duration = time.clock() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        print("  [{:.3f}{}] {}".format(duration, units, string), flush=True)


def module_to_dict(module, exclude=None):
    return dict([(x, getattr(module, x)) for x in dir(module)
                 if isclass(getattr(module, x)) and x not in exclude and getattr(module, x) not in exclude])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def copy_arguments(main_dict, main_filepath='', save_dir='./'):
    pycmd = 'python3 ' + main_filepath + ' \\\n'
    _main_dict = main_dict.copy()
    _main_dict['--name'] = _main_dict['--name'] + '_replicate'
    for k in _main_dict.keys():
        if 'batchNorm' in k:
            pycmd += ' ' + k + ' ' + str(_main_dict[k]) + ' \\\n'
        elif type(_main_dict[k]) == bool and _main_dict[k]:
            pycmd += ' ' + k + ' \\\n'
        elif type(_main_dict[k]) == list:
            pycmd += ' ' + k + ' ' + \
                ' '.join([str(f) for f in _main_dict[k]]) + ' \\\n'
        elif type(_main_dict[k]) != bool:
            pycmd += ' ' + k + ' ' + str(_main_dict[k]) + ' \\\n'
    pycmd = '#!/bin/bash\n' + pycmd[:-2]
    job_script = os.path.join(save_dir, 'job.sh')

    file = open(job_script, 'w')
    file.write(pycmd)
    file.close()

    return


def block_print():
    sys.stdout = open(os.devnull, 'w')


def get_pred_flag(height, width):
    pred_flag = np.ones((height, width, 3), dtype=np.uint8)
    pred_values = np.zeros((height, width, 3), dtype=np.uint8)

    hstart = int((192. / 1200) * height)
    wstart = int((224. / 1920) * width)
    h_step = int((24. / 1200) * height)
    w_step = int((32. / 1920) * width)

    pred_flag[hstart:hstart + h_step, -wstart + 0 * w_step:-wstart + 1 * w_step, :] = np.asarray([0, 0, 0])
    pred_flag[hstart:hstart + h_step, -wstart + 1 * w_step:-wstart + 2 * w_step, :] = np.asarray([0, 0, 0])
    pred_flag[hstart:hstart + h_step, -wstart + 2 * w_step:-wstart + 3 * w_step, :] = np.asarray([0, 0, 0])

    pred_values[hstart:hstart + h_step, -wstart + 0 * w_step:-wstart + 1 * w_step, :] = np.asarray([0, 0, 255])
    pred_values[hstart:hstart + h_step, -wstart + 1 * w_step:-wstart + 2 * w_step, :] = np.asarray([0, 255, 0])
    pred_values[hstart:hstart + h_step, -wstart + 2 * w_step:-wstart + 3 * w_step, :] = np.asarray([255, 0, 0])
    return pred_flag, pred_values


def create_pipe(pipe_filename, width, height, frame_rate=60, quite=True):
    # default extension and tonemapper
    pix_fmt = 'rgb24'
    out_fmt = 'yuv420p'
    codec = 'h264'

    command = [
        'ffmpeg',
        '-threads',
        '2',  # number of threads to start
        '-y',  # (optional) overwrite output file if it exists
        '-f',
        'rawvideo',  # input format
        '-vcodec',
        'rawvideo',  # input codec
        '-s',
        str(width) + 'x' + str(height),  # size of one frame
        '-pix_fmt',
        pix_fmt,  # input pixel format
        '-r',
        str(frame_rate),  # frames per second
        '-i',
        '-',  # The imput comes from a pipe
        '-an',  # Tells FFMPEG not to expect any audio
        '-codec:v',
        codec,  # output codec
        '-crf',
        '18',
        # compression quality for h264 (maybe h265 too?) - http://slhck.info/video/2017/02/24/crf-guide.html
        # '-compression_level', '10', # compression level for libjpeg if doing lossy depth
        '-strict',
        '-2',  # experimental 16 bit support nessesary for gray16le
        '-pix_fmt',
        out_fmt,  # output pixel format
        '-s',
        str(width) + 'x' + str(height),  # output size
        pipe_filename
    ]
    cmd = ' '.join(command)
    if not quite:
        print('openning a pip ....\n' + cmd + '\n')

    # open the pipe, and ignore stdout and stderr output
    dev_null = open(os.devnull, 'wb')
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=dev_null, stderr=dev_null, close_fds=True)


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    ry = 15
    yg = 6
    gc = 4
    cb = 11
    bm = 13
    mr = 6

    ncols = ry + yg + gc + cb + bm + mr

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # ry
    colorwheel[0:ry, 0] = 255
    colorwheel[0:ry, 1] = np.transpose(np.floor(255 * np.arange(0, ry) / ry))
    col += ry

    # yg
    colorwheel[col:col + yg, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, yg) / yg))
    colorwheel[col:col + yg, 1] = 255
    col += yg

    # gc
    colorwheel[col:col + gc, 1] = 255
    colorwheel[col:col + gc, 2] = np.transpose(np.floor(255 * np.arange(0, gc) / gc))
    col += gc

    # cb
    colorwheel[col:col + cb, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, cb) / cb))
    colorwheel[col:col + cb, 2] = 255
    col += cb

    # bm
    colorwheel[col:col + bm, 2] = 255
    colorwheel[col:col + bm, 0] = np.transpose(np.floor(255 * np.arange(0, bm) / bm))
    col += +bm

    # mr
    colorwheel[col:col + mr, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, mr) / mr))
    colorwheel[col:col + mr, 0] = 255

    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nan_idx = np.isnan(u) | np.isnan(v)
    u[nan_idx] = 0
    v[nan_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nan_idx)))

    return img


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    unknown_flow_threshold = 1e7

    idx_unknow = (abs(u) > unknown_flow_threshold) | (abs(v) > unknown_flow_threshold)
    u[idx_unknow] = 0
    v[idx_unknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idx_unknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)
