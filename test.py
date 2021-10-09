import cv2
import numpy
import torch

import networks
import utils
from parsers import parser

args = parser.parse_args()
args.network_class = utils.module_to_dict(networks)[args.model]
model = args.network_class()
assert torch.cuda.is_available(), 'Code supported for GPUs only at the moment'

model = model.cuda()
model = torch.nn.DataParallel(model, device_ids=range(len(args.gpu_ids)))
torch.manual_seed(args.seed)

checkpoint = torch.load(args.resume)
if 'state_dict' in checkpoint:
    input_dict = checkpoint['state_dict']
else:
    input_dict = checkpoint
curr_dict = model.module.state_dict()
state_dict = input_dict.copy()
for key in input_dict:
    if key not in curr_dict:
        print('error:', key)
        continue
    if curr_dict[key].shape != input_dict[key].shape:
        state_dict.pop(key)
        print("key {} skipped because of size mismatch.".format(key))
model.module.load_state_dict(state_dict, strict=False)

model.eval()
PATH = './examples/'
with torch.no_grad():
    im0 = cv2.imread(filename=PATH + '/im0.png', flags=-1)
    im1 = cv2.imread(filename=PATH + '/im1.png', flags=-1)
    h, w, _ = im0.shape
    im0 = torch.FloatTensor(numpy.ascontiguousarray(im0.transpose(2, 0, 1).astype(numpy.float32) *
                                                   (1.0 / 255.0))).cuda().unsqueeze(0)
    im1 = torch.FloatTensor(numpy.ascontiguousarray(im1.transpose(2, 0, 1).astype(numpy.float32) *
                                                   (1.0 / 255.0))).cuda().unsqueeze(0)
    pred = model(im0, im1, 0.5)
    pred = (pred[0].data.cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0) * 255.0).astype(numpy.uint8)
    cv2.imwrite(PATH + '/It_pred.png', pred)
