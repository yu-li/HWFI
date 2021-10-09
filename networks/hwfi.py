import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party import PWCNet
from third_party import softsplat


def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return F.interpolate(inputs, [h, w], mode=mode, align_corners=False)


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, is_prelu=True):
    if is_prelu:
        return nn.Sequential(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2,
                      bias=True), nn.PReLU())
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2,
                      bias=True))


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, residual=True, upsample=False, attention=True):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(nn.PReLU(), conv(in_planes, out_planes, kernel_size, stride, is_prelu=True),
                                  conv(out_planes, out_planes, kernel_size, stride=1, is_prelu=False))
        self.upsample = upsample
        self.residual = residual
        self.attention = attention
        self.is_shortcut_conv = False
        if residual and (in_planes != out_planes):
            self.is_shortcut_conv = True
            self.shortcut_conv = conv(in_planes, out_planes, kernel_size=1, is_prelu=False)
        if self.attention:
            self.ca = CALayer(out_planes)

    def forward(self, x, target_as=None):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        if self.upsample and target_as is not None:
            x = upsample2d_as(x, target_as)
        y = self.conv(x)
        if self.attention:
            y = self.ca(y)
        if self.residual:
            if self.is_shortcut_conv:
                x = self.shortcut_conv(x)
            y = y + x
        return y


class FlowNet(nn.Module):
    def __init__(self):
        super(FlowNet, self).__init__()
        self._model = PWCNet()

    @property
    def model(self):
        return self._model

    def forward(self, f0, f1):
        _, _, h, w = f0.shape
        new_h = h // 64 * 64
        new_w = w // 64 * 64
        f0_new = F.interpolate(f0, size=[new_h, new_w], mode="bilinear")
        f1_new = F.interpolate(f1, size=[new_h, new_w], mode="bilinear")
        x0 = torch.cat([f0_new, f1_new], 0)
        x1 = torch.cat([f1_new, f0_new], 0)
        flow = self._model(x0, x1)
        flow = F.interpolate(flow, size=[h // 4, w // 4], mode="bilinear")
        flow[:, 0:1, :, :] = flow[:, 0:1, :, :] * w / new_w
        flow[:, 1:2, :, :] = flow[:, 1:2, :, :] * h / new_h
        flow = flow * 5.0
        f0_1, f1_0 = torch.chunk(flow, 2, 0)
        return f0_1, f1_0

    def load_weights(self, path='third_party/network-default.pytorch'):
        self._model.load_state_dict(
            {strKey.replace('module', 'net'): tenWeight
             for strKey, tenWeight in torch.load(path).items()})


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1_1 = conv(3, 32)
        self.conv1_2 = conv(32, 32, is_prelu=False)
        self.conv1_2_prelu = nn.PReLU()
        self.conv2_1 = conv(32, 64, stride=2)
        self.conv2_2 = conv(64, 64, is_prelu=False)
        self.conv2_2_prelu = nn.PReLU()
        self.conv3_1 = conv(64, 96, stride=2)
        self.conv3_2 = conv(96, 96, is_prelu=False)

    def forward(self, x):
        output_dict = {}
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        output_dict['s1'] = x
        x = self.conv1_2_prelu(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        output_dict['s2'] = x
        x = self.conv2_2_prelu(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        output_dict['s3'] = x
        return output_dict


class GridNet(nn.Module):
    def __init__(self):
        super(GridNet, self).__init__()
        self.lateral1_head = ResBlock((32 + 9) * 2, 32)
        self.lateral1_1 = ResBlock(32, 32)
        self.lateral1_2 = ResBlock(32, 32)
        self.lateral1_3 = ResBlock(32, 32)
        self.lateral1_4 = ResBlock(32, 32)
        self.lateral1_5 = ResBlock(32, 32)
        self.lateral1_out = ResBlock(32, 32)

        self.lateral2_head = ResBlock(64 * 2, 64)
        self.lateral2_1 = ResBlock(64, 64)
        self.lateral2_2 = ResBlock(64, 64)
        self.lateral2_3 = ResBlock(64, 64)
        self.lateral2_4 = ResBlock(64, 64)
        self.lateral2_5 = ResBlock(64, 64)

        self.lateral3_head = ResBlock(96 * 2, 96)
        self.lateral3_1 = ResBlock(96, 96)
        self.lateral3_2 = ResBlock(96, 96)
        self.lateral3_3 = ResBlock(96, 96)
        self.lateral3_4 = ResBlock(96, 96)
        self.lateral3_5 = ResBlock(96, 96)

        self.downsample1_0 = ResBlock(32, 64, stride=2, residual=False)
        self.downsample2_0 = ResBlock(64, 96, stride=2, residual=False)
        self.downsample1_1 = ResBlock(32, 64, stride=2, residual=False)
        self.downsample2_1 = ResBlock(64, 96, stride=2, residual=False)
        self.downsample1_2 = ResBlock(32, 64, stride=2, residual=False)
        self.downsample2_2 = ResBlock(64, 96, stride=2, residual=False)

        self.upsample1_3 = ResBlock(64, 32, upsample=True, residual=False)
        self.upsample2_3 = ResBlock(96, 64, upsample=True, residual=False)
        self.upsample1_4 = ResBlock(64, 32, upsample=True, residual=False)
        self.upsample2_4 = ResBlock(96, 64, upsample=True, residual=False)
        self.upsample1_5 = ResBlock(64, 32, upsample=True, residual=False)
        self.upsample2_5 = ResBlock(96, 64, upsample=True, residual=False)

        self.lateral1_head_bw = ResBlock((32 + 9) * 2, 32)
        self.lateral1_1_bw = ResBlock(32, 32)
        self.lateral1_2_bw = ResBlock(32, 32)
        self.lateral1_3_bw = ResBlock(32, 32)
        self.lateral1_4_bw = ResBlock(32, 32)
        self.lateral1_5_bw = ResBlock(32, 32)
        self.lateral1_out_bw = ResBlock(32, 32)

        self.lateral2_head_bw = ResBlock(64 * 2, 64)
        self.lateral2_1_bw = ResBlock(64, 64)
        self.lateral2_2_bw = ResBlock(64, 64)
        self.lateral2_3_bw = ResBlock(64, 64)
        self.lateral2_4_bw = ResBlock(64, 64)
        self.lateral2_5_bw = ResBlock(64, 64)

        self.lateral3_head_bw = ResBlock(96 * 2, 96)
        self.lateral3_1_bw = ResBlock(96, 96)
        self.lateral3_2_bw = ResBlock(96, 96)
        self.lateral3_3_bw = ResBlock(96, 96)
        self.lateral3_4_bw = ResBlock(96, 96)
        self.lateral3_5_bw = ResBlock(96, 96)

        self.downsample1_0_bw = ResBlock(32, 64, stride=2, residual=False)
        self.downsample2_0_bw = ResBlock(64, 96, stride=2, residual=False)
        self.downsample1_1_bw = ResBlock(32, 64, stride=2, residual=False)
        self.downsample2_1_bw = ResBlock(64, 96, stride=2, residual=False)
        self.downsample1_2_bw = ResBlock(32, 64, stride=2, residual=False)
        self.downsample2_2_bw = ResBlock(64, 96, stride=2, residual=False)

        self.upsample1_3_bw = ResBlock(64, 32, upsample=True, residual=False)
        self.upsample2_3_bw = ResBlock(96, 64, upsample=True, residual=False)
        self.upsample1_4_bw = ResBlock(64, 32, upsample=True, residual=False)
        self.upsample2_4_bw = ResBlock(96, 64, upsample=True, residual=False)
        self.upsample1_5_bw = ResBlock(64, 32, upsample=True, residual=False)
        self.upsample2_5_bw = ResBlock(96, 64, upsample=True, residual=False)

        self.out = conv(64, 3, 3, is_prelu=False)

    def forward(self, f1, f2, e1, e2, im1, im2, f1_dict, f2_dict, f1_bw, f2_bw, e1_bw, e2_bw, f1_dict_bw, f2_dict_bw):
        x10 = self.lateral1_head([f1, f2, e1, e2, im1, im2, f1_dict['s1'], f2_dict['s1']])
        x11 = self.lateral1_1(x10)
        x12 = self.lateral1_2(x11)

        x20 = self.lateral2_head([f1_dict['s2'], f2_dict['s2']]) + self.downsample1_0(x10)
        x21 = self.lateral2_1(x20) + self.downsample1_1(x11)
        x22 = self.lateral2_2(x21) + self.downsample1_2(x12)

        x30 = self.lateral3_head([f1_dict['s3'], f2_dict['s3']]) + self.downsample2_0(x20)
        x31 = self.lateral3_1(x30) + self.downsample2_1(x21)
        x32 = self.lateral3_2(x31) + self.downsample2_2(x22)

        x10_bw = self.lateral1_head_bw([f1_bw, f2_bw, e1_bw, e2_bw, im1, im2, f1_dict_bw['s1'], f2_dict_bw['s1']])
        x11_bw = self.lateral1_1_bw(x10_bw)
        x12_bw = self.lateral1_2_bw(x11_bw)

        x20_bw = self.lateral2_head_bw([f1_dict_bw['s2'], f2_dict_bw['s2']]) + self.downsample1_0_bw(x10_bw)
        x21_bw = self.lateral2_1_bw(x20_bw) + self.downsample1_1_bw(x11_bw)
        x22_bw = self.lateral2_2_bw(x21_bw) + self.downsample1_2_bw(x12_bw)

        x30_bw = self.lateral3_head_bw([f1_dict_bw['s3'], f2_dict_bw['s3']]) + self.downsample2_0_bw(x20_bw)
        x31_bw = self.lateral3_1_bw(x30_bw) + self.downsample2_1_bw(x21_bw)
        x32_bw = self.lateral3_2_bw(x31_bw) + self.downsample2_2_bw(x22_bw)

        x30 = x30 + x30_bw
        x31 = x31 + x31_bw
        x32 = x32 + x32_bw

        x30_bw = x30 + x30_bw
        x31_bw = x31 + x31_bw
        x32_bw = x32 + x32_bw

        x33 = self.lateral3_3(x32)
        x33_bw = self.lateral3_3_bw(x32_bw)
        x34 = self.lateral3_4(x33)
        x34_bw = self.lateral3_4_bw(x33_bw)
        x35 = self.lateral3_5(x34)
        x35_bw = self.lateral3_5_bw(x34_bw)

        x23 = self.lateral2_3(x22) + self.upsample2_3(x33, target_as=x22)
        x23_bw = self.lateral2_3_bw(x22_bw) + self.upsample2_3_bw(x33_bw, target_as=x22_bw)
        x24 = self.lateral2_4(x23) + self.upsample2_4(x34, target_as=x23)
        x24_bw = self.lateral2_4_bw(x23_bw) + self.upsample2_4_bw(x34_bw, target_as=x23_bw)
        x25 = self.lateral2_5(x24) + self.upsample2_5(x35, target_as=x24)
        x25_bw = self.lateral2_5_bw(x24_bw) + self.upsample2_5_bw(x35_bw, target_as=x24_bw)

        x13 = self.lateral1_3(x12) + self.upsample1_3(x23, target_as=x12)
        x13_bw = self.lateral1_3_bw(x12_bw) + self.upsample1_3_bw(x23_bw, target_as=x12_bw)
        x14 = self.lateral1_4(x13) + self.upsample1_4(x24, target_as=x13)
        x14_bw = self.lateral1_4_bw(x13_bw) + self.upsample1_4_bw(x24_bw, target_as=x13_bw)
        x15 = self.lateral1_5(x14) + self.upsample1_5(x25, target_as=x14)
        x15_bw = self.lateral1_5_bw(x14_bw) + self.upsample1_5_bw(x25_bw, target_as=x14_bw)
        res_fw = self.lateral1_out(x15)
        res_bw = self.lateral1_out_bw(x15_bw)
        res = self.out(torch.cat([res_fw, res_bw], axis=1))

        return res


ten_grid = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.size()) not in ten_grid:
        ten_hor = torch.linspace(-1.0, 1.0,
                                 tenFlow.shape[3]).view(1, 1, 1,
                                                        tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2],
                                                                                 -1)
        ten_ver = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2],
                                                                   1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])

        ten_grid[str(tenFlow.size())] = torch.cat([ten_hor, ten_ver], 1).cuda()

    tenFlow = torch.cat([
        tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] /
        ((tenInput.shape[2] - 1.0) / 2.0)
    ], 1)

    return F.grid_sample(input=tenInput,
                         grid=(ten_grid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1),
                         mode='bilinear',
                         padding_mode='zeros')


class Metric(nn.Module):
    def __init__(self):
        super(Metric, self).__init__()
        self.alpha = nn.Parameter(-torch.ones(1, 1, 1, 1))

    def forward(self, ten_first, ten_second, tenFlow):
        return self.alpha * F.l1_loss(ten_first, backwarp(ten_second, tenFlow), reduction='none').mean(1, keepdim=True)


class HWFI(nn.Module):
    def __init__(self, finetune_flow=False):
        super(HWFI, self).__init__()
        self.finetune_flow = finetune_flow
        self.flow_net = FlowNet()
        if not self.finetune_flow:
            self.flow_net.load_weights()
            for p in self.parameters():
                p.requires_grad = False
            self.flow_net.eval()
        self.grid_net = GridNet()
        self.feature_extractor = FeatureExtractor()
        self.metric = Metric()

    def forward(self, im0, im1, t=0.5):
        if self.finetune_flow:
            f0_1_s3, f1_0_s3 = self.flow_net(im0, im1)
        else:
            with torch.no_grad():
                f0_1_s3, f1_0_s3 = self.flow_net(im0, im1)

        x = torch.cat([im0, im1], 0)
        edge0 = kornia.filters.sobel(im0)
        edge1 = kornia.filters.sobel(im1)
        f_dict = self.feature_extractor(x)
        f_dict_bw = self.feature_extractor(x)
        f0_dict = {'s1': None, 's2': None, 's3': None}
        f1_dict = {'s1': None, 's2': None, 's3': None}
        f0_dict_bw = {'s1': None, 's2': None, 's3': None}
        f1_dict_bw = {'s1': None, 's2': None, 's3': None}
        f0_dict['s1'], f1_dict['s1'] = torch.chunk(f_dict['s1'], 2, 0)
        f0_dict['s2'], f1_dict['s2'] = torch.chunk(f_dict['s2'], 2, 0)
        f0_dict['s3'], f1_dict['s3'] = torch.chunk(f_dict['s3'], 2, 0)

        f0_dict_bw['s1'], f1_dict_bw['s1'] = torch.chunk(f_dict_bw['s1'], 2, 0)
        f0_dict_bw['s2'], f1_dict_bw['s2'] = torch.chunk(f_dict_bw['s2'], 2, 0)
        f0_dict_bw['s3'], f1_dict_bw['s3'] = torch.chunk(f_dict_bw['s3'], 2, 0)

        f0_1_s2 = upsample2d_as(f0_1_s3, f0_dict['s2']) * 2.0
        f1_0_s2 = upsample2d_as(f1_0_s3, f0_dict['s2']) * 2.0

        f0_1_s1 = upsample2d_as(f0_1_s3, f0_dict['s1']) * 4.0
        f1_0_s1 = upsample2d_as(f1_0_s3, f0_dict['s1']) * 4.0

        z0_s1 = self.metric(im0, im1, f0_1_s1)
        z1_s1 = self.metric(im1, im0, f1_0_s1)
        z0_s2 = upsample2d_as(z0_s1, f0_dict['s2'])
        z1_s2 = upsample2d_as(z1_s1, f0_dict['s2'])
        z0_s3 = upsample2d_as(z0_s2, f0_dict['s3'])
        z1_s3 = upsample2d_as(z1_s2, f0_dict['s3'])

        f0 = softsplat.FunctionSoftsplat(tenInput=im0, tenFlow=f0_1_s1 * t, tenMetric=z0_s1, strType='softmax')
        f1 = softsplat.FunctionSoftsplat(tenInput=im1, tenFlow=f1_0_s1 * (1.0 - t), tenMetric=z1_s1, strType='softmax')

        f0_dict['s1'] = softsplat.FunctionSoftsplat(tenInput=f0_dict['s1'],
                                                    tenFlow=f0_1_s1 * t,
                                                    tenMetric=z0_s1,
                                                    strType='softmax')
        f0_dict['s2'] = softsplat.FunctionSoftsplat(tenInput=f0_dict['s2'],
                                                    tenFlow=f0_1_s2 * t,
                                                    tenMetric=z0_s2,
                                                    strType='softmax')
        f0_dict['s3'] = softsplat.FunctionSoftsplat(tenInput=f0_dict['s3'],
                                                    tenFlow=f0_1_s3 * t,
                                                    tenMetric=z0_s3,
                                                    strType='softmax')
        f1_dict['s1'] = softsplat.FunctionSoftsplat(tenInput=f1_dict['s1'],
                                                    tenFlow=f1_0_s1 * (1.0 - t),
                                                    tenMetric=z1_s1,
                                                    strType='softmax')
        f1_dict['s2'] = softsplat.FunctionSoftsplat(tenInput=f1_dict['s2'],
                                                    tenFlow=f1_0_s2 * (1.0 - t),
                                                    tenMetric=z1_s2,
                                                    strType='softmax')
        f1_dict['s3'] = softsplat.FunctionSoftsplat(tenInput=f1_dict['s3'],
                                                    tenFlow=f1_0_s3 * (1.0 - t),
                                                    tenMetric=z1_s3,
                                                    strType='softmax')
        edge0_fwd = softsplat.FunctionSoftsplat(tenInput=edge0, tenFlow=f0_1_s1 * t, tenMetric=z0_s1, strType='softmax')
        edge1_fwd = softsplat.FunctionSoftsplat(tenInput=edge1,
                                                tenFlow=f1_0_s1 * (1.0 - t),
                                                tenMetric=z1_s1,
                                                strType='softmax')

        f0_1_s1_bw = -(1 - t) * t * f0_1_s1 + t * t * f1_0_s1
        f0_1_s2_bw = -(1 - t) * t * f0_1_s2 + t * t * f1_0_s2
        f0_1_s3_bw = -(1 - t) * t * f0_1_s3 + t * t * f1_0_s3

        f1_0_s1_bw = (1 - t) * (1 - t) * f0_1_s1 - (1 - t) * t * f1_0_s1
        f1_0_s2_bw = (1 - t) * (1 - t) * f0_1_s2 - (1 - t) * t * f1_0_s2
        f1_0_s3_bw = (1 - t) * (1 - t) * f0_1_s3 - (1 - t) * t * f1_0_s3

        f0_bw = backwarp(tenInput=im0, tenFlow=f0_1_s1_bw)
        f1_bw = backwarp(tenInput=im1, tenFlow=f1_0_s1_bw)
        edge0_bw = backwarp(tenInput=edge0, tenFlow=f0_1_s1_bw)
        edge1_bw = backwarp(tenInput=edge1, tenFlow=f1_0_s1_bw)
        f0_dict_bw['s1'] = backwarp(tenInput=f0_dict_bw['s1'], tenFlow=f0_1_s1_bw)
        f0_dict_bw['s2'] = backwarp(tenInput=f0_dict_bw['s2'], tenFlow=f0_1_s2_bw)
        f0_dict_bw['s3'] = backwarp(tenInput=f0_dict_bw['s3'], tenFlow=f0_1_s3_bw)
        f1_dict_bw['s1'] = backwarp(tenInput=f1_dict_bw['s1'], tenFlow=f1_0_s1_bw)
        f1_dict_bw['s2'] = backwarp(tenInput=f1_dict_bw['s2'], tenFlow=f1_0_s2_bw)
        f1_dict_bw['s3'] = backwarp(tenInput=f1_dict_bw['s3'], tenFlow=f1_0_s3_bw)
        res = self.grid_net(f0, f1, edge0_fwd, edge1_fwd, im0, im1, f0_dict, f1_dict, f0_bw, f1_bw, edge0_bw, edge1_bw,
                            f0_dict_bw, f1_dict_bw)

        return res
