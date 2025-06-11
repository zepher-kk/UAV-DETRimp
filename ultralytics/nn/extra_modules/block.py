import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union
from collections import OrderedDict
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad, LightConv, ConvTranspose
from ..modules.block import get_activation, ConvNormLayer, WTConvNormLayer,BasicBlock, BottleNeck, RepC3, C3, C2f, Bottleneck
import math
from torch.nn import init
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import dill as pickle
import math
from ultralytics.nn.modules.conv import Conv
import numpy as np
import torch.nn.functional as F
import pywt
import pywt.data
__all__ = ['DySample','SPDConv','MFFF','FrequencyFocusedDownSampling','SemanticAlignmenCalibration',
           'WTConv2d_imp', 'WTConv2dMaxPool', 'FFC',
           'SEAdd', 'SEcatv2', 'SEcatv3'
    ,'WTConv2d','SEcatv1'
           ]


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def constant_init(self, module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    
    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x



class FFM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1, groups=dim)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        # res = x.clone()
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta


class ImprovedFFTKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.SiLU()

        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, groups=dim, bias=True)

        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ffm = FFM(dim)

        #通道注意力
        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.in_conv(x)
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')
        x_fca = torch.abs(x_fca)

        x_sca1 = self.conv1x1(x_fca)
        x_sca2 = self.conv3x3(x_fca)
        x_sca3 = self.conv5x5(x_fca)
        x_sca = x_sca1 + x_sca2 + x_sca3


        channel_weights = self.channel_attention(x_att)
        x_sca = x_sca * channel_weights

        x_sca = self.ffm(x_sca)

        out = x + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)

class MFFF(nn.Module): 
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = ImprovedFFTKernel(int(dim * self.e))

    def forward(self, x):
        c1 = round(x.size(1) * self.e)
        c2 = x.size(1) - c1
        ok_branch, identity = torch.split(self.cv1(x), [c1, c2], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))

class ADown(nn.Module):
    def __init__(self, c1, c2):  
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1,x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

class FrequencyFocusedDownSampling(nn.Module):
    def __init__(self, c1, c2):  
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
        self.ffm = FFM(self.c)  # FGM 模块处理 x2 分支
        self.conv_reduce = Conv(self.c * 2, self.c, 1, 1)
        self.conv_resize = Conv(self.c, self.c, 3, 2, 1)
    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)


        fgm_out = self.ffm(x2)
        fgm_out = self.conv_resize(fgm_out)
        pooled_out = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        pooled_out = self.cv2(pooled_out)

        x2 = torch.cat((fgm_out, pooled_out), 1)

        x2 = self.conv_reduce(x2)

        return torch.cat((x1, x2), 1)
    
    
class SemanticAlignmenCalibration(nn.Module):  # 
    def __init__(self, inc):
        super(SemanticAlignmenCalibration, self).__init__()
        hidden_channels = inc[0]

        self.groups = 2
        self.spatial_conv = Conv(inc[0], hidden_channels, 3)
        self.semantic_conv = Conv(inc[1], hidden_channels, 3)


        self.frequency_enhancer = FFM(hidden_channels)

        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)
        

        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64),
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)
        )

        self.init_weights()
        self.offset_conv[1].weight.data.zero_()

    def init_weights(self):

        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        coarse_features, semantic_features = x
        batch_size, _, out_h, out_w = coarse_features.size()


        semantic_features = self.semantic_conv(semantic_features)
        semantic_features = F.interpolate(semantic_features, coarse_features.size()[2:], mode='bilinear', align_corners=True)


        enhanced_frequency = self.frequency_enhancer(semantic_features)
        

        gate = torch.sigmoid(self.gating_conv(semantic_features))
        fused_features = semantic_features * (1 - gate) + enhanced_frequency * gate


        coarse_features = self.spatial_conv(coarse_features)


        conv_results = self.offset_conv(torch.cat([coarse_features, fused_features], 1))

        fused_features = fused_features.reshape(batch_size * self.groups, -1, out_h, out_w)
        coarse_features = coarse_features.reshape(batch_size * self.groups, -1, out_h, out_w)


        offset_low = conv_results[:, 0:self.groups * 2, :, :].reshape(batch_size * self.groups, -1, out_h, out_w)
        offset_high = conv_results[:, self.groups * 2:self.groups * 4, :, :].reshape(batch_size * self.groups, -1, out_h, out_w)


        normalization_factors = torch.tensor([[[[out_w, out_h]]]]).type_as(fused_features).to(fused_features.device)
        grid_w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        grid_h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        base_grid = torch.cat((grid_h.unsqueeze(2), grid_w.unsqueeze(2)), 2)
        base_grid = base_grid.repeat(batch_size * self.groups, 1, 1, 1).type_as(fused_features).to(fused_features.device)


        adjusted_grid_l = base_grid + offset_low.permute(0, 2, 3, 1) / normalization_factors
        adjusted_grid_h = base_grid + offset_high.permute(0, 2, 3, 1) / normalization_factors


        coarse_features = F.grid_sample(coarse_features, adjusted_grid_l, align_corners=True)
        fused_features = F.grid_sample(fused_features, adjusted_grid_h, align_corners=True)


        coarse_features = coarse_features.reshape(batch_size, -1, out_h, out_w)
        fused_features = fused_features.reshape(batch_size, -1, out_h, out_w)


        attention_weights = 1 + torch.tanh(conv_results[:, self.groups * 4:, :, :])
        final_features = fused_features * attention_weights[:, 0:1, :, :] + coarse_features * attention_weights[:, 1:2, :, :]

        return final_features


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):

    w = pywt.Wavelet(wave)

    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)

    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)


    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])


    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)


    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters.to(x.device), stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):

    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)

    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters.to(x.device), stride=2, groups=c, padding=pad)
    return x


# Define the WaveletTransform class
class WaveletTransform(Function):

    @staticmethod
    def forward(ctx, input, filters):

        ctx.filters = filters
        with torch.no_grad():
            x = wavelet_transform(input, filters)
        return x

    @staticmethod
    def backward(ctx, grad_output):

        grad = inverse_wavelet_transform(grad_output, ctx.filters)
        return grad, None


# Define the InverseWaveletTransform class
class InverseWaveletTransform(Function):


    @staticmethod
    def forward(ctx, input, filters):
        ctx.filters = filters
        with torch.no_grad():
            x = inverse_wavelet_transform(input, filters)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad = wavelet_transform(grad_output, ctx.filters)
        return grad, None



def wavelet_transform_init(filters):


    def apply(input):
        return WaveletTransform.apply(input, filters)

    return apply


# Initialize the InverseWaveletTransform
def inverse_wavelet_transform_init(filters):


    def apply(input):
        return InverseWaveletTransform.apply(input, filters)

    return apply

class WTConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1


        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)


        self.wt_function = wavelet_transform_init(self.wt_filter)
        self.iwt_function = inverse_wavelet_transform_init(self.iwt_filter)



        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])


        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )

        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter.to(x_in.device), bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):

            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            # 处理奇数尺寸（补零）
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:,:,0,:,:]

            # 处理高频分量
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            # 保存处理后的分量
            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])


        next_x_ll = 0
        for i in range(self.wt_levels-1, -1, -1):

            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)

            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]


        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0


        x = self.base_scale(self.base_conv(x))

        x = x + x_tag


        if self.do_stride is not None:
            x = self.do_stride(x)

        return x
class WTConv2d_imp(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d_imp, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1


        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)


        self.wt_function = wavelet_transform_init(self.wt_filter)
        self.iwt_function = inverse_wavelet_transform_init(self.iwt_filter)

        if in_channels != out_channels:
            self.channel_align = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.channel_align = nn.Identity()

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])


        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )

        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter.to(x_in.device), bias=None,
                                                   stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):

            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)

            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        if self.do_stride is not None:
            x = self.do_stride(x)
        x = self.channel_align(x)
        return x


class _ScaleModule(nn.Module):


    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConv2dMaxPool(nn.Module):
    def __init__(self, c1, c2, e=0.25, kernel_size=5, stride=2, use_avgpool=False, pool_kernel=3):
        super().__init__()
        assert 0 <= e <= 1, "e must be in [0,1]"
        base_conv_ch = int(round(c2 * e))
        wt_conv_ch = c2 - base_conv_ch

        self.use_avgpool = use_avgpool
        if self.use_avgpool:
            self.avgpool = nn.AvgPool2d(
                kernel_size=pool_kernel,
                stride=1,
                padding=pool_kernel // 2
            )
        else:
            self.avgpool = nn.Identity()

        self.base_conv = nn.Sequential(
            nn.Conv2d(c1, base_conv_ch * 2, kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(base_conv_ch * 2, base_conv_ch, kernel_size=1)
        )
        self.wt_conv = WTConv2d_imp(c1, wt_conv_ch, kernel_size=kernel_size, stride=stride
                                    )
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=pool_kernel, stride=stride, padding=pool_kernel // 2),
            nn.Conv2d(c1, wt_conv_ch, kernel_size=1)
        )
        self.merge_mode = "add"  # or "concat"
        self.outconv = nn.Sequential(nn.Conv2d(c2, c2, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(c2)
                                     )
    def forward(self, x):

        x = self.avgpool(x)

        x_base = self.base_conv(x)

        x_wt = self.wt_conv(x)
        x_mp = self.maxpool(x)

        if self.merge_mode == "add":
            xw = x_wt + x_mp  # 通道数需相同
        elif self.merge_mode == "concat":
            xw = torch.cat([x_wt, x_mp], dim=1)  # 通道数翻倍

        downresult = torch.cat([x_base, xw], dim=1)
        # 通道拼接
        return downresult

class SEnetV2(nn.Module):
    def __init__(self, in_channels, dim, reduction_ratio=4):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim*2
        mid_dim = self.dim // reduction_ratio

        self.h_avg_pool = nn.AdaptiveAvgPool2d((None, 1))  # 保持W维度
        self.w_avg_pool = nn.AdaptiveAvgPool2d((1, None))  # 保持H维度

        self.fc1 = nn.Sequential(
            nn.Linear(self.dim, mid_dim),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.dim, mid_dim),
            nn.ReLU(inplace=True)
        )
        # 双路特征融合
        self.fc3 = nn.Sequential(
            nn.Linear(self.dim, mid_dim),
            nn.Sigmoid()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(self.dim, mid_dim),
            nn.Sigmoid()
        )

        self.fc_all = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Sigmoid()
        )



    def forward(self, x):
        B, C, H, W = x.size()

        h_pool = self.h_avg_pool(x)  # [B, C, H, 1]
        w_pool = self.w_avg_pool(x)  # [B, C, 1, W]
        h_pool = h_pool.permute(0, 1, 3, 2)  # [B, C, 1, H]
        hw_pool = torch.cat([h_pool, w_pool], dim=3)  # [B, C, 1, H+W]
        hw_pool = hw_pool.squeeze(2) # [B, C, H+W]
        fc1 = self.fc1(hw_pool)
        fc2 = self.fc2(hw_pool)
        fc3 = self.fc3(hw_pool)
        fc4 = self.fc4(hw_pool)
        fc_all = torch.cat([fc1, fc2, fc3, fc4], dim=2)
        fc_all = self.fc_all(fc_all)
        assert H==W
        x1, x2 = fc_all[:, :, 0:H], fc_all[:, :, H:2*H]
        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(3)
        return x * x1 * x2
class SEcatv1(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        self.se = SEAttention(inc[1] * 2)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)

class SEAttention(nn.Module):
    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class SEcatv2(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = nn.Conv2d(inc[0], inc[1], 1)
        self.se = None
        #self.se = SEnetV2(inc[1] * 2, 160)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        if self.se is None:
            self.se = SEnetV2(x0.shape[1], x0.shape[2])
        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)

class SEcatv3(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = nn.Conv2d(inc[0], inc[1], 1)
        self.se = None
        #self.se = SEnetV2(inc[1] * 2, 160)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        if self.se is None:
            self.se = EfficientSE(x_concat.shape[1])
        x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)

class SEAdd(nn.Module):

    def __init__(self, inc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = nn.Conv2d(inc[0], inc[1], 1)

        self.se = SEnetV2(inc[1] * 2, 160)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        x0Ax1 = x1_weight + x0_weight
        assert x0Ax1.shape == x.shape, "input should be a square"
        return x0Ax1


class FFC(nn.Module):
    def __init__(self, c1, c2) -> None:
        super().__init__()

        self.conv_channel_fusion = Conv(c1, c2 // 2, k=1)
        self.conv_3x3_feature_extract = Conv(c2 // 2, c2 // 2, 3)
        self.conv_1x1 = Conv(c2 // 2, c2, 1)

    def forward(self, x):
        x = self.conv_1x1(self.conv_3x3_feature_extract(self.conv_channel_fusion(x)))
        return x


class EfficientSE(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.h_avg_pool = nn.AdaptiveAvgPool2d((None, 1))  # [B, C, H, 1]
        self.w_avg_pool = nn.AdaptiveAvgPool2d((1, None))  # [B, C, 1, W]

        mid_channels = max(8, in_channels // reduction)
        self.conv_h = nn.Conv2d(in_channels, mid_channels, 1)
        self.conv_w = nn.Conv2d(in_channels, mid_channels, 1)
        self.conv_fuse = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # 高度方向注意力
        h_pool = self.h_avg_pool(x)  # [B, C, H, 1]
        h_attn = self.conv_h(h_pool).sigmoid()  # [B, mid, H, 1]

        # 宽度方向注意力
        w_pool = self.w_avg_pool(x)  # [B, C, 1, W]
        w_attn = self.conv_w(w_pool).sigmoid()  # [B, mid, 1, W]

        # 融合空间注意力
        attn = self.conv_fuse(h_attn * w_attn)  # [B, C, H, W]
        return x * self.sigmoid(attn)
