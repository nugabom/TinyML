# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:58:20 2023

@author: kucha
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, reduce, repeat
import einops
from torchvision import datasets, transforms
from torchvision.models._utils import _make_divisible
from torchvision.models import mobilenet_v2
from typing import Callable, List, Optional
#from torchvision.utils import _log_api_usage_once
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, model_entrypoint, generate_default_cfgs, register_model_deprecations

__all__ = ['MobileNetV2', 'ConvNormActivation', 'InvertedResidual'] 
class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y
class ConvNormActivation(torch.nn.Sequential):
    """
    Configurable block used for Convolution-Normalzation-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalzation-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in wich case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolutiuon layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optinal): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        pad_layer : str = 'zero'
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [] 
        layers.append(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        )
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        #_log_api_usage_once(self)
        self.out_channels = out_channels

class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int,  block_id: int, pad_layer: str, norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
             # pw
             layers.append(
                 ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6, pad_layer=pad_layer)
                 )
        layers.extend(
                 [
                     # dw
                     ConvNormActivation(
                         hidden_dim,
                         hidden_dim,
                         stride=stride,
                         groups=hidden_dim,
                         norm_layer=norm_layer,
                         activation_layer=nn.ReLU6,
                         pad_layer=pad_layer
                         ),
                     # pw-linear
                     nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                     norm_layer(oup),
                     ]
                 )
        if self.use_res_connect:
            layers.extend([Add()])

        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return self.conv[-1](x, self.conv[:-1](x))
            #return x + self.conv(x)
        else:
            return self.conv(x)
            
class MobileNetV2(nn.Module):
    def __init__(
            self,
            pad_layer: str,
            num_classes: int = 1000,
            width_mult: float = 1.0,
            inverted_residual_setting = None,
            round_nearest: int = 8,
            block =None,
            norm_layer = None,
            dropout: float=0.2,
            **kwargs
            ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.with_feat = None
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = _make_divisible(32 * width_mult, round_nearest)
        self.last_channel = _make_divisible(1280 * width_mult, round_nearest)

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                    # t, c, n, s
                    [1, 16, 1, 1], #1 
                    [6, 24, 2, 2], #2
                    [6, 32, 3, 2], #3 4 5
                    [6, 64, 4, 2], # 6 7 8 9
                    [6, 96, 3, 1], 
                    [6, 160, 3, 2], 
                    [6, 320, 1, 1] 
            ]
        features = [ConvNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6,pad_layer=pad_layer)]
        n_block = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, block_id=n_block, norm_layer=norm_layer, pad_layer=pad_layer))
                input_channel = output_channel
                n_block += 1
        features.append(
            ConvNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6, pad_layer=pad_layer
            )
        )

        self.features = nn.Sequential(*features)
        #if self.split_block_idx != 1 or self.split_block_idx is not None:
        #    self.top = self.features[:self.split_block_idx]
        #    print(self.top)
        #    self.bottom = self.features[self.split_block_idx:]
        self.classifier = nn.Sequential(
                nn.Linear(self.last_channel, num_classes),
        )

    def update(self):
        if self.with_feat:
            for block in self.features:
                if getattr(block, 'use_res_connect', None):
                    setattr(block.conv[0][0], 'add', True)

    def forward_with_feature_s(self, x):
        x = self.features[:5](x)
        return x

        #print('student', self.per_patch_stage)
        feat = []
        y = None
        flag = False
        for name, layer in self.named_modules():
            if len(feat) == self.per_patch_stage:
                return feat
            if (isinstance(layer, (nn.Conv2d, nn.ReLU6)) and 'mid' not in name) or hasattr(layer, "mid_conv") or hasattr(layer, 'batch_norm'):
                if getattr(layer, "add", None):
                    y = x
                x = layer(x)
                if flag and isinstance(layer, nn.ReLU6):
                    pass
                    flag = False
                    feat.append(x)
                if (isinstance(layer, nn.Conv2d) and layer.kernel_size[0] > 1) or  hasattr(layer, 'mid_conv'):
                    #print('student', name, layer)
                    flag = True
            if isinstance(layer, Add):
                x = layer(x, y)
                y = None
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_with_feature_t(self, x):
        x = self.features[:5](x)
        return x

        feat = []
        y = None
        flag = False
        for name, layer in self.named_modules():
            if len(feat) == self.per_patch_stage:
                return feat
            if (isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU6)) and 'mid' not in name) or hasattr(layer, "mid_conv"):
                if getattr(layer, "add", None):
                    y = x
                x = layer(x)
                if flag and isinstance(layer, nn.ReLU6):
                    flag = False
                    feat.append(x)
                if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] > 1:
                    flag = True
            if isinstance(layer, Add):
                x = layer(x, y)
                y = None
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


    def get_features(self, x):
        pad1, pad2, pad3, pad4, pad5 = None, None, None, None, None

        pad1 = x.clone()
        x = self.features[0][0](x)
        if len(x) == 2:
            pad1 = x[0]
            x = x[1]
        #x = self.features[0][0](x)
        x = self.features[0][1](x)
        x = self.features[0][2](x)

        pad2 = x.clone()
        x = self.features[1].conv[0][0](x)
        #pad2, x = self.features[1].conv[0][0](x)
        if len(x) == 2:
            pad2 = x[0]
            x = x[1]
        x = self.features[1].conv[0][1](x)
        x = self.features[1].conv[0][2](x)
        x = self.features[1].conv[1:](x)

        x = self.features[2].conv[0](x)
        #pad3, x = self.features[2].conv[1][0](x)
        pad3 = x.clone()
        x = self.features[2].conv[1][0](x)
        if len(x) == 2:
            pad3 = x[0]
            x = x[1]
        x = self.features[2].conv[1][1](x)
        x = self.features[2].conv[1][2](x)
        x = self.features[2].conv[2:](x)

        y = x.clone()
        x = self.features[3].conv[0](x)
        pad4 = x.clone()
        x = self.features[3].conv[1][0](x)
        if len(x) == 2:
            pad4 = x[0]
            x = x[1]
        x = self.features[3].conv[1][1](x)
        x = self.features[3].conv[1][2](x)
        x = self.features[3].conv[2:4](x)
        x = self.features[3].conv[4](x, y)

        x = self.features[4].conv[0](x)
        pad5 = x.clone()
        x = self.features[4].conv[1][0](x)
        if len(x) == 2:
            pad5 = x[0]
            x = x[1]
        x = self.features[4].conv[1][1](x)
        x = self.features[4].conv[1][2](x)
        patch_out = self.features[4].conv[2:](x)
        final = self.features[5:](patch_out.clone())
        x = nn.functional.adaptive_avg_pool2d(final.clone(), (1, 1))
        x = torch.flatten(x,1)
        x = self.classifier(x) 

        return pad1, pad2, pad3, pad4, pad5, patch_out, final, x

    def get_features_base(self, x):
        x1 = x.clone()
        x = self.features[0][0](x)
        x = self.features[0][1](x)
        x2 = self.features[0][2](x)

        x = self.features[1].conv[0][0](x2.clone())
        x = self.features[1].conv[0][1](x)
        x = self.features[1].conv[0][2](x)
        x = self.features[1].conv[1:](x)

        x3 = self.features[2].conv[0](x)
        x = self.features[2].conv[1][0](x3.clone())
        x = self.features[2].conv[1][1](x)
        x = self.features[2].conv[1][2](x)
        x = self.features[2].conv[2:](x)

        y = x.clone()
        x4 = self.features[3].conv[0](x)
        x = self.features[3].conv[1][0](x4.clone())
        x = self.features[3].conv[1][1](x)
        x = self.features[3].conv[1][2](x)
        x = self.features[3].conv[2:4](x)
        x = self.features[3].conv[4](x, y)

        x5 = self.features[4].conv[0](x)
        x = self.features[4].conv[1][0](x5.clone())
        x = self.features[4].conv[1][1](x)
        x = self.features[4].conv[1][2](x)
        patch_out = self.features[4].conv[2:](x)
        final = self.features[5:](patch_out.clone())
        x = nn.functional.adaptive_avg_pool2d(final.clone(), (1, 1))
        x = torch.flatten(x,1)
        x = self.classifier(x) 

        return x1, x2, x3, x4, x5, patch_out, final, x

    def _forward_impl(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        if self.with_feat is None:
            return self._forward_impl(x)
        else:
            if self.teacher:
                return self.get_features_base(x)
            else:
                return self.get_features(x) 
    
# import torchsummary
# import torchprofile
# model = MobileNetV2(pad_layer='zero')
# torchsummary.summary(model, (3, 224, 224))
# from torchprofile import profile_macs
# inputs = torch.randn(1, 3, 224, 224)
# macs = profile_macs(model, inputs)
# print(macs)
def _create_mobilenetv2(variant, pretrained=False, **kwargs):
    for k, v in kwargs.items():
        print(f"{k} : {v}")
    return build_model_with_cfg(MobileNetV2, variant, pretrained, **kwargs)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 144, 144),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'classifier',
        **kwargs
    }

@register_model
def mobilenetv2_tiny_zero(pretrained=False, **kwargs) -> MobileNetV2:
    model_args = dict(width_mult=0.35, pad_layer='zero')
    return _create_mobilenetv2('mobilenetv2_tiny_zero', pretrained, **model_args)

@register_model
def mobilenetv2_tiny_zero_large(pretrained=False, **kwargs) -> MobileNetV2:
    model_args = dict(width_mult=0.5, pad_layer='zero')
    return _create_mobilenetv2('mobilenetv2_tiny_zero_large', pretrained, **model_args)

@register_model
def mobilenetv2_tiny_replicate(pretrained=False, **kwargs) -> MobileNetV2:
    model_args = dict(width_mult=0.35, pad_layer='replicate')
    return _create_mobilenetv2('mobilenetv2_tiny_replicate', pretrained, **model_args)

@register_model
def mobilenetv2_tiny_reflect(pretrained=False, **kwargs) -> MobileNetV2:
    model_args = dict(width_mult=0.35, pad_layer='reflect')
    return _create_mobilenetv2('mobilenetv2_tiny_reflect', pretrained, **model_args)
