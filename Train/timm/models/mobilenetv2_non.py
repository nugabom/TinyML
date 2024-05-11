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
from torchvision.utils import _log_api_usage_once
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, model_entrypoint, generate_default_cfgs, register_model_deprecations
from .custom_padding import *

__all__ = ['MobileNetV2_NON', 'ConvNormActivation', 'InvertedResidual'] 
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
        num_patches : int,
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
        
        if pad_layer == 'zero':
            layers = [ZeropatchPad2d(padding=padding, num_patches=num_patches)]
        elif pad_layer == 'replicate':
            layers = [ReplicationPad2d(padding=padding, num_patches=num_patches)]
        elif pad_layer == 'reflect':
            layers = [ReflectionPad2d(padding=padding, num_patches=num_patches)]
        elif pad_layer == 'default':
            layers = [torch.nn.ZeroPad2d(padding=padding)]
        elif pad_layer == 'mean_3px':
            layers = [Mean_3px_Pad2d(padding=padding, num_patches=num_patches)]
        elif pad_layer == 'mean_2px':
            layers = [Mean_2px_Pad2d(padding=padding, num_patches=num_patches)]
        elif pad_layer == 'weight':
            layers = [Weight_2px_Pad2d(padding=padding, num_patches=num_patches, C=in_channels)]
        elif pad_layer == 'dead':
            layers = [deadline4(padding=padding, num_patches=num_patches, C=in_channels)]
        else:
            raise NotImplementedError(f'{pad_layer} not in zero / replication / reflect')
        layers.append(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                0,
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
        _log_api_usage_once(self)
        self.out_channels = out_channels

class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int,  block_id: int, pad_layer: str, num_patches: int, norm_layer: Optional[Callable[..., nn.Module]] = None,
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
                 ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6, pad_layer='default', num_patches=1)
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
                         pad_layer=pad_layer,
                         num_patches=num_patches
                         ),
                     # pw-linear
                     nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                     norm_layer(oup),
                     ]
                 )

        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
            
class MobileNetV2_NON(nn.Module):
    def __init__(
            self,
            pad_layer: str,
            num_classes: int = 1000,
            width_mult: float = 1.0,
            patches : int = 1,
            split : int = 5,
            inverted_residual_setting = None,
            round_nearest: int = 8,
            block =None,
            norm_layer = None,
            dropout: float=0.2,
            **kwargs
            ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.patches = patches
        self.split = split
        mode = 'default'
        block_id = 0
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
        print(self.split, 'ssss')
        if block_id < self.split:
            mode = pad_layer
        block_id += 1
        features = [ConvNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6,pad_layer=mode, num_patches=patches)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                if block_id < self.split:
                    mode = pad_layer
                else:
                    mode = 'default'
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, block_id=block_id, norm_layer=norm_layer, pad_layer=mode, num_patches=patches))
                input_channel = output_channel
                block_id += 1
        features.append(
            ConvNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6, pad_layer='default', num_patches=1
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
        print(self.features)
        

    def forward(self, x):
        x = einops.rearrange(x, 'B C (p1 H) (p2 W) -> (B p1 p2) C H W', p1=self.patches, p2=self.patches)
        x = self.features[:self.split](x)
        x = einops.rearrange(x, '(B p1 p2) C H W -> B C (p1 H) (p2 W)', p1=self.patches, p2=self.patches)
        x = self.features[self.split:](x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# import torchsummary
# import torchprofile
# model = MobileNetV2_NON(pad_layer='zero')
# torchsummary.summary(model, (3, 224, 224))
# from torchprofile import profile_macs
# inputs = torch.randn(1, 3, 224, 224)
# macs = profile_macs(model, inputs)
# print(macs)
def _create_mobilenetv2_non(variant, pretrained=False, **kwargs):
    for k, v in kwargs.items():
        print(f"{k} : {v}")
    return build_model_with_cfg(MobileNetV2_NON, variant, pretrained, **kwargs)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'classifier',
        **kwargs
    }

@register_model
def mobilenetv2_non(pretrained=False, **kwargs) -> MobileNetV2_NON:
    model_args = dict(width_mult=1.0, patches=4, split=5, pad_layer='zero')
    return _create_mobilenetv2_non('mobilenetv2_non_zero', pretrained, **model_args)

@register_model
def mobilenetv2_tiny_non(pretrained=False, **kwargs) -> MobileNetV2_NON:
    model_args = dict(width_mult=0.35, patches=4, split=5, pad_layer='zero')
    return _create_mobilenetv2_non('mobilenetv2_tiny_non_zero', pretrained, **model_args)

@register_model
def mobilenetv2_weight_non(pretrained=False, **kwargs) -> MobileNetV2_NON:
    model_args = dict(width_mult=0.5, patches=4, split=5, pad_layer='weight')
    return _create_mobilenetv2_non('mobilenetv2_tiny_weight_zero', pretrained, **model_args)

@register_model
def mobilenetv2_non_replicate(pretrained=False, **kwargs) -> MobileNetV2_NON:
    model_args = dict(width_mult=0.35, patches=4, split=5, pad_layer='replicate')
    return _create_mobilenetv2_non('mobilenetv2_non_replicate', pretrained, **model_args)

@register_model
def mobilenetv2_non_tanh(pretrained=False, **kwargs) -> MobileNetV2_NON:
    model_args = dict(width_mult=0.5, patches=4, split=5, pad_layer='dead')
    return _create_mobilenetv2_non('mobilenetv2_non_tanh', pretrained, **model_args)

@register_model
def mobilenetv2_non_reflect(pretrained=False, **kwargs) -> MobileNetV2_NON:
    model_args = dict(width_mult=1.0, patches=4, split=5, pad_layer='reflect')
    return _create_mobilenetv2_non('mobilenetv2_non_reflect', pretrained, **model_args)

@register_model
def mobilenetv2_non_mean_3px(pretrained=False, **kwargs) -> MobileNetV2_NON:
    model_args = dict(width_mult=1.0, patches=4, split=5, pad_layer='mean_3px')
    return _create_mobilenetv2_non('mobilenetv2_non_mean_3px', pretrained, **model_args)

@register_model
def mobilenetv2_non_tiny_replicate(pretrained=False, **kwargs) -> MobileNetV2_NON:
    model_args = dict(width_mult=0.5, patches=4, split=5, pad_layer='replicate')
    return _create_mobilenetv2_non('mobilenetv2_non_tiny_replicate', pretrained, **model_args)

@register_model
def mobilenetv2_non_tiny_reflect(pretrained=False, **kwargs) -> MobileNetV2_NON:
    model_args = dict(width_mult=0.35, patches=4, split=5, pad_layer='reflect')
    return _create_mobilenetv2_non('mobilenetv2_non_tiny_reflect', pretrained, **model_args)

@register_model
def mobilenetv2_non_tiny_mean_2px(pretrained=False, **kwargs) -> MobileNetV2_NON:
    model_args = dict(width_mult=0.5, patches=4, split=5, pad_layer='mean_2px')
    return _create_mobilenetv2_non('mobilenetv2_non_mean_2px', pretrained, **model_args)

@register_model
def mobilenetv2_non_tiny_mean_2px_detach(pretrained=False, **kwargs) -> MobileNetV2_NON:
    model_args = dict(width_mult=0.35, patches=4, split=5, pad_layer='mean_2px_detach')
    return _create_mobilenetv2_non('mobilenetv2_non_mean_2px_detach', pretrained, **model_args)
