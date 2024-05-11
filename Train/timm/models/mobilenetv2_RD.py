# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:28:31 2023

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

__all__ = ['MobileNetV2_RD', 'ConvNormActivation', 'InvertedResidual'] 
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
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
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
        ]
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
        self, inp: int, oup: int, stride: int, expand_ratio: int,  block_id: int, norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        
        if block_id == 0:
            layers = [
                ConvNormActivation(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        else:
            # pw
            layers.append(
                ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
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
            
class MobileNetV2_RD(nn.Module):
    def __init__(
            self,
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
                    [6, 24, 1, 2], #2
                    [6, 32, 4, 2], #3 4 5
                    [6, 64, 4, 2], # 6 7 8 9
                    [6, 96, 3, 1], # 10 11 12
                    [6, 160, 4, 2], # 13 14 15 16
                    [6, 320, 1, 1] # 17
            ]
        features = [ConvNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)]
        n_block = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, block_id=n_block, norm_layer=norm_layer))
                input_channel = output_channel
                n_block += 1
        features.append(
            ConvNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
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

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def _create_mobilenetv2_RD(variant, pretrained=False, **kwargs):
    for k, v in kwargs.items():
        print(f"{k} : {v}")
    return build_model_with_cfg(MobileNetV2_RD, variant, pretrained, **kwargs)

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
def mobilenetv2_RD(pretrained=False, **kwargs) -> MobileNetV2_RD:
    model_args = dict(width_mult=1.0)
    return _create_mobilenetv2_RD('mobilenetv2_RD', pretrained, **model_args)

@register_model
def mobilenetv2_tiny_RD(pretrained=False, **kwargs) -> MobileNetV2_RD:
    model_args = dict(width_mult=0.35)
    return _create_mobilenetv2_RD('mobilenetv2_tiny_RD', pretrained, **model_args)

