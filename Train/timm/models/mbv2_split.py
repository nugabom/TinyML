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

__all__ = ['MobileNetV2_split', 'ConvNormActivation', 'InvertedResidual'] 
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
        mode: str = "zero"
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if mode == "zero":
            layers = [torch.nn.ConstantPad2d(padding=padding, value=0.0)]
        elif mode == "reflect":
            layers = [torch.nn.ReflectionPad2d(padding=padding)]
        elif mode == "replicate":
            layers = [torch.nn.ReplicationPad2d(padding=padding)]
        else:
            raise NotImplementedError(f"Mode {mode} in (zero_pad / reflect /replication)")
        if bias is None:
            bias = norm_layer is None
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
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None, mode='zero'
    ) -> None:
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6, mode=mode)
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
                    mode=mode
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
            
class MobileNetV2_split(nn.Module):
    def __init__(
            self,
            split_point,
            n_patches,
            mode,
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
        self.split_point = split_point
        self.n_patches = n_patches
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
                    [1, 16, 1, 1],
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1],
            ]
        features = [ConvNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6, mode=mode)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, mode=mode))
                input_channel = output_channel
        features.append(
            ConvNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6, mode=mode
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
        x = einops.rearrange(x, 'B C (p1 H) (p2 W) -> (B p1 p2) C H W', p1=self.n_patches, p2=self.n_patches)
        x = self.features[:self.split_point](x)
        x = einops.rearrange(x, '(B p1 p2) C H W -> B C (p1 H) (p2 W)', p1=self.n_patches, p2=self.n_patches)
        x = self.features[self.split_point:](x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def _create_mobilenet_split(variant, pretrained=False, **kwargs):
    print(kwargs)
    return build_model_with_cfg(MobileNetV2_split, variant, pretrained, **kwargs)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 160, 160),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }

@register_model
def mbv2_w35_split_5_p4_zero(pretrained=False, **kwargs) -> MobileNetV2_split:
  model_args = dict(split_point=5, num_classes=1000,  n_patches=4, width_mult=0.35, mode='zero', **kwargs)
  return _create_mobilenet_split('mbv2_w35_split_5_p4_zero', pretrained, **model_args)

@register_model
def mbv2_w35_split_5_p4_replicate(pretrained=False, **kwargs) -> MobileNetV2_split:
  model_args = dict(split_point=5, n_patches=4, width_mult=0.35, mode='replicate')
  return _create_mobilenet_split('mbv2_w35_split_5_p4_replicate', pretrained, **model_args)

@register_model
def mbv2_w35_split_5_p4_reflect(pretrained=False, **kwargs) -> MobileNetV2_split:
  model_args = dict(split_point=5, n_patches=4, width_mult=0.35, mode='reflect')
  return _create_mobilenet_split('mbv2_w35_split_5_p4_reflect', pretrained, model_args)
