import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision import datasets, transforms
from torchvision.models._utils import _make_divisible
import numpy as np
import os

def replace_module_by_names(model, modules_to_replace, Module_mapping):
    def helper(child: nn.Module):
        for n, c in child.named_children():
            if type(c) in Module_mapping.keys():
                for full_name, m in model.named_modules():
                    if c is m and full_name in modules_to_replace:
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)
    helper(model)
    return model

def find_modules_to_change_stride2(model):
    replaced_modules = {}
    for n, m in model.named_modules():
        if len(replaced_modules) > 2:
            break
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (3, 3) and m.stride == (2, 2):
                replaced_modules[n] = MyConv_function_stride2(m, 4)
    

    return replaced_modules

def find_modules_to_change_stride1(model):
    replaced_modules = {}
    for n, m in model.named_modules():
        if len(replaced_modules) > 1:
            break
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (3, 3) and m.stride == (1, 1):
                replaced_modules[n] = MyConv_function_stride1(m, 4)
    

    return replaced_modules




class MyConv_function_stride1(nn.Module):
    def __init__(self, ConvModule, padding,  num_patches):
        super(MyConv_function_stride1, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.num_patches = num_patches
        self.groups = ConvModule.groups
        self.padding = padding
        
    def forward(self, x):
        _, _, LH, LW = x.size()
        x_patches = rearrange(x, "B C (ph H) (pw W) -> (B ph pw) C H W", ph=self.num_patches, pw=self.num_patches)
        _, _, SH, SW = x_patches.size()
        assert LH == SH * self.num_patches, 'error'
        padded_x_patches = F.pad(x_patches, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0.0)
        out = self.mid_conv(padded_x_patches)
        out = rearrange(out, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.num_patches, pw=self.num_patches)
        return out
    
class MyConv_function_stride2(nn.Module):
    def __init__(self, ConvModule, padding, num_patches):
        super(MyConv_function_stride2, self).__init__()
        self.mid_conv = ConvModule
        self.num_patches = num_patches
        self.mid_conv.padding = (0, 0)
        self.groups = ConvModule.groups
        self.padding = padding
        
    def forward(self, x):
        _, _, LH, LW = x.size()
        x_patches = rearrange(x, "B C (ph H) (pw W) -> (B ph pw) C H W", ph=self.num_patches, pw=self.num_patches)
        _, _, SH, SW = x_patches.size()
        assert LH == SH * self.num_patches, 'error'
        padded_x_patches = F.pad(x_patches, (self.padding, self.padding - 1, self.padding, self.padding - 1), mode='constant', value=0.0)
        out = self.mid_conv(padded_x_patches)
        out = rearrange(out, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.num_patches, pw=self.num_patches)
        return out
    
Module_stride1_mapping = {
    nn.Conv2d : MyConv_function_stride1
}

Module_stride2_mapping = {
    nn.Conv2d : MyConv_function_stride2
}

module_to_mapping = {
    (1, 1): MyConv_function_stride1,
    (2, 2): MyConv_function_stride2
}

def is_spatial(layer):
    if isinstance(layer, nn.Conv2d):
        k = None
        if isinstance(layer.kernel_size, tuple):
            k = layer.kernel_size[0]
        else:
            pass
        if k > 1:
            return True
        return False
    return False

def get_stride_layer(layer):
    s = layer.stride
    if isinstance(s, int):
        s = (s, s)
    return s

def change_model_list(model, num_patches, Module_To_Mapping, patch_list):
    i = 0
    for name, target in model.named_modules():
        if i == len(patch_list):
            return model
        if is_spatial(target):
            i += 1
            if patch_list[i - 1] == 1:
                continue
            attrs = name.split('.')
            submodule = model
            for attr in attrs[:-1]:
                submodule = getattr(submodule, attr)
            pad = target.padding
            if isinstance(target.padding, tuple):
                pad = pad[0]
            replace = Module_To_Mapping[get_stride_layer(target)](target, pad, num_patches)
            setattr(submodule, attrs[-1], replace)

def change_model(model, num_patches, Module_To_Mapping, num_per_patch_stage):
    i = 0
    for name, target in model.named_modules():
        if i == num_per_patch_stage:
            return model
        if is_spatial(target):
            i += 1
            attrs = name.split('.')
            submodule = model
            for attr in attrs[:-1]:
                submodule = getattr(submodule, attr)
            pad = target.padding
            if isinstance(target.padding, tuple):
                pad = pad[0]
            replace = Module_To_Mapping[get_stride_layer(target)](target, pad, num_patches)
            setattr(submodule, attrs[-1], replace)
            
def replace_layer(model, block_id, patch_type, modules_to_replace):
    if patch_type == 1:
        return
    layers = model.features[block_id]
    target = layers

    for name, layer in layers.named_modules():
        if is_spatial(layer):
            attrs = name.split('.')
            for attr in attrs[:-1]:
                target = getattr(target, attr)

            check_stride = get_stride_layer(layer)
            replace = modules_to_replace[check_stride](layer, layer.padding[0], 4)
            setattr(target, attrs[-1], replace)

#model = mobilenet_v2(pretrained=True)

#modules_to_replace = find_modules_to_change_stride1(model)
# print(modules_to_replace)
#replace_module_by_names(model, modules_to_replace, Module_stride1_mapping)
#modules_to_replace = find_modules_to_change_stride2(model)
# print(modules_to_replace)
#replace_module_by_names(model, modules_to_replace, Module_stride2_mapping)
#print(model)
#model(torch.randn(1, 3, 224, 224))

#%%
#a = [1, 2, 3, 4, 5]
#print(a[2:])
