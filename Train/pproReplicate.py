# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:42:44 2023

@author: kucha
"""

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F 

error = 0.7
def get_patch(x, pad_tuple, ph, pw):
    _, _, h, w = x.size()
    patch_h = h // ph
    patch_w = w // pw
    padded_x = F.pad(x, pad_tuple, mode='constant', value=0.0)
    l, r, _, _ = pad_tuple
    out = padded_x.unfold(2, patch_h + l + r, patch_h).unfold(3, patch_w + l + r, patch_w)
    out = rearrange(out, "B C ph pw H W -> B ph pw C H W", ph=ph, pw=pw)
    return out

    
def replace_outlier(first, sec, pred, target, e):
    mae = torch.abs(first - sec)
    #e = mae.median(axis=[1, 2, -1, -2], keepdim=True)
    #e = mae.quantile(axis=[1 , 2, -1, -2]).unsqueeze(dim=[1, 2, -1, -2])
    #print(e.shape)
    zeros = torch.zeros_like(target)
    result = torch.where(mae < e, pred, )
    return result

def sampling_top(pred, target, mask):
    result = pred.clone()
    result[:, :, :, :, :, 0::2] = target[:, :, :, :, :, 0::2]
    return result

def sampling_left(pred, target, mask):
    result = pred.clone()
    result[:, :, :, :, 0::2, :] = target[:, :, :, :, ::2, :]
    return result
a="""
def replace_module_by_names(model, modules_to_replace):
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

def find_modules_to_change(model):
    replaced_modules = {}
    for n, m in model.named_modules():
        if len(replaced_modules) > 4:
            break
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (3, 3):
                replaced_modules[n] = Mean_2px_function(m, 1, 4)
    

    return replaced_modules
"""
class Mean_2px_stride1(nn.Module):
    def __init__(self, ConvModule, padding, num_patches):
        super(Mean_2px_stride1, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.padding = padding
        self.num_patches = num_patches
        self.groups = self.mid_conv.groups

        #self.top_border_weight = torch.nn.Parameter(self.mid_conv.weight)
        #self.bot_border_weight = torch.nn.Parameter(self.mid_conv.weight)
        #self.left_border_weight = torch.nn.Parameter(self.mid_conv.weight)
        #self.right_border_weight = torch.nn.Parameter(self.mid_conv.weight)

        #self.top_left_border_weight = torch.nn.Parameter(self.mid_conv.weight)
        #self.bot_left_border_weight = torch.nn.Parameter(self.mid_conv.weight)
        #self.top_right_border_weight = torch.nn.Parameter(self.mid_conv.weight)
        #self.bot_right_border_weight = torch.nn.Parameter(self.mid_conv.weight)
    def extra_repr(self):
        s = (f'{Mean_2px_stride1}: num-patches={self.num_patches}')
        return s.format(**self.__dict__)

    def forward(self, x):
        x_patches = rearrange(x, "B C (ph H) (pw W) -> B ph pw C H W", ph=self.num_patches, pw=self.num_patches)
       
        op = get_patch(x, (1, 1, 1, 1), self.num_patches, self.num_patches)

        target_top = op[:, :, :, :, :1, 1:-1]
        target_bot = op[:, :, :, :, -1:, 1:-1]
        target_left = op[:, :, :, :, 1:-1, :1]
        target_right = op[:, :, :, :, 1:-1, -1:]

        target_top_left = op[:, :, :, :, :1, :1]
        target_top_right = op[:, :, :, :, :1, -1:]
        target_bot_left = op[:, :, :, :, -1:, :1]
        target_bot_right = op[:, :, :, :, -1:, -1:]

        pad_top = x_patches[:, :, :, :, :1, :].mean(dim=-2, keepdim=True)
        pad_top[:, 0, :, :, :, :] *= 0

        pad_down = x_patches[:, :, :, :, -1:, :].mean(dim=-2, keepdim=True)
        pad_down[:, -1, :, :, :, :] *= 0

        pad_left = x_patches[:, :, :, :, :, :1].mean(dim=-1, keepdim=True)
        pad_left[:, :, 0, :, :, :] *= 0

        pad_right = x_patches[:, :, :, :, :, -1:].mean(dim=-1, keepdim=True)
        pad_right[:, :, -1, :, :, :] *= 0

        pad_topleft = x_patches[:, :, :, :, 0, 0].clone().unsqueeze(dim=-1).unsqueeze(-1)
        pad_topleft[:, 0, :, :, :, :] *= 0
        pad_topleft[:, :, 0, :, :, :] *= 0

        pad_topright = x_patches[:, :, :, :, 0, -1].clone().unsqueeze(dim=-1).unsqueeze(-1)
        pad_topright[:, 0, :, :, :, :] *= 0
        pad_topright[:, :, -1, :, :, :] *= 0

        
        pad_botleft = x_patches[:, :, :, :, -1, 0].clone().unsqueeze(dim=-1).unsqueeze(-1)
        pad_botleft[:, -1, :, :, :, :] *= 0
        pad_botleft[:, :, 0, :, :, :] *= 0
        
        pad_botright = x_patches[:, :, :, :, -1, -1].clone().unsqueeze(dim=-1).unsqueeze(-1)
        pad_botright[:, -1, :, :, :, :] *= 0
        pad_botright[:, :, -1, :, :, :] *= 0


        #print(pad_top.size(), target_top.size())
        #pad_top = replace_outlier(x_patches[:, :, :, :, :1, :], x_patches[:, :, :, :, 1:2, :], pad_top, target_top, error)
        #pad_down = replace_outlier(x_patches[:, :, :, :, -1:, :], x_patches[:, :, :, :, -2:-1, :],pad_down, target_bot, error)
        #pad_left = replace_outlier(x_patches[:, :, :, :, :, :1], x_patches[:, :, :, :, :, 1:2], pad_left, target_left, error)
        #pad_right = replace_outlier(x_patches[:, :, :, :, :, -1:], x_patches[:, :, :, :, :, -2:-1], pad_right, target_right, error)

        if self.spatial:
            pad_top = sampling_top(pad_top, target_top, self.mask)
            #pad_down = sampling_top(pad_down, target_bot, self.mask)
            pad_left = sampling_left(pad_left, target_left, self.mask)
            #pad_right = sampling_left(pad_right, target_right, self.mask)

            pad_topleft = target_top_left.clone()
            #pad_topright = target_top_right.clone()
            #pad_botleft = target_bot_left.clone()
            #pad_botright = target_bot_right.clone()


        pad_ex_top = torch.cat([pad_topleft, pad_top, pad_topright], dim=-1)
        pad_ex_down = torch.cat([pad_botleft, pad_down, pad_botright], dim=-1)
        x_patches = torch.cat([pad_left, x_patches, pad_right], dim = -1)

        x_patches = torch.cat([pad_ex_top, x_patches, pad_ex_down], dim = -2)
        x_patches = rearrange(x_patches, "B ph pw C H W -> (B ph pw) C H W", ph=self.num_patches, pw=self.num_patches)

        out = self.mid_conv(x_patches)
        a="""
        Mid = self.mid_conv(x_patches[:, :, 1:-1, 1:-1])
        out_top = F.conv2d(x_patches[:, :, :3, 1:-1], self.top_border_weight, groups=self.groups)
        out_bot = F.conv2d(x_patches[:, :, -3:, 1:-1], self.bot_border_weight, groups=self.groups)
        out_left = F.conv2d(x_patches[:, :, 1:-1, :3], self.left_border_weight, groups=self.groups)
        out_right = F.conv2d(x_patches[:, :, 1:-1, -3:], self.right_border_weight, groups=self.groups)

        out_top_left = F.conv2d(x_patches[:, :, :3, :3], self.top_left_border_weight, groups=self.groups)
        out_top_right = F.conv2d(x_patches[:, :, :3, -3:], self.top_right_border_weight, groups=self.groups)
        out_bot_left = F.conv2d(x_patches[:, :, -3:, :3], self.bot_left_border_weight, groups=self.groups)
        out_bot_right = F.conv2d(x_patches[:, :, -3:, -3:], self.bot_right_border_weight, groups=self.groups)

        out_ex_top = torch.cat([out_top_left, out_top, out_top_right], dim=-1)
        out_ex_down = torch.cat([out_bot_left, out_bot, out_bot_right], dim=-1)
        Mid = torch.cat([out_left, Mid, out_right], dim=-1)
        out = torch.cat([out_ex_top, Mid, out_ex_down], dim=-2)
        """
        out = rearrange(out, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.num_patches, pw=self.num_patches)
        
        return out

class Mean_2px_stride2(nn.Module):
    def __init__(self, ConvModule, padding, num_patches):
        super(Mean_2px_stride2, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.padding = padding
        self.num_patches = num_patches
        self.groups = self.mid_conv.groups
        #self.top_border_weight = torch.nn.Parameter(self.mid_conv.weight)
        #self.left_border_weight = torch.nn.Parameter(self.mid_conv.weight)
        #self.top_left_border_weight = torch.nn.Parameter(self.mid_conv.weight)

    def extra_repr(self):
        s = (f'{Mean_2px_stride2}: num-patches={self.num_patches}')
        return s.format(**self.__dict__)

    def forward(self, x):
        x_patches = rearrange(x, "B C (ph H) (pw W) -> B ph pw C H W", ph=self.num_patches, pw=self.num_patches)

        op = get_patch(x, (1, 0, 1, 0), self.num_patches, self.num_patches)

        target_top = op[:, :, :, :, :1, 1:]
        target_left = op[:, :, :, :, 1:, :1]

        target_top_left = op[:, :, :, :, :1, :1]
        pad_top = x_patches[:, :, :, :, :1, :].mean(dim=-2, keepdim=True)
        pad_top[:, 0, :, :, :, :] *= 0

        pad_left = x_patches[:, :, :, :, :, :1].mean(dim=-1, keepdim=True)
        pad_left[:, :, 0, :, :, :] *= 0

        pad_top_left = x_patches[:, :, :, :, 0, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
        pad_top_left[:, 0, :, :, :, :] *= 0
        pad_top_left[:, :, 0, :, :, :] *= 0

        #print(pad_top.size(), target_top.size())
        if self.spatial:
            pad_top = sampling_top(pad_top, target_top, self.mask)
            pad_left = sampling_left(pad_left, target_left, self.mask)
            pad_top_left = target_top_left.clone()

        x_patches = torch.cat([pad_left, x_patches], dim=-1)
        pad_ex_top = torch.cat([pad_top_left, pad_top], dim=-1)
        x_patches = torch.cat([pad_ex_top, x_patches], dim=-2)

        x_patches = rearrange(x_patches, "B ph pw C H W -> (B ph pw) C H W", ph=self.num_patches, pw=self.num_patches)
        out = self.mid_conv(x_patches)
        a="""
        Mid = self.mid_conv(x_patches[:, :, 2:, 2:])
        out_top = F.conv2d(x_patches[:, :, :3, 2:], self.top_border_weight, groups=self.groups, stride=2)
        out_left = F.conv2d(x_patches[:, :, 2:, :3], self.left_border_weight, groups=self.groups, stride=2)
        out_top_left = F.conv2d(x_patches[:, :, :3, :3], self.top_left_border_weight, groups=self.groups, stride=2)

        Mid = torch.cat([out_left, Mid], dim=-1)
        out_ex_top = torch.cat([out_top_left, out_top], dim=-1)
        out = torch.cat([out_ex_top, Mid], dim=-2)
        """
        out = rearrange(out, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.num_patches, pw=self.num_patches)
        return out

module_to_mapping = {
        (3, 3, 1): Mean_2px_stride1,
        (3, 3, 2): Mean_2px_stride2,
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

def get_attr(layer):
    k = layer.kernel_size
    s = layer.stride

    if isinstance(k, int):
        k = (k, k)
    if isinstance(s, tuple):
        s = s[0]

    return (*k, s)

def change_model_list(model, num_patches, Module_To_Mapping, patch_list):
    i = 0
    for name, target in model.named_modules():
        if i == len(patch_list):
            return model
        if is_spatial(target):
            i += 1
            if patch_list[i-1] == 1:
                continue
            attrs = name.split('.')
            submodule = model
            for attr in attrs[:-1]:
                submodule = getattr(submodule, attr)
            pad = target.padding
            if isinstance(target.padding, tuple):
                pad = pad[0]
            replace = Module_To_Mapping[get_attr(target)](target, pad, num_patches)
            if i in [1, 2]:
                replace.spatial = False
            else:
                replace.spatial = True
                if i in [3, 5]:
                    replace.mask = True
                else:
                    replace.mask = True
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
            replace = Module_To_Mapping[get_attr(target)](target, pad, num_patches)
            setattr(submodule, attrs[-1], replace)

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v2 as Net
    net = Net()
    num_patches = 4
    num_per_patch_stage = 5
    change_model(net, num_patches, module_to_mapping, num_per_patch_stage)
    #print(net)

