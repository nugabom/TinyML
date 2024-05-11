from itertools import product
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

def padding_prediction_all(patch):
    # Border
    a="""
    pad_top = patch[:, :, :1, :].mean(dim=-2, keepdim=True)
    pad_down = patch[:, :, -1:, :].mean(dim=-2, keepdim=True)
    pad_left = patch[:, :, :, :1].mean(dim=-1, keepdim=True)
    pad_right = patch[:, :, :, -1:].mean(dim=-1, keepdim=True)

    # Corner
    pad_topleft = patch[:, :, 0, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
    pad_topright = patch[:, :, 0, -1].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
    pad_botleft = patch[:, :, -1, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
    pad_botright = patch[:, :, -1, -1].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)

    pad_ex_top = torch.cat([pad_topleft, pad_top, pad_topright], dim=-1)
    mid = torch.cat([pad_left, patch, pad_right], dim=-1)
    pad_ex_bot = torch.cat([pad_botleft, pad_down, pad_botright], dim=-1)

    return torch.cat([pad_ex_top, mid, pad_ex_bot], dim=-2)
    """
    out = F.pad(patch, (1, 1, 1, 1), mode='reflect')
    return out
    

def padding_prediction_S2_remove_right(patch):
    # Border
    a="""
    pad_top = patch[:, :, :1, :].mean(dim=-2, keepdim=True)
    pad_down = patch[:, :, -1:, :].mean(dim=-2, keepdim=True)
    pad_left = patch[:, :, :, :1].mean(dim=-1, keepdim=True)

    # Corner
    pad_topleft = patch[:, :, 0, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
    pad_botleft = patch[:, :, -1, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)

    pad_ex_top = torch.cat([pad_topleft, pad_top], dim=-1)
    mid = torch.cat([pad_left, patch], dim=-1)
    pad_ex_bot = torch.cat([pad_botleft, pad_down], dim=-1)

    return torch.cat([pad_ex_top, mid, pad_ex_bot], dim=-2)
    """
    out = F.pad(patch, (1, 0, 1, 1), mode='reflect')
    return out

def padding_prediction_S2_remove_bot(patch):
    # Border
    a="""
    pad_top = patch[:, :, :1, :].mean(dim=-2, keepdim=True)
    pad_left = patch[:, :, :, :1].mean(dim=-1, keepdim=True)
    pad_right = patch[:, :, :, -1:].mean(dim=-1, keepdim=True)

    # Corner
    pad_topleft = patch[:, :, 0, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
    pad_topright = patch[:, :, 0, -1].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)


    pad_ex_top = torch.cat([pad_topleft, pad_top, pad_topright], dim=-1)
    mid = torch.cat([pad_left, patch, pad_right], dim=-1)

    return torch.cat([pad_ex_top, mid], dim=-2)
    """
    out = F.pad(patch, (1, 1, 1, 0), mode='reflect')
    return out

def padding_prediction_S2_remove_botright(patch):
    # Border
    a="""
    pad_top = patch[:, :, :1, :].mean(dim=-2, keepdim=True)
    pad_left = patch[:, :, :, :1].mean(dim=-1, keepdim=True)

    # Corner
    pad_topleft = patch[:, :, 0, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)

    pad_ex_top = torch.cat([pad_topleft, pad_top], dim=-1)
    mid = torch.cat([pad_left, patch], dim=-1)

    return torch.cat([pad_ex_top, mid], dim=-2)
    """
    out = F.pad(patch, (1, 0, 1, 0), mode='reflect')
    return out

def padding_prediction_S2_remove_left(patch):
    a="""
    pad_top = patch[:, :, :1, :].mean(dim=-2, keepdim=True)
    pad_down = patch[:, :, -1:, :].mean(dim=-2, keepdim=True)
    pad_right = patch[:, :, :, -1:].mean(dim=-1, keepdim=True)

    pad_topright = patch[:, :, 0, -1].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
    pad_botright = patch[:, :, -1, -1].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)

    pad_ex_top = torch.cat([pad_top, pad_topright], dim=-1)
    mid = torch.cat([patch, pad_right], dim=-1)
    pad_ex_bot = torch.cat([pad_down, pad_botright], dim=-1)

    return torch.cat([pad_ex_top, mid, pad_ex_bot], dim=-2)
    """
    out = F.pad(patch, (0, 1, 1, 1), mode='reflect')
    return out

def padding_prediction_S2_remove_leftright(patch):
    a="""
    pad_top = patch[:, :, -1:, :].mean(dim=-2, keepdim=True)
    pad_down = patch[:, :, :1, :].mean(dim=-2, keepdim=True)

    return torch.cat([pad_top, patch, pad_down], dim=-2)
    """
    out = F.pad(patch, (0, 0, 1, 1), mode='reflect')
    return out

def padding_prediction_S2_remove_top(patch):
    a="""
    pad_left = patch[:, :, :, :1].mean(dim=-1, keepdim=True)
    pad_right = patch[:, :, :, -1:].mean(dim=-1, keepdim=True)
    pad_down = patch[:, :, -1:, :].mean(dim=-2, keepdim=True)

    pad_botleft = patch[:, :, -1, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
    pad_botright = patch[:, :, -1, -1].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)

    mid = torch.cat([pad_left, patch, pad_right], dim=-1)
    pad_ex_bot = torch.cat([pad_botleft, pad_down, pad_botright], dim=-1)
    return torch.cat([mid, pad_ex_bot], dim=-2)
    """
    out = F.pad(patch, (1, 1, 0, 1), mode='reflect')
    return out

def padding_prediction_S2_remove_topleft(patch):
    a="""
    pad_right = patch[:, :, :, -1:].mean(dim=-1, keepdim=True)
    pad_down = patch[:, :, -1:, :].mean(dim=-2, keepdim=True)
    
    pad_botright = patch[:, :, -1, -1].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)

    mid = torch.cat([patch, pad_right], dim=-1)
    pad_ex_bot = torch.cat([pad_down, pad_botright], dim=-1)
    return torch.cat([mid, pad_ex_bot], dim=-2)
    """
    out = F.pad(patch, (0, 1, 0, 1), mode='reflect')
    return out

def padding_prediction_S2_remove_topleftright(patch):
    a="""
    pad_bot = patch[:, :, -1:, :].mean(dim=-2, keepdim=True)

    return torch.cat([patch, pad_bot], dim=-2)
    """
    out = F.pad(patch, (0, 0, 0, 1), mode='reflect')
    return out

def padding_prediction_S2_remove_topbot(patch):
    a="""
    pad_left = patch[:, :, :, :1].mean(dim=-1, keepdim=True)
    pad_right = patch[:, :, :, -1:].mean(dim=-1, keepdim=True)
    return torch.cat([pad_left, patch, pad_right], dim=-1)
    """
    out = F.pad(patch, (1, 1, 0, 0), mode='reflect')
    return out

def padding_prediction_S2_remove_topbotleft(patch):
    a="""
    pad_right = patch[:, :, :, -1:].mean(dim=-1, keepdim=True)
    return torch.cat([patch, pad_right], dim=-1)
    """
    out = F.pad(patch, (0, 1, 0, 0), mode='reflect')
    return out

class PPConv2d_S1(nn.Module):
    def __init__(self, ConvModule, padding, n_patch):
        super(PPConv2d_S1, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.groups = ConvModule.groups
        self.ph = n_patch
        self.pw = n_patch
        self.layer_id = self.mid_conv.layer_id
        self.patch_start = None
        self.kernel_size = self.mid_conv.kernel_size[0]
        self.padding = self.kernel_size // 2

    def extra_repr(self):
        s = (f"PPConv2d_S1: #patch = {self.ph}, padding={0, 0}")
        return s.format(**self.__dict__)

    def forward(self, x):
        ph = self.ph
        pw = self.pw

        ps = x.size()[-1] // ph

        p0 =  x[:, :, :self.patch_start, :self.patch_start]
        p1 =  x[:, :, :self.patch_start, self.patch_start:self.patch_start + ps *(pw -2)]
        p2 =  x[:, :, :self.patch_start, self.patch_start + ps * (pw - 2):]

        p3 =  x[:, :, self.patch_start:self.patch_start + ps * (ph - 2), :self.patch_start]
        p4 =  x[:, :, self.patch_start:self.patch_start + ps * (ph - 2), self.patch_start:self.patch_start + ps *(pw -2)]
        p5 =  x[:, :, self.patch_start:self.patch_start + ps * (ph - 2), self.patch_start + ps * (pw - 2):]

        p6 =  x[:, :, self.patch_start + ps * (ph - 2):, :self.patch_start]
        p7 =  x[:, :, self.patch_start + ps * (ph - 2):, self.patch_start:self.patch_start + ps *(pw -2)]
        p8 =  x[:, :, self.patch_start + ps * (ph - 2):, self.patch_start + ps * (pw - 2):]

      
        # top 
        o0 = padding_prediction_all(p0)
        o0[:, :, 0, :] *= 0.0
        o0[:, :, :, 0] *= 0.0
        o0 = self.mid_conv(o0)
        #o0 = self.mid_conv(F.pad(p0, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0.0))

        o1 = rearrange(p1, "B C H (n_patch W) -> (B n_patch) C H W", n_patch=pw-2)
        o1 = padding_prediction_all(o1)
        o1[:, :, 0, :] *= 0.0
        o1 = self.mid_conv(o1)
        #o1 = self.mid_conv(F.pad(o1, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0.0))
        o1 = rearrange(o1, "(B n_patch) C H W -> B C H (n_patch W)", n_patch=pw-2)

        o2 = padding_prediction_all(p2)
        o2[:, :, 0, :] *= 0.0
        o2[:, :, :, -1] *= 0.0
        o2 = self.mid_conv(o2)
        #o2 = self.mid_conv(F.pad(p2, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0.0))

        # mid
        o3 = rearrange(p3, "B C (n_patch H) W -> (B n_patch) C H W", n_patch=pw-2)
        o3 = padding_prediction_all(o3)
        o3[:, :, :, 0] *= 0.0
        o3 = self.mid_conv(o3)
        #o3 = self.mid_conv(F.pad(o3, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0.0))
        o3 = rearrange(o3, "(B n_patch) C H W -> B C (n_patch H) W", n_patch=pw-2)

        o4 = rearrange(p4, "B C (n_patch H) (n_patch1 W) -> (B n_patch n_patch1) C H W", n_patch=pw-2, n_patch1=pw-2)
        o4 = padding_prediction_all(o4)
        o4 = self.mid_conv(o4)
        #o4 = self.mid_conv(F.pad(o4, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0.0))
        o4 = rearrange(o4, "(B n_patch n_patch1) C H W -> B C (n_patch H) (n_patch1 W)", n_patch=pw-2, n_patch1=pw-2)

        o5 = rearrange(p5, "B C (n_patch H) W -> (B n_patch) C H W", n_patch=pw-2)
        o5 = padding_prediction_all(o5)
        o5[:, :, :, -1] *= 0.0
        o5 = self.mid_conv(o5)
        #o5 = self.mid_conv(F.pad(o5, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0.0))
        o5 = rearrange(o5, "(B n_patch) C H W -> B C (n_patch H) W", n_patch=pw-2)

        # bot
        o6 = padding_prediction_all(p6)
        o6[:, :, -1, :] *= 0.0
        o6[:, :, :, 0] *= 0.0
        o6 = self.mid_conv(o6)
        #o6 = self.mid_conv(F.pad(p6, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0.0))

        o7 = rearrange(p7, "B C H (n_patch W) -> (B n_patch) C H W", n_patch=pw-2)
        o7 = padding_prediction_all(o7)
        o7[:, :, -1, :] *= 0.0
        o7 = self.mid_conv(o7)
        #o7 = self.mid_conv(F.pad(o7, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0.0))
        o7 = rearrange(o7, "(B n_patch) C H W -> B C H (n_patch W)", n_patch=pw-2)

        o8 = padding_prediction_all(p8)
        o8[:, :, :, -1] *= 0.0
        o8[:, :, -1, :] *= 0.0
        o8 = self.mid_conv(o8)
        #o8 = self.mid_conv(F.pad(p8, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0.0))

        row0 = torch.cat([o0, o1, o2], dim=-1)
        row1 = torch.cat([o3, o4, o5], dim=-1)
        row2 = torch.cat([o6, o7, o8], dim=-1)

        out = torch.cat([row0, row1, row2], dim=-2)

        return out


class PPConv2d_S2(nn.Module):
    def __init__(self, ConvModule, padding, n_patch):
        super(PPConv2d_S2, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.groups = ConvModule.groups
        self.ph = n_patch
        self.pw = n_patch
        self.patch_start = None
        self.layer_id = self.mid_conv.layer_id
        self.kernel_size = self.mid_conv.kernel_size[0]
        self.padding = self.kernel_size // 2
        self.odd = None

    def extra_repr(self):
        s = (f"PPConv2d_S2: #patch = {self.ph}, padding={0, 0}")
        return s.format(**self.__dict__)

    def forward(self, x):
        ph = self.ph
        pw = self.pw

        ps = x.size()[-1] // ph

        p0 =  x[:, :, :self.patch_start, :self.patch_start]
        p1 =  x[:, :, :self.patch_start, self.patch_start:self.patch_start + ps *(pw -2)]
        p2 =  x[:, :, :self.patch_start, self.patch_start + ps * (pw - 2):]

        p3 =  x[:, :, self.patch_start:self.patch_start + ps * (ph - 2), :self.patch_start]
        p4 =  x[:, :, self.patch_start:self.patch_start + ps * (ph - 2), self.patch_start:self.patch_start + ps *(pw -2)]
        p5 =  x[:, :, self.patch_start:self.patch_start + ps * (ph - 2), self.patch_start + ps * (pw - 2):]

        p6 =  x[:, :, self.patch_start + ps * (ph - 2):, :self.patch_start]
        p7 =  x[:, :, self.patch_start + ps * (ph - 2):, self.patch_start:self.patch_start + ps *(pw -2)]
        p8 =  x[:, :, self.patch_start + ps * (ph - 2):, self.patch_start + ps * (pw - 2):]

        self.odd = p0.size()[-1] % 2 == 1

        if self.odd:
            # top
            o0 = padding_prediction_all(p0)
            o0[:,:, 0, :] *= 0.0
            o0[:, :, :, 0] *= 0.0
            o0 = self.mid_conv(o0)

            o1 = rearrange(p1, "B C H (n_patch W) -> (B n_patch) C H W", n_patch=pw-2)
            o1 = padding_prediction_S2_remove_left(o1)
            o1[:, :, 0, :] *= 0.0
            o1 = self.mid_conv(o1)
            o1 = rearrange(o1, "(B n_patch) C H W -> B C H (n_patch W)", n_patch=pw-2)

            o2 = padding_prediction_S2_remove_leftright(p2)
            o2[:, :, 0, :] *= 0.0
            o2 = self.mid_conv(o2)

            # mid
            o3 = rearrange(p3, "B C (n_patch H) W -> (B n_patch) C H W", n_patch=pw-2)
            o3 = padding_prediction_S2_remove_top(o3)
            o3[:, :, :, 0] *= 0.0
            o3 = self.mid_conv(o3)
            o3 = rearrange(o3, "(B n_patch) C H W -> B C (n_patch H) W", n_patch=pw-2)

            o4 = rearrange(p4, "B C (n_patch H) (n_patch1 W) -> (B n_patch n_patch1) C H W", n_patch=pw-2, n_patch1=pw-2)
            o4 = padding_prediction_S2_remove_topleft(o4)
            o4 = self.mid_conv(o4)
            o4 = rearrange(o4, "(B n_patch n_patch1) C H W -> B C (n_patch H) (n_patch1 W)", n_patch=pw-2, n_patch1=pw-2)
        
            o5 = rearrange(p5, "B C (n_patch H) W -> (B n_patch) C H W", n_patch=pw-2)
            o5 = padding_prediction_S2_remove_topleftright(o5)
            o5 = self.mid_conv(o5)
            o5 = rearrange(o5, "(B n_patch) C H W -> B C (n_patch H) W", n_patch=pw-2)

            # bot
            o6 = padding_prediction_S2_remove_topbot(p6)
            o6[:, :, :, 0] *= 0.0
            o6 = self.mid_conv(o6)

            o7 = rearrange(p7, "B C H (n_patch W) -> (B n_patch) C H W", n_patch=pw-2)
            o7 = padding_prediction_S2_remove_topbotleft(o7)
            o7 = self.mid_conv(o7)
            o7 = rearrange(o7, "(B n_patch) C H W -> B C H (n_patch W)", n_patch=pw-2)

            o8 = self.mid_conv(p8)
        else:
                    # top
            o0 = padding_prediction_S2_remove_botright(p0)
            o0[:,:, 0, :] *= 0.0
            o0[:, :, :, 0] *= 0.0
            o0 = self.mid_conv(o0)

            o1 = rearrange(p1, "B C H (n_patch W) -> (B n_patch) C H W", n_patch=pw-2)
            o1 = padding_prediction_S2_remove_botright(o1)
            o1[:, :, 0, :] *= 0.0
            o1 = self.mid_conv(o1)
            o1 = rearrange(o1, "(B n_patch) C H W -> B C H (n_patch W)", n_patch=pw-2)

            o2 = padding_prediction_S2_remove_botright(p2)
            o2[:, :, 0, :] *= 0.0
            o2 = self.mid_conv(o2)

            # mid
            o3 = rearrange(p3, "B C (n_patch H) W -> (B n_patch) C H W", n_patch=pw-2)
            o3 = padding_prediction_S2_remove_botright(o3)
            o3[:, :, :, 0] *= 0.0
            o3 = self.mid_conv(o3)
            o3 = rearrange(o3, "(B n_patch) C H W -> B C (n_patch H) W", n_patch=pw-2)

            o4 = rearrange(p4, "B C (n_patch H) (n_patch1 W) -> (B n_patch n_patch1) C H W", n_patch=pw-2, n_patch1=pw-2)
            o4 = padding_prediction_S2_remove_botright(o4)
            o4 = self.mid_conv(o4)
            o4 = rearrange(o4, "(B n_patch n_patch1) C H W -> B C (n_patch H) (n_patch1 W)", n_patch=pw-2, n_patch1=pw-2)
        
            o5 = rearrange(p5, "B C (n_patch H) W -> (B n_patch) C H W", n_patch=pw-2)
            o5 = padding_prediction_S2_remove_botright(o5)
            o5 = self.mid_conv(o5)
            o5 = rearrange(o5, "(B n_patch) C H W -> B C (n_patch H) W", n_patch=pw-2)

            # bot
            o6 = padding_prediction_S2_remove_botright(p6)
            o6[:, :, :, 0] *= 0.0
            o6 = self.mid_conv(o6)

            o7 = rearrange(p7, "B C H (n_patch W) -> (B n_patch) C H W", n_patch=pw-2)
            o7 = padding_prediction_S2_remove_botright(o7)
            o7 = self.mid_conv(o7)
            o7 = rearrange(o7, "(B n_patch) C H W -> B C H (n_patch W)", n_patch=pw-2)

            o8 = padding_prediction_S2_remove_botright(p8)
            o8 = self.mid_conv(o8)


        row0 = torch.cat([o0, o1, o2], dim=-1)
        row1 = torch.cat([o3, o4, o5], dim=-1)
        row2 = torch.cat([o6, o7, o8], dim=-1)

        out = torch.cat([row0, row1, row2], dim=-2)

        return out

        
def get_attr(layer):
    k = layer.kernel_size[0]
    s = layer.stride[0]
    return (k, s)
    
def change_training_model(model, num_patches, Module_To_Mapping, patch_list):
    i = 0
    config = []
    for n, target in model.named_modules():
        if i == len(patch_list):
            break
        if isinstance(target, nn.Conv2d) and target.kernel_size[0] > 1:
            attrs = n.split('.')
            submodule = model
            for attr in attrs[:-1]:
                submodule = getattr(submodule, attr)
            (k, s) = get_attr(target)
            if patch_list[i] == 0:
                replace = Module_To_Mapping[(k, s, patch_list[i])](target, target.padding, num_patches)
                setattr(submodule, attrs[-1], replace)

            config.append((patch_list[i], s))
            i += 1

    return config

def set_numbering(model):
    i = 0
    for n, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            mod.layer_id = i
            i += 1

def get_offset(conf, patch_size):
    offsets = [patch_size]
    feat_size = patch_size
    hb_size = patch_size
    for sconv in conf[:-1]:
        buf, stride = sconv
        feat_size = (feat_size - 3 + 2 - (stride-1)) // stride + 1
        hb_size = (hb_size - 3 + 2 - buf - (stride-1)) // stride + 1
        offset = feat_size - hb_size
        offsets.append(hb_size)
    offsets = offsets
    return offsets

def is_my_conv(layer):
    if (isinstance(layer, nn.Conv2d) and layer.kernel_size[0] > 1):
        return True
    elif isinstance(layer, (PPConv2d_S2, PPConv2d_S1)):
        return True
    return False

def set_start_point(model, start_point):
    i = 0
    exclude = []
    for n, mod in model.named_modules():
        if hasattr(mod, 'mid_conv'):
            exclude.append(mod.mid_conv)

    for n, mod in model.named_modules():
        if mod in exclude:
            continue
        if i == len(start_point):
            return 
        if is_my_conv(mod):
            mod.patch_start = start_point[i]
            #setattr(mod, 'patch_start', start_point[i])
            print(str(type(mod)), mod.patch_start)
            i += 1

module_to_mapping = {
    (3, 1, 0): PPConv2d_S1,
    (3, 2, 0): PPConv2d_S2,
}
