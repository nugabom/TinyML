from itertools import product
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from Mean2px import *

torch.set_printoptions(linewidth=200, sci_mode=False, precision=3)
def set_patch_id(model, pid):
    for n, mod in model.named_modules():
        mod.patch_id = pid

class Bar(nn.Module):
    def __init__(self):
        super(Bar, self).__init__()
        self.conv0 = nn.Conv2d(3, 3, 3, 2, padding=1)
        self.conv1 = nn.Conv2d(3, 3, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(3, 3, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(3, 3, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(3, 3, 3, 2, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class Toy(nn.Module):
    def __init__(self):
        super(Toy, self).__init__()
        self.conv0 = nn.Conv2d(3, 3, 3, 2, padding=1)
        self.conv1 = nn.Conv2d(3, 3, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(3, 3, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(3, 3, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(3, 3, 3, 2, padding=1)

    def _forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def forward(self, x):
        ph = gpatch
        pw = gpatch
        
        if "NConv" not in str(type(self.conv0)):
            x = self.conv0(x)

        x = rearrange(x, "B C (ph H) (pw W) -> (B ph pw) C H W", ph=ph, pw=pw)
        out_dict = {}
        for pid, y in enumerate(x):
            set_patch_id(self,pid)
            y = y.unsqueeze(dim=0)
            if "NConv" in str(type(self.conv0)):
                y = self.conv0(y)
            #print('conv0', y.size())
            y = self.conv1(y)
            #print('conv1', y.size())
            y = self.conv2(y)
            #print('conv2', y.size())
            y = self.conv3(y)
            #print('conv3', y.size())
            y = self.conv4(y)
            #print('conv4', y.size())
            out_dict[f"out{pid}"] = y
       
        out = [] 
        for j in range(ph):
            row = []
            for i in range(pw):
                row.append(out_dict[f"out{j * ph + i}"])
            print('logging', [t.size() for t in row])
            row = torch.cat(row, dim=-1)
            out.append(row)
        out = torch.cat(out, dim=-2)
        return out

class ConvBuf1(nn.Module):
    def __init__(self, ConvModule, padding, n_patch):
        super(ConvBuf1, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0,0)
        self.patch_id = None
        self.region = {}
        self.pad = 2
        self.ph = n_patch
        self.pw = n_patch
        self.layer_id = self.mid_conv.layer_id
        self.patch_start = None
 
    def extra_repr(self):
        s = (f"(ConvBuf1)") 
        return s.format(**self.__dict__)

    def hcat(self, tensors, patch=None):
        if len(tensors) == 2 and patch is not None:
            assert 1==0, f"tensors: {len(tensors)} and patch is not None"
       
        if patch is not None:
            #print(f"bug check: {tensors[0]}", tensors[0] in self.region.keys())
            out = torch.cat([self.region[f"{tensors[0]}"], patch], dim=-1)

            del self.region[f"{tensors[0]}"]
            return out

        if len(tensors) == 2:
            out = torch.cat([self.region[f"{tensors[0]}"], self.region[f"{tensors[1]}"]], dim=-1)
            del self.region[f"{tensors[0]}"]
            del self.region[f"{tensors[1]}"]

            return out

    def vcat(self, tensor, patch):
        out = torch.cat([self.region[f"{tensor[0]}"], patch], dim=-2)

        del self.region[f"{tensor[0]}"]     
        
        return out
    
    def fcat(self, top_tensors, left, patch):
        top = self.hcat(top_tensors)
        patch = self.hcat(left, patch)

        return torch.cat([top, patch], dim=-2)

    def store(self, x):
        ph = self.ph
        pw = self.pw

        if self.patch_id in [ph * (pw - 1) + i for i in range(pw)]:
            self.region[f"r{self.patch_id}"] = x[:, :, :, -2:].clone()
        elif self.patch_id in [pw - 1 + ph * i for i in range(ph - 1)]:
            self.region[f"b{self.patch_id}"] = x[:, :, -2:, :].clone()
        else:
            self.region[f"r{self.patch_id}"] = x[:, :, :, -2:].clone()
            self.region[f"b{self.patch_id}"]= x[:, :, -2:, :].clone()
            self.region[f"br{self.patch_id}"] = x[:, :, -2:, -2:].clone()

    def forward(self, x):
        pid = self.patch_id
        ph = self.ph
        pw = self.pw

        if pid != ph * pw - 1:
            self.store(x)

        if pid == 0:
            print(f"{self.layer_id} (ConvBuf1): {x.size()}, {self.patch_start}")

        t=0;b=0;l=0;r=0;
        if pid in [i for i in range(pw)]:
            t = 1
        if pid in [ph * (pw - 1) + i for i in range(pw)]:
            b = 1
        if pid in [ph * i for i in range(ph)]:
            l = 1
        if pid in [pw - 1 + ph * i for i in range(ph)]:
            r = 1

        row, col = pid // ph, pid % pw
        #print('buf1', self.region.keys())
        if pid == 0:
            pass
        elif pid in [i for i in range(1, pw)]:
            x = self.hcat([f"r{col - 1}"], x)
        elif pid in [ph * i for i in range(1, ph)]:
            x = self.vcat([f"b{(row - 1) * ph}"], x)
        else:
            x = self.fcat([f"br{ (row - 1) * ph + col - 1}", f"b{(row - 1) * ph + col}"], [f"r{row * ph + col - 1}"], x)

        x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)

        if pid == pw * ph - 1:
            assert not self.region, "error"
        out = self.mid_conv(x)
        #if pid == 0:
        #    print(self.layer_id, 'stride1 ',out.size())
        return out

class ConvBuf2(nn.Module):
    def __init__(self, ConvModule, padding, n_patch):
        super(ConvBuf2, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0,0)
        self.patch_id = None
        self.buf = {}
        self.pad = 2
        self.region = {}
        self.ph = n_patch
        self.pw = n_patch
        self.e = None
        self.layer_id = self.mid_conv.layer_id
        self.patch_start = None
 
    def extra_repr(self):
        s = (f"(ConvBuf1)") 
        return s.format(**self.__dict__)

    def hcat(self, tensors, patch=None):
        if len(tensors) == 2 and patch is not None:
            assert 1==0, f"tensors: {len(tensors)} and patch is not None"
        
        #print(f"bug check: {tensors[0]}")
        if patch is not None:
            out = torch.cat([self.region[f"{tensors[0]}"], patch], dim=-1)

            del self.region[f"{tensors[0]}"]
            return out

        if len(tensors) == 2:
            out = torch.cat([self.region[f"{tensors[0]}"], self.region[f"{tensors[1]}"]], dim=-1)
            del self.region[f"{tensors[0]}"]
            del self.region[f"{tensors[1]}"]

            return out

    def vcat(self, tensor, patch):
        out = torch.cat([self.region[f"{tensor[0]}"], patch], dim=-2)

        del self.region[f"{tensor[0]}"]     
        
        return out
    
    def fcat(self, top_tensors, left, patch):
        top = self.hcat(top_tensors)
        patch = self.hcat(left, patch)

        return torch.cat([top, patch], dim=-2)

    def store(self, x):
        ph = self.ph
        pw = self.pw

        if self.patch_id in [ph * (pw - 1) + i for i in range(pw)]:
            self.region[f"r{self.patch_id}"] = x[:, :, :, -2 + self.e:].clone()
        elif self.patch_id in [pw - 1 + ph * i for i in range(ph - 1)]:
            self.region[f"b{self.patch_id}"] = x[:, :, -2 + self.e:, :].clone()
        else:
            self.region[f"r{self.patch_id}"] = x[:, :, :, -2 + self.e:].clone()
            self.region[f"b{self.patch_id}"] = x[:, :, -2 + self.e:, :].clone()
            self.region[f"br{self.patch_id}"]= x[:, :, -2 + self.e:, -2 + self.e:].clone()

    def forward(self, x):
        
        pid = self.patch_id
        ph = self.ph
        pw = self.pw

        if pid == 0:
            print(f"{self.layer_id} (ConvBuf2): {x.size()}, {self.patch_start}")

        if self.e is None:
            _, _, h, w = x.size()
            if h % 2 == 0:
                self.e = 1
            else:
                self.e = 0

        if pid != ph * pw - 1:
            self.store(x)

        t=0;b=0;l=0;r=0;
        
        if pid in [i for i in range(pw)]:
            t = 1
        if pid in [ph * (pw - 1) + i for i in range(pw)]:
            b = 0
        if pid in [ph * i for i in range(ph)]:
            l = 1
        if pid in [pw - 1 + ph * i for i in range(ph)]:
            r = 0

        row, col = pid // ph, pid % pw
        
        #print('buf2', self.region.keys())
        if pid == 0:
            pass
        elif pid in [i for i in range(1, pw)]:
            x = self.hcat([f"r{col - 1}"], x)
        elif pid in [ph * i for i in range(1, ph)]:
            x = self.vcat([f"b{(row - 1) * ph}"], x)
        else:
            x = self.fcat([f"br{ (row - 1) * ph + col - 1}", f"b{(row - 1) * ph + col}"], [f"r{row * ph + col - 1}"], x)

        x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)

        if pid == pw * ph - 1:
            assert not self.region, "error"
        out = self.mid_conv(x)
        #if pid == 0:
        #    print(self.layer_id, 'stride2 ',out.size())
        return out


def padding_prediction_all(patch):
    # Border
    pad_top = patch[:, :, :2, :].mean(dim=-2, keepdim=True)
    pad_down = patch[:, :, -2:, :].mean(dim=-2, keepdim=True)
    pad_left = patch[:, :, :, :2].mean(dim=-1, keepdim=True)
    pad_right = patch[:, :, :, -2:].mean(dim=-1, keepdim=True)

    # Corner
    pad_topleft = patch[:, :, 0, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
    pad_topright = patch[:, :, 0, -1].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
    pad_botleft = patch[:, :, -1, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
    pad_botright = patch[:, :, -1, -1].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)

    pad_ex_top = torch.cat([pad_topleft, pad_top, pad_topright], dim=-1)
    mid = torch.cat([pad_left, patch, pad_right], dim=-1)
    pad_ex_bot = torch.cat([pad_botleft, pad_down, pad_botright], dim=-1)

    return torch.cat([pad_ex_top, mid, pad_ex_bot], dim=-2)

def padding_prediction_S2_remove_right(patch):
    # Border
    pad_top = patch[:, :, :2, :].mean(dim=-2, keepdim=True)
    pad_down = patch[:, :, -2:, :].mean(dim=-2, keepdim=True)
    pad_left = patch[:, :, :, :2].mean(dim=-1, keepdim=True)

    # Corner
    pad_topleft = patch[:, :, 0, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
    pad_botleft = patch[:, :, -1, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)

    pad_ex_top = torch.cat([pad_topleft, pad_top], dim=-1)
    mid = torch.cat([pad_left, patch], dim=-1)
    pad_ex_bot = torch.cat([pad_botleft, pad_down], dim=-1)

    return torch.cat([pad_ex_top, mid, pad_ex_bot], dim=-2)

def padding_prediction_S2_remove_bot(patch):
    # Border
    pad_top = patch[:, :, :2, :].mean(dim=-2, keepdim=True)
    pad_left = patch[:, :, :, :2].mean(dim=-1, keepdim=True)
    pad_right = patch[:, :, :, -2:].mean(dim=-1, keepdim=True)

    # Corner
    pad_topleft = patch[:, :, 0, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
    pad_topright = patch[:, :, 0, -1].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)


    pad_ex_top = torch.cat([pad_topleft, pad_top, pad_topright], dim=-1)
    mid = torch.cat([pad_left, patch, pad_right], dim=-1)

    return torch.cat([pad_ex_top, mid], dim=-2)

def padding_prediction_S2_remove_botright(patch):
    # Border
    pad_top = patch[:, :, :2, :].mean(dim=-2, keepdim=True)
    pad_left = patch[:, :, :, :2].mean(dim=-1, keepdim=True)

    # Corner
    pad_topleft = patch[:, :, 0, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)

    pad_ex_top = torch.cat([pad_topleft, pad_top], dim=-1)
    mid = torch.cat([pad_left, patch], dim=-1)

    return torch.cat([pad_ex_top, mid], dim=-2)

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

      
        # top 
        o0 = padding_prediction_all(p0)
        o0[:,:, 0, :] *= 0.0
        o0[:, :, :, 0] *= 0.0
        o0 = self.mid_conv(o0)
        #o0 = self.mid_conv(F.pad(p0, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0.0))

        o1 = rearrange(p1, "B C H (n_patch W) -> (B n_patch) C H W", n_patch=pw-2)
        o1 = padding_prediction_all(o1)
        o1[:, :, 0, :] *= 0.0
        o1 = self.mid_conv(o1)
        #o1 = self.mid_conv(F.pad(o1, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0.0))
        o1 = rearrange(o1, "(B n_patch) C H W -> B C H (n_patch W)", n_patch=pw-2)

        o2 = padding_prediction_S2_remove_right(p2)
        o2[:, :, 0, :] *= 0.0
        o2 = self.mid_conv(o2)
        #o2 = self.mid_conv(F.pad(p2, (self.padding, self.padding-1, self.padding, self.padding), mode='constant', value=0.0))

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
        o5 = padding_prediction_S2_remove_right(o5)
        o5 = self.mid_conv(o5)
        #o5 = self.mid_conv(F.pad(o5, (self.padding, self.padding-1, self.padding, self.padding), mode='constant', value=0.0))
        o5 = rearrange(o5, "(B n_patch) C H W -> B C (n_patch H) W", n_patch=pw-2)

        # bot
        o6 = padding_prediction_S2_remove_bot(p6)
        o6[:, :, :, 0] *= 0.0
        o6 = self.mid_conv(o6)
        #o6 = self.mid_conv(F.pad(p6, (self.padding, self.padding, self.padding, self.padding-1), mode='constant', value=0.0))

        o7 = rearrange(p7, "B C H (n_patch W) -> (B n_patch) C H W", n_patch=pw-2)
        o7 = padding_prediction_S2_remove_bot(o7)
        o7 = self.mid_conv(o7)
        #o7 = self.mid_conv(F.pad(o7, (self.padding, self.padding, self.padding, self.padding-1), mode='constant', value=0.0))
        o7 = rearrange(o7, "(B n_patch) C H W -> B C H (n_patch W)", n_patch=pw-2)

        o8 = padding_prediction_S2_remove_botright(p8)
        o8 = self.mid_conv(o8)
        #o8 = self.mid_conv(F.pad(p8, (self.padding, self.padding-1, self.padding, self.padding-1), mode='constant', value=0.0))

        row0 = torch.cat([o0, o1, o2], dim=-1)
        row1 = torch.cat([o3, o4, o5], dim=-1)
        row2 = torch.cat([o6, o7, o8], dim=-1)

        out = torch.cat([row0, row1, row2], dim=-2)

        return out



class NConv1(nn.Module):
    def __init__(self, ConvModule, padding, n_patch):
        super(NConv1, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.groups = ConvModule.groups
        self.ph = n_patch
        self.pw = n_patch
        self.layer_id = self.mid_conv.layer_id
        self.patch_start = None
        self.patch_id = None

    def extra_repr(self):
        s = (f"NConv1: #patch = {self.ph}, padding={0, 0}")
        return s.format(**self.__dict__)

    def forward(self, x):
        pid = self.patch_id
        ph = self.ph
        pw = self.pw
        if pid == 0:
            pass
            #print(f"{self.layer_id} (NConv1): {x.size()}, {self.patch_start}")

        pad_top = x[:, :, :2, :].mean(dim=-2, keepdim=True).detach()
        pad_down = x[:, :, -2:, :].mean(dim=-2, keepdim=True).detach()
        pad_left = x[:, :, :, :2].mean(dim=-1, keepdim=True).detach()
        pad_right = x[:, :, :, -2:].mean(dim=-1, keepdim=True).detach()

			
        pad_topleft = x[:, :, 0, 0].clone().unsqueeze(dim=-1).unsqueeze(-1).detach()
        pad_topright = x[:, :, 0, -1].clone().unsqueeze(dim=-1).unsqueeze(-1).detach()
        pad_botleft = x[:, :, -1, 0].clone().unsqueeze(dim=-1).unsqueeze(-1).detach()
        pad_botright = x[:, :, -1, -1].clone().unsqueeze(dim=-1).unsqueeze(-1).detach()

        if pid in [i for i in range(pw)]:
            pad_top[:, :, 0, :] *= 0.0
            pad_topleft[:, :, 0, :] *= 0.0
            pad_topright[:, :, 0, :] *= 0.0
        if pid in [pw * (ph - 1) + i for i in range(pw)]:
            pad_down[:, :, -1, :] *= 0.0
            pad_botleft[:, :, 0, :] *= 0.0
            pad_botright[:, :, 0, :] *= 0.0		
        if pid in [(pw - 1) + ph * i for i in range(ph)]:
            pad_right[:, :, :, -1] *= 0.0
            pad_topright[:, :, :, 0] *= 0.0
            pad_botright[:, :, :, 0] *= 0.0
        if pid in [ph * i for i in range(ph)]:
            pad_left[:, :, :, 0] *= 0.0
            pad_topleft[:, :, :, 0] *=0.0
            pad_botleft[:, :, :, 0] *= 0.0

        pad_ex_top = torch.cat([pad_topleft, pad_top, pad_topright], dim=-1)
        pad_ex_down = torch.cat([pad_botleft, pad_down, pad_botright], dim=-1)
        x = torch.cat([pad_left, x, pad_right], dim = -1)

        x  = torch.cat([pad_ex_top, x, pad_ex_down], dim = -2)
        out = self.mid_conv(x)
        return out


class NConv2(nn.Module):
    def __init__(self, ConvModule, padding, n_patch):
        super(NConv2, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.groups = ConvModule.groups
        self.ph = n_patch
        self.pw = n_patch
        self.patch_start = None
        self.layer_id = self.mid_conv.layer_id
        self.patch_id = None

    def extra_repr(self):
        s = (f"NConv2: #patch = {self.ph}, padding={0, 0}")
        return s.format(**self.__dict__)

    def forward(self, x):
        pid = self.patch_id
        ph = self.ph
        pw = self.pw
        if pid == 0 and self.layer_id != 0:
            pass
            #print(f"{self.layer_id} (NConv2): {x.size()}, {self.patch_start}")


        pid_full_pad = []
        
        for j in range(ph - 1):
            for i in range(pw - 1):
                pid_full_pad.append(j * ph + i)

        pid_right_pad = []

        for j in range(ph - 1):
            pid_right_pad.append(j * ph + pw - 1)

        pid_bot_pad = []
        for i in range(pw - 1):
            pid_bot_pad.append(ph * (pw - 1) + i)


        if pid in pid_full_pad:
            #x = padding_prediction_all(x)
            x = padding_prediction_S2_remove_botright(x)
        elif pid in pid_right_pad:
            x = padding_prediction_S2_remove_botright(x)
        elif pid in pid_bot_pad:
            x = padding_prediction_S2_remove_botright(x)
        elif pid == pw * ph - 1:
            x = padding_prediction_S2_remove_botright(x)
        else:
            raise NotImplementedError

   
        if pid in [i for i in range(pw)]:
            x[:, :, 0, :] *= 0.0
        if pid in [ph * i for i in range(ph)]:
            x[:, :, :, 0] *= 0.0

        out = self.mid_conv(x)
        return out

def get_attr(layer):
    k = layer.kernel_size[0]
    s = layer.stride[0]
    return (k, s)

def change_inference_model(model, num_patches, Module_To_Mapping, patch_list):
    i = 0
    config = []
    #config = [(0,2)]
    for n, target in model.named_modules():
        if i == len(patch_list):
            break
        if isinstance(target, nn.Conv2d):
            if i == 0 and patch_list[0] == 1:
                config.append((patch_list[0], 2))
                i += 1
                continue
            attrs = n.split('.')
            submodule = model
            for attr in attrs[:-1]:
                submodule = getattr(submodule, attr)
            (k, s) = get_attr(target)
            replace = Module_To_Mapping[(k, s, patch_list[i])](target, target.padding, gpatch)
            setattr(submodule, attrs[-1], replace)

            config.append((patch_list[i], s))
            i += 1

    return config

def change_training_model(model, num_patches, Module_To_Mapping, patch_list):
    i = 0
    config = []
    for n, target in model.named_modules():
        if i == len(patch_list):
            break
        if isinstance(target, nn.Conv2d):
            attrs = n.split('.')
            submodule = model
            for attr in attrs[:-1]:
                submodule = getattr(submodule, attr)
            (k, s) = get_attr(target)
            if patch_list[i] == 0:
                replace = Module_To_Mapping[(k, s, patch_list[i])](target, target.padding, gpatch)
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
        feat_size = (feat_size - 3 + 2) // stride + 1
        hb_size = (hb_size - 3 + 2 - buf) // stride + 1
        offset = feat_size - hb_size
        offsets.append(hb_size)
    offsets = offsets
    return offsets

def is_my_conv(layer):
    if (isinstance(layer, nn.Conv2d) and layer.kernel_size[0] > 1):
        return True
    elif isinstance(layer, (NConv1, NConv2, ConvBuf1, ConvBuf2, PPConv2d_S2, PPConv2d_S1)):
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

def list_equal(this, other):
    #print(f"this = {this} / other = {other}")
    if len(this) != len(other):
        return False

    for a, b in zip(this, other):
        #print(f"a={a}, b={b}")
        if (a[0] != b[0] or a[1] != b[1]):
            return False

    return True

model = Toy()
p = [0, 1, 0, 1, 0]
mapping = {
    (3, 1, 0): NConv1,
    (3, 2, 0): NConv2,
    (3, 1, 1): ConvBuf1,
    (3, 2, 1): ConvBuf2,
}

mapping2 = {
    (3, 1, 0): PPConv2d_S1,
    (3, 2, 0): PPConv2d_S2,
}

class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3, 2, padding=1)

    def forward(self, x):
        ph = gpatch
        pw = gpatch

        x = rearrange(x, "B C (ph H) (pw W) -> (B ph pw) C H W", ph=ph, pw=pw)
        out_dict = {}
        for pid, y in enumerate(x):
            set_patch_id(self, pid)
            y = y.unsqueeze(dim=0)
            y = self.conv(y)
            out_dict[f"out{pid}"] = y
        out = []
        for j in range(ph):
            row = []
            for i in range(pw):
                row.append(out_dict[f"out{j * ph + i}"])
            row = torch.cat(row, dim=-1)
            out.append(row)
        out = torch.cat(out, dim=-2)
        return out

class B(nn.Module):
    def __init__(self):
        super(B, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3, 2, padding=1)

    def forward(self, x):
        return self.conv(x)

feat_size = 144
gpatch = 3
x = torch.randn(1, 3, feat_size, feat_size)
patch_size = feat_size // gpatch
search = list(product([0, 1], repeat=5))

a="""
test1 = A()
test2 = B()
test1.load_state_dict(test2.state_dict())

set_numbering(test1)
set_numbering(test2)

change_inference_model(test1, gpatch, mapping, [0])
change_training_model(test2, gpatch, mapping2, [0])
#change_model_list(test2, gpatch, module_to_mapping, [0])
test2.conv.patch_start = 10
out1 = test1(x)
out2 = test2(x)
print(out1)
print(out2)
print(torch.allclose(out1, out2,rtol=0, atol=1e-5))

exit()
"""
# MAIN
for i, conf in enumerate(search):
    print(f"====================== {conf} ======================")
    model = Toy()
    target = Bar()

    target.load_state_dict(model.state_dict())
    
    set_numbering(model)
    set_numbering(target)

    _conf = change_inference_model(model, gpatch, mapping, conf)
    offsets = get_offset(_conf, patch_size)
    set_start_point(model, offsets)

    _conf1 = change_training_model(target, gpatch, mapping2, conf)
    #print(f"conf {_conf}\n conf1 {_conf1}")
    set_start_point(target, offsets)
    
    #assert list_equal(_conf, _conf1), "configuration not equal"
    model.eval()
    target.eval()
    a = x.clone()
    b = x.clone()
    out1 = model(a)
    #out2 = target(b)
    #assert torch.allclose(out1, out2, rtol=0.0, atol=1e-5)
    
    assert out1.size() == (1, 3, 18, 18), f'error: {conf} {out1.size()}'
