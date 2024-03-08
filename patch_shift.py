from itertools import product
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

gpatch = 4

def set_patch_id(model, pid):
    for n, mod in model.named_modules():
        mod.patch_id = pid

class Toy(nn.Module):
    def __init__(self):
        super(Toy, self).__init__()
        self.conv0 = nn.Conv2d(3, 3, 3, 2, padding=1)
        self.conv1 = nn.Conv2d(3, 3, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(3, 3, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(3, 3, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(3, 3, 3, 2, padding=1)

    def forward(self, x):
        ph = gpatch
        pw = gpatch
        x = rearrange(x, "B C (ph H) (pw W) -> (B ph pw) C H W", ph=ph, pw=pw)

        out_dict = {}
        for pid, y in enumerate(x):
            set_patch_id(model,pid)
            y = y.unsqueeze(dim=0)
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
        if pid == 0:
            print(self.layer_id, 'stride1 ',out.size())
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
        if pid == 0:
            print(self.layer_id, 'stride2 ',out.size())
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

    def extra_repr(self):
        s = (f"NConv1: #patch = {self.ph}, padding={0, 0}")
        return s.format(**self.__dict__)

    def forward(self, x):
        pid = self.patch_id
        ph = self.ph
        pw = self.pw

        x = F.pad(x, (1, 1, 1, 1), mode='constant', value=0.0)
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
        self.layer_id = self.mid_conv.layer_id

    def extra_repr(self):
        s = (f"NConv2: #patch = {self.ph}, padding={0, 0}")
        return s.format(**self.__dict__)

    def forward(self, x):
        pid = self.patch_id
        ph = self.ph
        pw = self.pw

        t=1;b=1;l=1;r=1;
        if pid in [ph * (pw - 1) + i for i in range(pw)]:
            b = 0
        if pid in [pw - 1 + ph * i for i in range(ph)]:
            r = 0

        x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)

        out = self.mid_conv(x)
        return out

def get_attr(layer):
    k = layer.kernel_size[0]
    s = layer.stride[0]
    return (k, s)

def change_model(model, num_patches, Module_To_Mapping, patch_list):
    i = 0
    for n, target in model.named_modules():
        if i == len(patch_list):
            break
        if isinstance(target, nn.Conv2d):
            if i == 0 and patch_list[0] == 1:
                i += 1
                continue
            attrs = n.split('.')
            submodule = model
            for attr in attrs[:-1]:
                submodule = getattr(submodule, attr)
            replace = mapping[(*get_attr(target), patch_list[i])](target, target.padding, gpatch)
            setattr(submodule, attrs[-1], replace)
            i += 1

# MAIN
def set_numbering(model):
    i = 0
    for n, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            mod.layer_id = i
            i += 1

model = Toy()
p = [0, 1, 0, 1, 0]
mapping = {
    (3, 1, 0): NConv1,
    (3, 2, 0): NConv2,
    (3, 1, 1): ConvBuf1,
    (3, 2, 1): ConvBuf2,
}

x = torch.randn(1, 3, 160, 160)
search = list(product([0, 1], repeat=5))

for conf in search:
    print(f'============== {conf} ======================')
    model = Toy()
    set_numbering(model)
    change_model(model, gpatch, mapping, conf)
    y = model(x)
    assert y.size() == (1, 3, 20, 20), f'error: {conf} {y.size()}'


