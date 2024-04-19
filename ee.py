import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pandas as pd
from collections import deque

torch.set_printoptions(profile='full', precision=3, linewidth=300, sci_mode=False)

torch.manual_seed(0)

B = 1
C = 3
res = 144
gpatch = 3

l,r = 13, 6
def overlapping_patch(x, l, r, ph, pw):
    padded_x = F.pad(x, (l, r, l, r), mode='constant', value=0.0)
    padded_x = padded_x.unfold(2, res//ph + l + r, res//ph).unfold(3, res//ph + l + r, res//ph)

    return rearrange(padded_x, "B C ph pw H W -> (B ph pw) C H W", ph=ph, pw=pw)
    
def bypass(out, l, r):
    out1 = rearrange(out, "(B ph) C H -> B C ph H", ph=patch)
    if l > 0:
        out1[:, :, 0, :l] *= 0.0
    if r > 0:
        out1[:, :, -1, -r:] *= 0.0
    return rearrange(out1, "B C ph H -> (B ph) C H", ph=patch)

def OPB(out, wl, wr, hl, hr):
    if hl > 0:
        out[:, :hl, :] *= 0.0
    if hr > 0:
        out[:, -hr:, :] *= 0.0
    if wl > 0:
        out[:, :, :wl] *= 0.0
    if wr > 0:
        out[:, :, -wr:] *= 0.0

    return out

def push(dq, item):
    dq.appendleft(item.clone())

def pop(dq):
    return dq.pop()

class StreamNet2D_S2(nn.Module):
    def __init__(self, ConvModule, n_patch):
        super(StreamNet2D_S2, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.patch_id = None
        self.buf = {}
        self.ph = n_patch
        self.pw = n_patch
        self.padding = None
        self.e = 0

    def combine_xy(self, xy, patch_out):
        #print(model.patch_id, 'combine_xy')
        #print(self.buf[xy].shape, patch_out.shape)
        out = torch.cat([self.buf[xy], patch_out], dim=-1)
        return out

    def combine_x(self, x, patch_out):
        #print(model.patch_id, 'combine_x')
        #print(self.buf[x].shape, patch_out.shape)
        out = torch.cat([self.buf[x], patch_out], dim=-1)
        return out
    
    def combine_y(self, y, patch_out):
        #print(model.patch_id, 'combine_y')
        #print(self.buf[y].shape, patch_out.shape)
        out = torch.cat([self.buf[y], patch_out], dim=-2)
        return out

    def forward(self, x):
        pid = self.patch_id
        ph = self.ph
        pw = self.pw

        row, col = pid // ph, pid % pw

        l = self.padding[0] if col == 0 else 0
        r = self.padding[1] if col == ph - 1 else 0
        t = self.padding[0] if row == 0 else 0
        b = self.padding[1] if row == pw - 1 else 0
        if pid == 0:
            out = self.compute_conv(x, l, r, t, b)
            self.buf[f"{pid}_y"] = out[:, :, -2 + self.e:, :].clone()
            self.buf[f"{pid}_x"] = out[:, :, :, -2 + self.e:].clone()

            self.get_mem()
        elif row == 0 and col < pw - 1:
            out = self.compute_conv(x, l, r, t, b)
            out = self.combine_x(f"{col - 1}_x", out)
             
            self.buf[f"{pid}_y"] = out[:, :, -2 + self.e:, :].clone()
            self.buf[f"{pid}_x"] = out[:, :, :, -2 + self.e:].clone()

            self.get_mem()
            del self.buf[f"{col - 1}_x"]

        elif row == 0 and col == pw - 1:
            out = self.compute_conv(x, l, r, t, b)
            out = self.combine_x(f"{col - 1}_x", out)

            self.buf[f"{pid}_y"] = out[:, :, -2 + self.e:, :].clone()

            self.get_mem()
            del self.buf[f"{col - 1}_x"]

        elif col == 0 and row < ph - 1:
            out = self.compute_conv(x, l, r, t, b)
            
            self.buf[f"{pid}_xy"] = out[:, :, :, -2 + self.e:].clone()
            out = self.combine_y(f"{(row - 1) * pw}_y", out)
            self.buf[f"{pid}_y"] = out[:, :, -2 + self.e:, :].clone()
            self.get_mem()
            del self.buf[f"{(row - 1) * pw}_y"]

        elif col == 0 and row == ph - 1:
            out = self.compute_conv(x, l, r, t, b)
            self.buf[f"{pid}_xy"] = out[:, :, :, -2 + self.e:].clone()

            out = self.combine_y(f"{(row - 1) * pw}_y", out)
            self.get_mem()
            
            del self.buf[f"{(row - 1) * pw}_y"]
        elif row < ph - 1 and col < pw - 1:
            out = self.compute_conv(x, l, r, t, b)
            out = self.combine_xy(f"{row  * pw + col - 1}_xy", out)
            self.buf[f"{pid}_xy"] = out[:, :, :, -2 + self.e:].clone()

            out = self.combine_y(f"{(row -1) * pw + col}_y", out)
            self.buf[f"{pid}_y"] = out[:, :, -2 + self.e:, :].clone()
            self.get_mem()

            del self.buf[f"{row * pw + col - 1}_xy"]
            del self.buf[f"{(row - 1)* pw + col}_y"]

        elif row == ph - 1 and col < pw - 1:
            out = self.compute_conv(x, l, r, t, b)
            out = self.combine_xy(f"{row * pw + col - 1}_xy", out)
            self.buf[f"{pid}_xy"] = out[:, :, :, -2 + self.e:].clone()

            out = self.combine_y(f"{(row - 1) * pw + col}_y", out)
            self.get_mem()

            del self.buf[f"{row * pw + col - 1}_xy"]
            del self.buf[f"{(row - 1) * pw + col}_y"]
        elif col == pw - 1 and row < ph - 1:
            out = self.compute_conv(x, l, r, t, b)
            out = self.combine_xy(f"{(row * pw + col - 1)}_xy", out)
            out = self.combine_y(f"{(row - 1) * pw + col}_y", out)

            self.buf[f"{pid}_y"] = out[:, :, -2 + self.e:, :].clone()
            self.get_mem()
            del self.buf[f"{row * pw + col - 1}_xy"]
            del self.buf[f"{(row  - 1)* pw + col}_y"]

        else:
            out = self.compute_conv(x, l, r, t, b)
            out = self.combine_xy(f"{row  * pw + col - 1}_xy", out)
            out = self.combine_y(f"{(row - 1) * pw + col}_y", out)

            self.get_mem()
            del self.buf[f"{row  * pw + col - 1}_xy"]
            del self.buf[f"{(row -1) * pw + col}_y"]
            assert len(self.buf) == 0, "buffering error"
        return out
        
        
    def compute_conv(self, x, l, r, t, b):
        #print('padding', self.padding, self.mid_conv.padding)
        out = self.mid_conv(x)
        l, r, t, b = self.output_padding(l, r, t, b)
        out = self.output_padding_bypass(out, l, r, t, b)
        return out
       
    def get_mem(self):
        mem = sum([np.prod(seg.size()) for seg in self.buf.values()])
        print([seg.size() for seg in self.buf.values()])
        print(self.name, len(self.buf), mem)

    def output_padding_bypass(self, out, l, r, t, b):
        if t > 0:
            out[:, :, :t, :] *= 0.0
        if b > 0:
            out[:, :, -b:, :] *= 0.0
        if l > 0:
            out[:, :, :, :l] *= 0.0
        if r > 0:
            out[:, :, :, -r:] *= 0.0

        return out

    def output_padding(self, pad_l, pad_r, pad_t, pad_b):
        pad_l //= 2
        pad_r //= 2
        pad_t //= 2
        pad_b //= 2
        return (pad_l, pad_r, pad_t, pad_b)

class StreamNet2D_S1(nn.Module):
    def __init__(self, ConvModule, n_patch):
        super(StreamNet2D_S1, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.patch_id = None
        self.buf = {}
        self.ph = n_patch
        self.pw = n_patch
        self.padding = None
        self.e = 0
    def combine_xy(self, xy, patch_out):
        #print(model.patch_id, 'combine_xy')
        #print(self.buf[xy].shape, patch_out.shape)
        out = torch.cat([self.buf[xy], patch_out], dim=-1)
        return out

    def combine_x(self, x, patch_out):
        #print(model.patch_id, 'combine_x')
        #print(self.buf[x].shape, patch_out.shape)
        out = torch.cat([self.buf[x], patch_out], dim=-1)
        return out
    
    def combine_y(self, y, patch_out):
        #print(model.patch_id, 'combine_y')
        #print(self.buf[y].shape, patch_out.shape)
        out = torch.cat([self.buf[y], patch_out], dim=-2)
        return out

    def forward(self, x):
        pid = self.patch_id
        ph = self.ph
        pw = self.pw

        row, col = pid // ph, pid % pw

        l = self.padding[0] if col == 0 else 0
        r = self.padding[1] if col == ph - 1 else 0
        t = self.padding[0] if row == 0 else 0
        b = self.padding[1] if row == pw - 1 else 0

        if pid == 0:
            out = self.compute_conv(x, l, r, t, b)

            self.buf[f"{pid}_y"] = out[:, :, -2 + self.e:, :].clone()
            self.buf[f"{pid}_x"] = out[:, :, :, -2 + self.e:].clone()
            self.get_mem()
        elif row == 0 and col < pw - 1:
            out = self.compute_conv(x, l, r, t, b)
            out = self.combine_x(f"{col - 1}_x", out)
             
            self.buf[f"{pid}_y"] = out[:, :, -2 + self.e:, :].clone()
            self.buf[f"{pid}_x"] = out[:, :, :, -2 + self.e:].clone()
            self.get_mem()

            del self.buf[f"{col - 1}_x"]

        elif row == 0 and col == pw - 1:
            out = self.compute_conv(x, l, r, t, b)
            out = self.combine_x(f"{col - 1}_x", out)

            self.buf[f"{pid}_y"] = out[:, :, -2 + self.e:, :].clone()
            self.get_mem()

            del self.buf[f"{col - 1}_x"]

        elif col == 0 and row < ph - 1:
            out = self.compute_conv(x, l, r, t, b)
            
            self.buf[f"{pid}_xy"] = out[:, :, :, -2 + self.e:].clone()
            out = self.combine_y(f"{(row - 1) * pw}_y", out)
            self.buf[f"{pid}_y"] = out[:, :, -2 + self.e:, :].clone()
            self.get_mem()
            del self.buf[f"{(row - 1) * pw}_y"]

        elif col == 0 and row == ph - 1:
            out = self.compute_conv(x, l, r, t, b)
            self.buf[f"{pid}_xy"] = out[:, :, :, -2 + self.e:].clone()

            out = self.combine_y(f"{(row - 1) * pw}_y", out)
            self.get_mem()
            
            del self.buf[f"{(row - 1) * pw}_y"]
        elif row < ph - 1 and col < pw - 1:
            out = self.compute_conv(x, l, r, t, b)
            out = self.combine_xy(f"{row  * pw + col - 1}_xy", out)
            self.buf[f"{pid}_xy"] = out[:, :, :, -2 + self.e:].clone()

            out = self.combine_y(f"{(row -1) * pw + col}_y", out)
            self.buf[f"{pid}_y"] = out[:, :, -2 + self.e:, :].clone()

            self.get_mem()
            del self.buf[f"{row * pw + col - 1}_xy"]
            del self.buf[f"{(row - 1)* pw + col}_y"]

        elif row == ph - 1 and col < pw - 1:
            out = self.compute_conv(x, l, r, t, b)
            out = self.combine_xy(f"{row * pw + col - 1}_xy", out)
            self.buf[f"{pid}_xy"] = out[:, :, :, -2 + self.e:].clone()

            out = self.combine_y(f"{(row - 1) * pw + col}_y", out)
            self.get_mem()

            del self.buf[f"{row * pw + col - 1}_xy"]
            del self.buf[f"{(row - 1) * pw + col}_y"]
        elif col == pw - 1 and row < ph - 1:
            out = self.compute_conv(x, l, r, t, b)
            out = self.combine_xy(f"{(row * pw + col - 1)}_xy", out)
            out = self.combine_y(f"{(row - 1) * pw + col}_y", out)

            self.buf[f"{pid}_y"] = out[:, :, -2 + self.e:, :].clone()
            self.get_mem()
            del self.buf[f"{row * pw + col - 1}_xy"]
            del self.buf[f"{(row  - 1)* pw + col}_y"]

        else:
            out = self.compute_conv(x, l, r, t, b)
            out = self.combine_xy(f"{row  * pw + col - 1}_xy", out)
            out = self.combine_y(f"{(row - 1) * pw + col}_y", out)

            self.get_mem()
            del self.buf[f"{row  * pw + col - 1}_xy"]
            del self.buf[f"{(row -1) * pw + col}_y"]
            assert len(self.buf) == 0, "buffering error"
        return out
        
        
    def compute_conv(self, x, l, r, t, b):
        #print('padding', self.padding, self.mid_conv.padding)
        out = self.mid_conv(x)
        l, r, t, b = self.output_padding(l, r, t, b)
        out = self.output_padding_bypass(out, l, r, t, b)
        return out
   
    def get_mem(self):
        mem = sum([np.prod(seg.shape) for seg in self.buf.values()])
        print([seg.size() for seg in self.buf.values()])
        print(self.name, len(self.buf), mem)
    def output_padding_bypass(self, out, l, r, t, b):
        if t > 0:
            out[:, :, :t, :] *= 0.0
        if b > 0:
            out[:, :, -b:, :] *= 0.0
        if l > 0:
            out[:, :, :, :l] *= 0.0
        if r > 0:
            out[:, :, :, -r:] *= 0.0

        return out

    def output_padding(self, pad_l, pad_r, pad_t, pad_b):
        pad_l -= 1
        pad_r -= 1
        pad_t -= 1
        pad_b -= 1
        return (pad_l, pad_r, pad_t, pad_b)


class Toy(nn.Module):
    def __init__(self):
        super(Toy, self).__init__()
        self.conv0 = nn.Conv2d(1, 1, 3, padding=1, bias=False, stride=2)
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1, bias=False, stride=2)
        self.conv3 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(1, 1, 3, padding=1, bias=False, stride=2)

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out

def set_patch_id(model, pid):
    for n, mod in model.named_modules():
        mod.patch_id = pid

def set_padding(model, padding_list):
    i = 0
    exclude = []
    for n, mod in model.named_modules():
        if i == len(padding_list):
            return 
        if "Stream" in str(type(mod)):
            mod.padding = padding_list[i]
            i += 1



def get_attr(layer):
    k = layer.kernel_size[0]
    s = layer.stride[0]
    return (k, s)

module_to_mapping = {
    (3, 1): StreamNet2D_S1,
    (3, 2): StreamNet2D_S2,
}

def change_model_list(model, num_patch, Module_To_Mapping, patch_list):
    i = 0
    for n, target in model.named_modules():
        if i == len(patch_list):
            break
        if isinstance(target, nn.Conv2d) and target.kernel_size[0] > 1:
            if patch_list[i] == 0:
                target.padding = (0, 0)
                i += 1
                continue

            attrs = n.split('.')
            submodule = model
            for attr in attrs[:-1]:
                submodule = getattr(submodule, attr)
            replace = Module_To_Mapping[get_attr(target)](target, gpatch)

            setattr(submodule, attrs[-1], replace)
            i += 1


class Toy1(nn.Module):
    def __init__(self):
        super(Toy1, self).__init__()
        # MBV2-w0.5
        a="""
        self.conv0 = nn.Conv2d(3, 16, 3, padding=0, bias=False, stride=2)
        self.conv1 = nn.Conv2d(16, 16, 3, padding=0, bias=False, groups=16)
        self.pw = nn.Conv2d(16, 8*6, 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(8*6, 8*6, 3, padding=0, bias=False, stride=2, groups=48)
        self.pw1 = nn.Conv2d(48, 16*6, 1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(16*6, 16*6, 3, padding=0, bias=False, groups=96)
        self.conv4 = nn.Conv2d(16*6, 16*6, 3, padding=0, bias=False, stride=2, groups=96)
        """
        self.conv0 = nn.Conv2d(3, 16, 3, padding=0, bias=False, stride=2)
        self.conv1 = nn.Conv2d(16, 16, 3, padding=0, bias=False, groups=16)
        self.pw = nn.Conv2d(16, 8*6, 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(8*6, 8*6, 3, padding=0, bias=False, stride=2, groups=48)
        self.conv3 = nn.Conv2d(8*6, 8*6, 3, padding=0, bias=False, groups=48)
        self.conv4 = nn.Conv2d(8*6, 8*6, 3, padding=0, bias=False, stride=2, groups=48)
        self.l = l
        self.r = r

    def forward(self, x):
        ph = gpatch
        pw = gpatch
        
        _, _, H, W = x.size()
        x = F.pad(x, (self.l, self.r, self.l, self.r), mode='constant', value=0.0)
        x = x.unfold(2, H//ph + self.l + self.r, H//ph).unfold(3, W//ph + self.l + self.r, W//pw)
        x = rearrange(x, "B C ph pw H W -> (B ph pw) C H W", ph=ph, pw=pw)
        tensor_list = []
        for i, p in enumerate(x):
            row, col = i // ph, i % pw
            if i == 0:
                tensor_list.append(p)
            elif row == 0:
                tensor_list.append(p[:, :, self.l + self.r - 1:])
            elif col == 0:
                tensor_list.append(p[:, self.l + self.r - 1:, :])
            else:
                tensor_list.append(p[:, self.l + self.r - 1:, self.l + self.r - 1:])


        out = []
        out_dict = {}
        for pid, y in enumerate(tensor_list):
            set_patch_id(model, pid)
            print(f'=================== {pid} ======================')
            y = y.unsqueeze(dim=0)
            print(y.shape)
            y = self.conv0(y)
            y = self.conv1(y)
            y = self.pw(y)
            y = self.conv2(y)
            #y = self.pw1(y)
            y = self.conv3(y)
            y = self.conv4(y)
            out_dict[f"out{pid}"] = y.clone()

        for j in range(ph):
            row = []
            for i in range(pw):
                row.append(out_dict[f"out{j * pw + i}"])
            row = torch.cat(row, dim=-1)
            out.append(row)
        out = torch.cat(out, dim=-2)
        return out

#reference = Toy()
model = Toy1()
#model.load_state_dict(reference.state_dict())
change_model_list(model, gpatch, module_to_mapping, [1, 1, 1, 1, 0])

#STOP
i = 0
for n, mod in model.named_modules():
   if "StreamNet" in str(type(mod)):
       mod.name = f"{i} th layer"
       i += 1

set_padding(model, [(13,6), (6, 3), (5, 2), (2, 1)])

x = torch.randn(B, C, res, res)
#out = reference(x.clone())
out1 = model(x.clone())

_, _, H, W = out1.shape
#print('reference')
#print(out.shape)
print('model out')
print(out1.shape)
a = """
x = torch.randn(B, C, res, res)
w = torch.randn(C, C, 3, 3)
w1 = torch.randn(C, C, 3, 3)

o = F.conv2d(x, w, padding=1)
print(o.reshape(3, 3))


over_x = overlapping_patch(x, l, r, patch, patch)
tensor_list = []

for i, p in enumerate(over_x):
    row, col = i // patch, i % patch
    if row == 0 and col == 0:
        tensor_list.append(p)
    elif row == 0:
        tensor_list.append(p[:, :, l + r - 2:])
    elif col == 0:
        tensor_list.append(p[:, l + r - 2:, :])
    else:
        tensor_list.append(p[:, l + r - 2:, l + r - 2:])

temp = {}
out_temp = []
out_list = []

for i, p in enumerate(tensor_list):
    row, col = i // patch, i % patch
    pl = l if col == 0 else 0
    pr = r if col == patch - 1 else 0
    pt = l if row == 0 else 0
    pb = r if row == patch - 1 else 0
    print(f"============={i}==({pl}, {pr}, {pt}, {pb})=============")
    print(p)

    if i == 0:
        out_temp = F.conv2d(p, w, padding=0)
        pl -= 1;pt-=1
        out_temp = OPB(out_temp, pl, pr, pt, pb)
        temp[f'{i}_y'] = out_temp[:, -2:, :]
        temp[f'{i}_x'] = out_temp[:, :, -2:]
       
        print(out_temp) 
        out_temp = F.conv2d(out_temp, w1, padding=0)
        out_list.append(out_temp)

    elif row == 0 and col < patch - 1:
        out_temp = F.conv2d(p, w, padding=0)
        pt -= 1
        print('before OPB','\n',out_temp) 
        out_temp = OPB(out_temp, pl, pr, pt, pb)

        overlap = temp[f"{col - 1}_x"]
        del temp[f"{col - 1}_x"]

        out_temp = torch.concat([overlap, out_temp], dim=-1)

        temp[f"{i}_y"] = out_temp[:, -2:, :]
        temp[f"{i}_x"] = out_temp[:, :, -2:]

        print('after OPB','\n',out_temp) 
        out_temp = F.conv2d(out_temp, w1, padding=0)
        out_list.append(out_temp)

    elif row == 0 and col == patch - 1:
        out_temp = F.conv2d(p, w, padding=0)
        pt -= 1;pr -=1
        out_temp = OPB(out_temp, pl, pr, pt, pb)

        overlap = temp[f"{col - 1}_x"]
        del temp[f"{col - 1}_x"]

        out_temp = torch.concat([overlap, out_temp], dim=-1)
        print(out_temp) 

        temp[f"{i}_y"] = out_temp[:, -2:, :]

        out_temp = F.conv2d(out_temp, w1, padding=0)
        out_list.append(out_temp)

    elif col == 0 and row < patch - 1:
        out_temp = F.conv2d(p, w, padding=0)
        pl -= 1
        out_temp = OPB(out_temp, pl, pr, pt, pb)
        
        temp[f"{i}_xy"] = out_temp[:, :, -2:]
        overlap = temp[f"{(row-1)*patch}_y"]
        del temp[f"{(row - 1) * patch}_y"]

        out_temp = torch.concat([overlap, out_temp], dim=-2)
        print(out_temp) 
        temp[f"{i}_y"] = out_temp[:, -2:, :]
        
        out_temp = F.conv2d(out_temp, w1, padding=0)
        out_list.append(out_temp)

    elif col == 0 and row == patch - 1:
        out_temp = F.conv2d(p, w, padding=0)
        pl -= 1;pb -= 1
        out_temp = OPB(out_temp, pl, pr, pt, pb)

        temp[f"{i}_xy"] = out_temp[:, :, -2:]

        overlap = temp[f"{(row-1)*patch}_y"]
        del temp[f"{(row - 1) * patch}_y"]

        out_temp = torch.concat([overlap, out_temp], dim=-2)
        print(out_temp) 
        out_temp = F.conv2d(out_temp, w1, padding=0)
        out_list.append(out_temp)

    elif row < patch - 1 and col < patch - 1:
        out_temp = F.conv2d(p, w, padding=0)

        overlap = temp[f"{(row - 1)*patch + col}_y"]
        xy = temp[f'{row * patch + col - 1}_xy']

        del temp[f"{(row - 1)*patch + col}_y"]
        del temp[f'{row * patch + col - 1}_xy']

        print(xy.shape, out_temp.shape, overlap.shape) 
        out_temp = torch.concat([xy, out_temp], dim=-1)
        temp[f"{i}_xy"] = out_temp[:, :, -2:]
        out_temp = torch.concat([overlap, out_temp], dim=-2)
        temp[f"{i}_y"] = out_temp[:, -2:, :]

        out_temp = F.conv2d(out_temp, w1, padding=0)
        out_list.append(out_temp)

    elif row == patch - 1 and col < patch - 1:
        out_temp = F.conv2d(p, w, padding=0)
        pb -= 1;
        out_temp = OPB(out_temp, pl, pr, pt, pb)

        overlap = temp[f"{(row - 1)*patch + col}_y"]
        xy = temp[f'{row * patch + col - 1}_xy']

        print(xy.shape, out_temp.shape, overlap.shape) 
        del temp[f"{(row - 1)*patch + col}_y"]
        del temp[f'{row * patch + col - 1}_xy']

        out_temp = torch.concat([xy, out_temp], dim=-1)
        temp[f"{i}_xy"] = out_temp[:, :, -2:]
        out_temp = torch.concat([overlap, out_temp], dim=-2)

        print(out_temp) 
        out_temp = F.conv2d(out_temp, w1, padding=0)
        out_list.append(out_temp)
    elif col == patch - 1 and row < patch - 1:
        out_temp = F.conv2d(p, w, padding=0)
        pr -= 1;
        out_temp = OPB(out_temp, pl, pr, pt, pb)

        overlap = temp[f"{(row - 1)*patch + col}_y"]
        xy = temp[f'{row * patch + col - 1}_xy']

        del temp[f"{(row - 1)*patch + col}_y"]
        del temp[f'{row * patch + col - 1}_xy']
        print(xy.shape, out_temp.shape, overlap.shape)
        out_temp = torch.concat([xy, out_temp], dim=-1)
        out_temp = torch.concat([overlap, out_temp], dim=-2)
        temp[f"{i}_y"] = out_temp[:, -2:, :]
        print(out_temp) 

        out_temp = F.conv2d(out_temp, w1, padding=0)
        out_list.append(out_temp)

    else:
        out_temp = F.conv2d(p, w, padding=0)
        pb -= 1; pr -=1;
        out_temp = OPB(out_temp, pl, pr, pt, pb)

        overlap = temp[f"{(row - 1)*patch + col}_y"]
        xy = temp[f'{row * patch + col - 1}_xy']

        del temp[f"{(row - 1)*patch + col}_y"]
        del temp[f'{row * patch + col - 1}_xy']
        print(overlap.shape, xy.shape, out_temp.shape)
        out_temp = torch.concat([xy, out_temp], dim=-1)
        out_temp = torch.concat([overlap, out_temp], dim=-2)

        print(out_temp) 
        out_temp = F.conv2d(out_temp, w1, padding=0)
        out_list.append(out_temp)

        
out_temp0 = torch.concat(out_list[:3], dim=-1)
out_temp1 = torch.concat(out_list[3:6], dim=-1)
out_temp2 = torch.concat(out_list[6:], dim=-1)
out = torch.concat([out_temp0, out_temp1, out_temp2], dim=-2)
print('reference')
print(o1)
print('output')
print(out)
"""



