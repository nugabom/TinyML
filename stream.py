import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
import numpy as np

torch.set_printoptions(precision=3, linewidth=600, sci_mode=False, threshold=20000)

def output_bypassing(out, l, r, t, b):
    if t:
        out[:, :, :t, :] *= 0.0
    if b:
        out[:, :, -b:, :] *= 0.0
    if l:
        out[:, :, :, :l] *= 0.0
    if r:
        out[:, :, :, -r:] *= 0.0

    return out

def fused_opt_overlap(x, l, r, patch_info):
    N, a, b, c = patch_info
    rows = []
    temp = []
    outs = []
    padded_x = F.pad(x, (l, r, l, r), mode='constant', value=0.0)
    for pid in range(N ** 2):
        row, col = pid // N, pid % N

        height,width = 0, 0
        start_y, start_x = 0, 0

        if row == 0:
            start_y = 0
            height = a

        elif row in [i for i in range(1, N-1)]:
            start_y = a + (row - 1) * b
            height = b

        elif row in [N - 1]:
            start_y = a + (N - 2) * b
            height = c

        if col == 0:
            start_x = 0
            width = a

        elif col in [i for i in range(1, N-1)]:
            start_x = a + (col - 1) * b
            width = b

        elif col in [N - 1]:
            start_x = a + (N - 2) * b
            width = c


        # start_y = max(0, start_y - l)
        # start_x = max(0, start_x - l)
        overlap_patch = padded_x[:, :, start_y:start_y + height + l + r, start_x:start_x + width + l + r].clone()
        outs.append(overlap_patch)
    return outs 

class StreamNet_K3_S1(nn.Module):
    def __init__(self, conv, patch_info):
        super(StreamNet_K3_S1, self).__init__()
        self.mid_conv = conv
        self.patch_info = patch_info
        self.mid_conv.padding = (0, 0)
        N, a, b, c = patch_info
        self.patch_idx = None
        self.buf = {}
        self.ph = N
        self.pw = N

        self.large_padding = None
        self.small_padding = None
        self.right_pixels = c


        self.buf_region = self.mid_conv.kernel_size[0] - self.mid_conv.stride[0]

    def delete(self):
        deleted_keys = []
        for key, item in self.buf.items():
            if getattr(item, 'delete', False):
                deleted_keys.append(key)

        for key in deleted_keys:
            del self.buf[key]
    def store_right(self, x):
        y = x[:, :, :, -self.buf_region:].clone()
        self.buf[f"r{self.patch_idx}"] = y

    def store_bot(self, x):
        y = x[:, :, -self.buf_region:, :].clone()
        self.buf[f"b{self.patch_idx}"] = y

    def hcat(self, buf_idx, x):
        out = torch.cat([self.buf[buf_idx], x], dim=-1)
        self.buf[buf_idx].delete = True
        return out
        # if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
        #     self.store_right(x)
    def vcat(self, buf_idx, x):
        out = torch.cat([self.buf[buf_idx], x], dim=-2)
        self.buf[buf_idx].delete = True
        # del self.buf[buf_idx]
        return out
    
    def forward(self, x):
        pid = self.patch_idx 
        self.right_padding_cnt = ceil(self.small_padding/self.right_pixels)
        row, col = pid // self.ph, pid % self.pw
        
        t, b, l, r = 0, 0, 0, 0

        if row == 0:
            t = self.large_padding
        if row in [self.ph - i -1 for i in range(0, self.right_padding_cnt)]:
            b = self.small_padding - (self.ph - 1 - row) * self.right_pixels
        if col == 0:
            l = self.large_padding
        if col in [self.pw - i -1 for i in range(0, self.right_padding_cnt)]:
            r = self.small_padding - (self.pw -1- col) * self.right_pixels

        print((row, col), b)
        x = output_bypassing(x, l, r, t, b)

        if pid == 0:
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
        elif pid in [i for i in range(1, self.pw)]:
            x = self.hcat(f"r{col - 1}", x)
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
        elif pid in [self.ph * i for i in range(self.ph)]:
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
            x = self.vcat(f"b{(row - 1) * self.ph}", x)
        else:
            x = self.hcat(f"r{(col - 1) + row * self.ph}", x)
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)       
            x = self.vcat(f"b{(row - 1) * self.ph + col}", x)   

        if pid not in [(self.ph - 1) * self.ph + i for i in range(self.ph)]:
            self.store_bot(x)

        # print((row, col), sum([np.prod(v.shape) for k, v in self.buf.items()]))
        self.delete()
        out = self.mid_conv(x)
        return out
        
class StreamNet_K3_S2(nn.Module):
    def __init__(self, conv, patch_info):
        super(StreamNet_K3_S2, self).__init__()
        self.mid_conv = conv
        self.patch_info = patch_info
        self.mid_conv.padding = (0, 0)
        N, a, b, c = patch_info
        self.patch_idx = None
        self.buf = {}
        self.ph = N
        self.pw = N

        self.large_padding = None
        self.small_padding = None
        self.right_pixels = c


        self.buf_region = self.mid_conv.kernel_size[0] - self.mid_conv.stride[0]

    def store_right(self, x):
        y = x[:, :, :, -self.buf_region:].clone()
        self.buf[f"r{self.patch_idx}"] = y

    def store_bot(self, x):
        y = x[:, :, -self.buf_region:, :].clone()
        self.buf[f"b{self.patch_idx}"] = y

    def hcat(self, buf_idx, x):
        out = torch.cat([self.buf[buf_idx], x], dim=-1)
        del self.buf[buf_idx]
        return out
        # if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
        #     self.store_right(x)
    def vcat(self, buf_idx, x):
        out = torch.cat([self.buf[buf_idx], x], dim=-2)
        del self.buf[buf_idx]
        return out
    
    def forward(self, x):
        pid = self.patch_idx 
        self.right_padding_cnt = ceil((self.small_padding )/self.right_pixels)    
        row, col = pid // self.ph, pid % self.pw
        
        t, b, l, r = 0, 0, 0, 0

        if row == 0:
            t = self.large_padding
        if row in [self.ph - i -1 for i in range(0, self.right_padding_cnt)]:
            b = self.small_padding - (self.ph - 1 - row) * self.right_pixels
        if col == 0:
            l = self.large_padding
        if col in [self.pw - i -1 for i in range(0, self.right_padding_cnt)]:
            r = self.small_padding - (self.pw -1- col)* self.right_pixels

        x = output_bypassing(x, l, r, t, b)

        if pid == 0:
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
        elif pid in [i for i in range(1, self.pw)]:
            x = self.hcat(f"r{col - 1}", x)
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
        elif pid in [self.ph * i for i in range(self.ph)]:
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
            x = self.vcat(f"b{(row - 1) * self.ph}", x)
        else:
            x = self.hcat(f"r{(col - 1) + row * self.ph}", x)
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)       
            x = self.vcat(f"b{(row - 1) * self.ph + col}", x)   

        if pid not in [(self.ph - 1) * self.ph + i for i in range(self.ph)]:
            self.store_bot(x)
        
        out = self.mid_conv(x)
        return out
    
class StreamNet_K5_S1(nn.Module):
    def __init__(self, conv, patch_info):
        super(StreamNet_K5_S1, self).__init__()
        self.mid_conv = conv
        self.patch_info = patch_info
        self.mid_conv.padding = (0, 0)
        N, a, b, c = patch_info
        self.patch_idx = None
        self.buf = {}
        self.ph = N
        self.pw = N

        self.large_padding = None
        self.small_padding = None
        self.right_pixels = c


        self.buf_region = self.mid_conv.kernel_size[0] - self.mid_conv.stride[0]

    def store_right(self, x):
        y = x[:, :, :, -self.buf_region:].clone()
        self.buf[f"r{self.patch_idx}"] = y

    def store_bot(self, x):
        y = x[:, :, -self.buf_region:, :].clone()
        self.buf[f"b{self.patch_idx}"] = y

    def hcat(self, buf_idx, x):
        out = torch.cat([self.buf[buf_idx], x], dim=-1)
        del self.buf[buf_idx]
        return out
        # if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
        #     self.store_right(x)
    def vcat(self, buf_idx, x):
        out = torch.cat([self.buf[buf_idx], x], dim=-2)
        del self.buf[buf_idx]
        return out
    
    def forward(self, x):
        pid = self.patch_idx 
        self.right_padding_cnt = ceil(self.small_padding/self.right_pixels)
        row, col = pid // self.ph, pid % self.pw
        
        t, b, l, r = 0, 0, 0, 0


        if row == 0:
            t = self.large_padding
        if row in [self.ph - i -1 for i in range(0, self.right_padding_cnt)]:
            b = self.small_padding - (self.ph - 1 - row) * self.right_pixels
        if col == 0:
            l = self.large_padding
        if col in [self.pw - i -1 for i in range(0, self.right_padding_cnt)]:
            r = self.small_padding - (self.pw -1- col)* self.right_pixels

        x = output_bypassing(x, l, r, t, b)

        if pid == 0:
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
        elif pid in [i for i in range(1, self.pw)]:
            x = self.hcat(f"r{col - 1}", x)
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
        elif pid in [self.ph * i for i in range(self.ph)]:
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
            x = self.vcat(f"b{(row - 1) * self.ph}", x)
        else:
            x = self.hcat(f"r{(col - 1) + row * self.ph}", x)
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)       
            x = self.vcat(f"b{(row - 1) * self.ph + col}", x)   

        if pid not in [(self.ph - 1) * self.ph + i for i in range(self.ph)]:
            self.store_bot(x)
        
        out = self.mid_conv(x)
        return out

class StreamNet_K5_S2(nn.Module):
    def __init__(self, conv, patch_info):
        super(StreamNet_K5_S2, self).__init__()
        self.mid_conv = conv
        self.patch_info = patch_info
        self.mid_conv.padding = (0, 0)
        N, a, b, c = patch_info
        self.patch_idx = None
        self.buf = {}
        self.ph = N
        self.pw = N

        self.large_padding = None
        self.small_padding = None
        self.right_pixels = c


        self.buf_region = self.mid_conv.kernel_size[0] - self.mid_conv.stride[0]

    def store_right(self, x):
        y = x[:, :, :, -self.buf_region:].clone()
        self.buf[f"r{self.patch_idx}"] = y

    def store_bot(self, x):
        y = x[:, :, -self.buf_region:, :].clone()
        self.buf[f"b{self.patch_idx}"] = y

    def hcat(self, buf_idx, x):
        out = torch.cat([self.buf[buf_idx], x], dim=-1)
        del self.buf[buf_idx]
        return out
        # if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
        #     self.store_right(x)
    def vcat(self, buf_idx, x):
        out = torch.cat([self.buf[buf_idx], x], dim=-2)
        del self.buf[buf_idx]
        return out
    
    def forward(self, x):
        pid = self.patch_idx 
        self.right_padding_cnt = ceil((self.small_padding )/self.right_pixels)    
        row, col = pid // self.ph, pid % self.pw
        
        t, b, l, r = 0, 0, 0, 0

        if row == 0:
            t = self.large_padding
        if row in [self.ph - i -1 for i in range(0, self.right_padding_cnt)]:
            b = self.small_padding - (self.ph - 1 - row) * self.right_pixels
        if col == 0:
            l = self.large_padding
        if col in [self.pw - i -1 for i in range(0, self.right_padding_cnt)]:
            r = self.small_padding - (self.pw -1- col)* self.right_pixels

        x = output_bypassing(x, l, r, t, b)

        if pid == 0:
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
        elif pid in [i for i in range(1, self.pw)]:
            x = self.hcat(f"r{col - 1}", x)
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
        elif pid in [self.ph * i for i in range(self.ph)]:
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
            x = self.vcat(f"b{(row - 1) * self.ph}", x)
        else:
            x = self.hcat(f"r{(col - 1) + row * self.ph}", x)
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)       
            x = self.vcat(f"b{(row - 1) * self.ph + col}", x)   

        if pid not in [(self.ph - 1) * self.ph + i for i in range(self.ph)]:
            self.store_bot(x)
        out = self.mid_conv(x)
        return out

class StreamNet_K7_S1(nn.Module):
    def __init__(self, conv, patch_info):
        super(StreamNet_K7_S1, self).__init__()
        self.mid_conv = conv
        self.patch_info = patch_info
        self.mid_conv.padding = (0, 0)
        N, a, b, c = patch_info
        self.patch_idx = None
        self.buf = {}
        self.ph = N
        self.pw = N

        self.large_padding = None
        self.small_padding = None
        self.right_pixels = c


        self.buf_region = self.mid_conv.kernel_size[0] - self.mid_conv.stride[0]

    def store_right(self, x):
        y = x[:, :, :, -self.buf_region:].clone()
        self.buf[f"r{self.patch_idx}"] = y

    def store_bot(self, x):
        y = x[:, :, -self.buf_region:, :].clone()
        self.buf[f"b{self.patch_idx}"] = y

    def hcat(self, buf_idx, x):
        out = torch.cat([self.buf[buf_idx], x], dim=-1)
        del self.buf[buf_idx]
        return out
        # if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
        #     self.store_right(x)
    def vcat(self, buf_idx, x):
        out = torch.cat([self.buf[buf_idx], x], dim=-2)
        del self.buf[buf_idx]
        return out
    
    def forward(self, x):
        pid = self.patch_idx 
        self.right_padding_cnt = ceil(self.small_padding/self.right_pixels)
        row, col = pid // self.ph, pid % self.pw
        
        t, b, l, r = 0, 0, 0, 0

        if row == 0:
            t = self.large_padding
        if row in [self.ph - i -1 for i in range(0, self.right_padding_cnt)]:
            b = self.small_padding - (self.ph - 1 - row) * self.right_pixels
        if col == 0:
            l = self.large_padding
        if col in [self.pw - i -1 for i in range(0, self.right_padding_cnt)]:
            r = self.small_padding - (self.pw -1- col)* self.right_pixels

        x = output_bypassing(x, l, r, t, b)

        if pid == 0:
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
        elif pid in [i for i in range(1, self.pw)]:
            x = self.hcat(f"r{col - 1}", x)
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
        elif pid in [self.ph * i for i in range(self.ph)]:
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
            x = self.vcat(f"b{(row - 1) * self.ph}", x)
        else:
            x = self.hcat(f"r{(col - 1) + row * self.ph}", x)
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)       
            x = self.vcat(f"b{(row - 1) * self.ph + col}", x)   

        if pid not in [(self.ph - 1) * self.ph + i for i in range(self.ph)]:
            self.store_bot(x)
        
        out = self.mid_conv(x)
        return out

class StreamNet_K7_S2(nn.Module):
    def __init__(self, conv, patch_info):
        super(StreamNet_K7_S2, self).__init__()
        self.mid_conv = conv
        self.patch_info = patch_info
        self.mid_conv.padding = (0, 0)
        N, a, b, c = patch_info
        self.patch_idx = None
        self.buf = {}
        self.ph = N
        self.pw = N

        self.large_padding = None
        self.small_padding = None
        self.right_pixels = c
        
        self.buf_region = self.mid_conv.kernel_size[0] - self.mid_conv.stride[0]

    def store_right(self, x):
        y = x[:, :, :, -self.buf_region:].clone()
        self.buf[f"r{self.patch_idx}"] = y

    def store_bot(self, x):
        y = x[:, :, -self.buf_region:, :].clone()
        self.buf[f"b{self.patch_idx}"] = y

    def hcat(self, buf_idx, x):
        out = torch.cat([self.buf[buf_idx], x], dim=-1)
        # del self.buf[buf_idx]
        self.buf[buf_idx].delete = True
        return out

    def vcat(self, buf_idx, x):
        out = torch.cat([self.buf[buf_idx], x], dim=-2)
        # del self.buf[buf_idx]
        self.buf[buf_idx].delete = True
        return out
    
    def delete(self):
        for key, item in self.buf.items():
            if getattr(item, 'delete', False):
                del self.buf[key]
    def forward(self, x):
        pid = self.patch_idx 

        self.right_padding_cnt = ceil((self.small_padding )/self.right_pixels)    
        row, col = pid // self.ph, pid % self.pw
        
        t, b, l, r = 0, 0, 0, 0
        if row == 0:
            t = self.large_padding
        if row in [self.ph - i -1 for i in range(0, self.right_padding_cnt)]:
            b = self.small_padding - (self.ph - 1 - row) * self.right_pixels
        if col == 0:
            l = self.large_padding
        if col in [self.pw - i -1 for i in range(0, self.right_padding_cnt)]:
            r = self.small_padding - (self.pw -1- col)* self.right_pixels

        x = output_bypassing(x, l, r, t, b)

        if pid == 0:
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
        elif pid in [i for i in range(1, self.pw)]:
            x = self.hcat(f"r{col - 1}", x)
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
        elif pid in [self.ph * i for i in range(self.ph)]:
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)
            x = self.vcat(f"b{(row - 1) * self.ph}", x)
        else:
            x = self.hcat(f"r{(col - 1) + row * self.ph}", x)
            if pid not in [self.ph - 1 + self.ph * i for i in range(self.ph)]:
                self.store_right(x)       
            x = self.vcat(f"b{(row - 1) * self.ph + col}", x)   

        if pid not in [(self.ph - 1) * self.ph + i for i in range(self.ph)]:
            self.store_bot(x)
        
        # print((row, col), sum([np.prod(k.shape) for k, v in self.buf.items()]))
        self.delete()
        out = self.mid_conv(x)
        return out

# MI3 clear
# PL clear
# MI4 clear

B, C, H, W = 1, 1, 160, 160
l = 67
r = 36
# x = torch.arange(B * C * H * W).reshape(B, C, H, W) + 1.
x = torch.randn(B, C, H, W)
N = 20

S = 8
first = 3 - 2
a = S + l + r
b = S + first
c = S + first

inputs = []
for pid in range(N**2):
    padded_x = F.pad(x, (l, r , l, r), mode='constant', value=0.0)

    row, col = pid //N, pid % N
    
    if row in [0]:
        start_h = 0
        height = a
    elif row in [i for i in range(1, N-1)]:
        start_h = a + (row - 1) * S - first
        height = b
    elif row in [N - 1]:
        start_h = a + (N - 2) * S - first
        height = c
    
    if col in [0]:
        start_w = 0
        width = a
    elif col in [i for i in range(1, N-1)]:
        start_w = a + (col - 1) * S - first
        width = b
    elif col in [N - 1]:
        start_w = a + (N - 2) * S - first
        width = c
    # print(f"{row, col}, {start_h}, {height} / {start_w}, {width}")
    inputs.append(padded_x[:, :, start_h:start_h + height, start_w:start_w + width])

conv1 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
conv2 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
# conv2 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
# conv3 = nn.Conv2d(1, 1, 7, stride=2, padding=3)
# conv4 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
# conv5 = nn.Conv2d(1, 1, 5, stride=1, padding=2)
# conv6 = nn.Conv2d(1, 1, 7, stride=2, padding=3)
# conv7 = nn.Conv2d(1, 1, 5, stride=2, padding=2)
# conv4 = nn.Conv2d(1, 1, 5, padding=2, stride=2)

patch_info2 = (N, 43, 4, 4)
# patch_info3 = (N, 41, 4, 4)
# patch_info4 = (N, 13, 2, 2)
# patch_info5 = (N, 11, 2, 2)
# patch_info6 = (N, 7, 2, 2)
# patch_info7 = (N, 5, 2, 2)

y = conv1(x)
y1 = conv2(y)
# y = conv3(y)
# y = conv4(y)
# y = conv5(y)
# y1 = conv6(y)
# y1 = conv7(y)

# y1 = conv4(y)

conv1.padding = (0, 0)
stream2 = StreamNet_K3_S1(conv2, patch_info2)
# stream3 = StreamNet_K7_S2(conv3, patch_info3)
# stream4 = StreamNet_K3_S1(conv4, patch_info4)
# stream5 = StreamNet_K5_S1(conv5, patch_info5)
# stream6 = StreamNet_K7_S2(conv6, patch_info6)
# stream7 = StreamNet_K5_S2(conv7, patch_info7)

stream2.large_padding = 16
stream2.small_padding = 13

# stream3.large_padding = 15
# stream3.small_padding = 12

# stream4.large_padding = 6
# stream4.small_padding = 5

# stream5.large_padding = 5
# stream5.small_padding = 4

# stream6.large_padding = 3
# stream6.small_padding = 2

# stream7.large_padding = 2
# stream7.small_padding = 1
# stream3.large_padding = 3
# stream3.small_padding = 2

# stream4.large_padding = 2
# stream4.small_padding = 1
outs = []
for pid, x in enumerate(inputs):
    row, col = pid // N, pid % N
    conv1.patch_idx = pid
    stream2.patch_idx = pid
    # stream3.patch_idx = pid
    # stream4.patch_idx = pid
    # stream5.patch_idx = pid
    # stream6.patch_idx = pid
    # stream7.patch_idx = pid

    y = conv1(x)
    # print(row, col)
    # print(y)
    y = stream2(y)
    # y = stream3(y)
    # y = stream4(y)
    # y = stream5(y)
    # y = stream6(y)
    # y = stream7(y)
    # y = stream4(y)
    outs.append(y.clone())

rows = []
while outs:
    rows.append(torch.cat(outs[:N], dim=-1))
    outs = outs[N:]
out = torch.cat(rows, dim=-2)
# print(y1[0, 0, :, :])
# print(out[0, 0, :, :])

# print((12 + 12) * 6 + 9* 36)
