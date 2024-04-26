import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pandas as pd

torch.set_printoptions(profile='full')
torch.set_printoptions(precision=3)
torch.set_printoptions(linewidth=200)

torch.manual_seed(0)
# weight shape
# (out_ch, in_ch, k, k)

# padding

tp = 3
lp = 3
bp = 3
rp = 3

ph = 4
pw = 4

res = 64
in_ch = 1
mid_ch = 1
out_ch = 1

def show(input):
    _, _, h, w = input.size()
    input = input.reshape(h, w)
    print(input)

def logging(input, filename, over):
    if over:
        input = rearrange(input, '(B ph pw) C H W -> B C (ph H) (pw W)', ph=ph, pw=pw)
    _, _, h, w = input.size()
    log = input.reshape(h, w).numpy()
    log = pd.DataFrame(log).astype('float')
    log.to_csv(f'{filename}.csv')

def overlap(input, weight, stride, lp, rp, tp, bp):
    out = rearrange(input, '(B ph pw) C H W -> B ph pw C H W', ph=ph, pw=pw)
    # overlapping remove
    out[:, 0, :, :, :tp, :] = 0
    out[:, -1, :, :, -bp:, :] = 0
    out[:, :, 0, :, :, :lp] = 0
    out[:, :, -1, :, :, -rp:] = 0

    out = rearrange(out, 'B ph pw C H W -> (B ph pw) C H W', ph=ph, pw=pw)
    out = F.conv2d(out, weight, stride=stride, padding=0)
    
    return out 

w1 = torch.randn((in_ch, mid_ch, 3, 3))
w2 = torch.randn((mid_ch, out_ch, 3, 3))
w3 = torch.randn((out_ch, out_ch, 3, 3))

input = torch.randn((1, 1, res, res))
input = torch.arange(1, in_ch * res * res + 1).reshape(1, in_ch, res, res).float()
pad_input = F.pad(input.clone(), (lp, rp, tp, bp), 'constant', 0)
no_pad_input = rearrange(input, 'B C (ph H) (pw W) -> (B ph pw) C H W', ph=ph, pw=pw)
over_input = pad_input.unfold(2, res//ph+tp+bp, res//ph).unfold(3, res//pw+lp+rp, res//pw)
over_input = rearrange(over_input, 'B C ph pw H W -> (B ph pw) C H W', ph = ph, pw=pw)


aux1 = {}
aux2 = {}
aux3 = {}

# Padding + NO
l = 0
r = 0
t = 0
b = 0

patches = [torch.unsqueeze(x, dim=0) for x in no_pad_input]
first = 0
mid = 0
mid1 = 0

# 0
l=1;t=1;r=0;b=0
aux1['r0'] = patches[0][:, :, :, -1:]
aux1['b0'] = patches[0][:, :,-1:, : ]
aux1['br0'] = patches[0][:, :, -1:, -1:]
first = max(first, patches[0].numel())
x = F.pad(patches[0], (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w1, stride=2, padding=0)

aux2['r0'] = x[:, :, :, -2:]
aux2['b0'] = x[:, :, -2:, :]
aux2['br0'] = x[:, :, -2:, -2:]
mid = max(mid, x.numel())

l=1;t=1;r=0;b=0;
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w2, stride=1, padding=0)

aux3['r0'] = x[:, :, :, -2:]
aux3['b0'] = x[:, :, -2:, :]
aux3['br0'] = x[:, :, -2:, -2:]
mid1 = max(mid1, x.numel())

l=1;t=1;r=0;b=0;
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
out0 = F.conv2d(x, w3, stride=1, padding=0)


# 1
l=0;r=0;t=1;b=0
x= torch.cat([aux1['r0'], patches[1]], dim=-1)
del aux1['r0']
aux1['r1'] = patches[1][:, :, :, -1:]
aux1['b1'] = patches[1][:, :, -1:, :]
aux1['br1'] = patches[1][:, :, -1:, -1:]
first = max(first, x.numel())
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w1, stride=2, padding=0)

#show(x)

aux2['r1'] = x[:, :, :, -2:]
aux2['b1'] = x[:, :, -2:, :]
aux2['br1'] = x[:, :, -2:, -2:]
x = torch.cat([aux2['r0'], x], dim=-1)
del aux2['r0']
mid = max(mid, x.numel())
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w2, stride=1, padding=0)
#show(x)
aux3['r1'] = x[:, :, :, -2:]
aux3['b1'] = x[:, :, -2:, :]
aux3['br1'] = x[:, :, -2:, -2:]
x = torch.cat([aux3['r0'], x], dim=-1)
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
out1 = F.conv2d(x, w3, stride=1, padding=0)
#show(out2)


# 4
l=1;r=0;t=0;b=0;
x = torch.cat([aux1['b0'], patches[4]], dim=-2)
del aux1['b0']
aux1['r4'] = patches[4][:, :, :, -1:]
aux1['b4'] = patches[4][:, :, -1:, :]
aux1['br4'] = patches[4][:, :, -1:, -1:]
first = max(first, x.numel())
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w1, stride=2, padding=0)
#show(x)
aux2['r4'] = x[:, :, :, -2:]
aux2['b4'] = x[:, :, -2:, :]
aux2['br4'] = x[:, :, -2:, -2:]

#print(x.shape, aux1['b0'].shape)
x = torch.cat([aux2['b0'], x], dim=-2)
del aux2['b0']
mid = max(mid, x.numel())
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w2, stride=1, padding=0)
#show(x)
aux3['r4'] = x[:, :, :, -2:]
aux3['b4'] = x[:, :, -2:, :]
aux3['br4'] = x[:, :, -2:, -2:]
x = torch.cat([aux3['b0'], x], dim=-2)
del aux3['b0']
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
out4  = F.conv2d(x, w3, stride=1, padding=0)

# 5
l=0;r=0;t=0;b=0;
top = torch.cat([aux1['br0'], aux1['b1']], dim=-1)
del aux1['br0']
del aux1['b1']
x = torch.cat([aux1['r4'], patches[5]], dim=-1)
del aux1['r4']
x = torch.cat([top, x], dim=-2)
aux1['r5'] = patches[5][:, :, :, -1:]
aux1['br5'] = patches[5][:, :, -1:, -1:]
aux1['b5'] = patches[5][:, :, -1:, :]
first = max(first, x.numel())
x = F.conv2d(x, w1, stride=2, padding=0)
aux2['r5'] = x[:, :, :, -2:]
aux2['b5'] = x[:, :, -2:, :]
aux2['br5'] = x[:, :, -2:, -2:]
top = torch.cat([aux2['br0'], aux2['b1']], dim=-1)
del aux2['br0']
del aux2['b1']
x = torch.cat([aux2['r4'],x], dim=-1)
del aux2['r4']
x = torch.cat([top, x], dim=-2)
mid = max(mid, x.numel())
x = F.conv2d(x, w2, stride=1, padding=0)

aux3['r5'] = x[:, :, :, -2:]
aux3['b5'] = x[:, :, -2:, :]
aux3['br5'] = x[:, :, -2:, -2:]

top = torch.cat([aux3['br0'], aux3['b1']], dim=-1)
del aux3['br0']
del aux3['b1']
x = torch.cat([aux3['r4'], x], dim=-1)
del aux3['r4']
x = torch.cat([top, x], dim=-2)
out5 = F.conv2d(x, w3, stride=1, padding=0)

# 2
l=0;r=0;t=1;b=0;
x = torch.cat([aux1['r1'], patches[2]], dim=-1)
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
del aux1['r1']
aux1['r2'] = patches[2][:, :, :, -1:]
aux1['b2'] = patches[2][:, :, -1:, :]
aux1['br2'] = patches[2][:, :, -1:, -1:]
first = max(first, x.numel())
x = F.conv2d(x, w1, stride=2, padding=0)
aux2['r2'] = x[:, :, :, -2:]
aux2['b2'] = x[:, :, -2:, :]
aux2['br2'] = x[:, :, -2:, -2:]
x = torch.cat([aux2['r1'], x], dim=-1)
del aux2['r1']
mid = max(mid, x.numel())
x = F.pad(x ,(l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w2, stride=1, padding=0)

aux3['r2'] = x[:, :, :, -2:]
aux3['b2'] = x[:, :, -2:, :]
aux3['br2'] = x[:, :, -2:, -2:]

x = torch.cat([aux3['r1'], x], dim=-1)
del aux3['r1']
x = F.pad(x ,(l, r, t, b), mode='constant', value=0.0)
out2 = F.conv2d(x, w3, stride=1, padding=0)

#6
l=0;r=0;t=0;b=0;
top = torch.cat([aux1['br1'], aux1['b2']], dim=-1)
del aux1['br1']
del aux1['b2']
x = torch.cat([aux1['r5'], patches[6]], dim=-1)
del aux1['r5']
x = torch.cat([top, x], dim=-2)

aux1['r6'] = patches[6][:, :, :, -1:]
aux1['br6'] = patches[6][:, :, -1:, -1:]
aux1['b6'] = patches[6][:, :, -1:, :]

first = max(first, x.numel())
x = F.conv2d(x, w1, stride=2, padding=0)
aux2['r6'] = x[:, :, :, -2:]
aux2['b6'] = x[:, :, -2:, :]
aux2['br6'] = x[:, :, -2:, -2:]

top = torch.cat([aux2['br1'], aux2['b2']], dim=-1)
del aux2['br1']
del aux2['b2']
x = torch.cat([aux2['r5'], x], dim=-1)
del aux2['r5']
x = torch.cat([top, x], dim=-2)
mid = max(mid, x.numel())
x = F.conv2d(x, w2, stride=1, padding=0)

aux3['r6'] = x[:, :, :, -2:]
aux3['b6'] = x[:, :, -2:, :]
aux3['br6'] = x[:, :, -2:, -2:]

top = torch.cat([aux3['br1'], aux3['b2']], dim=-1)
del aux3['br1']
del aux3['b2']
x = torch.cat([aux3['r5'], x], dim=-1)
del aux3['r5']
x = torch.cat([top, x], dim=-2)
out6 = F.conv2d(x, w3, stride=1, padding=0)

# 8
l=1;r=0;t=0;b=0;
x = torch.cat([aux1['b4'], patches[8]], dim=-2)
del aux1['b4']
aux1['r8'] = patches[8][:, :, :, -1:]
aux1['b8'] = patches[8][:, :, -1:, :]
aux1['br8'] = patches[8][:, :, -1:, -1:]
first = max(first, x.numel())
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w1, stride=2, padding=0)

aux2['r8'] = x[:, :, :, -2:]
aux2['b8'] = x[:, :, -2:, :]
aux2['br8'] = x[:, :, -2:, -2:]

x = torch.cat([aux2['b4'], x], dim=-2)
del aux2['b4']
mid = max(mid, x.numel())
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w2, stride=1, padding=0)


aux3['r8'] = x[:, :, :, -2:]
aux3['b8'] = x[:, :, -2:, :]
aux3['br8'] = x[:, :, -2:, -2:]

x = torch.cat([aux3['b4'], x], dim=-2)
del aux3['b4']
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
out8 = F.conv2d(x, w3, stride=1, padding=0)

# 9
l=0;r=0;t=0;b=0;
aux1['r9'] = patches[9][:, :, :, -1:]
aux1['b9'] = patches[9][:, :, -1:, :]
aux1['br9'] = patches[9][:, :, -1:, -1:]
top = torch.cat([aux1['br4'], aux1['b5']], dim=-1)
del aux1['br4']
del aux1['b5']
x = torch.cat([aux1['r8'], patches[9]], dim=-1)
del aux1['r8']
x = torch.cat([top, x], dim=-2)
first = max(first, x.numel())
x = F.conv2d(x, w1, stride=2, padding=0)
aux2['r9'] = x[:, :, :, -2:]
aux2['b9'] = x[:, :, -2:, :]
aux2['br9'] = x[:, :, -2:, -2:]
top = torch.cat([aux2['br4'], aux2['b5']], dim=-1)
del aux2['br4']
del aux2['b5']
x = torch.cat([aux2['r8'], x], dim=-1)
del aux2['r8']
x = torch.cat([top, x], dim=-2)
mid = max(mid, x.numel())
x = F.conv2d(x, w2, stride=1, padding=0)

aux3['r9'] = x[:, :, :, -2:]
aux3['b9'] = x[:, :, -2:, :]
aux3['br9'] = x[:, :, -2:, -2:]

top = torch.cat([aux3['br4'], aux3['b5']], dim=-1)
del aux3['br4']
del aux3['b5']
x = torch.cat([aux3['r8'], x], dim=-1)
del aux3['r8']
x = torch.cat([top, x], dim=-2)
out9 = F.conv2d(x, w3, stride=1, padding=0)

# 10
l=0;r=0;t=0;b=0;
top = torch.cat([aux1['br5'], aux1['b6']], dim=-1)
del aux1['br5']
del aux1['b6']
aux1['r10'] = patches[10][:, :, :, -1:]
aux1['br10'] = patches[10][:, :, -1:, -1:]
aux1['b10'] = patches[10][:, :, -1:, :]

x = torch.cat([aux1['r9'], patches[10]], dim=-1)
del aux1['r9']
x = torch.cat([top, x], dim=-2)
first = max(first, x.numel())
x = F.conv2d(x, w1, stride=2, padding=0)
aux2['r10'] = x[:, :, :, -2:]
aux2['b10'] = x[:, :, -2:, :]
aux2['br10'] = x[:, :, -2:, -2:]
top = torch.cat([aux2['br5'], aux2['b6']], dim=-1)
del aux2['br5']
del aux2['b6']
x = torch.cat([aux2['r9'], x], dim=-1)
del aux2['r9']
x = torch.cat([top, x], dim=-2)
mid = max(mid, x.numel())
x = F.conv2d(x, w2, stride=1, padding=0)

aux3['r10'] = x[:, :, :, -2:]
aux3['b10'] = x[:, :, -2:, :]
aux3['br10'] = x[:, :, -2:, -2:]
top = torch.cat([aux3['br5'], aux3['b6']], dim=-1)
del aux3['br5']
del aux3['b6']
x = torch.cat([aux3['r9'], x], dim=-1)
del aux3['r9']
x = torch.cat([top, x], dim=-2)
#mid = max(mid, x.numel())
out10 = F.conv2d(x, w3, stride=1, padding=0)

# 12
l=1;r=0;t=0;b=0
x = torch.cat([aux1['b8'], patches[12]], dim=-2)
del aux1['b8']
aux1['r12'] = patches[12][:, :, :, -1:]
first = max(first, x.numel())
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w1, stride=2, padding=0)

aux2['r12'] = x[:, :, :, -2:]
x = torch.cat([aux2['b8'], x], dim=-2)
del aux2['b8']
mid = max(mid, x.numel())
l=1;r=0;t=0;b=1
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w2, stride=1, padding=0)

aux3['r12'] = x[:, :, :, -2:]
x = torch.cat([aux3['b8'], x], dim=-2)
del aux3['b8']
mid = max(mid, x.numel())
l=1;r=0;t=0;b=1
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
out12 = F.conv2d(x, w3, stride=1, padding=0)

# 13
l=0;r=0;t=0;b=0
top = torch.cat([aux1['br8'], aux1['b9']], dim=-1)
del aux1['br8']
del aux1['b9']
x = torch.cat([aux1['r12'], patches[13]], dim=-1)
del aux1['r12']
x = torch.cat([top, x], dim=-2)
aux1['r13'] = patches[13][:, :, :, -1:]
first = max(first, x.numel())
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w1, stride=2, padding=0)
aux2['r13'] = x[:, :, :, -2:]

top = torch.cat([aux2['br8'], aux2['b9']], dim=-1)
del aux2['br8']
del aux2['b9']
x = torch.cat([aux2['r12'], x], dim=-1)
del aux2['r12']
x = torch.cat([top, x], dim=-2)
mid = max(mid, x.numel())
l=0;r=0;t=0;b=1
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w2, stride=1, padding=0)
aux3['r13'] = x[:, :, :, -2:]

top = torch.cat([aux3['br8'], aux3['b9']], dim=-1)
del aux3['br8']
del aux3['b9']
x = torch.cat([aux3['r12'], x], dim=-1)
del aux3['r12']
x = torch.cat([top, x], dim=-2)
mid = max(mid, x.numel())
l=0;r=0;t=0;b=1;
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
out13 = F.conv2d(x, w3, stride=1, padding=0)

# 14
l=0;r=0;t=0;b=0
top = torch.cat([aux1['br9'], aux1['b10']], dim=-1)
del aux1['br9']
del aux1['b10']
x = torch.cat([aux1['r13'], patches[14]], dim=-1)
del aux1['r13']
x = torch.cat([top, x], dim=-2)
aux1['r14'] = patches[14][:, :, :, -1:]
first = max(first, x.numel())
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w1, stride=2, padding=0)
aux2['r14'] = x[:, :, :, -2:]

top = torch.cat([aux2['br9'], aux2['b10']], dim=-1)
del aux2['br9']
del aux2['b10']
x = torch.cat([aux2['r13'], x], dim=-1)
del aux2['r13']
x = torch.cat([top, x], dim=-2)
mid = max(mid, x.numel())
l=0;r=0;t=0;b=1;
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w2, stride=1, padding=0)
aux3['r14'] = x[:, :, :, -2:]

top = torch.cat([aux3['br9'], aux3['b10']], dim=-1)
del aux3['br9']
del aux3['b10']
x = torch.cat([aux3['r13'], x], dim=-1)
del aux3['r13']
x = torch.cat([top, x], dim=-2)
mid = max(mid, x.numel())
l=0;r=0;t=0;b=1;
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
out14 = F.conv2d(x, w3, stride=1, padding=0)

# 3
l=0;r=0;t=1;b=0;
x = torch.cat([aux1['r2'], patches[3]], dim=-1)
del aux1['r2']
aux1['b3'] = patches[3][:, :, -1:, :]
first = max(first, x.numel())
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w1, stride=2, padding=0)

l=0;r=1;t=1;b=0;
aux2['b3'] = x[:, :, -2:, :]
x = torch.cat([aux2['r2'], x], dim=-1)
del aux2['r2']
mid = max(mid, x.numel())
l=0;r=1;t=1;b=0;
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w2, stride=1, padding=0)

aux3['b3'] = x[:, :, -2:, :]
x = torch.cat([aux3['r2'], x], dim=-1)
del aux3['r2']
mid = max(mid, x.numel())
l=0;r=1;t=1;b=0;
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
out3 = F.conv2d(x, w3, stride=1, padding=0)

# 7
l=0;r=0;t=0;b=0
top = torch.cat([aux1['br2'], aux1['b3']], dim=-1)
del aux1['br2']
del aux1['b3']
x = torch.cat([aux1['r6'], patches[7]], dim=-1)
del aux1['r6']
aux1['b7'] = patches[7][:, :, -1:, :]
x = torch.cat([top, x], dim=-2)
first = max(first, x.numel())
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w1, stride=2, padding=0)

aux2['b7'] = x[:, :, -2:, :]
top = torch.cat([aux2['br2'], aux2['b3']], dim=-1)
del aux2['br2']
del aux2['b3']
x = torch.cat([aux2['r6'], x], dim=-1)
del aux2['r6']
x = torch.cat([top, x], dim=-2)
mid = max(mid, x.numel())
l=0;r=1;t=0;b=0;
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w2, stride=1, padding=0)

aux3['b7'] = x[:, :, -2:, :]
top = torch.cat([aux3['br2'], aux3['b3']], dim=-1)
del aux3['br2']
del aux3['b3']
x = torch.cat([aux3['r6'], x], dim=-1)
del aux3['r6']
x = torch.cat([top, x], dim=-2)
mid = max(mid, x.numel())
l=0;r=1;t=0;b=0;
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
out7 = F.conv2d(x, w3, stride=1, padding=0)

# 11
l=0;r=0;t=0;b=0
top = torch.cat([aux1['br6'], aux1['b7']], dim=-1)
del aux1['br6']
del aux1['b7']
x = torch.cat([aux1['r10'], patches[11]], dim=-1)
del aux1['r10']
aux1['b11'] = patches[11][:, :, -1:, :]
x = torch.cat([top, x], dim=-2)
first = max(first, x.numel())
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w1, stride=2, padding=0)

aux2['b11'] = x[:, :, -2:, :]
top = torch.cat([aux2['br6'], aux2['b7']], dim=-1)
del aux2['br6']
del aux2['b7']
x = torch.cat([aux2['r10'], x], dim=-1)
del aux2['r10']
x = torch.cat([top, x], dim=-2)
mid = max(mid, x.numel())
l=0;r=1;t=0;b=0;
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w2, stride=1, padding=0)

aux3['b11'] = x[:, :, -2:, :]
top = torch.cat([aux3['br6'], aux3['b7']], dim=-1)
del aux3['br6']
del aux3['b7']
x = torch.cat([aux3['r10'], x], dim=-1)
del aux3['r10']
x = torch.cat([top, x], dim=-2)
mid = max(mid, x.numel())
l=0;r=1;t=0;b=0;
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
out11 = F.conv2d(x, w3, stride=1, padding=0)
# 15
l=0;r=0;t=0;b=0;
top = torch.cat([aux1['br10'], aux1['b11']], dim=-1)
del aux1['br10']
del aux1['b11']
x = torch.cat([aux1['r14'], patches[15]], dim=-1)
del aux1['r14']
x = torch.cat([top, x], dim=-2)
first = max(first, x.numel())
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w1, stride=2, padding=0)

top = torch.cat([aux2['br10'], aux2['b11']], dim=-1)
del aux2['br10']
del aux2['b11']
x = torch.cat([aux2['r14'], x], dim=-1)
del aux2['r14']
x = torch.cat([top, x], dim=-2)
mid = max(mid, x.numel())
l=0;r=1;t=0;b=1;
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
x = F.conv2d(x, w2, stride=1, padding=0)

top = torch.cat([aux3['br10'], aux3['b11']], dim=-1)
del aux3['br10']
del aux3['b11']
x = torch.cat([aux3['r14'], x], dim=-1)
del aux3['r14']
x = torch.cat([top, x], dim=-2)
mid = max(mid, x.numel())
l=0;r=1;t=0;b=1;
x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
out15 = F.conv2d(x, w3, stride=1, padding=0)

row1 = torch.cat([out0, out1, out2, out3], dim=-1)
row2 = torch.cat([out4, out5, out6, out7], dim=-1)
row3 = torch.cat([out8, out9, out10, out11], dim=-1)
row4 = torch.cat([out12, out13, out14, out15], dim=-1)
out = torch.cat([row1, row2, row3, row4], dim=-2)
logging(out, 'result', False)
exit()
#exit()
a="""
for i, x in enumerate(no_pad_input):
    l = 0
    r = 0
    t = 0
    b = 0
    if i in [0, 4, 8, 12]:
        l = 1
    if i in [3, 7, 11, 15]:
        r = 1
    if i in [0, 1, 2, 3]:
        t = 1
    if i in [12, 13, 14, 15]:
        b = 1

    x = torch.unsqueeze(x, dim=0)
    x = F.pad(x, (l, r, t, b), mode='constant',value= 0.0)
    x = F.conv2d(x, w1, stride=1, padding=0)
    x = F.pad(x, (l, r, t, b), mode='constant',value= 0.0)
    x = F.conv2d(x, w2, stride=1, padding=0)
    print(f"========== {i} ===============")
    print(x)
exit()
"""
over_input = pad_input.unfold(2, res//ph+tp+bp, res//ph).unfold(3, res//pw+lp+rp, res//pw)

over_input = rearrange(over_input, 'B C ph pw H W -> (B ph pw) C H W', ph = ph, pw=pw)
out1 = F.conv2d(input, w1, stride=1, padding=1)
out2 = overlap(over_input, w1, 1, 2, 2, 2, 2)

out1 = F.conv2d(out1, w2, stride=1, padding=1)
out2 = overlap(out2, w2, 1, 1, 1, 1, 1)
logging(out1, 'conv', False)
logging(out2, 'patchpadding', True)

out2 = rearrange(out2, '(B ph pw) C H W -> B C (ph H) (pw W)', ph=ph, pw=pw)
#print(torch.allclose(out1.float(), out2.float()))
#dum = nn.Conv2d(in_channels=1, out_channels=1, kernel_numel= ,stride=1, padding=0)
