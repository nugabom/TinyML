import torch
import numpy as np
import torch.nn as nn
from torchvision.models import mobilenet_v2 as Net
from torchvision.models.mobilenetv2 import Add
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
model = Net()
features = []
stages = []
ops = []

add = False

class Stage:
    def __init__(self,op):
        self.ops = op
        self.input_shape = op.input_shape
        self.output_shape = op.output_shape
        self.residual = op.residual
        self.n_patch = 1
        self.pt = False
        self.bt = False
        self.spatial = op.spatial
        self.input_buf = op.input_shape
        self.output_buf = op.output_shape
    def __repr__(self):
        return f"op={self.op}, input_shape={self.input_shape}, output_shape={self.output_shape}, n_patch={self.n_patch}, residual={self.residual}"

def factor_patch(last):
    B, C, H, W = last.output_shape
    patch_candidate = []
    for i in range(2, H + 1):
        if H % i == 0:
            patch_candidate.append(i)
    return patch_candidate

def compute_overlap_patch(stage_in_windows, N=4):
    left_pad = 0
    right_pad = 0

    try_stages = deepcopy(stage_in_windows)
    print(try_stages)
    input_buf = try_stages[0].input_buf
    out_buf = try_stages[0].output_buf
    
    Br, Cr, Hr, Wr = 0, 0, 0, 0

    for stage in reversed(try_stages):
        Bo, Co, Ho, Wo = stage.op.output_shape

        assert Ho % N == 0, 'not valid shape'
        stage.output_shape = (Bo, Co, Ho // N + left_pad + right_pad, Wo // N + left_pad + right_pad)
        if stage.spatial == True:
            stride = stage.op.stride[0]
            left_pad = stride * left_pad + 1
            right_pad = stride * right_pad
            right_pad += 0 if stride == 2 else 1
        Bi, Ci, Hi, Wi = stage.op.input_shape

        assert Hi % N == 0, 'not valid shape'
        stage.input_shape = (Bo, Co, Hi // N + left_pad + right_pad, Wi // N + left_pad + right_pad)
        stage.pt = True
        stage.n_patch = N
        stage.input_buf = input_buf
        stage.output_buf = out_buf
        if "Add" in str(stage.op):
            Br, Cr, Hr, Wr = stage.output_shape

        if getattr(stage, "residual", None):
            stage.residual = (Br, Cr, Hr, Wr)

    return try_stages

def get_window(stages, target_idx, ws):
    left, right = target_idx - 1, target_idx + 1

    left_ws = ws
    right_ws = ws

    while left >= 0:
        
        if stages[left].spatial == True:
            left_ws -= 1

        if left_ws == 0:
            left = left + 1
            break

        left -= 1
   
    while len(stages) > right:
        if stages[right].spatial == True:
            right_ws -= 1

        if right_ws == 0:
            right = right - 1
            break
        right += 1
    return left, right
    
def get_stage_with_peak(stages):
    max_idx = 0
    max_peak = -1

    for i, stage in enumerate(stages):
        if isinstance(stage, list):
            pass
        else:
            mem = np.prod(stage.input_shape) + np.prod(stage.output_shape)
            if getattr(stage, 'residual', None):
                mem += np.prod(stage.residual)
        if mem > max_peak:
            max_idx = i
            max_peak = mem
    return max_idx, max_peak/1024
    
def conv_module_name_filter(name):
    filters = {
        'kernel_size' : 'k',
        'stride' : 's',
        'padding' : 'pad',
        'bias': 'b',
        'groups' : 'g'
    }
    for k in filters:
        name = name.replace(k, filters[k])
    return name

def module_profiling(self, input, output):
    ins = input[0].size()
    outs = output.size()

    self.input_shape = ins
    self.output_shape = outs

for n, mod in model.named_modules():
    if isinstance(mod, (nn.Conv2d, Add)):
        mod.name = conv_module_name_filter(mod.__repr__())
        mod.register_forward_hook(lambda m, input, output: module_profiling(m, input, output))

model(torch.randn(1, 3, 144, 144))

    

for block in model.features:
    if "Conv2dNormActivation" in str(type(block)):
        for l in block.modules():
            if isinstance(l, nn.Conv2d):
                spatial = True if l.kernel_size[0] > 1 else False
                setattr(l, 'residual', None)
                setattr(l, 'spatial', spatial)
                ops.append(l)
    if "InvertedResidual" in str(type(block)):
        add = False
        spatial = False
        for l in block.conv.modules():
            if isinstance(l, Add):
                add = l.input_shape
                
        for l in block.conv.modules():
            if isinstance(l, Add):
                setattr(l, 'residual', None)
                setattr(l, 'spatial', False)
                ops.append(l)
            if isinstance(l, nn.Conv2d):
                spatial = True if l.kernel_size[0] > 1 else False
                setattr(l, 'residual', add)
                setattr(l, 'spatial', spatial)
                ops.append(l)


print('index, conv, spatial, input, output, residual_mem')
data = []
indices = []
for i, layer in enumerate(ops):
    #print(i, layer.name, layer.spatial, layer.input_shape, layer.output_shape, layer.residual)
    data.append([np.prod(layer.input_shape), np.prod(layer.output_shape), np.prod(layer.residual), int(layer.spatial)])
    indices.append(layer.name)

df = pd.DataFrame(data, columns=["Input size", "Output size", "Residual", "Spatial"], index=indices)
df = df.fillna(0)

ax = df.plot(kind='bar', stacked=True)

for i, spine in enumerate(ax.containers[0]):
    spatial = df.iloc[i, df.columns.get_loc("Spatial")]
    if spatial:
        spine.set_edgecolor('r')
        spine.set_linewidth(2)

for op in ops:
    stages.append(Stage(op))
#plt.show()
#print(df)

idx, peak_mem = get_stage_with_peak(stages)
left, right = get_window(stages, idx, 2)

temp = compute_overlap_patch(stages[:14], 3)
for t in temp:
    print(t)
