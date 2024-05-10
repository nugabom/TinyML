import torch
import numpy as np
import torch.nn as nn
from torchvision.models import mobilenet_v2 as Net
from torchvision.models.mobilenetv2 import Add
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
set_N = 4
USE_BUFFER = True

pd.options.display.max_rows = None
pd.options.display.max_columns = None
model = Net(width_mult=0.5)
features = []
stages = []
ops = []

add = False

def find_buffer_windows(stages, peak_idx):
    input_buf_iter, output_buf_iter = peak_idx, peak_idx
    input_buf_idx, output_buf_idx = 0, 0
    input_buf_min, output_buf_min = 100e+6, 100e+6
    
    while input_buf_iter >= 0:
        candidate = np.prod(stages[input_buf_iter].ops[0].input_shape)
        if isinstance(stages[input_buf_iter].ops[0], nn.Conv2d) and stages[input_buf_iter].ops[0].in_channels == 3:
            candidate = 0

        if input_buf_min > candidate:
            input_buf_idx = input_buf_iter
            input_buf_min = candidate

        input_buf_iter -= 1
        
    while len(stages) > output_buf_iter:
        candidate = np.prod(stages[output_buf_iter].ops[0].output_shape)

        if output_buf_min > candidate:
            output_buf_idx = output_buf_iter
            output_buf_min = candidate

        output_buf_iter += 1
    return input_buf_idx, output_buf_idx

def Stage_DW(op):
    ret = Stage(op)
    if hasattr(ret.ops[0], 'groups') and ret.ops[0].groups > 1:
        ret.input_shapes[0] = (0, 0, 0, 0)
        ret.output_shapes[0] = (0, 0, 0, 0)
    return ret
def get_search(stages):
    ret_ops = []
    
    for stage in stages:
        ret_ops.extend(stage.ops)
    ret_ops = [Stage(op) for op in ret_ops]
    return ret_ops

class Stage:
    def __init__(self,op):
        self.ops = [op]
        self.input_shapes = [op.input_shape]
        self.output_shapes = [op.output_shape]
#if hasattr(op, 'groups') and op.groups == 1 else (0, 0, 0, 0)]
        self.residuals = [op.residual]
        self.n_patch = 1
        self.pt = [False]
        self.bt = [False]
        self.spatials = [op.spatial]
        self.input_buf = op.input_shape
        self.output_buf = op.output_shape
        self.overlap_input = [0]
        self.overlap_output = [0]
        self.buffer_mem = 0
        self.buffer_overlap = [2]

    def __repr__(self):
        input_shapes = []
        output_shapes = []

        for i, input_shape in enumerate(self.input_shapes):
            Bi, Ci, Hi, Wi = input_shape
            input_shapes.append((Bi, Ci, Hi//self.n_patch + self.overlap_input[i], Wi//self.n_patch +  self.overlap_input[i]))

        for i, output_shape in enumerate(self.output_shapes):
            Bo, Co, Ho, Wo = output_shape
            output_shapes.append((Bo, Co, Ho//self.n_patch + self.overlap_output[i], Wo//self.n_patch + self.overlap_output[i]))

        return f"op={self.ops}, input_shape={self.input_shapes}, output_shape={self.output_shapes}, n_patch={self.n_patch}, residual={self.residuals}"

def factor_patch(last):
    B, C, H, W = last.output_shape
    patch_candidate = []
    for i in range(2, H + 1):
        if H % i == 0:
            patch_candidate.append(i)
    if len(patch_candidate) == 1:
        patch_candidate = [1]
    if len(patch_candidate) == 2:
        patch_candidate = [2]
    return patch_candidate

def compute_overlap(stage_in_windows):
    left_pad = 0
    right_pad = 0

    try_stages = deepcopy(stage_in_windows)
    #print(try_stages)
    input_buf = try_stages[0].input_buf
    out_buf = try_stages[-1].output_buf
    
    Br, Cr, Hr, Wr = 0, 0, 0, 0

    for stage in reversed(try_stages):
        stage.overlap_output = [left_pad + right_pad]
        if stage.spatials[0] == True:
            k = stage.ops[0].kernel_size[0] // 2
            stride = stage.ops[0].stride[0]
            left_pad = stride * left_pad + k
            right_pad = stride * right_pad + k
            right_pad -= 1 if stride == 2 else 0
       
        stage.overlap_input = [left_pad + right_pad]
        stage.pt = True
        stage.input_buf = input_buf
        stage.output_buf = out_buf
        if "Add" in str(stage.ops[0]):
            Br, Cr, Hr, Wr = stage.output_shapes[0]

        if getattr(stage, "residual", None):
            stage.residual = (Br, Cr, Hr, Wr)

    return try_stages

def get_window(total_stages, target_idx, wleft, wright):
    left_idx, right_idx = target_idx - 1, target_idx + 1

    flags = False
    while left_idx >= 0:
        for spatial_layers in reversed(total_stages[left_idx].spatials):
            if spatial_layers == True:
                wleft -= 1
                break
        if wleft < 0:
            flags = True
        if flags:
            left_idx += 1
            break
        left_idx -= 1

    flags = False

    while len(total_stages) > right_idx:
        for spatial_layers in total_stages[right_idx].spatials:
            if spatial_layers == True:
                wright -= 1
                break
        if wright < 0:
            flags = True
        if flags:
            right_idx -= 1
            break
        right_idx += 1

    return left_idx, right_idx
    
def inplace(op):
    if hasattr(op, 'groups') and op.groups > 1:
        #print(op)
        return False
    return False

def get_stage_with_peak_s(total_stages):
    max_idx = 0
    max_mem = -1

    stages_mems = []
    for i, stages in enumerate(total_stages):
        stages_mem = 0
        for op, ifmap, ofmap, res in zip(stages.ops, stages.input_shapes, stages.output_shapes, stages.residuals):
            if inplace(op):
                mem = max(np.prod(ifmap), np.prod(ofmap))
                if res is not None:
                    mem += np.prod(res)
                stages_mem = max(stages_mem, mem)
                continue
            mem = np.prod(ifmap) + np.prod(ofmap)
            if res is not None:
                mem += np.prod(res)
            stages_mem = max(stages_mem, mem)
        
        if len(stages.ops) > 1:
            stages_mem += np.prod(stages.input_buf) +  np.prod(stages.output_buf) + stages.buffer_mem
        stages_mems.append(stages_mem)

    arr = np.array(stages_mems)
    max_mem = np.max(arr)
    max_indices = np.where(arr == max_mem)[0].astype(int)
    return max_indices, max_mem/1024
    
def get_stage_with_peak(total_stages):
    max_idx = 0
    max_mem = -1

    stages_mems = []
    for i, stages in enumerate(total_stages):
        stages_mem = stages.buffer_mem
        for op, ifmap, ofmap, res in zip(stages.ops, stages.input_shapes, stages.output_shapes, stages.residuals):
            if inplace(op):
                mem = max(np.prod(ifmap), np.prod(ofmap))
                if res is not None:
                    mem += np.prod(res)
                stages_mem = max(stages_mem, mem)
                continue
            mem = np.prod(ifmap) + np.prod(ofmap)
            if res is not None:
                mem += np.prod(res)
            stages_mem = max(stages_mem, mem)
        
        if len(stages.ops) > 1:
            stages_mem += np.prod(stages.input_buf) +  np.prod(stages.output_buf) + stages.buffer_mem
        if max_mem < stages_mem:
            max_idx = i
            max_mem = stages_mem

    return max_idx, max_mem/1024
def conv_module_name_filter(name):
    filters = {
        'kernel_size' : 'k',
        'stride' : 's',
        'padding' : 'p',
        'bias': 'b',
        'groups' : 'g'
    }
    for k in filters:
        name = name.replace(k, filters[k])
    return name

def module_profiling(self, input, output):
    ins = input[0].size()
    outs = output.size()

    if isinstance(self, Add):
        B, C, H, W = ins
        ins = (1, C, H, W)
    self.input_shape = ins
    self.output_shape = outs

for n, mod in model.named_modules():
    if isinstance(mod, (nn.Conv2d, Add)):
        mod.name = conv_module_name_filter(mod.__repr__())
        mod.register_forward_hook(lambda m, input, output: module_profiling(m, input, output))

model(torch.randn(1, 3, 160, 160))

    
layer_id = 0
for block in model.features:
    if "Conv2dNormActivation" in str(type(block)):
        for l in block.modules():
            if isinstance(l, nn.Conv2d):
                spatial = True if l.kernel_size[0] > 1 else False
                setattr(l, 'residual', None)
                setattr(l, 'spatial', spatial)
                setattr(l, 'layer_id', layer_id)
                layer_id += 1
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
                setattr(l, 'layer_id', layer_id)
                layer_id += 1
                ops.append(l)
            if isinstance(l, nn.Conv2d):
                spatial = True if l.kernel_size[0] > 1 else False
                setattr(l, 'residual', add)
                setattr(l, 'spatial', spatial)
                setattr(l, 'layer_id', layer_id)
                layer_id += 1
                ops.append(l)


print('index, conv, spatial, input, output, residual_mem')
data = []
indices = []
for i, layer in enumerate(ops):
    #print(i, layer.name, layer.spatial, layer.input_shape, layer.output_shape, layer.residual)
    data.append([np.prod(layer.input_shape)/1024, np.prod(layer.output_shape)/1024, np.prod(layer.residual)/1024 if layer.residual is not None else None, int(layer.spatial)])
    indices.append(layer.name)

#df = pd.DataFrame(data, columns=["Input size", "Output size", "Residual", "Spatial"], index=indices)
#df = df.fillna(0)
#print(df)
#exit()
#ax = df.plot(kind='bar', stacked=True)

#for i, spine in enumerate(ax.containers[0]):
#    spatial = df.iloc[i, df.columns.get_loc("Spatial")]
#    if spatial:
#        spine.set_edgecolor('r')
#        spine.set_linewidth(2)

for op in ops:
    stages.append(Stage(op))
#plt.show()
#print(df)

def show_layer_mem(total_stages):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 'orange', 'purple']
    op_names = []
    for stages in total_stages:
        for op in stages.ops:
            op_names.append(op.name)

    fusion_box = []
    op_inputs = []
    op_outs = []
    op_residual = []
    op_layers = []
    op_stream = []
    for stages in total_stages:
        if len(stages.ops) > 1:
            buf_size = np.prod(stages.input_buf) + np.prod(stages.output_buf)
            print('debug', len(stages.ops), len(stages.input_shapes), len(stages.output_shapes), len(stages.residuals)) 
            for idx_fusion, (op, i, o, r) in enumerate(zip(stages.ops, stages.input_shapes, stages.output_shapes, stages.residuals)):
                #print(op.name)    
                if idx_fusion == 0:
                    start = op.layer_id
                
                if idx_fusion == len(stages.ops) - 1:
                    end = op.layer_id
                    fusion_box.append((start, end))
                input_size = np.prod(i)
                if hasattr(op, 'groups') and op.groups > 1:
                    out_size = 0
                out_size = np.prod(o) + input_size
                residual_size = out_size
                if r is not None:
                    residual_size += np.prod(r)
                
                op_inputs.append(input_size)
                op_outs.append(out_size)
                op_residual.append(residual_size)
                op_layers.append(residual_size + buf_size)
                op_stream.append(residual_size + buf_size + stages.buffer_mem)

        else:
            input_size = stages.input_shapes[0]
            out_size = stages.output_shapes[0]
            residual = stages.residuals[0]

            input_size = np.prod(input_size)
            out_size = np.prod(out_size) + input_size
            residual_size = out_size

            op_inputs.append(input_size)
            op_outs.append(out_size)

            if residual is not None:
                residual_size += np.prod(residual)
            else:
                pass

            op_residual.append(residual_size)
            op_layers.append(residual_size)
            op_stream.append(residual_size)

    plt.figure(figsize=(16, 8))
    x = range(len(op_names))
    print(len(op_layers), len(op_residual), len(op_outs), len(op_inputs))
    peak_mem = max(op_stream)/1024
    for i, box_info in enumerate(fusion_box):
        left, right = box_info
        left -= 0.4
        right += 0.4
        plt.axvspan(left, right, linestyle='--', color=colors[i], alpha=0.3)
    plt.axhline(peak_mem,0.0, max(x), color='r', linestyle='--')
    plt.text((0 + max(x))/2, peak_mem + 2, f"Peak Mem = {peak_mem:.1f} KB", color='r', fontsize=10)
    plt.bar(x, np.array(op_stream)/1024, label='Stream buffer')
    plt.bar(x, np.array(op_layers)/1024, label='stage I/O buffer')
    plt.bar(x, np.array(op_residual)/1024, label='residual')
    plt.bar(x, np.array(op_outs)/1024, label='output')
    plt.bar(x, np.array(op_inputs)/1024, label='input')
    plt.xticks(range(len(op_names)), op_names, rotation=90, fontsize=8)
    plt.ylabel('layer-wise peak memory (KB)')
    plt.legend()
    plt.tight_layout()

def get_buffer_memory_in_fusion(fused_stage):
    n_patch = fused_stage.n_patch

    input_buffering_idx = [i for i, buf_overlap in enumerate(fused_stage.buffer_overlap) if buf_overlap != 2]

    ret_mem = 0

    for i, buf_layer in enumerate(input_buffering_idx):
        if i == 0:
            continue
        _, ch, IH, _ = fused_stage.input_shapes[buf_layer]
        overlap = 2 - fused_stage.buffer_overlap[buf_layer]

        pixels = (n_patch + 2) * IH
        ret_mem += ch * (overlap * pixels + n_patch * overlap * overlap)
    return ret_mem
def get_stream_buffer_memory_in_fusion(fused_stage):
    n_patch = fused_stage.n_patch
    
    output_buffering_idx = [i for i, buf_overlap in enumerate(fused_stage.buffer_overlap) if buf_overlap != 2]

    ret_mem = 0
    #print(output_buffering_idx)
    #print([c for c in fused_stage.buffer_overlap if c != 2])
    #print('================')
    for i in range(len(output_buffering_idx)):
        # last spatial layer
        if i == len(output_buffering_idx) - 1:
            break

        cur_layer_idx = output_buffering_idx[i]
        next_layer_input_idx = output_buffering_idx[i + 1]

        _, _, layer_OH, _ = fused_stage.ops[cur_layer_idx].output_shape
        _, ch, fused_OH, _ = fused_stage.output_shapes[cur_layer_idx]

        overlap = 2 - fused_stage.buffer_overlap[next_layer_input_idx]
        non_overlap_OH = layer_OH // n_patch
        other_OH = non_overlap_OH + overlap
             
        # first_patch is so large
        pixels = 0
        if fused_OH > other_OH:
            if n_patch == 2:
                pixels = 2 * fused_OH + other_OH
            else:
                pixels = (n_patch - 1) * other_OH + 3 * fused_OH
        else:
            if n_patch == 2:
                pixels = fused_OH + other_OH + non_overlap_OH
            else:
                pixels = 2 * non_overlap_OH + n_patch * other_OH + fused_OH
        #print(ch * overlap * pixels)
        ret_mem += ch * overlap * pixels 

        
    return ret_mem


#setting
def try_fusion(stages, input_buf_idx, output_buf_idx, N=set_N):
    try_stages = deepcopy(stages[input_buf_idx:output_buf_idx + 1])
    #print(try_stages)
    _, _, _, H = try_stages[-1].ops[0].output_shape
    a=""" 
    if H % N != 0:
        N = 14 
    if H % N != 0:
        N = 7
    if H % N != 0:
        N = 4
    if H % N != 0:
        N = 2
    if H % N != 0:
        N = 1
    """
    if H % N != 0:
        N = 10
    if H % N != 0:
        N = 5
    if H % N != 0:
        N = 4
    if H % N != 0:
        N = 2
    if H % N != 0:
        N = 1
    if N == 1:
        print("N == 1 case")
        return try_stages, 0
    fused_stage = Stage(try_stages[-1].ops[0])
    fused_stage.input_buf = try_stages[0].ops[0].input_shape
    if isinstance(try_stages[0].ops[0], nn.Conv2d) and try_stages[0].ops[0].in_channels == 3:
        fused_stage.input_buf = (0, 0, 0, 0)
    fused_stage.n_patch = N
    #print(fused_stage.input_buf, fused_stage.output_buf)

    Br, Cr, Hr, Wr = 0, 0, 0, 0
    
    left_pad, right_pad = 0, 0

    use_buffering = False if len(try_stages) < 2 else True
    for i, stage in enumerate(reversed(try_stages)):
        overlap_out = left_pad + right_pad
        #overlap_out = 0
        Bo, Co, Ho, Wo = stage.output_shapes[0]
        output_shape = (Bo, Co, Ho//N + overlap_out, Wo//N + overlap_out)

        buffer_overlap = 2
        if stage.spatials[0] == True:
            stride = stage.ops[0].stride[0]
            left_pad = stride * left_pad + 1
            right_pad = stride * right_pad
            right_pad += 0 if stride == 2 else 1
            buffer_overlap = 1 if stride == 2 else 0

        overlap_input = left_pad + right_pad
        overlap_input = 0
        Bi, Ci, Hi, Wi = stage.input_shapes[0]
        input_shape = (Bi, Ci, Hi//N + overlap_input, Wi//N + overlap_input)
    
            
        if "Add" in str(stage.ops[0]):
            Br, Cr, Hr, Wr = output_shape

        spatial = stage.spatials[0]

        residual = stage.residuals[0]
        if residual is not None:
            residual = (Br, Cr, Hr, Wr)
        if i != 0:
            fused_stage.ops.insert(0, stage.ops[0])
            fused_stage.input_shapes.insert(0, input_shape)
            fused_stage.output_shapes.insert(0, output_shape)
            fused_stage.spatials.insert(0,spatial)
            fused_stage.overlap_input.insert(0, overlap_input)
            fused_stage.overlap_output.insert(0,overlap_out)
            fused_stage.residuals.insert(0, residual)
            fused_stage.pt.insert(0, True)
            fused_stage.bt.insert(0, use_buffering)
            fused_stage.buffer_overlap.insert(0, buffer_overlap)
        else:
            fused_stage.input_shapes[0] = input_shape
            fused_stage.output_shapes[0] = output_shape
            fused_stage.overlap_input[0] = overlap_input
            fused_stage.overlap_output[0] = overlap_out
            fused_stage.residuals[0] = residual
            fused_stage.pt[0] = True
            fused_stage.bt[0] = use_buffering
            fused_stage.buffer_overlap[0] = buffer_overlap
  
    # last spatial layer don't use buffer
    if USE_BUFFER:
        #fused_stage.buffer_mem = get_buffer_memory_in_fusion(fused_stage)
        last_spatial = [i for i, spatial_value in enumerate(fused_stage.buffer_overlap) if spatial_value != 2][-1]
        fused_stage.bt[last_spatial] = False
        fused_stage.buffer_mem = get_stream_buffer_memory_in_fusion(fused_stage)
    
    _, new_peak_stage = get_stage_with_peak([fused_stage])
    return fused_stage, new_peak_stage

def valid_check(stages):
    valid_ops = []
    for s in stages:
        for o in s.ops:
            valid_ops.append(o)
    
    assert len(valid_ops) == len(ops), 'ops error'
    for valid_layer, layer in zip(valid_ops, ops):
        assert valid_layer.layer_id == layer.layer_id, f"{valid_layer.name}, {layer.name}"

def concat_stages(total_stages, stage_window, fused_stages, left_idx, input_buf_idx, output_buf_idx, right_idx):
    full = []

    block0 = total_stages[:max(left_idx, 0)]
    if isinstance(block0, list):
        full.extend(block0)
    else:
        full.append(block0)

    block1 = stage_window[:input_buf_idx]
    if isinstance(block1, list):
        full.extend(block1)
    else:
        full.append(block1)

    block2 = fused_stages
    if isinstance(block2, list):
        full.extend(block2)
    else:
        full.append(block2)

    block3 = stage_window[output_buf_idx + 1:]
    if isinstance(block3, list):
        full.extend(block3)
    else:
        full.append(block3)

    block4 = total_stages[right_idx + 1:]
    if isinstance(block4, list):
        full.extend(block4)
    else:
        full.append(block4)
    
    #print("Typing", type(block0), type(block1), type(block2), type(block3), type(block4)) 
    return full

def foo(total_stages, peak_idx, peak_mem):
    current_total_stages = deepcopy(total_stages)

    # 0-window
    left, right = get_window(total_stages, peak_idx, 0, 0)
    stages_window = get_search(total_stages[max(left, 0): right + 1])
    temp = compute_overlap(stages_window)
    peak_idx_window, _ = get_stage_with_peak(temp)
    input_buf, output_buf = find_buffer_windows(temp, peak_idx_window)
    zero_fused_stages, sb = try_fusion(temp, input_buf, output_buf)
    #print('zero ????')
    #try_zero_fusion = total_stages[:max(left, 0)] + stages_window[:input_buf] + [zero_fused_stages] + stages_window[output_buf+1:] + total_stages[right+1:]
    try_zero_fusion = concat_stages(total_stages, stages_window, zero_fused_stages, left, input_buf, output_buf, right)
    #i = 0
    #for s in try_zero_fusion:
    #    if isinstance(s, np.ndarray):
    #        print('=========== error source = ===')
    #        print(len(s))
    #        assert 1==0, 'stop'
    #    for k in s.ops:
    #        print(i, k)
    #        
    #        i += 1
    _, try_zero_peak = get_stage_with_peak(try_zero_fusion)

    valid_check(try_zero_fusion)
    # left-window
    left, right = get_window(total_stages, peak_idx, 1, 0)
    stages_window = get_search(total_stages[max(left, 0): right + 1])
    temp = compute_overlap(stages_window)
    peak_idx_window, _ = get_stage_with_peak(temp)
    input_buf, output_buf = find_buffer_windows(temp, peak_idx_window)
    left_fused_stages, sb = try_fusion(temp, input_buf, output_buf)
    #print('left ????')
    
    #try_left_fusion = total_stages[:max(left, 0)] + stages_window[:input_buf] + [left_fused_stages] + stages_window[output_buf+1:] + total_stages[right+1:]
    try_left_fusion = concat_stages(total_stages, stages_window, left_fused_stages, left, input_buf, output_buf, right)
    _, try_left_peak = get_stage_with_peak(try_left_fusion)

    valid_check(try_left_fusion)
    # right-window
    left, right = get_window(total_stages, peak_idx, 0, 1)
    stages_window = get_search(total_stages[max(left, 0): right + 1])
    temp = compute_overlap(stages_window)
    peak_idx_window, _ = get_stage_with_peak(temp)
    input_buf, output_buf = find_buffer_windows(temp, peak_idx_window)
    right_fused_stages, sb = try_fusion(temp, input_buf, output_buf)

    #try_right_fusion = total_stages[:max(left, 0)] + stages_window[:input_buf] + [right_fused_stages] + stages_window[output_buf+1:] + total_stages[right+1:]
    try_right_fusion = concat_stages(total_stages, stages_window, right_fused_stages, left, input_buf, output_buf, right)
   
    check=""" 
    for i, s in enumerate(total_stages[:max(left, 0)]):
        for j,k in enumerate(s.ops):
            print(f"left {i} group / {j} layer / {k.name}")

    for i, s in enumerate(stages_window):
        for j,k in enumerate(s.ops):
            print(f"input-buf {i} group / {j} layer / {k.name}")

    for i, s in enumerate(stages_window[output_buf + 1:]):
        for j,k in enumerate(s.ops):
            print(f"output-buf {i} group / {j} layer / {k.name}")

    print(right_fused_stages)
    for i, s in enumerate([right_fused_stages]):
        for j,k in enumerate(s.ops):
            print(f"main {i} group / {j} layer / {k.name}")
    """
    _, try_right_peak = get_stage_with_peak(try_right_fusion)

    valid_check(try_right_fusion)
    # left-right-window
    left, right = get_window(total_stages, peak_idx, 1, 1)
    stages_window = get_search(total_stages[max(left, 0): right + 1])
    temp = compute_overlap(stages_window)
    peak_idx_window, _ = get_stage_with_peak(temp)
    input_buf, output_buf = find_buffer_windows(temp, peak_idx_window)
    left_right_fused_stages, sb = try_fusion(temp, input_buf, output_buf)

    for s in left_right_fused_stages.ops:
        print(s)
    #try_left_right_fusion = total_stages[:max(left, 0)] + stages_window[:input_buf] + [left_right_fused_stages] + stages_window[output_buf+1:] + total_stages[right+1:]
    try_left_right_fusion = concat_stages(total_stages, stages_window, left_right_fused_stages, left, input_buf, output_buf, right)
    _, try_left_right_peak = get_stage_with_peak(try_left_right_fusion)

    valid_check(try_left_right_fusion)
    a="""
    print("zero-window")
    for op in zero_fused_stages.ops:
        print(op.name)

    print("right-window")
    for op in right_fused_stages.ops:
        print(op.name)

    print("left-window")
    for op in left_fused_stages.ops:
        print(op.name)

    print("left_right-window")
    for op in left_right_fused_stages.ops:
        print(op.name)
    """
    memories = np.array([try_zero_peak, try_left_peak, try_right_peak, try_left_right_peak, peak_mem])
    print(memories)
    lowest = np.where(memories == np.min(memories))[0]

    nodes = np.array([try_zero_fusion, try_left_fusion, try_right_fusion, try_left_right_fusion, current_total_stages], dtype=object)
    nodes = nodes[lowest]

    idx = 0
    minimum = 1000
    for i, node in enumerate(nodes):
        if minimum > len(node):
            idx = i
            minimum = len(node)
    return nodes[idx], lowest[idx]

    
def get_step_iterate(total_stages, peak_idx, peak_mem, look=False):
    # 0-step
    op_id = get_layer_id_local(total_stages[peak_idx])
    current_total_stages = total_stages
    
    left, right = get_window(total_stages, peak_idx, 1, 1)
    stages_window = get_search(total_stages[max(left, 0):right + 1])
    temp = compute_overlap(stages_window)
    peak_idx_window = -1
    for i, op in enumerate(temp):
        if op_id == op.ops[0].layer_id:
            peak_idx_window = i
    assert peak_idx_window != -1, 'error'
    if look:
        for i, s in enumerate(stages_window):
            print(s.ops[0])
    #peak_idx_window, current_peak = get_stage_with_peak(temp)
    input_buf, output_buf = find_buffer_windows(temp, peak_idx_window)
    #print(total_stages[max(left, 0):right+1], '\n', input_buf, output_buf, len(temp))
    fused_stage, sb= try_fusion(temp, input_buf, output_buf, N=4)
    if True:
        print('fusion result', sb)
        for op, ifmap, ofmap in zip(fused_stage.ops, fused_stage.input_shapes, fused_stage.output_shapes):
            print(op.name ,"->", ifmap, "->", ofmap) 
    try_total_stages = total_stages[:max(left, 0)] + stages_window[:input_buf] + [fused_stage] + stages_window[output_buf+1:]+total_stages[right + 1:]

    
    print('stages')
    valid_ops = []
    for i, s in enumerate(try_total_stages):
        for k in s.ops:
            valid_ops.append(k)
    assert len(ops) == len(valid_ops), 'ops error'
    for valid_layer, layer in zip(valid_ops, ops):
        assert valid_layer.layer_id == layer.layer_id, f'{valid_layer.name}, {layer.name}'
    _, new_peak_stage = get_stage_with_peak(try_total_stages)

    if new_peak_stage >= peak_mem:
        return current_total_stages
    # recursive step
    return try_total_stages
    current_total_stages = try_total_stages

    #show_layer_mem(try_total_stages)
    peak_idx = current_total_stages.index(fused_stage)
    prev_peak_stage = new_peak_stage

    # 1 - step
   
    left, right = get_window(current_total_stages, peak_idx, 0, 0)
    left_stages_window = get_search(current_total_stages[max(left, 0):right + 1])
    #print(len(left_stages_window), left, right)
    left_stages_window = compute_overlap(left_stages_window)
    peak_idx_window, _ = get_stage_with_peak(left_stages_window)

    input_buf, output_buf = find_buffer_windows(left_stages_window, peak_idx_window)

    fused_left_stages, new_peak_stage = try_fusion(left_stages_window, input_buf, output_buf, N=4)

    try_total_stages = current_total_stages[:left] + left_stages_window[:input_buf]+[fused_left_stages] + left_stages_window[output_buf+1:]+current_total_stages[left:]
    _, new_peak_stage = get_stage_with_peak(try_total_stages)

    if new_peak_stage >= prev_peak_stage:
        return current_total_stages

    return try_total_stages
#MAIN

def goo(stages, idx, peak_mem):
    if isinstance(idx, int):
        s, _ = foo(stages, idx, peak_mem)
        return s
    temp = []
    mems = []
    for e_idx in idx:
        s, mem = foo(stages, e_idx, peak_mem)
        temp.append(s)
        mems.append(mem)

    for s in temp:
        print('goo', type(s))
    ret = temp[np.argmin(np.array(mems))]
    if isinstance(ret, np.ndarray):
        ret = ret.tolist()
    return ret
#stages = try_fusion(stages, 0, 20)
#print(stages[0].buffer_mem)
#exit()
show_layer_mem(stages)
idx, peak_mem = get_stage_with_peak_s(stages)
print(f'fusion1 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)

idx, peak_mem = get_stage_with_peak_s(stages)
print(f'fusion2 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)

idx, peak_mem = get_stage_with_peak_s(stages)
print(f'fusion3 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)

idx, peak_mem = get_stage_with_peak_s(stages)
print(f'fusion4 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)

idx, peak_mem = get_stage_with_peak_s(stages)
print(f'fusion5 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)
idx, peak_mem = get_stage_with_peak_s(stages)
print(f'fusion6 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)
idx, peak_mem = get_stage_with_peak_s(stages)
print(f'fusion7 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)

idx, peak_mem = get_stage_with_peak_s(stages)
print(f'fusion8 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)

idx, peak_mem = get_stage_with_peak_s(stages)
print(f'fusion9 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)

idx, peak_mem = get_stage_with_peak_s(stages)
print(f'fusion10 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)

idx, peak_mem = get_stage_with_peak_s(stages)
print(f'fusion11 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)

idx, peak_mem = get_stage_with_peak_s(stages)
print(f'fusion11 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)
idx, peak_mem = get_stage_with_peak_s(stages)
print(f'fusion11 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)
print(f'fusion11 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)
print(f'fusion11 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)
print(f'fusion11 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)
print(f'fusion11 : {idx}, {peak_mem}')
stages = goo(stages, idx, peak_mem)
show_layer_mem(stages)

i = 0
group = 0
# kkk
print('fusion')
for s in stages:
    if len(s.ops) > 1:
        for op, ifmap, ofmap, res in zip(s.ops, s.input_shapes, s.output_shapes, s.residuals):
            print(f"============ {group} ({s.n_patch}) {i} layer  (stream{s.buffer_mem/1024} / )==================")
            print('op\t\t',op.name)
            print('ifmap\t\t', ifmap)
            print('ofmap\t\t',ofmap)
            print('residual\t\t',res)
            i += 1
        group += 1
show_layer_mem(stages)
plt.show()
#print(idx, peak_mem)
#left, right = get_window(stages, idx, 0, 0)
#stages_window = get_search(stages[left:right+1])
#get_step_
#print(stages_window)
#temp = compute_overlap(stages_window)
#peak_idx_window, _ = get_stage_with_peak(temp)
#input_buf, output_buf = find_buffer_windows(temp, peak_idx_window)
#print(input_buf, output_buf)
