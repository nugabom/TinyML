import torch
import numpy as np
import torch.nn as nn
from torchvision.models import mobilenet_v2 as Net
from torchvision.models.mobilenetv2 import Add
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy


def main():
    set_N = 10
    USE_BUFFER = True

    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    model = Net(width_mult=0.5)
    features = []
    stages = []
    ops = []

    add = False

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

    for op in ops:
        stages.append(Stage(op))

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

if __name__ == "__main__":
    main()
