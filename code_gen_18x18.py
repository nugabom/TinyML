import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
#t.manual_seed(1)
from functools import partial
from typing import Any, Callable, List, Optional

from torch import nn, Tensor

from torchvision.ops.misc import Conv2dNormActivation
from torchvision.transforms._presets import ImageClassification
from torchvision.models._utils import _make_divisible
from torchvision.models import mobilenet_v2
from allocator.firstFit import FirstFit
#from zero import *


## configuration

# change for 3x3
ph = 4
pw = 4

out_ch= 32
allocator = FirstFit(10 * 1024 * 1024, False)
tensors = []
ops = []
gpid = 0

def patch_size(x, l, r, ph, pw):
    b, c, h, w = x.size()
    patch_h = h // ph
    patch_w = w // pw
    out = F.pad(x, (l, r, l, r), mode='constant', value=0.0)
    out = out.unfold(2, patch_h + l + r, patch_h).unfold(3, patch_w + l + r
    , patch_w)
    out = rearrange(out, 'B C ph pw H W -> B C (ph H) (pw W)', ph=ph, pw=pw)
    return out

def get_buffer(loc, offset):
    if loc == "front":
        return f"&buffer0[{offset}]"
    elif loc == "end":
        return f"&buffer0[{offset}]"
    elif loc == "residual":
        return f"&buffer1[{offset}]"
    else:
        raise NotImplementedError

class OutPutBuffer:
    def __init__(self):
        self.params = {}
        self.out_add = 0
        self.cur_x = 0
        self.cur_y = 0
        self.cur_c = 0
        self.H = 0
        self.W = 0
        self.final_out = None

    def insert(self, pid, t):
        _, c, h, w = t.size()
        self.cur_c = c
        self.params[f'patch{pid}_start_x'] = self.cur_x
        self.params[f'patch{pid}_start_y'] = self.cur_y
        self.params[f'patch{pid}_x'] = w
        self.params[f'patch{pid}_y'] = h
        self.params[f'patch{pid}_c'] = c
        self.params[f"patch{pid}_add"] =  self.out_add + (self.cur_y * self.W + self.cur_x) * self.cur_c

    def set_out_size(self, out):
        _, _, h, w = out.size()
        self.H = h
        self.W = w
        self.final_out = out.clone()
        assert self.final_out is not None, "error"

    def recap(self, add):
        for i in range(ph * pw):
            self.params[f"patch{i}_add"] += add

    # change for 3x3
    def loop(self, outs, cnt):
        for i in range(cnt ** 2):
            out = outs[f"out{i}"]
            if i in [cnt * j for j in range(1, cnt)]:
                self.cur_y += h

            if i % cnt == 0:
                self.cur_x = 0
            _, c, h, w = out.size()
            self.insert(i, out)
            self.cur_x += w
output_buffer = OutPutBuffer()

def execution_code_gen(pid, patch_size, ph, out_buf): 
    out_ptr = get_buffer(out_buf.params["output_buf_add"], out_buf.params[f"patch{pid}_add"])
    patch_out = get_buffer(out_buf.params[f"patch_out{pid}_buf_add"], out_buf.params[f"patch_out{pid}_buf_add_offset"])


    t, b, l, r = 0, 0, 0, 0
    row, col = pid // ph, pid % ph

    if row == 0:
        t = 1
    elif row == ph - 1:
        b = 0
    
    if col == 0:
        l = 1
    elif col == ph - 1:
        r = 0

    out_x = out_buf.params[f"patch{pid}_x"]
    out_y = out_buf.params[f"patch{pid}_y"]
    out_c = out_buf.params[f"patch{pid}_c"]

    non_overlapping = """
    main = f"\t/* patch excution {pid} */\n"
    main += f"\t\tpatch_input = getInput();\n"
    main += f"\t\tstart_x = {max(patch_size * col - 1, 0)};\n"
    main += f"\t\tstart_y = {max(patch_size * row - 1, 0)};\n"
    main += f"\t\timg_ptr = &img[(start_x + start_y * {patch_size * ph}) * 3];\n"
    main += f"\t\tpatch_input += {l * 3};\n"
    main += f"\n\t\t\tfor(h={t}; h<{patch_size + 1 - b}; h++)" + "{\n"
    main += f"\t\t\tint bytes = {(patch_size + 1 - (l + r)) * 3};\n"
    main += f"\t\t\tmemcpy (patch_input, img_ptr, bytes);\n"
    main += f"\t\t\timg_ptr += {patch_size * ph * ph * 3};\n"
    main += f"\t\t\tpatch_input += bytes;\n"
    main += f"\t\t\tpatch_input += {r * 3};\n"
    main += "\t\t\t}\n"
    main += f"\t\t\tpatch{pid}_invoke();\n"
    main += f"\t\t\toutput_ptr = {out_ptr};\n"
    main += f"\t\t\tpatch_output = {patch_out};\n"

    main += f"\t\t\tfor (h = 0; h < {out_y}; h++) " + "{\n"
    main += f"\t\t\t\tfor (w = 0; w < {out_x}; w++) " + "{\n"
    main += f"\t\t\t\t\tfor (c = 0; c < {out_c}; c++) " + "{\n"

    main += f"\t\t\t\t\t\tint output_idx = (w + h * {out_buf.W}) * {out_c} + c;\n"
    main += f"\t\t\t\t\t\toutput_ptr[output_idx] = patch_output[(w + h * {out_x}) * {out_c} + c];\n"
    main += "\t\t\t\t\t}\n"
    main += "\t\t\t\t}\n"
    main += "\t\t\t}\n"
    main += "\n"
    """
    
    #first_non_overlapping="""
    main = f"\t/* patch excution {pid} */\n"
    main += f"\t\tpatch_input = getInput();\n"
    main += f"\t\tstart_x = {patch_size * col};\n"
    main += f"\t\tstart_y = {patch_size * row};\n"
    main += f"\t\timg_ptr = &img[(start_x + start_y * {patch_size * ph}) * 3];\n"
    main += f"\n\t\t\tfor(h=0; h<{patch_size}; h++)" + "{\n"
    main += f"\t\t\tint bytes = {patch_size * 3};\n"
    main += f"\t\t\tmemcpy (patch_input, img_ptr, bytes);\n"
    main += f"\t\t\timg_ptr += {patch_size * ph * ph * 3};\n"
    main += f"\t\t\tpatch_input += bytes;\n"
    main += "\t\t\t}\n"
    main += f"\t\t\tpatch{pid}_invoke();\n"
    main += f"\t\t\toutput_ptr = {out_ptr};\n"
    main += f"\t\t\tpatch_output = {patch_out};\n"

    main += f"\t\t\tfor (h = 0; h < {out_y}; h++) " + "{\n"
    main += f"\t\t\t\tfor (w = 0; w < {out_x}; w++) " + "{\n"
    main += f"\t\t\t\t\tfor (c = 0; c < {out_c}; c++) " + "{\n"

    main += f"\t\t\t\t\t\tint output_idx = (w + h * {out_buf.W}) * {out_c} + c;\n"
    main += f"\t\t\t\t\t\toutput_ptr[output_idx] = patch_output[(w + h * {out_x}) * {out_c} + c];\n"
    main += "\t\t\t\t\t}\n"
    main += "\t\t\t\t}\n"
    main += "\t\t\t}\n"
    main += "\n"
    #"""
    return main




def gen_model(ph, pw, patch_size, ops, out_buf, schedule):
    out_buf.recap(out_buf.params["output_buf_add_offset"])

    # change for 3x3
    for i in range(ph * pw + 1):
        globals()[f"patch{i}"] = []

    for op in ops:
        if "Store" in op.op_type:
            continue
        if "Delete" in op.op_type:
            continue
        pid = op.params['patch_id']
        globals()[f"patch{pid}"].append(op)

    main = ""
    header="""
    /* Automatically generated source file */
#include <float.h>
#include "arm_nnfunctions.h"

#include "genNN.h"
#include "genModel.h"

#include "tinyengine_function.h"
#include "tinyengine_function_fp.h"


/* Variables used by all ops */
ADD_params add_params;
//Conv_Params conv_params;
//Depthwise_Params dpconv_params;
int i;
int8_t *int8ptr,*int8ptr2;
int32_t *int32ptr;
float *fptr,*fptr2,*fptr3;

signed char* getInput() {
    return &buffer0[9216];
}
signed char* getImg() {
    return &buffer0[58720];
}
signed char* getOutput() {
    return NNoutput;
}

    """

    # change for 3x3
    for i in range(ph * pw):
        main += f"void patch{i}_invoke()" + " {\n"
        for op in globals()[f"patch{i}"]:
            main += op.inference_str()
        main += "}\n"
    main += "\n\n void end2endinference(q7_t* img) {\n"
    main += "\t//stage 1\n"
    main += "\tint i, j, w, h, c, pid=0, start_x=0,start_y=0;\n" 
    main += "\tq7_t *patch_input, *img_ptr, *patch_output, *output_ptr;\n"
    for pid in schedule:
        main += execution_code_gen(pid, patch_size, ph, out_buf)
    main += "\n\n}"

    return header + main

## additional module
class INPUT(nn.Module):
    def __init__(self):
        super(INPUT, self).__init__()

    def forward(self, x):
        return x
        
class StoreBase(nn.Module):
    def __init__(self, patch_id, pad):
        super(StoreBase, self).__init__()
        self.params = {}
        self.params['patch_id'] = patch_id
        self.params['pad'] = pad
    def forward(self, x):
        return x

class RStore(StoreBase):
    def __init__(self, patch_id, pad):
        super(RStore, self).__init__(patch_id, pad)
    def forward(self, x):
        return x

class BStore(StoreBase):
    def __init__(self, patch_id, pad):
        super(BStore, self).__init__(patch_id, pad)
    def forward(self, x):
        return x

class BRStore(StoreBase):
    def __init__(self, patch_id, pad):
        super(BRStore, self).__init__(patch_id, pad)
    def forward(self, x):
        return x

class DeleteBase(nn.Module):
    def __init__(self, patch_id, pad):
        super(DeleteBase, self).__init__()
        self.params = {}
        self.params['patch_id'] = patch_id
        self.params['pad'] = pad
    def forward(self, x):
        return x 

class RDelete(DeleteBase):
    def __init__(self, patch_id, pad):
        super(RDelete, self).__init__(patch_id, pad)
    def forward(self, x):
        return x

class BDelete(DeleteBase):
    def __init__(self, patch_id, pad):
        super(BDelete, self).__init__(patch_id, pad)
    def forward(self, x):
        return x

class BRDelete(DeleteBase):
    def __init__(self, patch_id, pad):
        super(BRDelete, self).__init__(patch_id, pad)
    def forward(self, x):
        return x

class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
        self.params = {}

    def forward(self, x, y):
        #return y
        return x + y
        
## OpNode
class OpNode:
    def __init__(self, op_node):
        self.input_tensors = []
        self.output_tensors = []
        self.op_type = None
        self.params = {}
        self.op_node = op_node
        self.padding = gpadding
        self.psk = gpsk
    def getBuffer(self, loc, offset):
        #print(loc)
        if loc == "front":
            return f"&buffer0[{offset}]"
        elif loc == "end":
            return f"&buffer0[{offset}]"
        elif loc == "residual":
            return f"&buffer1[{offset}]"
        else:
            raise NotImplementedError

    def set_op_type(self, _type):
        self.op_type = _type
    
    def set_input_tensors(self, t):
        self.input_tensors.append(t)
        #print(t.graph_id)
    
    def set_output_tensors(self, t):
        self.output_tensors.append(t)
        #print(t.graph_id)

    # change for 3x3
    def inference_str(self):
        p = self.params
        call_str = f"/* layer {p['layer_id']}: {self.op_type} */\n"
        if self.op_type == "DepthwiseConv2d":
            pid = p['patch_id']
            stride = p['stride']
            
            l=0;r=0;t=0;b=0
            if pid in [i for i in range(pw)]:
                t=1
            if pid in [ph * (pw - 1) + i for i in range(pw)]:
                l=1
            if pid in [ph + i for i in range(ph)]:
                if stride == 1:
                    r=1
            if pid in [pw - 1 + ph * i for i in range(ph)]:
                if stride == 1:
                    b=1

            name = f"depthwise_kernel3x3_stride{p['stride']}_inplace_CHW"
            suffix, ho_buf = hybrid_patch_execution(self)
            if (self.psk and pid !=9) and not suffix:
                name += "_PSK"
            name += f"{suffix}"
            name += "(" 
            call_str += name
            
            # input
            call_str += f"\n\t{self.getBuffer(p['input_buf_add'], p['input_buf_add_offset'])}, "
            
            # input_x / input_y / input_c
            call_str += f"\n\t{p['input_x']}, {p['input_y']}, {p['input_c']},"
            
            # (const q7_t*) weight / offsetBias / offsetRBias / shift / multiplier
            call_str += f"\n\t(const q7_t*) CHWweight{p['weight_id']},"
            call_str += f"\n\toffsetBias{p['weight_id']},"
            call_str += f"\n\toffsetRBias{p['weight_id']},"
            call_str += f"\n\tshift{p['layer_id']},"
            call_str += f"\n\tmultiplier{p['layer_id']},"
            
            # output_offset / input_offset
            call_str += f"\n\t{p['output_zero_point']}, {p['input_zero_point']},"

            # output_activation_min / output_activation_max
            call_str += f"\n\t-128, 127,"

            # output
            call_str += f"\n\t{self.getBuffer(p['output_buf_add'], p['output_buf_add_offset'])},"
            
            # output_x / output_y / output_c
            call_str += f"\n\t{p['output_x']}, {p['output_y']}, {p['output_c']},"

            # runtime_buf
            call_str += f"\n\tsbuf,"
            call_str += f"\n\t{p['input_zero_point']}"

            if ho_buf:
                call_str += f"\n\t,{l}, {r}, {t}, {b}"
                call_str += f",{ho_buf}"
            
            if self.psk and pid != ph * pw and not suffix:
                call_str += f"\n\t, (const q7_t*) fold_weight{p['weight_id']}"
        elif self.op_type == "Add":
            #if self.padding:
            #    return ""
            p = {**p, **self.op_node.params}
            name = f"add_fpreq"
            
            suffix, ho_buf = hybrid_patch_execution(self)
            if not suffix:
                name += f"{suffix}"
                name += "("
                call_str += name
                # size
                call_str += f"\n\t{p['input_y'] * p['input_x'] * p['input_c']},"

                # input_data1
                call_str += f"\n\t{self.getBuffer(p['input_buf_add'], p['input_buf_add_offset'])}, "

                # input1_scale / input1_zero
                call_str += f"\n\t{p['input_scale']}, {p['input_zero_point']},"

                # input_data2
                call_str += f"\n\t{self.getBuffer(p['input2_buf_add'], p['input2_buf_add_offset'])},"

                # input2_scale / input2_zero
                call_str += f"\n\t{p['input2_scale']}, {p['input2_zero_point']},"
                # output_scale / zero_y
                call_str += f"\n\t{p['output_scale']}, {p['output_zero_point']},"

                # output_data
                call_str += f"\n\t{self.getBuffer(p['output_buf_add'], p['output_buf_add_offset'])}"
            else:
                pid = p['patch_id']
                input_c = p[f'patch{pid}_c']
                input_y = p[f'patch{pid}_y']
                input_x = p[f'patch{pid}_x']

                name += f"{suffix}"
                name += "("
                call_str += name

                # trunc_h / size /input_c
                call_str += f"\n\t{input_y}, "
                call_str += f"\n\t{input_x * input_c}, "
                call_str += f"\n\t{input_c}, "

                # input_data1
                call_str += f"\n\t{self.getBuffer(p['input_buf_add'], p['input_buf_add_offset'])}, "

                # input1_scale / input1_zero
                #call_str += f"\n\t{p['input_scale']}, {p['input_zero_point']},"
                call_str += f"\n\t0.00034423457691445947,-7, "

                # input_data2
                call_str += f"\n\t{self.getBuffer(p['input2_buf_add'], p['input2_buf_add_offset'])},"

                # input2_scale / input2_zero
                call_str += "\n\t2.3277170839719474e-05,3,0.00034770756610669196,-7,"
                #call_str += f"\n\t{p['input2_scale']}, {p['input2_zero_point']},"
                # output_scale / zero_y
                #call_str += f"\n\t{p['output_scale']}, {p['output_zero_point']},"

                # output_data
                call_str += f"\n\t{self.getBuffer(p['output_buf_add'], p['output_buf_add_offset'])} "
                call_str += f",{ho_buf}"
        elif self.op_type == "Conv2d":
            pid = p['patch_id']
            if p['kernel'] == 1:
                #if self.padding:
                #    return ""
                ch = p['input_c']
                if ch in [8, 16, 24, 48]:
                    name = f"convolve_1x1_s8_ch{ch}" +  " ("
                else:
                    name = f"convolve_1x1_s8" +  " ("
                call_str += name
                
                # input
                call_str += f"\n\t{self.getBuffer(p['input_buf_add'], p['input_buf_add_offset'])}, "
                
                # input_x / input_y / input_c
                call_str += f"\n\t{p['input_x']}, {p['input_y']}, {p['input_c']},"
                
                # (const q7_t*) weight / bias  / shift / multiplier
                call_str += f"\n\t(const q7_t*) weight{p['weight_id']},"
                call_str += f"\n\tbias{p['weight_id']},"
                call_str += f"\n\tshift{p['weight_id']},"
                call_str += f"\n\tmultiplier{p['weight_id']},"
                if p['layer_id'] == 2:
                    call_str += f"\n\t-5, 128,"
                elif p['layer_id'] == 3:
                    call_str += f"\n\t-128, 5,"
                elif p['layer_id'] == 5:
                    call_str += f"\n\t-7, 128,"
                elif p['layer_id'] == 6:
                    call_str += f"\n\t-128, 7,"
                elif p['layer_id'] == 8:
                    call_str += f"\n\t3, 128,"
                elif p['layer_id'] == 10:
                    call_str += f"\n\t-128, 7,"
                elif p['layer_id'] == 12:
                    call_str += f"\n\t-5, 128,"
                # output_offset / input_offset
                #call_str += f"\n\t{p['output_zero_point']}, {p['input_zero_point']},"
    
                # output_activation_min / output_activation_max
                call_str += f"\n\t-128, 127,"
    
                # output
                call_str += f"\n\t{self.getBuffer(p['output_buf_add'], p['output_buf_add_offset'])},"
                
                # output_x / output_y / output_c
                call_str += f"\n\t{p['output_x']}, {p['output_y']}, {p['output_c']},"
    
                # runtime_buf
                call_str += f"\n\tsbuf "
            elif isinstance(self.op_node, nn.Conv2d) and self.op_node.padding == (0, 0):
                pid = p['patch_id']
                l=0;r=0;t=0;b=0;
                if pid in [i for i in range(pw)]:
                    t = 1
                if pid in [ph * i for i in range(pw)]:
                    l = 1

                name = f"patchpadding_kbuf_convolve_s8_kernel3_inputch3_stride2"
                
                call_str += name
                call_str += "("

                call_str += f"\n\t{self.getBuffer(p['input_buf_add'], p['input_buf_add_offset'])}, "
                
                # input_x / input_y / input_c
                call_str += f"\n\t{p['input_x']}, {p['input_y']}, {p['input_c']},"
                
                # (const q7_t*) weight / offsetBias / shift / multiplier
                call_str += f"\n\t(const q7_t*) weight{p['weight_id']},"
                call_str += f"\n\tkbuf,"
                call_str += f"\n\tbias{p['weight_id']},"
                call_str += f"\n\tshift{p['weight_id']},"
                call_str += f"\n\tmultiplier{p['weight_id']},"
                
                # output_offset / input_offset
                #call_str += f"\n\t{p['output_zero_point']}, {p['input_zero_point']},"
                call_str += f"\n\t-128, -3,"
    
                # output_activation_min / output_activation_max
                call_str += f"\n\t-128, 127,"
    
                # output
                call_str += f"\n\t{self.getBuffer(p['output_buf_add'], p['output_buf_add_offset'])},"
                
                # output_x / output_y / output_c
                call_str += f"\n\t{p['output_x']}, {p['output_y']}, {p['output_c']},"
    
                # runtime_buf
                call_str += f"\n\tsbuf,"

                # zero point
                call_str += f"\n\t{p['input_zero_point']},"

                # pad
                call_str += f"\n\t{t}, {b}, {l}, {r}"

            else:
                name = f"convolve_s8_kernel3_inputch3_stride2_pad1"
                
                call_str += name
                if self.psk and pid != ph * pw:
                    call_str += "_PSK"
                call_str += "("
                # input
                call_str += f"\n\t{self.getBuffer(p['input_buf_add'], p['input_buf_add_offset'])}, "
                
                # input_x / input_y / input_c
                call_str += f"\n\t{p['input_x']}, {p['input_y']}, {p['input_c']},"
                
                # (const q7_t*) weight / offsetBias / shift / multiplier
                call_str += f"\n\t(const q7_t*) weight{p['weight_id']},"
                call_str += f"\n\tbias{p['weight_id']},"
                call_str += f"\n\tshift{p['weight_id']},"
                call_str += f"\n\tmultiplier{p['weight_id']},"
                
                # output_offset / input_offset
                call_str += f"\n\t{p['output_zero_point']}, {p['input_zero_point']},"
    
                # output_activation_min / output_activation_max
                call_str += f"\n\t-128, 127,"
    
                # output
                call_str += f"\n\t{self.getBuffer(p['output_buf_add'], p['output_buf_add_offset'])},"
                
                # output_x / output_y / output_c
                call_str += f"\n\t{p['output_x']}, {p['output_y']}, {p['output_c']},"
    
                # runtime_buf
                call_str += f"\n\tsbuf, kbuf, "
                call_str += f"\n\t{p['input_zero_point']}"
                if self.psk and pid != 9:
                    call_str += f"\n\t, (const q7_t*)fold_weight{p['weight_id']}"
        call_str += "\n);\n"
        return call_str
 
class buffer:
    def __init__(self, tensor, gid, op_type=None):
        self.tensor = tensor.clone()
        self.graph_id = gid
        self.allocator_idx = None
        self.op_type=op_type

      
## graph extraction fucntion
def check_in_tensors(x):
    flagging = None
    for i, y in enumerate(tensors):
        if t.equal(x.tensor, y.tensor):
            if flagging is not None:
                assert 1 == 0, "Same Tensor"
            flagging = i
    
    return flagging

def get_computation(self, input, output):
    op = OpNode(self)
    op.params['patch_id'] = gpid
    if isinstance(self, (StoreBase, DeleteBase)):
        op.params['pad'] = self.params['pad']
        pass
    else:
        op.params['layer_id'] = self.layer_id
    
    if hasattr(self, 'weight_id'):
        op.params['weight_id'] = self.weight_id

    t = None
    if isinstance(self, INPUT):
        import torch
        op.set_op_type("INPUT")
        tensors.append(buffer(1000*torch.ones_like(input[0]), -2))
        op.set_input_tensors(buffer(1000*torch.ones_like(input[0]), -2))
        ops.append(op)

        return
    if isinstance(self, (nn.Conv2d,Hconv2d_stride2, Hconv2d_stride1)):
        op.params['stride'] = self.stride[0]
        op.params['kernel'] = self.kernel_size[0]
        if self.groups > 1:
            t = "DepthwiseConv2d"
        else:
            t = "Conv2d"
    elif isinstance(self, (Add, Hadd)):
        t = "Add"
    elif isinstance(self, nn.AdaptiveAvgPool2d):
        t = "Avg"
    elif isinstance(self, nn.Linear):
        t = "Linear"
    elif isinstance(self, StoreBase):
        if isinstance(self, RStore):
            t = "RStore"
        elif isinstance(self, BStore):
            t = "BStore"
        elif isinstance(self, BRStore):
            t = "BRStore"

    elif isinstance(self, DeleteBase):
        if isinstance(self, RDelete):
            t = "RDelete"
        elif isinstance(self, BDelete):
            t = "BDelete"
        elif isinstance(self, BRDelete):
            t = "BRDelete"
    op.set_op_type(str(t))
    #print(t)
    assert t != None, "error"
    if t == "Store":
        gid = check_in_tensors(buffer(input[0], -1))
        if gid is None:
            gid = len(tensors)
        tensors.append(buffer(input[0], gid, str(type(self))))
        op.set_input_tensors(buffer(input[0], gid, str(type(self))))
        ops.append(op)
        return
    if t == "Delete":
        gid = check_in_tensors(buffer(input[0], -1))
        if gid is None:
            gid = len(tensors)
        tensors.append(buffer(input[0], gid,str(type(self))))
        op.set_input_tensors((buffer(input[0], gid, str(type(self)))))
        ops.append(op)
        return

    if t == "Linear":
        import torch
        in_buf = buffer(torch.randn(1, 1280), len(tensors) - 1,"Linear")
        out_buf = buffer(torch.randn(1280, 10), len(tensors),"Linear")
        tensors.append(out_buf)
        op.set_input_tensors(in_buf) 
        op.set_output_tensors(out_buf) 
        ops.append(op)
        return
    for i, x in enumerate(input):
        gid = check_in_tensors(buffer(x, -1))
        if gid is None:
            gid = len(tensors)
            tensors.append(buffer(x, gid,t))
            assert check_in_tensors(buffer(x, gid)) is not None, 'error'
        op.set_input_tensors(buffer(x, gid,t))
        if i == 1:
            _, input2_c, input2_y, input2_x = x.size()
            op.params['input2_c'] = input2_c
            op.params['input2_y'] = input2_y
            op.params['input2_x'] = input2_x
            op.params['input2_zero_point'] = 128
            op.params['input2_scale'] = random.uniform(-127, 128)
        elif i == 0:
            _, input_c, input_y, input_x = x.size()
            op.params['input_c'] = input_c
            op.params['input_y'] = input_y
            op.params['input_x'] = input_x
            op.params['input_zero_point'] = 128
            op.params['input_scale'] = random.uniform(0, 1)

    for i, x in enumerate(output):
        x = x.unsqueeze(dim=0)
        gid = check_in_tensors(buffer(x, -1))
        if gid is None:
            gid = len(tensors)
            tensors.append(buffer(x, gid,t))
            assert check_in_tensors(buffer(x, gid)) is not None, 'error'
        op.set_output_tensors(buffer(x, gid,t))
        _, output_c, output_y, output_x = x.size()
        op.params['output_c'] = output_c
        op.params['output_y'] = output_y
        op.params['output_x'] = output_x
        op.params['output_zero_point'] = 128
        op.params['output_scale'] = random.uniform(0, 1)

    ops.append(op)
     
## Memory scheduler code
## set model graph using in-place depth-wise convolution
def inplace_memory_fusion(ops):
    for i, op in enumerate(ops):
        if op.op_type == "DepthwiseConv2d":
            previous_output_idx = op.output_tensors[0].graph_id
            op.output_tensors[0].graph_id = op.input_tensors[0].graph_id

            if(
                i + 1 < len(ops) \
                and len(ops[i + 1].input_tensors) > 0 \
                and str(ops[i + 1].input_tensors[0].graph_id) == str(previous_output_idx)
            ):
                ops[i + 1].input_tensors[0].graph_id = op.input_tensors[0].graph_id
            
            for following_idx in range(i, len(ops)):
                for cnt, inp_tensor in enumerate(ops[following_idx].input_tensors):
                    if str(inp_tensor.graph_id) == str(previous_output_idx):
                        inp_tensor.graph_id = op.input_tensors[0].graph_id

# Memory Scheduler         
def memory_schedule(ops):
    flags = False
    for i, op in enumerate(ops):
        flag = False
        color = 'inference'
        unallocated_tensors = []
        if "Store" in op.op_type:
            flag = True
        #print('ku: ', op.op_type)
        for te in op.input_tensors:
            if te.allocator_idx is None:
                unallocated_tensors.append(te)
        for cnt, te in enumerate(op.output_tensors):
            if (
                cnt == 0 \
                and op.op_type == "DepthwiseConv2d"
            ):
                if te.allocator_idx is None:
                    unallocated_tensors.append(te)
            else:
                if te.allocator_idx is None:
                    unallocated_tensors.append(te)
        if "Delete" in op.op_type:
            assert len(unallocated_tensors) == 0, "error"
        for cnt, te in enumerate(unallocated_tensors):
            start_idx = i
            end_idx = i + 1

            for idx in range(i + 1, len(ops)):
                for input_t in ops[idx].input_tensors:
                    if str(te.graph_id) == str(input_t.graph_id):
                        if flag:
                            pass
                            #print(ops[idx].op_type)
                        end_idx = idx + 1
            if hasattr(op.op_node, "per_patch"):
                end_idx = start_idx
            if hasattr(op.op_node, "is_start_of_normal_inference_block"):
                if getattr(op.op_node, "is_start_of_normal_inference_block"):
                    print(f"kucha, {start_idx} / {end_idx}")
                    if te in op.input_tensors:
                        color='out_buf'
                        print(te.tensor.size(), te.graph_id)
                        start_idx = 0
            if flag:
                pass
                #print(f"sibal {op.op_type} / {ops[end_idx -1].op_type }: {start_idx} -> {end_idx}")    
            te.allocator_idx = allocator.addTensor(start_idx, end_idx, np.prod(te.tensor.shape), name=te.graph_id, type=color)
            for j in range(i + 1, len(ops)):
                oop = ops[j]
                for tt in oop.input_tensors:
                    if str(te.graph_id) == str(tt.graph_id):
                       tt.allocator_idx = te.allocator_idx
                for tt in oop.output_tensors:
                    if str(te.graph_id) == str(tt.graph_id):
                        tt.allocator_idx = te.allocator_idx
    for i, op in enumerate(ops):
        if (op.op_type == "DepthwiseConv2d" \
           and op.params['stride'] == 2
           ):
            if op.input_tensors[0].allocator_idx == op.output_tensors[0].allocator_idx:
                allocator.rectangles[op.input_tensors[0].allocator_idx]["stride2_inplace_idx"] = i

# HO module class
class Hadd(nn.Module):
    def __init__(self, AddModule, parent):
        super(Hadd, self).__init__()
        self.mid_add = AddModule
        self.region = {}
        self.patch_id = None
        self.pad = 1
        self.layer_id = self.mid_add.layer_id
        self.params = {}
        self.hb = 0
    def extra_repr(self):
        s = (f"(Hadd)")
        return s.format(**self.__dict__)

    def hcat(self, tensors, patch=None):
        if len(tensors) == 2 and patch is not None:
            assert 1 == 0, f"tensors: {len(tensors)} and patch: not None"
        if patch is not None:
            out = t.cat([self.region[f"{tensors[0]}"], patch], dim=-1)
            temp = RDelete(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"{tensors[0]}"])
            del self.region[f"{tensors[0]}"]
            return out
        if len(tensors) == 2:
            out = t.cat([self.region[f"{tensors[0]}"], self.region[f"{tensors[1]}"]], dim=-1)
            temp = BRDelete(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"{tensors[0]}"])
            del self.region[f"{tensors[0]}"]
            temp = BDelete(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"{tensors[1]}"])
            del self.region[f"{tensors[1]}"]
            return out

    def vcat(self, tensor, patch):
        temp=BDelete(self.patch_id, self.pad)
        temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
       
        out = t.cat([self.region[f"{tensor[0]}"], patch], dim=-2)
        temp(self.region[f"{tensor[0]}"])
        del self.region[f"{tensor[0]}"]
        return out
    
    def fcat(self, top_tensors, left_tensor, patch):
        top = self.hcat(top_tensors)
        #print(top.size(), patch.size())
        patch = self.hcat(left_tensor, patch)
        return t.cat([top, patch], dim=-2)
    
    # change for 3x3
    def store_to_next(self, x):
        if self.patch_id in [ph * (pw -1)+ i for i in range(ph)]:
            self.region[f"r{self.patch_id}"] = x[:, :, :, -1:].clone()
            temp = RStore(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"r{self.patch_id}"])

        elif self.patch_id in [pw - 1 + ph * i for i in range(0, ph - 1)]:
            self.region[f"b{self.patch_id}"] = x[:, :, -1:, :].clone()
            temp = BStore(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"b{self.patch_id}"])

        else:
            self.region[f"r{self.patch_id}"] = x[:, :, :-1, -1:].clone()
            temp=RStore(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"r{self.patch_id}"])

            self.region[f"b{self.patch_id}"] = x[:, :, -1:, :-1].clone()
            temp=BStore(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"b{self.patch_id}"])

            self.region[f"br{self.patch_id}"] = x[:, :, -1:, -1:].clone()
            temp=BRStore(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"br{self.patch_id}"])

    # change for 3x3
    def forward(self, x, y):
        pid = self.patch_id
        if pid != ph * pw - 1:
            self.store_to_next(x)
        mem = sum([np.prod(seg.size()) for seg in self.region.values()])
        self.hb = mem
        #print(f"{self.patch_id}, {mem}")
        remove_right = []
        remove_bot = []
        for j in range(ph):
            for i in range(pw -1):
                remove_right.append(ph * j + i)

        for j in range(ph - 1):
            for i in range(pw):
                remove_bot.append(ph * j + i)

        if pid in remove_right:
            x = x[:, :, :, :-1]
        if pid in remove_bot:
            x = x[:, :, :-1, :]
        _, input_c, input_y, input_x = x.size()
        self.params[f"patch{pid}_c"] = input_c
        self.params[f"patch{pid}_y"] = input_y
        self.params[f"patch{pid}_x"] = input_x

        row, col = pid//ph, pid % pw

        if pid == 0:
            pass
        elif pid in [i for i in range(1,ph)]:
            x = self.hcat([f"r{col - 1}"], x)
        elif pid in [ph * i for i in range(ph)]:
            x = self.vcat([f"b{(row - 1) * ph}"], x)
        else:
            x = self.fcat([f"br{ (row - 1) * ph + col - 1}", f"b{(row - 1) * ph + col}"], [f"r{row * ph + col - 1}"], x)

        out = self.mid_add(x, y)
        return out

class Hconv2d_stride1(nn.Module):
    def __init__(self, ConvModule, parent):
        super(Hconv2d_stride1, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.patch_id = None
        self.region = {}
        self.even = None
        self.n_macs = 0
        self.pad = 2
        self.layer_id = self.mid_conv.layer_id
        self.kernel_size = self.mid_conv.kernel_size
        self.stride = self.mid_conv.stride
        self.groups = self.mid_conv.groups
        self.weight_id = self.mid_conv.weight_id
        self.hb = 0

    def extra_repr(self):
        s = (f"(Hconv): padding={self.mid_conv.padding}")
        return s.format(**self.__dict__)
    
    def hcat(self, tensors, patch=None):
        if len(tensors) == 2 and patch is not None:
            assert 1 == 0, f"tensors: {len(tensors)} and patch: not None"
        if patch is not None:
            out = t.cat([self.region[f"{tensors[0]}"], patch], dim=-1)
            temp = RDelete(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"{tensors[0]}"])
            del self.region[f"{tensors[0]}"]
            return out
        if len(tensors) == 2:
            out = t.cat([self.region[f"{tensors[0]}"], self.region[f"{tensors[1]}"]], dim=-1)
            temp = BRDelete(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"{tensors[0]}"])
            del self.region[f"{tensors[0]}"]
            temp = BDelete(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"{tensors[1]}"])
            del self.region[f"{tensors[1]}"]
            return out

    def vcat(self, tensor, patch):
        out = t.cat([self.region[f"{tensor[0]}"], patch], dim=-2)
        temp = BDelete(self.patch_id, self.pad)
        temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
        temp(self.region[f"{tensor[0]}"])
        del self.region[f"{tensor[0]}"]
        return out
    
    def fcat(self, top_tensors, left_tensor, patch):
        top = self.hcat(top_tensors)
        #print(top.size(), patch.size())
        patch = self.hcat(left_tensor, patch)
        return t.cat([top, patch], dim=-2)

    # change for 3x3
    def store_to_next(self, x):
        # bottom
        if self.patch_id in [ph * (pw - 1) + i for i in range(pw)]:
            self.region[f"r{self.patch_id}"] = x[:, :, :, -2:].clone()
            temp = RStore(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"r{self.patch_id}"])
        # right
        elif self.patch_id in [pw - 1 + ph * i for i in range(ph - 1)]:
            self.region[f"b{self.patch_id}"] = x[:, :, -2:, :].clone()
            temp = BStore(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"b{self.patch_id}"])
        else:
            self.region[f"r{self.patch_id}"] = x[:, :, :, -2:].clone()
            temp = RStore(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"r{self.patch_id}"])

            self.region[f"b{self.patch_id}"] = x[:, :, -2:, :].clone()
            temp = BStore(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"b{self.patch_id}"])

            self.region[f"br{self.patch_id}"] = x[:, :, -2:, -2:].clone()
            temp = BRStore(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"br{self.patch_id}"])

    # change for 3x3
    def forward(self, x):
        pid = self.patch_id
        # 27 profiling
        if pid != pw * ph - 1:
            self.store_to_next(x)
        mem = sum([np.prod(seg.size()) for seg in self.region.values()])
        self.hb = mem
        #print('s1 in', x.size())
        #print(f"{self.patch_id}, {mem}")
        l=0;r=0;t=0;b=0;

        if pid in [i for i in range(pw)]:
            t = 1
        if pid in [ph * (pw - 1) + i for i in range(pw)]:
            b = 1
        if pid in [ph * i for i in range(ph)]:
            l = 1
        if pid in [pw - 1 + ph * i for i in range(ph)]:
            r = 1

        row, col = pid//ph, pid % pw

        if pid == 0:
            pass
        elif pid in [i for i in range(1, pw)]:
            x = self.hcat([f"r{col - 1}"], x)
        elif pid in [ph * i for i in range(ph)]:
            x = self.vcat([f"b{(row - 1) * ph}"], x)
        else:
            x = self.fcat([f"br{ (row - 1) * ph + col - 1}", f"b{(row - 1) * ph + col}"], [f"r{row * ph + col - 1}"], x)

        x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
        #print('s1 out', x.size())
        if pid == pw * ph -1:
            assert not self.region, "error"
        out = self.mid_conv(x)
        return out

class Hconv2d_stride2(nn.Module):
    def __init__(self, ConvModule, parent):
        super(Hconv2d_stride2, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.patch_id = None
        self.region = {}
        self.even = None
        self.pad = 2
        self.layer_id = self.mid_conv.layer_id
        self.kernel_size = self.mid_conv.kernel_size
        self.stride = self.mid_conv.stride
        self.groups = self.mid_conv.groups
        self.weight_id = self.mid_conv.weight_id
        self.hb = 0

    def extra_repr(self):
        s = (f"(Hconv): padding={self.mid_conv.padding}")
        return s.format(**self.__dict__)
    
    def hcat(self, tensors, patch=None):
        if len(tensors) == 2 and patch is not None:
            assert 1 == 0, f"tensors: {len(tensors)} and patch: not None"
        if patch is not None:
            out = t.cat([self.region[f"{tensors[0]}"], patch], dim=-1)
            temp = RDelete(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"{tensors[0]}"])
            del self.region[f"{tensors[0]}"]
            return out
        if len(tensors) == 2:
            out = t.cat([self.region[f"{tensors[0]}"], self.region[f"{tensors[1]}"]], dim=-1)
            temp = BRDelete(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"{tensors[0]}"])
            del self.region[f"{tensors[0]}"]
            temp = BDelete(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"{tensors[1]}"])
            del self.region[f"{tensors[1]}"]
            return out

    def vcat(self, tensor, patch):
        temp = BDelete(self.patch_id, self.pad)
        temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
        out = t.cat([self.region[f"{tensor[0]}"], patch], dim=-2)
        temp(self.region[f"{tensor[0]}"])
        del self.region[f"{tensor[0]}"]
        return out
    
    def fcat(self, top_tensors, left_tensor, patch):
        top = self.hcat(top_tensors)
        #print(top.size(), patch.size())
        patch = self.hcat(left_tensor, patch)
        return t.cat([top, patch], dim=-2)

    # change for 3x3
    def store_to_next(self, x):
        # bottom
        if self.patch_id in [ph * (pw - 1) + i for i in range(pw)]:
            temp = RStore(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            self.region[f"r{self.patch_id}"] = x[:, :, :, -2 + self.even:].clone()
            temp(self.region[f"r{self.patch_id}"])
        # right
        elif self.patch_id in [pw - 1 + ph * i for i in range(ph - 1)]:
            temp = BStore(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            self.region[f"b{self.patch_id}"] = x[:, :, -2 + self.even:, :].clone()
            temp(self.region[f"b{self.patch_id}"])
        else:
            self.region[f"r{self.patch_id}"] = x[:, :, :, -2 + self.even:].clone()
            temp = RStore(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            temp(self.region[f"r{self.patch_id}"])

            temp = BStore(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            self.region[f"b{self.patch_id}"] = x[:, :, -2 + self.even:, :].clone()
            temp(self.region[f"b{self.patch_id}"])

            temp = BRStore(self.patch_id, self.pad)
            temp.register_forward_hook(lambda m, input, output: get_computation(m, input, output))
            self.region[f"br{self.patch_id}"] = x[:, :, -2 + self.even:, -2 + self.even:].clone()
            temp(self.region[f"br{self.patch_id}"])

    # change for 3x3
    def forward(self, x):
        pid = self.patch_id
        if self.even is None:
            _, _, h, w = x.size()
            if h % 2 == 0:
                self.even = 1
                self.pad = 1
            else:
                self.even = 0
        if pid != ph * pw - 1:
            self.store_to_next(x)
        mem = sum([np.prod(seg.size()) for seg in self.region.values()])
        self.hb = mem
        #print(f"{self.patch_id}, {mem}")
        l=0;r=0;t=0;b=0;
        #print('s2 in',x.size())

        if pid in [i for i in range(pw)]:
            t = 1
        if pid in [ph * (pw - 1) + i for i in range(pw)]:
            b = 1
            if x.size()[-1] % 2 == 0:
                b = 0
        if pid in [ph * i for i in range(ph)]:
            l = 1
        if pid in [pw - 1 + ph * i for i in range(ph)]:
            r = 1
            if x.size()[-1] % 2 == 0:
                r = 0

        row, col = pid//ph, pid % pw

        if pid == 0:
            pass
        elif pid in [i for i in range(1, pw)]:
            x = self.hcat([f"r{col - 1}"], x)
        elif pid in [ph * i for i in range(ph)]:
            x = self.vcat([f"b{(row - 1) * ph}"], x)
        else:
            x = self.fcat([f"br{ (row - 1) * ph + col - 1}", f"b{(row - 1) * ph + col}"], [f"r{row * ph + col - 1}"], x)

        x = F.pad(x, (l, r, t, b), mode='constant', value=0.0)
        #print('s2 out',x.size())
        out = self.mid_conv(x)
        return out

## model replace dict

module_to_mapping = {
    (3, 3, 1): Hconv2d_stride1,
    (3, 3, 2): Hconv2d_stride2,
}

## replace Module sub functions

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
    k = layer.stride
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
    
## replace Module

def change_model(model, num_patches, Module_To_Mapping, patch_list):
    i = 0
    for name, target in model.named_modules():
        parent = None
        if i == len(patch_list):
            break
        if is_spatial(target):
            i += 1
            if patch_list[i - 1] == 0:
                continue
            attrs = name.split('.')
            submodule = model
            for attr in attrs[:-1]:
                submodule = getattr(submodule, attr)
                if 'InvertedResidual' in str(type(submodule)):
                    parent = submodule

            replace = Module_To_Mapping[get_attr(target)](target, parent)
            setattr(submodule, attrs[-1], replace)
        if isinstance(target, Add):
            if patch_list[i - 1] == 0:
                continue
            attrs = name.split('.')
            submodule = model
            for attr in attrs[:-1]:
                submodule = getattr(submodule, attr)
                if 'InvertedResidual' in str(type(submodule)):
                    parent = submodule

            replace = Hadd(target, parent)
            setattr(submodule, attrs[-1], replace)
    a="""
    for name, mod in model.named_modules():
        if isinstance(mod, Add):
            submodule = model
            attrs = name.split('.')
            for attr in attrs[:-2]:
                submodule = getattr(submodule, attr)
                if 'InvertedResidual' in str(type(submodule)):
                    parent = submodule
            sub = getattr(submodule, '1')
            sub = getattr(sub, '0')
            if 'Hconv2d' in str(type(sub)):
                for attr in attrs[-2:-1]:
                    submodule = getattr(submodule, attr)
                replace = Hadd(target, parent)
                setattr(submodule, attrs[-1], replace)
    """
    
## IRB block

class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            pass
            #norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.region = {}
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                #Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
                nn.Conv2d(in_channels=inp, out_channels=hidden_dim, kernel_size=1) 
            )
        layers.extend(
            [
                # dw
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    padding=1,
                    kernel_size=3
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                #norm_layer(oup),
            ]
        )
        if self.use_res_connect:
            layers.append(Add())
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

        for n, m in self.conv.named_modules():
            #print(type(m))
            if not isinstance(m, (nn.Conv2d, Add, nn.Sequential)):
                assert 1 == 0, 'error'

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return self.conv[-1](x, self.conv[:-1](x))
        else:
            return self.conv(x)               

def set_patch_id(model, pid):
    for name, mod in model.named_modules():
        mod.patch_id = pid
        #if hasattr(mod, 'patch_id'):
        #    mod.patch_id = pid

class MB(nn.Module):
    def __init__(self):
        super(MB, self).__init__()
        IRB = InvertedResidual
        self.patching = None
        # LARGE MODEL CONFIG
        a="""
        self.conv0 = Conv2dNormActivation(3, 32, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        self.conv1 = IRB(inp=32, oup=16, stride=1, expand_ratio=1)
        self.conv2 = IRB(inp=16, oup=24, stride=2, expand_ratio=6)
        self.conv3 = IRB(inp=24, oup=24, stride=1, expand_ratio=6)
        self.conv4 = IRB(inp=24, oup=32, stride=2, expand_ratio=6)
        self.conv5 = IRB(inp=32, oup=32, stride=1, expand_ratio=6)
        self.conv6 = IRB(inp=32, oup=32, stride=1, expand_ratio=6)
        self.conv7 = IRB(inp=32, oup=64, stride=2, expand_ratio=6)
        self.conv8 = IRB(inp=64, oup=64, stride=1, expand_ratio=6)
        self.conv9 = IRB(inp=64, oup=64, stride=1, expand_ratio=6)
        self.conv10 = IRB(inp=64, oup=64, stride=1, expand_ratio=6)
        self.conv11 = IRB(inp=64, oup=96, stride=1, expand_ratio=6)
        self.conv12 = IRB(inp=96, oup=96, stride=1, expand_ratio=6)
        self.conv13 = IRB(inp=96, oup=96, stride=1, expand_ratio=6)
        self.conv14 = IRB(inp=96, oup=160, stride=2, expand_ratio=6)
        self.conv15 = IRB(inp=160, oup=160, stride=1, expand_ratio=6)
        self.conv16 = IRB(inp=160, oup=160, stride=1, expand_ratio=6)
        self.conv17 = IRB(inp=160, oup=320, stride=1, expand_ratio=6)
        self.conv18 = Conv2dNormActivation(320, 1280, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, 1000)
        """
        # TINY MODEL w0.5 r160
        self.conv0 = Conv2dNormActivation(3, 16, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        self.conv1 = IRB(inp=16, oup=8, stride=1, expand_ratio=1)
        self.conv2 = IRB(inp=8, oup=16, stride=2, expand_ratio=6)
        self.conv3 = IRB(inp=16, oup=16, stride=1, expand_ratio=6)
        self.conv4 = IRB(inp=16, oup=16, stride=2, expand_ratio=6)
        self.conv5 = IRB(inp=16, oup=16, stride=1, expand_ratio=6)
        self.conv6 = IRB(inp=16, oup=16, stride=1, expand_ratio=6)
        self.conv7 = IRB(inp=16, oup=32, stride=2, expand_ratio=6)
        self.conv8 = IRB(inp=32, oup=32, stride=1, expand_ratio=6)
        self.conv9 = IRB(inp=32, oup=32, stride=1, expand_ratio=6)
        self.conv10 = IRB(inp=32, oup=32, stride=1, expand_ratio=6)
        self.conv11 = IRB(inp=32, oup=48, stride=1, expand_ratio=6)
        self.conv12 = IRB(inp=48, oup=48, stride=1, expand_ratio=6)
        self.conv13 = IRB(inp=48, oup=48, stride=1, expand_ratio=6)
        self.conv14 = IRB(inp=48, oup=80, stride=2, expand_ratio=6)
        self.conv15 = IRB(inp=80, oup=80, stride=1, expand_ratio=6)
        self.conv16 = IRB(inp=80, oup=80, stride=1, expand_ratio=6)
        self.conv17 = IRB(inp=80, oup=160, stride=1, expand_ratio=6)
        self.conv18 = Conv2dNormActivation(160, 640, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        self.classifier = nn.Linear(640, 1000)
        a="""
        # TINY MODEL w0.35 r144
        self.conv0 = Conv2dNormActivation(3, 16, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        self.conv1 = IRB(inp=16, oup=8, stride=1, expand_ratio=1)
        self.conv2 = IRB(inp=8, oup=8, stride=2, expand_ratio=6)
        self.conv3 = IRB(inp=8, oup=8, stride=1, expand_ratio=6)
        self.conv4 = IRB(inp=8, oup=16, stride=2, expand_ratio=6)
        self.conv5 = IRB(inp=16, oup=16, stride=1, expand_ratio=6)
        self.conv6 = IRB(inp=16, oup=16, stride=1, expand_ratio=6)
        self.conv7 = IRB(inp=16, oup=24, stride=2, expand_ratio=6)
        self.conv8 = IRB(inp=24, oup=24, stride=1, expand_ratio=6)
        self.conv9 = IRB(inp=24, oup=24, stride=1, expand_ratio=6)
        self.conv10 = IRB(inp=24, oup=24, stride=1, expand_ratio=6)
        self.conv11 = IRB(inp=24, oup=32, stride=1, expand_ratio=6)
        self.conv12 = IRB(inp=32, oup=32, stride=1, expand_ratio=6)
        self.conv13 = IRB(inp=32, oup=32, stride=1, expand_ratio=6)
        self.conv14 = IRB(inp=32, oup=56, stride=2, expand_ratio=6)
        self.conv15 = IRB(inp=56, oup=56, stride=1, expand_ratio=6)
        self.conv16 = IRB(inp=56, oup=56, stride=1, expand_ratio=6)
        self.conv17 = IRB(inp=56, oup=112, stride=1, expand_ratio=6)
        self.conv18 = Conv2dNormActivation(112, 1280, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        self.classifier = nn.Linear(1280, 1000)
        """
        setattr(self.conv4.conv[-1], 'per_patch', True)
        setattr(self.conv5.conv[0], "is_start_of_normal_inference_block", True)
        #self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def is_hybrid(self):
        for name, mod in self.named_modules():
            if hasattr(mod, 'patch_id'):
                return True

        return False


    # change for 3x3
    def patch_forward(self, x):
        #print("Patch execution")
        global gpid
        global per_patch_buffer
        global tensors
        # patch_resizer
        pad = 0
        _, _, H, W = x.size()
        ps = H // ph 
        input_patches = []
        accum_y = pad + ps 
        start_y = 0
        for j in range(ph):
            accum_x = ps + pad
            for i in range(pw):
                if i == 0:
                    start_x = 0
                
                if i == pw - 1:
                    input_patches.append(x[:, :, start_y:accum_y, start_x:])
                    break
                
                if j == ph - 1:
                    input_patches.append(x[:, :, start_y:, start_x:accum_x])

                else:
                    input_patches.append(x[:, :, start_y:accum_y, start_x:accum_x])
                start_x = accum_x; accum_x +=ps
            start_y = accum_y; accum_y += ps
        for i, c in enumerate(input_patches):
            row, col = i // ph, i % ph
            #print(f"({row}, {col}) -> {c.size()}")
        #exit()
        #x = rearrange(x, 'B C (ph H) (pw W) -> (B ph pw) C H W', ph=ph, pw=pw)
        #tensors.append((buffer(x, 0)))
        #schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        schedule = list(range(ph * pw)) 
        out_dict = {}
        for pid, x1 in enumerate(input_patches):
            set_patch_id(self, pid)
            gpid = pid
            #x1 = x[pid]
            #x1 = t.unsqueeze(x1, dim=0) 

            x1 = self.conv0(x1)
            x1 = self.conv1(x1)
            x1 = self.conv2(x1)
            x1 = self.conv3(x1)
            x1 = self.conv4(x1)
            out_dict[f"out{pid}"] = x1

        out = []
        for j in range(ph):
            row = []
            for i in range(pw):
                row.append(out_dict[f"out{j * ph + i}"])
            row = t.cat(row, dim=-1)
            out.append(row)
        out = t.cat(out, dim=-2)
        print(out.size())
        #print(out.size()) 
        #print(row1.size(), row2.size(), row3.size())
        global out_buf
        out_buf = np.prod(out.size())
        
        output_buffer.set_out_size(out)
        output_buffer.loop(out_dict, ph)
        #allocator.addTensor(
        #te.allocator_idx = allocator.addTensor(start_idx, end_idx, np.prod(te.tensor.shape), name=te.graph_id, type='inference')
        #allocator.addTensor(0, 25000, np.prod(out.shape), name="output_buffer", type='inference')
        per_patch_buffer = out 
        gpid = ph * pw
        x = self.conv5(out)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = x.mean([2, 3])
        #x = self.pool(x)
        x = self.classifier(x)
        a="""
    def patch_forward(self, x):
        print("Patch execution")
        x = self.conv0(x)
        x = self.conv1(x)
        x1 = self.conv2(x)

        x1 = rearrange(x1, 'B C (ph H) (pw W) -> (B ph pw) C H W', ph=ph, pw=pw)
        schedule = [0, 1, 4, 5, 2, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15]
        
        out_dict = {}
        for pid in schedule:
            set_patch_id(self, pid)
            x = x1[pid]
            set_patch_id(self, pid)
            x = t.unsqueeze(x, dim=0) 
            x = self.conv3(x)
            x = self.conv4(x)
            out_dict[f"out{pid}"] = x

        row1 = t.cat([out_dict["out0"], out_dict["out1"], out_dict["out2"], out_dict["out3"]], dim=-1)
        row2 = t.cat([out_dict["out4"], out_dict["out5"], out_dict["out6"], out_dict["out7"]], dim=-1)
        row3 = t.cat([out_dict["out8"], out_dict["out9"], out_dict["out10"], out_dict["out11"]], dim=-1)
        row4 = t.cat([out_dict["out12"], out_dict["out13"], out_dict["out14"], out_dict["out15"]], dim=-1)
        out = t.cat([row1, row2, row3, row4], dim=-2)
        
        x = self.conv5(out)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = x.mean([2, 3])
        #x = self.pool(x)
        x = self.classifier(x)
        return x
        """
        return x
    def norm_forward(self, x):
        print("Normal execution")
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        #x = self.pool(x)
        #x = x.reshape(1, -1)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x    

    def forward(self, x):
        #if hasattr(self, "patch"):
        if self.patching is not None:
            return self.patch_forward(x)
        #elif self.is_hybrid():
        #    return self.patch_forward(x)
        return self.norm_forward(x)
# TEST function


def anaylsis_hybrid(ops):
    for i, op in enumerate(ops):
        if op.op_type == "DepthwiseConv2d" or op.op_type == "Add":
            for find_store_op_idx in range(i - 1, 0, -1):
                prev = ops[find_store_op_idx]
                if "Store" in prev.op_type or "Delete" in prev.op_type:
                    op.params[prev.op_type] = prev
                else:
                    break
def allocate_SRAM(ops):
  for i, op in enumerate(ops):
      for cnt, te in enumerate(op.input_tensors):
          if t.equal(te.tensor, output_buffer.final_out):
              output_buffer.params['output_buf_add_offset'] = allocator.getIdxAddress(te.allocator_idx)
              output_buffer.params['output_buf_add'] = "front"
          if cnt == 0:
              op.params["input_buf_add_offset"] = allocator.getIdxAddress(te.allocator_idx)
              op.params["input_buf_add"] = "front"
          elif cnt == 1:
              op.params["input2_buf_add_offset"] = allocator.getIdxAddress(te.allocator_idx)
              op.params["input2_buf_add"] = "front"
          elif cnt == 2:
              op.params["input3_buf_add_offset"] = allocator.getIdxAddress(te.allocator_idx)
              op.params["input3_buf_add"] = "front"
          op.input_tensors[cnt].buffer_name = "buffer0"
          op.input_tensors[cnt].buffer_address = allocator.getIdxAddress(te.allocator_idx)
      
      for cnt, te in enumerate(op.output_tensors):
          if cnt == 0:
              op.params["output_buf_add_offset"] = allocator.getIdxAddress(te.allocator_idx)
              op.params["output_buf_add"] = "front"
              op.output_tensors[cnt].buffer_name = "buffer0"
              op.output_tensors[cnt].buffer_address = allocator.getIdxAddress(te.allocator_idx)
              if hasattr(op.op_node, 'per_patch'):
                  pid = op.params['patch_id']
                  output_buffer.params[f"patch_out{pid}_buf_add_offset"] = allocator.getIdxAddress(te.allocator_idx)
                  output_buffer.params[f"patch_out{pid}_buf_add"] = "front"
          if cnt == 1:
              op.params["output2_buf_add_offset"] = allocator.getIdxAddress(te.allocator_idx)
              op.params["output2_buf_add"] = "front"
              op.output_tensors[cnt].buffer_name = "buffer0"
              op.output_tensors[cnt].buffer_address = sel.allocator.getIdxAddress(te.allocator_idx)

def set_numbering(model):  
    execution_order = 0
    weight_order = 0
    for n, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, Add, nn.Linear)):
            block_name = n.split('.')[0]
            mod.layer_id = execution_order
            mod.block_id = block_name
            execution_order +=1 
            if isinstance(mod, Add):
                continue
            mod.weight_id = weight_order
            weight_order += 1

def hybrid_patch_execution(op):
    if op.op_type == "DepthwiseConv2d" or op.op_type == "Add":
        r_store = None
        b_store = None
        br_store = None
        r_delete = None
        b_delete = None
        br_delete = None

        store_type = None
        delete_type = None

        p = op.params
        pid = p['patch_id']
        
        if "RStore" in p:
            r_store = p["RStore"]
        if "BStore" in p:
            b_store = p["BStore"]
        if "BRStore" in p:
            br_store = p["BRStore"]
        if "RDelete" in p:
            r_delete = p["RDelete"]
        if "BDelete" in p:
            b_delete = p["BDelete"]
        if "BRDelete" in p:
            br_delete = p["BRDelete"]
        
        # aggregate
        if r_store is not None and b_store is not None and br_store is not None:
            store_type = 3 
        elif r_store is not None:
            store_type = 1
        elif b_store is not None:
            store_type = 2
        elif r_store is None and b_store is None and br_store is None:
            store_type = 0
        else:
            assert 1 == 0, "baga store type"

        if r_delete is not None and b_delete is not None and br_delete is not None:
            delete_type = 3
        elif r_delete is not None:
            delete_type = 1
        elif b_delete is not None:
            delete_type = 2
        elif r_delete is None and b_delete is None and br_delete is None:
            delete_type = 0
        else:
            assert 1 == 0, "baga delete type"
        # case 0 Store Full / Delete X

        name = None

        if op.op_type == "DepthwiseConv2d":
            name = f"depthwise_kernel_3x3_stride{op.params['stride']}_inplace_CHW"
        if op.op_type == "Add":
            name = "Add"
        suffix=""
        data=""
        #print(f"{op.op_type} at {pid}: ", end='')
        if store_type == 3 and delete_type == 0:
            suffix += "_FStore"
            data += f"\n\t{r_store.getBuffer(r_store.params['input_buf_add'], r_store.params['input_buf_add_offset'])}, "
            data += f"\n\t{b_store.getBuffer(b_store.params['input_buf_add'], b_store.params['input_buf_add_offset'])}, "
            data += f"\n\t{br_store.getBuffer(br_store.params['input_buf_add'], br_store.params['input_buf_add_offset'])},"
            
            data += f"\n\t{r_store.params['pad']}"
        if store_type == 3 and delete_type == 1:
            suffix += "_FStore_RLoad"
            data += f"\n\t{r_store.getBuffer(r_store.params['input_buf_add'], r_store.params['input_buf_add_offset'])}, "
            data += f"\n\t{b_store.getBuffer(b_store.params['input_buf_add'], b_store.params['input_buf_add_offset'])}, "
            data += f"\n\t{br_store.getBuffer(br_store.params['input_buf_add'], br_store.params['input_buf_add_offset'])}, "

            data += f"\n\t{r_delete.getBuffer(r_delete.params['input_buf_add'], r_delete.params['input_buf_add_offset'])}, "
            data += f"\n\t{r_store.params['pad']}"
        if store_type == 2 and delete_type == 1:
            suffix += "_BStore_RLoad"
            data += f"\n\t{b_store.getBuffer(b_store.params['input_buf_add'], b_store.params['input_buf_add_offset'])}, "

            data += f"\n\t{r_delete.getBuffer(r_delete.params['input_buf_add'], r_delete.params['input_buf_add_offset'])},"
            data += f"\n\t{b_store.params['pad']}"
        if store_type == 3 and delete_type == 2:
            suffix += "_FStore_BLoad"
            data += f"\n\t{r_store.getBuffer(r_store.params['input_buf_add'], r_store.params['input_buf_add_offset'])}, "
            data += f"\n\t{b_store.getBuffer(b_store.params['input_buf_add'], b_store.params['input_buf_add_offset'])}, "
            data += f"\n\t{br_store.getBuffer(br_store.params['input_buf_add'], br_store.params['input_buf_add_offset'])}, "

            data += f"\n\t{b_delete.getBuffer(b_delete.params['input_buf_add'], b_delete.params['input_buf_add_offset'])},"
            data += f"\n\t{r_store.params['pad']}"
        if store_type == 3 and delete_type == 3:
            suffix += "_FStore_FLoad"
            data += f"\n\t{r_store.getBuffer(r_store.params['input_buf_add'], r_store.params['input_buf_add_offset'])}, "
            data += f"\n\t{b_store.getBuffer(b_store.params['input_buf_add'], b_store.params['input_buf_add_offset'])}, "
            data += f"\n\t{br_store.getBuffer(br_store.params['input_buf_add'], br_store.params['input_buf_add_offset'])}, "

            data += f"\n\t{r_delete.getBuffer(r_delete.params['input_buf_add'], r_delete.params['input_buf_add_offset'])}, "
            data += f"\n\t{b_delete.getBuffer(b_delete.params['input_buf_add'], b_delete.params['input_buf_add_offset'])}, "
            data += f"\n\t{br_delete.getBuffer(br_delete.params['input_buf_add'], br_delete.params['input_buf_add_offset'])},"
            data += f"\n\t{r_store.params['pad']}"
        if store_type == 2 and delete_type == 3:
            suffix += "_BStore_FLoad"
            data += f"\n\t{b_store.getBuffer(b_store.params['input_buf_add'], b_store.params['input_buf_add_offset'])}, "

            data += f"\n\t{r_delete.getBuffer(r_delete.params['input_buf_add'], r_delete.params['input_buf_add_offset'])}, "
            data += f"\n\t{b_delete.getBuffer(b_delete.params['input_buf_add'], b_delete.params['input_buf_add_offset'])}, "
            data += f"\n\t{br_delete.getBuffer(br_delete.params['input_buf_add'], br_delete.params['input_buf_add_offset'])},"
            data += f"\n\t{b_store.params['pad']}"
        if store_type == 1 and delete_type == 2:
            suffix += "_RStore_BLoad"
            data += f"\n\t{r_store.getBuffer(r_store.params['input_buf_add'], r_store.params['input_buf_add_offset'])}, "

            data += f"\n\t{b_delete.getBuffer(b_delete.params['input_buf_add'], b_delete.params['input_buf_add_offset'])},"
            data += f"\n\t{r_store.params['pad']}"
        if store_type == 1 and delete_type == 3:
            suffix += "_RStore_FLoad"
            data += f"\n\t{r_store.getBuffer(r_store.params['input_buf_add'], r_store.params['input_buf_add_offset'])}, "

            data += f"\n\t{r_delete.getBuffer(r_delete.params['input_buf_add'], r_delete.params['input_buf_add_offset'])}, "
            data += f"\n\t{b_delete.getBuffer(b_delete.params['input_buf_add'], b_delete.params['input_buf_add_offset'])}, "
            data += f"\n\t{br_delete.getBuffer(br_delete.params['input_buf_add'], br_delete.params['input_buf_add_offset'])},"
            data += f"\n\t{r_store.params['pad']}"
        if store_type == 0 and delete_type == 3:
            suffix += "_FLoad"

            data += f"\n\t{r_delete.getBuffer(r_delete.params['input_buf_add'], r_delete.params['input_buf_add_offset'])}, "
            data += f"\n\t{b_delete.getBuffer(b_delete.params['input_buf_add'], b_delete.params['input_buf_add_offset'])}, "
            data += f"\n\t{br_delete.getBuffer(br_delete.params['input_buf_add'], br_delete.params['input_buf_add_offset'])},"
            data += f"\n\t{r_delete.params['pad']}"
        if store_type == 0 and delete_type == 0:
            pass
        
        #print(f"{name}_{suffix} ({data})\n\n")
        return (suffix, data) 
        
def record_io_size(self, x, y):
    x = x[0]
    pid = self.patch_id
    if not hasattr(self, 'params'):
        self.params = {}

    self.params[f"{pid}_input_size"] = np.prod(x.size()[1:])
    self.params[f"{pid}_output_size"] = np.prod(y.size()[1:])
    if hasattr(self,'hb'):
        self.params[f"{pid}_buffer"] = self.hb
    print(pid, self.params.keys())

def set_io_profiling(model):
    exclude = []
    
    for n, mod, in model.named_modules():
        if 'mid' in n:
            exclude.append(mod)

    for n, mod in model.named_modules():
        if mod in exclude:
            continue
        if isinstance(mod, (nn.Conv2d, Hadd, nn.AdaptiveAvgPool2d, nn.Linear,  Hconv2d_stride1, Hconv2d_stride2)):
            mod.register_forward_hook(lambda m, input, output: record_io_size(m, input, output))

def set_computation_profiling(model):
    exclude = []
    for n, mod in model.named_modules():
        if 'mid' in n:
            exclude.append(mod)

    for n, mod in model.named_modules():
        if mod in exclude:
            continue
        if isinstance(mod, (nn.Conv2d, Add, nn.AdaptiveAvgPool2d, nn.Linear, Hadd, Hconv2d_stride1, Hconv2d_stride2)):
            mod.register_forward_hook(lambda m, input, output: get_computation(m, input, output))

gpadding =False
gpsk = False
# change for 3x3
#schedule = [0, 1, 3, 4, 2, 5, 6, 7, 8]
schedule = list(range(ph * pw))
# MAIN
left=1;right=0
x = t.randn(1, 3, 160, 160)
x = patch_size(x, left, right, ph, pw)

model = MB()
#b = MB()
model.conv0[0].padding = (0, 0)
#model.conv1.conv[0].padding = (0, 0)
#model.conv2.conv[1].padding = (0, 0)
#model.conv3.conv[1].padding = (0, 0)
#model.conv4.conv[1].padding = (0, 0)
#model.load_state_dict(b.state_dict())
#base.eval()
#setattr(model, 'patch', True)
set_numbering(model)
#change_model(model, ph, module_to_mapping, [0, 1, 1, 1, 1])
model.patching = True
#set_io_profiling(model)
set_computation_profiling(model)
model(x)
a="""
blocks = [
    [model.conv0[0]],
    model.conv1.conv,
    model.conv2.conv,
    model.conv3.conv,
    model.conv4.conv,
]

block_memory = []
features = []
buffers = []
#blocks = [model.conv2]
for pid in range(ph * pw):
    result = 0
    for i, mod in enumerate(blocks):
        buf = 0
        buf += sum([sum([m.params[f"{pid}_buffer"] for m in l if hasattr(m, "region")]) for l in blocks[:i]])   
        if pid > 0:
            buf += sum([sum([m.params[f"{pid - 1}_buffer"] for m in l if hasattr(m, "region")]) for l in blocks[i+1:]])   
        print('buf', buf)
        if True:
            if len(mod) == 4:
                #print([l for l in mod[1].params if "buffer" in l])
                #print([l for l in mod[3].params if "buffer" in l])

                #result = buf  + \
                if hasattr(mod[1].params, f'{pid}_buffer'):
                    buf = buf + mod[1].params[f"{pid}_buffer"]  
                buf = 0
                result = buf + \
                        max(
                            mod[0].params[f"{pid}_input_size"] + mod[0].params[f"{pid}_output_size"],   
                            mod[1].params[f"{pid}_input_size"] + mod[0].params[f"{pid}_input_size"],
                            mod[2].params[f"{pid}_input_size"] + mod[2].params[f"{pid}_output_size"]
                            )
            elif len(mod) == 2:
                #result = buf +\
                buf = buf + mod[0].params[f"{pid}_buffer"] 
                result = buf + \
                        max(
                            mod[0].params[f"{pid}_input_size"],
                            mod[1].params[f"{pid}_input_size"] + mod[1].params[f"{pid}_output_size"]
                            )
                
            elif len(mod) == 3:
                #result = buf  + 
                buf= buf + mod[1].params[f"{pid}_buffer"] 
                result = buf +\
                        max(
                            mod[0].params[f"{pid}_input_size"] + mod[0].params[f"{pid}_output_size"],   
                            mod[1].params[f"{pid}_input_size"],
                            mod[2].params[f"{pid}_input_size"] + mod[2].params[f"{pid}_output_size"]
                            )
            elif len(mod) == 1:
                buf = buf + mod[0].params[f"{pid}_buffer"] 
                result = buf +  mod[0].params[f"{pid}_input_size"] + mod[0].params[f"{pid}_output_size"]
                #result = buf + mod[0].params[f"{pid}_input_size"] + mod[0].params[f"{pid}_output_size"]
            else:
                raise NotImplementedError
            features.append((result-buf)/1024)
            block_memory.append(result)
            buffers.append(buf)
print(max(buffers), max(features))
exit()
bm = []
print('output_buf', out_buf/1024)
for block in block_memory:
    bm.append((block + out_buf)/1024)
block_memory = [b/1024 for b in block_memory]

label_x = []
x = list(range(ph * pw*5))
for i in range(ph * pw):
    label_x.extend([0, 1, 2, 3, 4])

for i, block in enumerate(bm):
    pas
    #print(f"=================== patch {i//5} ===============")
    #print(f"\t{block}")
print(len(x), len(bm), len(buffers), len(features))
plt.bar(x, bm, color='r', label="Output buffer")
plt.bar(x, block_memory, color='g', label="Stream buffer")
plt.bar(x, features, color='b', label="Activation")
plt.axhline(y=(32256/(6*6) + out_buf)/1024, color='k')
plt.xticks(x, label_x)
plt.title("MBV2-w0.35-144 Patch 18x18")
plt.ylabel("Peak Mem (kB)")
plt.xlabel("per-patch inf block id")
plt.legend()
plt.show()
plt.savefig("./All_Buffering_Patch18x18_MB_W35_R144png")
exit()
"""
inplace_memory_fusion(ops)
memory_schedule(ops)
allocator.sortSize()
allocator.allocate()
allocate_SRAM(ops)
anaylsis_hybrid(ops)
print(allocator.get_peak())
allocator.visualize('./life_Large.png')
#print(gen_model(ph, pw, 144//ph, ops, output_buffer, schedule))
exit()

def produce_sorting(rec):
    return rec['start']
life_time = sorted(allocator.rectangles,key=produce_sorting)
for te in life_time:
    print(te)
for i, op in enumerate(ops):
    if 'layer_id' not in op.params:
        continue
    mem_peak = 0
    op_tensors = [rec for rec in life_time if not (rec['end'] < i)]
    op_tensors = [rec for rec in op_tensors if rec['end'] <= i+1]

    mem_peak = sum([rec['size'] for rec in op_tensors])
 
