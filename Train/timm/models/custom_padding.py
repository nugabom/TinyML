import torch
import einops
import torch.nn.functional as F

__all__ = ['ZeropatchPad2d', 'ReplicationPad2d', 'ReflectionPad2d', 'Patch2Feature', 'Mean_3px_Pad2d', 'Mean_2px_Pad2d', 'Mean_2px_Pad2d_detach', 'deadline4', 'Weight_2px_Pad2d']
# definition for patch with custom padding
# input : patches
# output : patches

class PaddingModule(torch.nn.Module):
    def __init__(self, padding, num_patches):
        super(PaddingModule, self).__init__()
        self.padding = torch.tensor(padding)
        self.torch_pad2d = (padding, padding, padding, padding)
        self.num_patches = torch.tensor(num_patches)
    
    def forward(self, x):
        pass

    def get_loss(self, input):
        pass
        

# post processing
class Patch2Feature(torch.nn.Module):
    def __init__(self, num_patches):
        super(Patch2Feature, self).__init__()
        self.num_patches = num_patches

    def forward(self, x):
        return einops.rearrange(x, '(B p1 p2) C H W -> B C (p1 H) (p2 W)', p1=self.num_patches, p2=self.num_patches)
class Weight_2px_Pad2d(PaddingModule):
   def __init__(self, padding, num_patches, C):
        super().__init__(padding, num_patches)
        self.topW = torch.nn.Parameter(1/2*torch.ones(size=(C,)))
        self.botW = torch.nn.Parameter(1/2*torch.ones(size=(C,)))
        self.leftW = torch.nn.Parameter(1/2*torch.ones(size=(C,)))
        self.rightW = torch.nn.Parameter(1/2*torch.ones(size=(C,)))
        
        self.topleftW = torch.nn.Parameter(torch.ones(size=(C,)))
        self.toprightW = torch.nn.Parameter(torch.ones(size=(C,)))
        self.botleftW = torch.nn.Parameter(torch.ones(size=(C,)))
        self.botrightW = torch.nn.Parameter(torch.ones(size=(C,)))
        
   def forward(self, x):
        b, C, H, W = x.size()
        P = self.num_patches
        B = b // (P ** 2)
        
        top = torch.arange(0, P, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        bot = torch.arange(P ** 2 - P, P ** 2, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        left = torch.arange(0, b, P)
        right = torch.arange(P-1, b, P)

        top_bot_kernel = torch.ones(size=(C, 1, 2, 1), device=x.device)
        left_right_kernel = torch.ones(size=(C, 1, 1, 2), device=x.device)

        topleftW = self.topleftW.view(1, -1)
        toprightW = self.toprightW.view(1, -1)
        botleftW = self.botleftW.view(1, -1)
        botrightW = self.botrightW.view(1, -1)

        topW = self.topW.view(-1, 1, 1, 1)
        botW = self.botW.view(-1, 1, 1, 1)
        leftW = self.leftW.view(-1, 1, 1, 1)
        rightW = self.rightW.view(-1, 1, 1, 1)
        
        top_pad = F.conv2d(x[:, :, :2, :], topW * top_bot_kernel, stride=1, groups=C)
        bot_pad = F.conv2d(x[:, :, H-2:, :], botW * top_bot_kernel, stride=1, groups=C)
        left_pad = F.conv2d(x[:, :, :, :2], leftW * left_right_kernel, stride=1, groups=C)
        right_pad = F.conv2d(x[:, :, :, W-2:], rightW * left_right_kernel, stride=1, groups=C)

        
        x = F.pad(x, self.torch_pad2d, mode='constant', value=0.)
        

        x[:, :, :self.padding, self.padding:W+self.padding] = top_pad
        x[:, :, H+self.padding:, self.padding:W + self.padding] = bot_pad
        x[:, :, self.padding:H + self.padding, :self.padding] = left_pad
        x[:, :, self.padding:H + self.padding, W + self.padding:] = right_pad
        
        
        x[:, :, 0, 0] = topleftW * x[:, :, 1, 1].clone()
        x[:, :, 0, -1] = toprightW * x[:, :, 1, -2].clone()
        x[:, :, -1, 0] = botleftW * x[:, :, -2, 1].clone()
        x[:, :, -1, -1] = botrightW * x[:, :, -2, -2].clone()

        x[top, :, :self.padding, :] = 0
        x[bot, :, H+self.padding, :] = 0
        x[left, :, :, :self.padding] = 0
        x[right, :, :, W+self.padding:] = 0
        return x
# for patch per patch layer
# otherwise use pytorch ZeropPad2d
class ZeropatchPad2d(PaddingModule):
    def __init__(self, padding, num_patches):
        super().__init__(padding, num_patches)
    
    def forward(self, x):
        # pad patch with custom padding
        x = F.pad(x, self.torch_pad2d, mode='constant', value=0.0)
        
        P = self.num_patches

        # remove for border of feature map
        b, C, pad_patch_h, pad_patch_w = x.size()
        B = b // (P ** 2)
        top = torch.arange(0, P, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        bot = torch.arange(P ** 2 - P, P ** 2, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        left = torch.arange(0, b, P)
        right = torch.arange(P-1, b, P)

        x[top, :, :self.padding, :] = 0
        x[bot, :, pad_patch_h-self.padding, :] = 0
        x[left, :, :, :self.padding] = 0
        x[right, :, :, pad_patch_w-self.padding:] = 0

        return x

class ReplicationPad2d(PaddingModule):
    def __init__(self, padding, num_patches):
        super().__init__(padding, num_patches)
    
    def forward(self, x):
        # pad patch with custom padding
        x = F.pad(x, self.torch_pad2d, mode='replicate')

        P = self.num_patches

        # remove for border of feature map
        b, C, pad_patch_h, pad_patch_w = x.size()
        B = b // (P ** 2)
        top = torch.arange(0, P, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        bot = torch.arange(P ** 2 - P, P ** 2, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        left = torch.arange(0, b, P)
        right = torch.arange(P-1, b, P)

        x[top, :, :self.padding, :] = 0
        x[bot, :, pad_patch_h-self.padding, :] = 0
        x[left, :, :, :self.padding] = 0
        x[right, :, :, pad_patch_w-self.padding:] = 0

        return x

class Mean_3px_Pad2d(PaddingModule):
    def __init__(self, padding, num_patches):
        super().__init__(padding, num_patches)
    def forward(self, x):
        b, C, H, W = x.size()
        P = self.num_patches
        B = b // (P ** 2)
        
        top = torch.arange(0, P, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        bot = torch.arange(P ** 2 - P, P ** 2, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        left = torch.arange(0, b, P)
        right = torch.arange(P-1, b, P)
        
        mean_kernel = torch.ones(size=(C, 1, 1, 3), device=x.device)
        mean_kernel /= 3
        
        top_pad = F.conv2d(F.pad(x[:, :, :self.padding, :], (0, 2, 0, 0), mode='constant', value=0.0), mean_kernel, stride=1, groups=C, padding=0)
        bot_pad = F.conv2d(F.pad(x[:, :, H-self.padding:, :], (0, 2, 0, 0), mode='constant', value=0.0), mean_kernel, stride=1, groups=C, padding=0)
        left_pad = F.conv2d(x[:, :, :, :3], mean_kernel, stride=1, groups=C, padding=0)
        right_pad = F.conv2d(x[:, :, :, W-3:], mean_kernel, stride=1, groups=C, padding=0)
        
        x = F.pad(x, self.torch_pad2d, mode='replicate')
        
        x[:, :, :self.padding, self.padding:W+self.padding] = top_pad
        x[:, :, H+self.padding:, self.padding:W + self.padding] = bot_pad
        x[:, :, self.padding:H + self.padding, :self.padding] = left_pad
        x[:, :, self.padding:H + self.padding, W + self.padding:] = right_pad

        x[top, :, :self.padding, :] = 0
        x[bot, :, H+self.padding, :] = 0
        x[left, :, :, :self.padding] = 0
        x[right, :, :, W+self.padding:] = 0

        return x
class ReflectionPad2d(PaddingModule):
    def __init__(self, padding, num_patches):
        super().__init__(padding, num_patches)
    
    def forward(self, x):
        # pad patch with custom padding
        x = F.pad(x, self.torch_pad2d, mode='reflect')
        
        P = self.num_patches
        # remove for border of feature map
        b, C, pad_patch_h, pad_patch_w = x.size()
        B = b // (P ** 2)
        top = torch.arange(0, P, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        bot = torch.arange(P ** 2 - P, P ** 2, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        left = torch.arange(0, b, P)
        right = torch.arange(P-1, b, P)

        x[top, :, :self.padding, :] = 0
        x[bot, :, pad_patch_h-self.padding, :] = 0
        x[left, :, :, :self.padding] = 0
        x[right, :, :, pad_patch_w-self.padding:] = 0

        return x

class Mean_2px_Pad2d(PaddingModule):
    def __init__(self, padding, num_patches):
        super().__init__(padding, num_patches)
    
    def forward(self, x):
        b, C, H, W = x.size()
        P = self.num_patches
        B = b // (P ** 2)
        
        top = torch.arange(0, P, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        bot = torch.arange(P ** 2 - P, P ** 2, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        left = torch.arange(0, b, P)
        right = torch.arange(P-1, b, P)

        top_bot_kernel = torch.ones(size=(C, 1, 2, 1), device=x.device)
        left_right_kernel = torch.ones(size=(C, 1, 1, 2), device=x.device)

        top_bot_kernel /= 2
        left_right_kernel /= 2

        top_pad = F.conv2d(x[:, :, :2, :], top_bot_kernel, stride=1, groups=C)
        bot_pad = F.conv2d(x[:, :, H-2:, :], top_bot_kernel, stride=1, groups=C)
        left_pad = F.conv2d(x[:, :, :, :2], left_right_kernel, stride=1, groups=C)
        right_pad = F.conv2d(x[:, :, :, W-2:], left_right_kernel, stride=1, groups=C)

        x = F.pad(x, self.torch_pad2d, mode='replicate')
        
        x[:, :, :self.padding, self.padding:W+self.padding] = top_pad
        x[:, :, H+self.padding:, self.padding:W + self.padding] = bot_pad
        x[:, :, self.padding:H + self.padding, :self.padding] = left_pad
        x[:, :, self.padding:H + self.padding, W + self.padding:] = right_pad

        x[top, :, :self.padding, :] = 0
        x[bot, :, H+self.padding, :] = 0
        x[left, :, :, :self.padding] = 0
        x[right, :, :, W+self.padding:] = 0

        return x

class Mean_2px_Pad2d_detach(PaddingModule):
    def __init__(self, padding, num_patches):
        super().__init__(padding, num_patches)
    
    def forward(self, x):
        b, C, H, W = x.size()
        P = self.num_patches
        B = b // (P ** 2)
        
        top = torch.arange(0, P, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        bot = torch.arange(P ** 2 - P, P ** 2, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        left = torch.arange(0, b, P)
        right = torch.arange(P-1, b, P)

        top_bot_kernel = torch.ones(size=(C, 1, 2, 1), device=x.device)
        left_right_kernel = torch.ones(size=(C, 1, 1, 2), device=x.device)

        top_bot_kernel /= 2
        left_right_kernel /= 2

        top_pad = F.conv2d(x[:, :, :2, :], top_bot_kernel, stride=1, groups=C)
        bot_pad = F.conv2d(x[:, :, H-2:, :], top_bot_kernel, stride=1, groups=C)
        left_pad = F.conv2d(x[:, :, :, :2], left_right_kernel, stride=1, groups=C)
        right_pad = F.conv2d(x[:, :, :, W-2:], left_right_kernel, stride=1, groups=C)

        x = F.pad(x, self.torch_pad2d, mode='replicate')
        
        x[:, :, :self.padding, self.padding:W+self.padding] = top_pad.detach()
        x[:, :, H+self.padding:, self.padding:W + self.padding] = bot_pad.detach()
        x[:, :, self.padding:H + self.padding, :self.padding] = left_pad.detach()
        x[:, :, self.padding:H + self.padding, W + self.padding:] = right_pad.detach()

        x[top, :, :self.padding, :] = 0
        x[bot, :, H+self.padding, :] = 0
        x[left, :, :, :self.padding] = 0
        x[right, :, :, W+self.padding:] = 0

        return x

class deadline4(PaddingModule):
    def __init__(self, padding, num_patches, C):
        super().__init__(padding, num_patches) 
        self.register_buffer('scaling_factor', torch.tensor(0.0))
        self.reg_func = deadline_function3.apply
        self.topW = torch.nn.Parameter(0.5108*torch.ones(size=(C,)))
        self.botW = torch.nn.Parameter(0.5108*torch.ones(size=(C,)))
        self.leftW = torch.nn.Parameter(0.5108*torch.ones(size=(C,)))
        self.rightW = torch.nn.Parameter(0.5108*torch.ones(size=(C,)))
        
        self.topleftW = torch.nn.Parameter(1.0986 *torch.ones(size=(C,)))
        self.toprightW = torch.nn.Parameter(1.0986 *torch.ones(size=(C,)))
        self.botleftW = torch.nn.Parameter(1.0986 * torch.ones(size=(C,)))
        self.botrightW = torch.nn.Parameter(1.0986 * torch.ones(size=(C,)))
        
    def forward(self, x):
        out = self.reg_func(x, 
                            self.topW, self.botW, self.leftW, self.rightW,
                            self.topleftW, self.toprightW, self.botleftW, self.botrightW,
                            self.padding, self.num_patches,self.scaling_factor)
        return out


class deadline_function3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, 
                topW, botW, leftW, rightW,
                topleftW, toprightW, botleftW, botrightW,
                padding, num_patches, scaling_factor):
        
        b, C, patch_h, patch_w = x.size()
        P = num_patches
        B = b // (P ** 2)
        
        pre_tl_tar_br_region = torch.arange(1, P, 1).repeat(P - 1) + torch.arange(P, P ** 2, P).repeat_interleave(P-1)
        pre_tl_tar_br_region = pre_tl_tar_br_region.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave((P - 1) ** 2)
        
        pre_tr_tar_bl_region = torch.arange(0, P - 1, 1).repeat(P - 1) + torch.arange(P, P ** 2, P).repeat_interleave(P-1)
        pre_tr_tar_bl_region = pre_tr_tar_bl_region.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave((P - 1) ** 2)
        
        pre_bl_tar_tr_region = torch.arange(1, P, 1).repeat(P - 1) + torch.arange(0, P ** 2 - P, P).repeat_interleave(P-1)
        pre_bl_tar_tr_region = pre_bl_tar_tr_region.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave((P - 1) ** 2)
        
        pre_br_tar_tl_region = torch.arange(0, P - 1, 1).repeat(P - 1) + torch.arange(0, P**2-P, P).repeat_interleave(P-1)
        pre_br_tar_tl_region = pre_br_tar_tl_region.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave((P - 1) ** 2)
        
        # border region
        pre_bot_tar_top = torch.arange(0, P * (P - 1), 1)
        pre_bot_tar_top = pre_bot_tar_top.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P * (P - 1))
        
        pre_top_tar_bot = torch.arange(P, P ** 2, 1)
        pre_top_tar_bot = pre_top_tar_bot.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P * (P - 1))
        
        pre_right_tar_left = torch.arange(0, P - 1 , 1).repeat(P) + torch.arange(0, P**2, P).repeat_interleave(P - 1)
        pre_right_tar_left = pre_right_tar_left.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P * (P - 1))
        
        pre_left_tar_right = torch.arange(1, P , 1).repeat(P) + torch.arange(0, P**2, P).repeat_interleave(P - 1)
        pre_left_tar_right = pre_left_tar_right.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P * (P - 1))        
        
        topW = 2 * torch.tanh(topW /2.0).view(1, C, 1)
        botW = 2 * torch.tanh(botW / 2.0).view(1, C, 1)
        leftW = 2 * torch.tanh(leftW / 2.0).view(1, C, 1)
        rightW = 2 * torch.tanh(rightW / 2.0).view(1, C, 1)
        
        topleftW = 2 * torch.tanh(topleftW / 2.0).view(1, C)
        toprighttW = 2 * torch.tanh(toprightW / 2.0).view(1, C)
        botleftW = 2 * torch.tanh(botleftW / 2.0).view(1, C)
        botrightW = 2 * torch.tanh(botrightW / 2.0).view(1, C)
        
        # prediction
        # corner
        prediction_top_left_corner = topleftW * x[pre_tl_tar_br_region, :, 0, 0]
        prediction_top_right_corner = toprightW * x[pre_tr_tar_bl_region, :, 0, -1]
        prediction_bot_left_corner =  botleftW  * x[pre_bl_tar_tr_region, :, -1, 0]
        prediction_bot_right_corner = botrightW * x[pre_br_tar_tl_region, :, -1, -1]
        
        #error
        prediction_top_border = topW * x[pre_top_tar_bot, :, 0, :] + (1-topW) * x[pre_top_tar_bot, :, 1, :]
        prediction_bot_border = botW * x[pre_bot_tar_top, :, -1, :] + (1-botW) * x[pre_bot_tar_top, :, -2, :]
        prediction_left_border = leftW * x[pre_left_tar_right, :, :, 0] + (1-leftW) * x[pre_left_tar_right, :, :, 1]
        prediction_right_border = rightW * x[pre_right_tar_left, :, :, -1] + (1-rightW) * x[pre_right_tar_left, :, :, -2]
        
        out = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0.)
        
        out[pre_tl_tar_br_region, :, 0, 0] = prediction_top_left_corner
        out[pre_tr_tar_bl_region, :, 0, -1] = prediction_top_right_corner
        out[pre_bl_tar_tr_region, :, -1, 0] = prediction_bot_left_corner
        out[pre_br_tar_tl_region, :, -1, -1] = prediction_bot_right_corner
        
        out[pre_top_tar_bot, :, 0, 1:1+patch_w] = prediction_top_border
        out[pre_bot_tar_top, :, -1, 1:1+patch_w] = prediction_bot_border
        out[pre_left_tar_right, :, 1:patch_h + 1, 0] = prediction_left_border
        out[pre_right_tar_left, :, 1:patch_h + 1, -1] = prediction_right_border
        
        ctx.save_for_backward( x,
                       topW, botW, leftW, rightW,
                       topleftW, toprightW, botleftW, botrightW,
                       padding, 
                       num_patches,
                       scaling_factor
                      )
        return out
    @staticmethod
    def backward(ctx, grad_outputs):
        (x,
         topW, botW, leftW, rightW,
         topleftW, toprightW, botleftW, botrightW,
         padding, 
         num_patches,
         scaling_factor) = ctx.saved_tensors
        b, C, patch_h, patch_w = x.size()
        P = num_patches
        B = b//(P ** 2)   
        
        pre_tl_tar_br_region = torch.arange(1, P, 1).repeat(P - 1) + torch.arange(P, P ** 2, P).repeat_interleave(P-1)
        pre_tl_tar_br_region = pre_tl_tar_br_region.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave((P - 1) ** 2)
        
        pre_tr_tar_bl_region = torch.arange(0, P - 1, 1).repeat(P - 1) + torch.arange(P, P ** 2, P).repeat_interleave(P-1)
        pre_tr_tar_bl_region = pre_tr_tar_bl_region.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave((P - 1) ** 2)
        
        pre_bl_tar_tr_region = torch.arange(1, P, 1).repeat(P - 1) + torch.arange(0, P ** 2 - P, P).repeat_interleave(P-1)
        pre_bl_tar_tr_region = pre_bl_tar_tr_region.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave((P - 1) ** 2)
        
        pre_br_tar_tl_region = torch.arange(0, P - 1, 1).repeat(P - 1) + torch.arange(0, P**2-P, P).repeat_interleave(P-1)
        pre_br_tar_tl_region = pre_br_tar_tl_region.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave((P - 1) ** 2)
        
        # border region
        pre_bot_tar_top = torch.arange(0, P * (P - 1), 1)
        pre_bot_tar_top = pre_bot_tar_top.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P * (P - 1))
        
        pre_top_tar_bot = torch.arange(P, P ** 2, 1)
        pre_top_tar_bot = pre_top_tar_bot.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P * (P - 1))
        
        pre_right_tar_left = torch.arange(0, P - 1 , 1).repeat(P) + torch.arange(0, P**2, P).repeat_interleave(P - 1)
        pre_right_tar_left = pre_right_tar_left.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P * (P - 1))
        
        pre_left_tar_right = torch.arange(1, P , 1).repeat(P) + torch.arange(0, P**2, P).repeat_interleave(P - 1)
        pre_left_tar_right = pre_left_tar_right.repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P * (P - 1)) 
        
        # corner
        compute_top_left_corner = x[pre_tl_tar_br_region, :, 0, 0]
        compute_top_right_corner = x[pre_tr_tar_bl_region, :, 0, -1]
        compute_bot_left_corner =  x[pre_bl_tar_tr_region, :, -1, 0]
        compute_bot_right_corner = x[pre_br_tar_tl_region, :, -1, -1]
        
        # border
        compute_top_border = x[pre_top_tar_bot, :, 0, :] - x[pre_top_tar_bot, :, 1, :]
        compute_bot_border = x[pre_bot_tar_top, :, -1, :] - x[pre_bot_tar_top, :, -2, :]
        compute_left_border = x[pre_left_tar_right, :, :, 0] - x[pre_left_tar_right, :, :, 1]
        compute_right_border =  x[pre_right_tar_left, :, :, -1] - x[pre_right_tar_left, :, :, -2]
        
        # error
        #error_top_left_corner = x[pre_br_tar_tl_region, :, -1, -1] - topleftW * compute_top_left_corner
        #error_top_right_corner = x[pre_bl_tar_tr_region, :, -1, 0] - toprightW * compute_top_right_corner
        #error_bot_left_corner = x[pre_tr_tar_bl_region, :, 0, -1] - botleftW * compute_bot_left_corner
        #error_bot_right_corner = x[pre_tl_tar_br_region, :, 0, 0] - botrightW * compute_bot_right_corner
        
        # # # border
        #error_top_border = x[pre_bot_tar_top, :, -1, :] - topW * x[pre_top_tar_bot, :, 0, :] - (1-topW) * x[pre_top_tar_bot, :, 1, :]
        #error_bot_border = x[pre_top_tar_bot, :, 0, :] - botW * x[pre_bot_tar_top, :, -1, :] - (1-botW) * x[pre_bot_tar_top, :, -2, :]
        #error_left_border = x[pre_right_tar_left, :, :, -1] - leftW * x[pre_left_tar_right, :, :, 0] - (1-leftW) * x[pre_left_tar_right, :, :, 1]
        #error_right_border = x[pre_left_tar_right, :, :, 0] - rightW * x[pre_right_tar_left, :, :, -1] - (1-rightW) * x[pre_right_tar_left, :, :, -2]
        
        #N = 4 * error_top_border.numel() +  4 * error_top_left_corner.numel()
        grad_inputs = grad_outputs[:, :, 1:patch_h + 1, 1: patch_w + 1]
        # get padding grad
        
        grad_outputs_top_left_corner = grad_outputs[pre_tl_tar_br_region, :, 0, 0] 
        grad_outputs_top_right_corner = grad_outputs[pre_tr_tar_bl_region, :, 0, -1] 
        grad_outputs_bot_left_corner = grad_outputs[pre_bl_tar_tr_region, :, -1, 0] 
        grad_outputs_bot_right_corner =  grad_outputs[pre_br_tar_tl_region, :, -1, -1]
        
        # border grad
        grad_outputs_top_border =  grad_outputs[pre_top_tar_bot, :, 0, 1:patch_w + 1] 
        grad_outputs_bot_border =  grad_outputs[pre_bot_tar_top, :, -1, 1:patch_w + 1] 
        grad_outputs_left_border = grad_outputs[pre_left_tar_right, :, 1:patch_h + 1, 0] 
        grad_outputs_right_border =  grad_outputs[pre_right_tar_left, :, 1:patch_h + 1, -1] 

        # grad
        #print(compute_top_left_corner.size(),compute_top_left_corner.size() )
        top_left_grad = torch.sum(compute_top_left_corner * grad_outputs_top_left_corner * (1-topleftW ** 2/ 4), axis=0)
        top_right_grad = torch.sum(compute_top_right_corner * grad_outputs_top_right_corner * (1-toprightW ** 2/ 4), axis=0)
        bot_left_grad = torch.sum(compute_bot_left_corner * grad_outputs_bot_left_corner * (1-botleftW ** 2 / 4), axis=0)
        bot_right_grad = torch.sum(compute_bot_right_corner * grad_outputs_bot_right_corner * (1-botrightW ** 2 / 4),  axis=0)
        
        top_grad = torch.sum(grad_outputs_top_border * compute_top_border * (1-topW ** 2 / 4), axis=[0, 2]) 
        bot_grad = torch.sum(grad_outputs_bot_border  * compute_bot_border * (1-botW ** 2 / 4), axis=[0, 2]) 
        left_grad = torch.sum(grad_outputs_left_border * compute_left_border * (1-leftW ** 2 / 4), axis=[0, 2])
        right_grad = torch.sum(grad_outputs_right_border  * compute_right_border * (1-rightW ** 2 / 4), axis=[0, 2]) 
        
        grad_inputs[pre_tl_tar_br_region, :, 0, 0] += topleftW* grad_outputs_top_left_corner 
        grad_inputs[pre_tr_tar_bl_region, :, 0, - 1] += toprightW * grad_outputs_top_right_corner
        grad_inputs[pre_bl_tar_tr_region, :, - 1, 0] +=  botleftW *grad_outputs_bot_left_corner 
        grad_inputs[pre_br_tar_tl_region, :, - 1, -1] += botrightW *grad_outputs_bot_right_corner 
        

        grad_inputs[pre_top_tar_bot, :, 0 ,:] +=  topW* grad_outputs_top_border
        grad_inputs[pre_top_tar_bot, :, 1, :] +=  (1-topW) * grad_outputs_top_border 

        grad_inputs[pre_bot_tar_top, :, -1, :] += botW * grad_outputs_bot_border 
        grad_inputs[pre_bot_tar_top, :, -2 , :] += (1 - botW) * grad_outputs_bot_border 

        grad_inputs[pre_left_tar_right, :, :, 0] += leftW * grad_outputs_left_border
        grad_inputs[pre_left_tar_right, :, :, 1] += (1-leftW) * grad_outputs_left_border 

        grad_inputs[pre_right_tar_left, :, :, -1] += rightW * grad_outputs_right_border 
        grad_inputs[pre_right_tar_left, :, :, -2] += (1-rightW) * grad_outputs_right_border 
        
        return grad_inputs, top_grad, bot_grad, left_grad, right_grad, top_left_grad, top_right_grad, bot_left_grad, bot_right_grad, None, None, None
