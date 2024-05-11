import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def get_patch(feature, pad_tuple, ph, pw):
    _, _, h, w = feature.size()
    patch_h = h // ph
    patch_w = w // pw
    padded_x = F.pad(feature, pad_tuple, mode='constant', value=0.0)
    l, r, _, _ = pad_tuple
    out = padded_x.unfold(2, patch_h + l + r, patch_h).unfold(3, patch_w + l + r, patch_w)
    return out

def sampling_top(pred, target):
    result = pred.clone()
    result[:, :, 2, :, :, :] = traget[:, :, 2, :, :, :]
    return result

def sampling_left(pred, target):
    result = pred.clone()
    result[:, :, :, 2, :, :] = traget[:, :, :, 2, :, :]
    return result

class Interpolate_S1(nn.Module):
    def __init__(self, ConvModule, padding, n_patch):
        super(Interpolate_S1, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0,0)
        self.ph = n_patch
        self.pw = n_patch
        self.pad = self.mid_conv.kernel_size[0] // 2
        self.pad_tuple = (self.pad, self.pad, self.pad, self.pad)

        self.top1 = nn.Parameter(0.5*torch.ones(1, self.mid_conv.in_channels, 1, 1, 2, 1))
        self.bot1 = nn.Parameter(0.5*torch.ones(1, self.mid_conv.in_channels, 1, 1, 2, 1))
        self.left1 = nn.Parameter(0.5*torch.ones(1, self.mid_conv.in_channels, 1, 1, 1, 2))
        self.right1 = nn.Parameter(0.5*torch.ones(1, self.mid_conv.in_channels, 1, 1, 1, 2))

        self.top2 = nn.Parameter(0.5*torch.ones(1, self.mid_conv.in_channels, 1, 1, 2, 1))
        self.bot2 = nn.Parameter(0.5*torch.ones(1, self.mid_conv.in_channels, 1, 1, 2, 1))
        self.left2 = nn.Parameter(0.5*torch.ones(1, self.mid_conv.in_channels, 1, 1, 1, 2))
        self.right2 = nn.Parameter(0.5*torch.ones(1, self.mid_conv.in_channels, 1, 1, 1, 2))

        self.fitting = True
        self.lr = 1.0

        self.top_m = nn.Parameter(torch.zeros(1, self.mid_conv.in_channels, 1, 1, 2, 1))
        self.bot_m = nn.Parameter(torch.zeros(1, self.mid_conv.in_channels, 1, 1, 2, 1))

        self.left_m = nn.Parameter(torch.zeros(1, self.mid_conv.in_channels, 1, 1, 1, 2))
        self.right_m = nn.Parameter(torch.zeros(1, self.mid_conv.in_channels, 1, 1, 1, 2))

    def alpha_update(self, patches, x):
        B, C, H, W = x.size()
        patch2 = rearrange(x, "B C (ph H) (pw W) -> B C ph pw H W", ph=self.ph, pw=self.pw)

        target_top = patches[:, :, :, :, :1, 1:-1]
        target_bot = patches[:, :, :, :, -1:, 1:-1]
        target_left = patches[:, :, :, :, 1:-1, :1]
        target_right = patches[:, :, :, :, 1:-1, -1:]

        pred_top = (self.top * patch2[:, :, :, :, :2, :]).sum(dim=-2, keepdim=True)
        # B, C, ph, pw, H, W
        pred_top[:, :, 0, :, 0, :] *= 0.0

        #pred_bot = self.bot[0] * patch2[:, :, :, :, -1, :] + self.bot[1] * patch2[:, :, :, :, -2, :]
        pred_bot = (self.bot * patch2[:, :, :, :, -2:, :]).sum(dim=-2, keepdim=True)
        pred_bot[:, :, -1, :, -1, :] *= 0.0

        #pred_left = patch2[:, :, :, :, :, 1] + self.left * (patch2[:, :, :, :, :, 0] - patch2[:, :, :, :, :, 1])
        #pred_left = self.left[0] * patch2[:, :, :, :, :, 0] + self.left[1] * patch2[:, :, :, :, :, 1]
        pred_left = (self.left * patch2[:, :, :, :, :, :2]).sum(dim=-1, keepdim=True)
        pred_left[:, :, :, 0, :, 0] *= 0.0
        #diff_left = patch2[:, :, :, :, :, 0] - patch2[:, :, :, :, :, 1]

        #pred_right = patch2[:, :, :, :, :, -2] + self.right * (patch2[:, :, :, :, :, -1] - patch2[:, :, :, :, :, -2])
        #pred_right = self.right[0] * patch2[:, :, :, :, :, -1] + self.right[1] * patch2[:, :, :, :, :, -2]
        pred_right = (self.right * patch2[:, :, :, :, :, -2:]).sum(dim=-1, keepdim=True)
        pred_right[:, :, :, -1, :, -1] *= 0.0
        #diff_right = patch2[:, :, :, :, :, -1] - patch2[:, :, :, :, :, -2]

        mse_loss_top = 2 * (pred_top - target_top) * patch2[:, :, :, :, :2, :]
        mse_loss_bot = 2 * (pred_bot - target_bot) * patch2[:, :, :, :, -2:, :]
        mse_loss_left = 2 * (pred_left - target_left) * patch2[:, :, :, :, :, :2]
        mse_loss_right = 2 * (pred_right - target_right) * patch2[:, :, :, :, :, -2:]

        mse_loss_top = mse_loss_top.sum(axis=[0, 2, 3, 5]) / ((self.ph - 1) * H * B)
        mse_loss_bot = mse_loss_bot.sum(axis=[0, 2, 3, 5]) / ((self.ph - 1) * H * B)
        mse_loss_left = mse_loss_left.sum(axis=[0, 2, 3, 4]) / ((self.ph - 1) * H * B)
        mse_loss_right = mse_loss_right.sum(axis=[0, 2, 3, 4]) / ((self.ph - 1) * H * B)

        #total_loss = (mse_loss_top + mse_loss_bot + mse_loss_left + mse_loss_right) / (4 * (self.ph - 1) * H * B)
        self.top_m = nn.Parameter(0.90 * self.top_m - self.lr * mse_loss_top.reshape(self.top.size()))
        self.bot_m = nn.Parameter(0.90 * self.bot_m - self.lr * mse_loss_bot.reshape(self.bot.size()))
        self.left_m = nn.Parameter(0.90 * self.left_m - self.lr * mse_loss_left.reshape(self.left.size()))
        self.right_m = nn.Parameter(0.90 * self.right_m - self.lr * mse_loss_right.reshape(self.right.size()))

        self.top = nn.Parameter(self.top_m + self.top)
        self.bot = nn.Parameter(self.bot_m + self.bot)
        self.left = nn.Parameter(self.left_m + self.left)
        self.right = nn.Parameter(self.right_m + self.right)

    def predict(self, x):
        patch = rearrange(x, "B C (ph H) (pw W) -> B C ph pw H W", ph=self.ph, pw=self.pw)
       
        pad_top = (self.top1 * patch[:, :, :, :, :2, :]).sum(dim=-2, keepdim=True)
        pad_bot = (self.bot1 * patch[:, :, :, :, -2:, :]).sum(dim=-2, keepdim=True)
        pad_left = (self.left1 * patch[:, :, :, :, :, :2]).sum(dim=-1, keepdim=True)
        pad_right = (self.right1 * patch[:, :, :, :, :, -2:]).sum(dim=-1, keepdim=True)

        pad_topleft = patch[:, :, :, :, 0, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
        pad_topright = patch[:, :, :, :, 0, -1].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
        pad_botleft = patch[:, :, :, :, -1, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)
        pad_botright = patch[:, :, :, :, -1, -1].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)

        pad_top[:, :, 0, :, 0, :] *= 0.0
        pad_bot[:, :, -1, :, -1, :] *= 0.0
        pad_left[:, :, :, 0, :, 0] *= 0.0
        pad_right[:, :, :, -1, :, -1] *= 0.0

        pad_topleft[:, :, 0, :, :, :] *= 0.0
        pad_topleft[:, :, :, 0, :, :] *= 0.0

        pad_botleft[:, :, -1, :, :, :] *= 0.0
        pad_botleft[:, :, :, 0, :, :] *= 0.0

        pad_topright[:, :, 0, :, :, :] *= 0.0
        pad_topright[:, :, :, -1, :, :] *= 0.0

        pad_botright[:, :, -1, :, :, :] *= 0.0
        pad_botright[:, :, :, -1, :, :] *= 0.0

        pad_ex_top = torch.cat([pad_topleft, pad_top, pad_topright], dim=-1)
        pad_mid = torch.cat([pad_left, patch, pad_right], dim=-1)
        pad_ex_bot = torch.cat([pad_botleft, pad_bot, pad_botright], dim=-1)
        
        padded_patch = torch.cat([pad_ex_top, pad_mid, pad_ex_bot], dim=-2)
        padded_patch1 = rearrange(padded_patch, "B C ph pw H W -> (B ph pw) C H W", ph=self.ph, pw=self.pw)
        out = self.mid_conv(padded_patch1)
        out = rearrange(out, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.ph, pw=self.pw)
        if not self.training:
            return out
        return padded_patch, out

    def forward(self, x):
        if self.fitting:
            patches = get_patch(x, self.pad_tuple, self.ph, self.pw)
            self.alpha_update(patches, x)
            x = F.pad(x, self.pad_tuple, mode='constant', value=0.0)
            return self.mid_conv(x)
        else:
            return self.predict(x)

class Interpolate_S2(nn.Module):
    def __init__(self, ConvModule, padding, n_patch):
        super(Interpolate_S2, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0,0)
        self.ph = n_patch
        self.pw = n_patch
        self.pad = self.mid_conv.kernel_size[0] // 2
        self.pad_tuple = (self.pad, self.pad - 1, self.pad, self.pad - 1)
        self.top = nn.Parameter(0.5*torch.ones(1, self.mid_conv.in_channels, 1, 1, 2, 1))
        self.left = nn.Parameter(0.5*torch.ones(1, self.mid_conv.in_channels, 1, 1, 1, 2))

        self.fitting = True
        self.lr = 1.0
        self.top_m = nn.Parameter(torch.zeros(1, self.mid_conv.in_channels, 1, 1, 2, 1))
        self.left_m = nn.Parameter(torch.zeros(1, self.mid_conv.in_channels, 1, 1, 1, 2))

    def alpha_update(self, patches, x):
        B, C, H, W = x.size()
        patch2 = rearrange(x, "B C (ph H) (pw W) -> B C ph pw H W", ph=self.ph, pw=self.pw)

        target_top = patches[:, :, :, :, :1, 1:]
        target_left = patches[:, :, :, :, 1:, :1]
        
        
        #pred_top = patch2[:, :, :, :, 1, :] + self.top * (patch2[:, :, :, :, 0, :] - patch2[:, :, :, :, 1, :])
        pred_top = (self.top * patch2[:, :, :, :, :2, :]).sum(dim=-2, keepdim=True)
        pred_top[:, :, 0, :, 0, :] *= 0.0

        #pred_left = patch2[:, :, :, :, :, 1] + self.left * (patch2[:, :, :, :, :, 0] - patch2[:, :, :, :, :, 1])
        pred_left = (self.left * patch2[:, :, :, :, :, :2]).sum(dim=-1, keepdim=True)
        pred_left[:, :, :, 0, :, 0] *= 0.0

        mse_loss_top = 2 * (pred_top - target_top) * patch2[:, :, :, :, :2, :]
        mse_loss_left = 2 * (pred_left - target_left) * patch2[:, :, :, :, :, :2]

        mse_loss_top = mse_loss_top.sum(axis=[0, 2, 3, 5]) / ((self.ph - 1) * H * B)
        mse_loss_left = mse_loss_left.sum(axis=[0, 2, 3, 4]) / ((self.ph - 1) * H * B)

        #total_loss = (mse_loss_top + mse_loss_left) / (2 * (self.ph - 1) * H * B)
        self.top_m = nn.Parameter(0.90 * self.top_m - self.lr * mse_loss_top.reshape(self.top.size()))
        self.left_m = nn.Parameter(0.90 * self.left_m - self.lr * mse_loss_left.reshape(self.left.size()))

        self.top = nn.Parameter(self.top_m + self.top)
        self.left = nn.Parameter(self.left_m + self.left)

    def predict(self, x):
        patch = rearrange(x, "B C (ph H) (pw W) -> B C ph pw H W", ph=self.ph, pw=self.pw)
        
        pad_top = (self.top1 * patch[:, :, :, :, :2, :]).sum(dim=-2, keepdim=True)
        pad_left = (self.left1 * patch[:, :, :, :, :, :2]).sum(dim=-1, keepdim=True)

        pad_top1 = (self.top2 * patch[:, :, :, :, :2, :]).sum(dim=-2, keepdim=True)
        pad_left1 = (self.left2 * patch[:, :, :, :, :, :2]).sum(dim=-1, keepdim=True)

        pad_top = sampling_top(pad_top, pad_top1)
        pad_left = sampling_left(pad_left, pad_left1)

        pad_topleft = patch[:, :, :, :, 0, 0].clone().unsqueeze(dim=-1).unsqueeze(dim=-1)

        pad_top[:, :, 0, :, 0, :] *= 0.0
        pad_left[:, :, :, 0, :, 0] *= 0.0

        pad_topleft[:, :, 0, :, :, :] *= 0.0
        pad_topleft[:, :, :, 0, :, :] *= 0.0

        pad_ex_top = torch.cat([pad_topleft, pad_top], dim=-1)
        pad_mid = torch.cat([pad_left, patch], dim=-1)
        
        padded_patch = torch.cat([pad_ex_top, pad_mid], dim=-2)
        padded_patch1 = rearrange(padded_patch, "B C ph pw H W -> (B ph pw) C H W", ph=self.ph, pw=self.pw)
        out = self.mid_conv(padded_patch1)
        out = rearrange(out, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.ph, pw=self.pw)
        if not self.training:
            return out
        return padded_patch, out

    def forward(self, x):
        if self.fitting:
            patches = get_patch(x, self.pad_tuple, self.ph, self.pw)
            self.alpha_update(patches, x)
            x = F.pad(x, self.pad_tuple, mode='constant', value=0.0)
            return self.mid_conv(x)
        else:
            return self.predict(x)

def get_attr(layer):
    k = layer.kernel_size[0]
    s = layer.stride[0]
    return (k, s)

def change_model_list(model, num_patches, Module_To_Mapping, patch_list):
    i = 0
    for n, target in model.named_modules():
        if i == len(patch_list):
            break
        if isinstance(target, nn.Conv2d) and target.kernel_size[0] > 1:
            attrs = n.split('.')
            submodule = model
            for attr in attrs[:-1]:
                submodule = getattr(submodule, attr)
            (k, s) = get_attr(target)
            if patch_list[i] == 0:
                replace = Module_To_Mapping[(k, s, patch_list[i])](target, target.padding, num_patches)
                setattr(submodule, attrs[-1], replace)
            i += 1

def tuning_mode(model):
    for n, mod in model.named_modules():
        if hasattr(mod, 'fitting'):
            mod.fitting = True

def predict_mode(model):
    for n, mod in model.named_modules():
        if hasattr(mod, 'fitting'):
            mod.fitting = False
            mod.top_m = nn.Parameter(torch.zeros_like(mod.top_m).to(mod.top_m))
            mod.left_m = nn.Parameter(torch.zeros_like(mod.left_m).to(mod.left_m))
            if hasattr(mod, 'bot_m'):
                mod.bot_m = nn.Parameter(torch.zeros_like(mod.bot_m).to(mod.bot_m))
                mod.right_m = nn.Parameter(torch.zeros_like(mod.right_m).to(mod.right_m))


def show_status(model):
    for n, mod in model.named_modules():
        if hasattr(mod, 'fitting'):
            print(mod, mod.fitting)

module_to_mapping = {
    (3, 1, 0): Interpolate_S1,
    (3, 2, 0): Interpolate_S2,
}
def set_forward_lr(model, lr):
    for n, mod in model.named_modules():
        if hasattr(mod, 'fitting'):
            mod.lr = lr

def show_alpha(model):
    for n, mod in model.named_modules():
        if hasattr(mod, 'fitting'):
            print('top', mod.top.reshape(-1))
            print('left', mod.left.reshape(-1))
            if hasattr(mod, 'right'):
                print('right', mod.top.reshape(-1))
                print('bot', mod.left.reshape(-1))


def alternative_tuning(model, progressive):
    i = 0
    for n, mod in model.named_modules():
        if i == len(progressive):
            break
        if hasattr(mod, 'fitting'):
            if progressive[i] == 1:
                mod.fitting = True
            else:
                mod.fitting = False
            i += 1
a="""
# MAIN
from torchvision.models import mobilenet_v2
model = mobilenet_v2(pretrained=True)
change_model_list(model, 4, module_to_mapping, [0, 0, 0, 0, 0])
show_status(model)
predict_mode(model)
show_status(model)
tuning_mode(model)
from torchvision.datasets import FakeData
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

preprocessing = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.458, 0.456, 0.406],
                         std=[0.229, 0.224, 0.224])
])
train_dataset = FakeData(1024, image_size=(3, 224, 224), num_classes=1000, transform=preprocessing)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
model.with_feat = None
model.eval()
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

for epoch in range(5):
    progressive = [0, 0, 0, 0, 0]
    progressive[epoch] = 1
    alternative_tuning(model, progressive)
    print(f"=============================== epoch = {epoch} =============================")
    show_status(model)
    optimizer.param_groups[0]["lr"] = 0.1
    print(scheduler.get_lr()[0])
    for batch_idx, (input, target) in enumerate(train_loader):
        input.cuda()
        target.cuda()
        model(input)
        set_forward_lr(model, scheduler.get_last_lr()[0])
        #show_alpha(model)
        scheduler.step()
"""
