import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class NonOverlappedConv2d(nn.Module):
    def __init__(self, conv, args, shift=0):
        super(NonOverlappedConv2d, self).__init__()
        self.conv = conv
        self.conv.padding = (0,0)
        self.ph = args.n_patch_h
        self.pw = args.n_patch_w
        self.coefficient_train = args.coefficient_train
        self.sampling = args.sampling
        self.sample_width = args.sample_width
        self.pad_method = args.pad_method
        self.stride = conv.stride[0]
        self.kernel_size = conv.kernel_size[0]
        self.output_shift = shift
        self.input_shift = shift * self.stride

        # For GEMM-based Convolution (shift != 0)
        self.hw_indices = None
        self.ih = 0
        self.iw = 0

        if self.pad_method == "predict":
            self.d = conv.weight.device
            v = args.coefficient_value
            if shift == 0:
                pre_shape = (1, self.conv.in_channels, 1, 1)
                if args.coefficient_random_init:
                    self.t = nn.Parameter(torch.randn(*pre_shape, 2, 1).to(self.d))
                    self.b = nn.Parameter(torch.randn(*pre_shape, 2, 1).to(self.d))
                    self.l = nn.Parameter(torch.randn(*pre_shape, 1, 2).to(self.d))
                    self.r = nn.Parameter(torch.randn(*pre_shape, 1, 2).to(self.d))
                else:
                    self.t = nn.Parameter(v * torch.ones(*pre_shape, 2, 1).to(self.d))
                    self.b = nn.Parameter(v * torch.ones(*pre_shape, 2, 1).to(self.d))
                    self.l = nn.Parameter(v * torch.ones(*pre_shape, 1, 2).to(self.d))
                    self.r = nn.Parameter(v * torch.ones(*pre_shape, 1, 2).to(self.d))
            else:
                if args.coefficient_random_init:
                    self.t = nn.Parameter(torch.randn(1, self.conv.in_channels, 1, 2, 1).to(self.d))
                    self.b = nn.Parameter(torch.randn(1, self.conv.in_channels, 1, 2, 1).to(self.d))
                    self.l = nn.Parameter(torch.randn(1, self.conv.in_channels, 1, 1, 2).to(self.d))
                    self.r = nn.Parameter(torch.randn(1, self.conv.in_channels, 1, 1, 2).to(self.d))
                else:
                    self.t = nn.Parameter(v * torch.ones(1, self.conv.in_channels, 1, 2, 1).to(self.d))
                    self.b = nn.Parameter(v * torch.ones(1, self.conv.in_channels, 1, 2, 1).to(self.d))
                    self.l = nn.Parameter(v * torch.ones(1, self.conv.in_channels, 1, 1, 2).to(self.d))
                    self.r = nn.Parameter(v * torch.ones(1, self.conv.in_channels, 1, 1, 2).to(self.d))

    def _forward_non_overlapping_shift0(self, x):
        patch = rearrange(x, "B C (ph H) (pw W) -> B C ph pw H W", ph=self.ph, pw=self.pw)
        B, C, _, _, H, W = patch.size()

        if self.sampling:
            padded_x = F.pad(x, (1, 1, 1, 1), mode='constant', value=0.0)
            overlapped_patch = padded_x.unfold(2, H + 2, H).unfold(3, W + 2, W)
            # Pad Top, Left, Top-Left
            pad_t   = overlapped_patch[:, :, :, :,  0:1, 1:-1].clone()
            pad_l   = overlapped_patch[:, :, :, :, 1:-1,  0:1].clone()
            pad_t_l = overlapped_patch[:, :, :, : , 0:1,  0:1].clone()
            if self.sample_width > 1:
                if self.pad_method == "zero":
                    new_pad_t = 0
                    new_pad_l = 0
                elif self.pad_method == "mean":
                    new_pad_t = torch.mean(self.t * patch[:, :, :, :, :2,  1::2], dim=-2, keepdim=True)
                    new_pad_l = torch.mean(self.l * patch[:, :, :, :, 1::2,  :2], dim=-1, keepdim=True)
                elif self.pad_method == "replicate":
                    new_pad_t = patch[:, :, :, :, :1,  1::2].clone()
                    new_pad_l = patch[:, :, :, :, 1::2,  :1].clone()
                elif self.pad_method == "reflect":
                    new_pad_t = patch[:, :, :, :, 1:2,  1::2].clone()
                    new_pad_l = patch[:, :, :, :, 1::2,  1:2].clone()
                elif self.pad_method == "predict":
                    new_pad_t = torch.sum(self.t * patch[:, :, :, :, :2,  1::2], dim=-2, keepdim=True)
                    new_pad_l = torch.sum(self.l * patch[:, :, :, :, 1::2,  :2], dim=-1, keepdim=True)
                pad_t[:, :, :, :, :, 1::2] = new_pad_t
                pad_l[:, :, :, :, 1::2, :] = new_pad_l
        else:
            # If you use zero padding with non-sampling
            if self.pad_method == "zero":
                non_overlapped_patch = F.pad(patch, (1, 1, 1, 1), mode='constant', value=0.0)
                return non_overlapped_patch

            if self.pad_method == "mean":
                pad_t = torch.mean(patch[:, :, :, :,  :2,   :], dim=-2, keepdim=True)
                pad_l = torch.mean(patch[:, :, :, :,   :,  :2], dim=-1, keepdim=True)
                pad_t_l = patch[:, :, :, :,  :1,  :1].clone()
            elif self.pad_method == "replicate":
                pad_t = patch[:, :, :, :,  0:1,   :].clone()
                pad_l = patch[:, :, :, :,   :,  0:1].clone()
                pad_t_l = patch[:, :, :, :,  0:1,  0:1].clone()
            elif self.pad_method == "reflect":
                pad_t = patch[:, :, :, :,  1:2,   :].clone()
                pad_l = patch[:, :, :, :,   :,  1:2].clone()
                pad_t_l = patch[:, :, :, :,  1:2,  1:2].clone()
            elif self.pad_method == "predict":
                pad_t = torch.sum(self.t * patch[:, :, :, :,  :2,   :], dim=-2, keepdim=True)
                pad_l = torch.sum(self.l * patch[:, :, :, :,   :,  :2], dim=-1, keepdim=True)
                pad_t_l = patch[:, :, :, :,  :1,  :1].clone()

            # Boundary must be zero
            pad_t[:, :, 0, :, 0, :] *= 0.0
            pad_l[:, :, :, 0, :, 0] *= 0.0
            pad_t_l[:, :, 0, :, :, :] *= 0.0
            pad_t_l[:, :, :, 0, :, :] *= 0.0

        # When Stride=1, Use Bottom / Left
        if self.stride == 1:
            if self.pad_method == "zero":
                pad_b = torch.zeros(B, C, self.ph, self.pw, 1, W).to(self.d)
                pad_r = torch.zeros(B, C, self.ph, self.pw, H, 1).to(self.d)
                pad_corner = torch.zeros(B, C, self.ph, self.pw, 1, 1).to(self.d)
                pad_t_r = pad_corner
                pad_b_l = pad_corner
                pad_b_r = pad_corner
            elif self.pad_method == "mean":
                pad_b = torch.mean(patch[:, :, :, :, -2:,   :], dim=-2, keepdim=True)
                pad_r = torch.mean(patch[:, :, :, :,   :, -2:], dim=-1, keepdim=True)
                pad_t_r = patch[:, :, :, :,  0:1, -1:].clone()
                pad_b_l = patch[:, :, :, :, -1:,  0:1].clone()
                pad_b_r = patch[:, :, :, :, -1:, -1:].clone()
            elif self.pad_method == "replicate":
                pad_b = patch[:, :, :, :, -1:,   :].clone()
                pad_r = patch[:, :, :, :,   :, -1:].clone()
                pad_t_r = patch[:, :, :, :,  0:1, -1:].clone()
                pad_b_l = patch[:, :, :, :, -1:,  0:1].clone()
                pad_b_r = patch[:, :, :, :, -1:, -1:].clone()
            elif self.pad_method == "reflect":
                pad_b = patch[:, :, :, :, -2:-1,   :].clone()
                pad_r = patch[:, :, :, :,   :, -2:-1].clone()
                pad_t_r = patch[:, :, :, :,  1:2, -2:-1].clone()
                pad_b_l = patch[:, :, :, :, -2:-1,  1:2].clone()
                pad_b_r = patch[:, :, :, :, -2:-1, -2:-1].clone()
            elif self.pad_method == "predict":
                pad_b = torch.sum(self.b * patch[:, :, :, :, -2:,   :], dim=-2, keepdim=True)
                pad_r = torch.sum(self.r * patch[:, :, :, :,   :, -2:], dim=-1, keepdim=True)
                pad_t_r = patch[:, :, :, :,  0:1, -1:].clone()
                pad_b_l = patch[:, :, :, :, -1:,  0:1].clone()
                pad_b_r = patch[:, :, :, :, -1:, -1:].clone()

            # Boundary must be zero
            pad_b[:, :, -1,  :, -1,  :] *= 0.0
            pad_r[:, :,  :, -1,  :, -1] *= 0.0
            pad_t_r[:, :, 0,  :, :, :] *= 0.0
            pad_t_r[:, :, :, -1, :, :] *= 0.0
            pad_b_l[:, :, -1, :, :, :] *= 0.0
            pad_b_l[:, :,  :, 0, :, :] *= 0.0
            pad_b_r[:, :, -1,  :, :, :] *= 0.0
            pad_b_r[:, :,  :, -1, :, :] *= 0.0

            patch_top = torch.cat([pad_t_l, pad_t, pad_t_r], dim=-1)
            patch_mid = torch.cat([pad_l, patch, pad_r], dim=-1)
            patch_bot = torch.cat([pad_b_l, pad_b, pad_b_r], dim=-1)
            non_overlapped_patch = torch.cat([patch_top, patch_mid, patch_bot], dim=-2)
        else:
            patch_top = torch.cat([pad_t_l, pad_t], dim=-1)
            patch_mid = torch.cat([pad_l, patch], dim=-1)
            non_overlapped_patch = torch.cat([patch_top, patch_mid], dim=-2)

        return non_overlapped_patch

    def _forward_non_overlapping_shift(self, x):
        N, C, IH, IW = x.size()

        # Generate indices for GEMM-based Convolution (with im2col)
        if self.hw_indices == None and self.ih != IH and self.iw != IW:
            ph_size = IH // self.ph
            pw_size = IW // self.pw
            i = []
            OH = (IH + self.stride - 1) // self.stride
            OW = (IW + self.stride - 1) // self.stride
            self.ih = IH
            self.iw = IW
            self.oh = OH
            self.ow = OW

            # index
            zero_index = IH * IW
            t_index = zero_index + 1
            b_index = t_index + (self.ph - 1) * IW
            l_index = b_index + (self.ph - 1) * IW
            r_index = l_index + (self.pw - 1) * IH

            if self.pad_method in ["mean", "predict"]:
                self.pad_t_index = []
                self.pad_b_index = []
                self.pad_l_index = []
                self.pad_r_index = []
                for ph_idx in range(self.ph):
                    if ph_idx > 0:
                        oh_start = ph_size * ph_idx + self.output_shift
                        self.pad_t_index.append(oh_start)
                        self.pad_t_index.append(oh_start+1)
                    if ph_idx < self.ph - 1:
                        oh_end = ph_size * (ph_idx + 1) + self.output_shift
                        self.pad_b_index.append(oh_end-2)
                        self.pad_b_index.append(oh_end-1)
                for pw_idx in range(self.pw):
                    if pw_idx > 0:
                        ow_start = pw_size * pw_idx + self.output_shift
                        self.pad_l_index.append(ow_start)
                        self.pad_l_index.append(ow_start+1)
                    if pw_idx < self.pw - 1:
                        ow_end = pw_size * (pw_idx + 1) + self.output_shift
                        self.pad_r_index.append(ow_end-2)
                        self.pad_r_index.append(oh_end-1)


            for ph_idx in range(self.ph):
                # Set oh_start
                if ph_idx == 0:
                    oh_start = 0
                else:
                    oh_start = ph_size * ph_idx + self.output_shift

                # Set oh_end
                if ph_idx == self.ph - 1:
                    oh_end = OH
                else:
                    oh_end = ph_size * (ph_idx + 1) + self.output_shift

                for oh in range(oh_start, oh_end):
                    for pw_idx in range(self.pw):
                        # Set ow_start
                        if pw_idx == 0:
                            ow_start = 0
                        else:
                            ow_start = pw_size + pw_idx + self.output_shift

                        # Set ow_end
                        if pw_idx == self.pw - 1:
                            ow_end = OW
                        else:
                            ow_end = pw_size * (pw_idx + 1) + self.output_shift


                        for ow in range(ow_start, ow_end):
                            patch_t = (oh == oh_start)
                            patch_b = (oh == oh_end - 1)
                            patch_l = (ow == ow_start)
                            patch_r = (ow == ow_end - 1)
                            patch_t_l = (patch_t and patch_l)
                            patch_t_r = (patch_t and patch_r)
                            patch_b_l = (patch_b and patch_l)
                            patch_b_r = (patch_b and patch_r)

                            is_sample_t = (patch_t and ph_idx > 0)
                            is_sample_l = (patch_l and pw_idx > 0)
                            is_sample_candidate = (self.sampling and (is_sample_t or is_sample_l))

                            if self.pad_method == "zero":
                                for kh in [-1, 0, 1]:
                                    for kw in [-1, 0, 1]:
                                        ih = oh * self.stride + kh
                                        iw = ow * self.stride + kw

                                        if ih == -1 or ih == IH or iw == -1 or iw == IW:
                                            i.append(zero_index)
                                        elif is_sample_candidate and ((is_sample_t and kh == -1) or (is_sample_l and kw == -1)):
                                            i.append(ih * IW + iw)
                                        elif (patch_t and kh == -1) or (patch_b and kh == 1) or (patch_l and kw == -1) or (patch_r and kw == 1):
                                            i.append(zero_index)
                                        else:
                                            i.append(ih * IW + iw)
                            elif self.pad_method in ["replicate", "reflect"]:
                                offset = 1 if self.pad_method == "replicate" else 2
                                for kh in [-1, 0, 1]:
                                    for kw in [-1, 0, 1]:
                                        ih = oh * self.stride + kh
                                        iw = ow * self.stride + kw

                                        if ih == -1 or ih == IH or iw == -1 or iw == IW:
                                            i.append(zero_index)
                                        elif is_sample_candidate and ((is_sample_t and kh == -1) or (is_sample_l and kw == -1)):
                                            i.append(ih * IW + iw)
                                        elif patch_t_l and kh == -1 and kw == -1:
                                            i.append((ih + offset) * IW + iw + offset)
                                        elif patch_t_r and kh == -1 and kw == 1:
                                            i.append((ih + offset) * IW + iw - offset)
                                        elif patch_b_l and kh == 1 and kw == -1:
                                            i.append((ih - offset) * IW + iw + offset)
                                        elif patch_b_r and kh == 1 and kw == 1:
                                            i.append((ih - offset) * IW + iw - offset)
                                        elif patch_t and kh == -1:
                                            i.append((ih + offset) * IW + iw)
                                        elif patch_b and kh == 1:
                                            i.append((ih - offset) * IW + iw)
                                        elif patch_l and kw == -1:
                                            i.append(ih * IW + iw + offset)
                                        elif patch_r and kw == 1:
                                            i.append((ih * IW + iw - offset))
                                        else:
                                            i.append(ih * IW + iw)
                            elif self.pad_method in ["mean", "predict"]:
                                for kh in [-1, 0, 1]:
                                    for kw in [-1, 0, 1]:
                                        ih = oh * self.stride + kh
                                        iw = ow * self.stride + kw

                                        if ih == -1 or ih == IH or iw == -1 or iw == IW:
                                            i.append(zero_index)
                                        elif is_sample_candidate and ((is_sample_t and kh == -1) or (is_sample_l and kw == -1)):
                                            i.append(ih * IW + iw)
                                        elif patch_t_l and kh == -1 and kw == -1:
                                            i.append((ih + 1) * IW + iw + 1)
                                        elif patch_t_r and kh == -1 and kw == 1:
                                            i.append((ih + 1) * IW + iw - 1)
                                        elif patch_b_l and kh == 1 and kw == -1:
                                            i.append((ih - 1) * IW + iw + 1)
                                        elif patch_b_r and kh == 1 and kw == 1:
                                            i.append((ih - 1) * IW + iw - 1)
                                        elif patch_t and kh == -1:
                                            i.append(t_index + (ph_idx - 1) * IW + iw)
                                        elif patch_b and kh == 1:
                                            i.append(b_index + (ph_idx) * IW + iw)
                                        elif patch_l and kw == -1:
                                            i.append(l_index + (pw_idx - 1) * IH + ih)
                                        elif patch_r and kw == 1:
                                            i.append(r_index + (pw_idx) * IH + ih)
                                        else:
                                            i.append(ih * IW + iw)
            self.hw_indices = torch.LongTensor(i).to(self.d)

        cat_list = [x.reshape(N, C, IH * IW), torch.zeros(N, C, 1).to(self.d)]

        if self.pad_method == "mean":
            pad_t = torch.mean(x[:, :, self.pad_t_index, :].reshape(N, C, self.ph-1, 2, IW), dim=-2).reshape(N, C, -1)
            pad_b = torch.mean(x[:, :, self.pad_b_index, :].reshape(N, C, self.ph-1, 2, IW), dim=-2).reshape(N, C, -1)
            pad_l = torch.mean(x[:, :, :, self.pad_l_index].reshape(N, C, IH, self.pw-1, 2), dim=-1).reshape(N, C, -1)
            pad_r = torch.mean(x[:, :, :, self.pad_r_index].reshape(N, C, IH, self.pw-1, 2), dim=-1).reshape(N, C, -1)
            cat_list.extend([pad_t, pad_b, pad_l, pad_r])
        elif self.pad_method == "predict":
            pad_t = torch.sum(self.t * x[:, :, self.pad_t_index, :].reshape(N, C, self.ph-1, 2, IW), dim=-2).reshape(N, C, -1)
            pad_b = torch.sum(self.b * x[:, :, self.pad_b_index, :].reshape(N, C, self.ph-1, 2, IW), dim=-2).reshape(N, C, -1)
            pad_l = torch.sum(self.l * x[:, :, :, self.pad_l_index].reshape(N, C, IH, self.pw-1, 2), dim=-1).reshape(N, C, -1)
            pad_r = torch.sum(self.r * x[:, :, :, self.pad_r_index].reshape(N, C, IH, self.pw-1, 2), dim=-1).reshape(N, C, -1)
            cat_list.extend([pad_t, pad_b, pad_l, pad_r])

        im2col_x = torch.cat(cat_list, dim=-1)[:, :, self.hw_indices]  # (N, C, OH * OW * KH * KW)
        if self.conv.groups == 1:
            im2col_x = im2col_x.permute(0, 2, 1).reshape(N, -1, 9 * C)
        else:
            im2col_x = im2col_x.reshape(-1, 9)
        out_channels = self.conv.out_channels
        transposed_weight = self.conv.weight.permute(2, 3, 1, 0).reshape(-1, out_channels)
        out = torch.matmul(im2col_x, transposed_weight)
        out = out.reshape(-1, self.oh, self.ow, out_channels).permute(0, 3, 1, 2)
        return out


    def forward(self, x):
        if self.output_shift == 0:
            non_overlapped_patch_6d =  self._forward_non_overlapping_shift0(x)
            non_overlapped_patch_4d = rearrange(non_overlapped_patch_6d, "B C ph pw H W -> (B ph pw) C H W", ph=self.ph, pw=self.pw)
            out = self.conv(non_overlapped_patch_4d)
            out = rearrange(out, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.ph, pw=self.pw)
            if self.coefficient_train and self.training:
                return non_overlapped_patch_6d, out
            return out
        else:
            return self._forward_non_overlapping_shift(x)


def get_parent_module_by_name(model, name):
    attrs = name.split('.')
    submodule = model
    for attr in attrs[:-1]:
        submodule = getattr(submodule, attr)
    return submodule

def replace_conv2d_to_non_overlapped_conv2d(parent_mod, mod, mod_name, args, shift):
    replaced_mod = NonOverlappedConv2d(mod, args, shift=shift)
    setattr(parent_mod, mod_name, replaced_mod)

def transform_with_non_overlapping(model, args):
    # In patch list, 0 means non-overlapping patch with padding
    assert len(args.patch_list) > 0
    assert args.pivot_idx < len(args.patch_list)

    conv3x3_list = []
    i = 0
    for name, mod in model.named_modules():
        # Currently, our implementation support only kernel_size=3
        if isinstance(mod, nn.Conv2d) and mod.kernel_size[0] == 3:
            parent_mod = get_parent_module_by_name(model, name)
            mod_name = name.split('.')[-1]
            conv3x3_list.append([parent_mod, mod, mod_name])
            i += 1
            if i == len(args.patch_list):
                break

    shift_backward = 0
    shift_forward = 0
    # Set equal patch size for pivot conv3x3
    if args.patch_list[args.pivot_idx] == 0:
        replace_conv2d_to_non_overlapped_conv2d(*conv3x3_list[args.pivot_idx], args, shift=0)
    elif conv3x3_list[args.pivot_idx][1].stride[0] == 1:
        shift_backward = 1

    # Propagate patch size backward
    for i in range(args.pivot_idx-1, 0, -1):
        stride = conv3x3_list[i][1].stride[0]

        if args.patch_list[i] == 0:
            replace_conv2d_to_non_overlapped_conv2d(*conv3x3_list[i], args, shift=shift_forward)
        elif stride == 1:
            shift_backward += 1

        if stride == 2:
            shift_backward *= 2

    # Propagate patch size forward
    for i in range(args.pivot_idx+1, len(args.patch_list), 1):
        stride = conv3x3_list[i][1].stride[0]

        if stride == 2:
            shift_forward = (shift_forward + 1) // 2

        if args.patch_list[i] == 0:
            replace_conv2d_to_non_overlapped_conv2d(*conv3x3_list[i], args, shift=shift_backward)
        elif stride == 1:
            shift_forward -= 1

def main():
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(pretrained=True)
    change_model_list(model, 4, module_to_mapping, [0, 0, 0, 0, 0])
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
        print(f"=============================== epoch = {epoch} =============================")
        show_status(model)
        optimizer.param_groups[0]["lr"] = 0.1
        print(scheduler.get_lr()[0])
        for batch_idx, (input, target) in enumerate(train_loader):
            input.cuda()
            target.cuda()
            model(input)
            scheduler.step()

if __name__ == "__main__":
    main()
