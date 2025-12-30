# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional

import kornia.augmentation as K
import kornia
from kornia.augmentation import AugmentationBase2D
from kornia.geometry.transform import resize

import utils_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiffJPEG(nn.Module):
    def __init__(self, quality=50):
        super().__init__()
        self.quality = quality
    
    def forward(self, x):
        # Ensure x on correct device
        x = x.to(device)
        with torch.no_grad():
            img_clip = utils_img.clamp_pixel(x)
            img_jpeg = utils_img.jpeg_compress(img_clip, self.quality)
            img_gap = img_jpeg - x
            img_gap = img_gap.detach()
        return x + img_gap

class RandomDiffJPEG(AugmentationBase2D):
    def __init__(self, p=0.5, low=10, high=100, step=10):
        super().__init__(p=p)
        # Pre-create DiffJPEG modules on device
        self.diff_jpegs = nn.ModuleList(
            [DiffJPEG(qf).to(device) for qf in range(low, high, step)]
        )

    def generate_parameters(self, input_shape: torch.Size):
        # Sample index of JPEG quality
        qf_idx = torch.randint(len(self.diff_jpegs), (input_shape[0],), device=device)
        return dict(qf=qf_idx)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        # input shape: (B, C, H, W)
        B = input.shape[0]
        output = torch.empty_like(input, device=device)
        for i in range(B):
            idx = params['qf'][i].item()
            output[i] = self.diff_jpegs[idx](input[i:i+1])
        return output

class RandomBlur(AugmentationBase2D):
    def __init__(self, blur_size=17, p=0.5):
        super().__init__(p=p)
        # Pre-create GaussianBlur modules for odd kernel sizes
        kernels = list(range(1, blur_size, 2))
        self.blurs = nn.ModuleList(
            [K.RandomGaussianBlur((k, k), (k * 0.15 + 0.35, k * 0.15 + 0.35), p=1.0).to(device)
             for k in kernels]
        )

    def generate_parameters(self, input_shape: torch.Size):
        idx = torch.randint(len(self.blurs), (input_shape[0],), device=device)
        return dict(blur_idx=idx)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B = input.shape[0]
        output = torch.empty_like(input, device=device)
        for i in range(B):
            idx = params['blur_idx'][i].item()
            output[i] = self.blurs[idx](input[i:i+1])
        return output
    
class HiddenAug(nn.Module):
    """
    A flexible augmentation sequence including identity, flips, crops, resized crops,
    blurs, JPEG, affine, and color jitter, all moved to GPU.
    """
    def __init__(
        self,
        img_size,
        p_crop=0.3,
        p_res=0.3,
        p_blur=0.3,
        p_jpeg=0.3,
        p_rot=0.3,
        p_color_jitter=0.3,
    ):
        super().__init__()
        augmentations = []
        # Identity and horizontal flip
        augmentations += [nn.Identity(), K.RandomHorizontalFlip(p=1.0).to(device)]

        # Random crops
        if p_crop > 0:
            for ratio in [0.3, 0.7]:
                size = int(img_size * np.sqrt(ratio))
                augmentations.append(
                    K.RandomCrop((size, size), p=1.0).to(device)
                )

        # Random resized crops (resolution scaling)
        if p_res > 0:
            for ratio in [0.3, 0.7]:
                size = int(img_size * np.sqrt(ratio))
                augmentations.append(
                    K.RandomResizedCrop((size, size), scale=(1.0, 1.0), p=1.0).to(device)
                )

        # Gaussian blurs
        if p_blur > 0:
            augmentations.append(
                K.RandomGaussianBlur((11,11), (2.0,2.0), p=1.0).to(device)
            )

        # JPEG quality variations
        if p_jpeg > 0:
            augmentations += [DiffJPEG(50), DiffJPEG(80)]
            for mj in augmentations[-2:]:
                mj.to(device)

        # Affine transformations
        if p_rot > 0:
            for deg in [(-10,10), (90,90), (-90,-90)]:
                augmentations.append(
                    K.RandomAffine(degrees=deg, p=1.0).to(device)
                )

        # Color jitter (Color Jitter)
        if p_color_jitter > 0:
            for params in [
                (1.5,0,0,0),
                (0,1.5,0,0),
                (0,0,1.5,0),
                (0,0,0,0.25),
            ]:
                augmentations.append(
                    K.ColorJitter(*params, p=1.0).to(device)
                )

        # Compose as sequential with random apply
        self.hidden_aug = K.AugmentationSequential(*augmentations, random_apply=1).to(device)

    def forward(self, x):
        x = x.to(device)
        return self.hidden_aug(x)


    
class KorniaAug(nn.Module):
    """
    Combined Kornia augmentation pipeline with all ops moved to device.
    """
    def __init__(
        self,
        img_size=224,
        p_crop=0.5,
        p_aff=0.5,
        p_blur=0.5,
        p_color_jitter=0.5,
        p_diff_jpeg=0.5,
        degrees=30,
        crop_scale=(0.2, 1.0),
        crop_ratio=(3/4, 4/3),
        blur_size=17,
        color_jitter=(1.0, 1.0, 1.0, 0.3),
        diff_jpeg_low=10,
        diff_jpeg_high=100,
        diff_jpeg_step=10,
    ):
        super().__init__()
        # Instantiate and move to device
        self.jitter = K.ColorJitter(*color_jitter, p=p_color_jitter).to(device)
        self.aff = K.RandomAffine(degrees=degrees, p=p_aff).to(device)
        self.crop = K.RandomResizedCrop((img_size, img_size), scale=crop_scale, ratio=crop_ratio, p=p_crop).to(device)
        self.hflip = K.RandomHorizontalFlip(p=0.5).to(device)
        self.blur = RandomBlur(blur_size, p_blur)
        self.diff_jpeg = RandomDiffJPEG(p_diff_jpeg, low=diff_jpeg_low, high=diff_jpeg_high, step=diff_jpeg_step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        x = self.diff_jpeg(x)
        x = self.aff(x)
        x = self.crop(x)
        x = self.blur(x)
        x = self.jitter(x)
        x = self.hflip(x)
        return x
