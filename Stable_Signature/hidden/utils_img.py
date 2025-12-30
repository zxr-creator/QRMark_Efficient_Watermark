# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torchvision import transforms
from torchvision.transforms import functional
from augly.image import functional as aug_functional

import kornia.augmentation as K

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
image_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

normalize_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
unnormalize_rgb = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
normalize_yuv = transforms.Normalize(mean=[0.5, 0, 0], std=[0.5, 1, 1])
unnormalize_yuv = transforms.Normalize(mean=[-0.5/0.5, 0, 0], std=[1/0.5, 1/1, 1/1])


def normalize_img(x):
    """ Normalize image to approx. [-1,1] """
    return (x - image_mean.to(x.device)) / image_std.to(x.device)

def unnormalize_img(x):
    """ Unnormalize image to [0,1] """
    return (x * image_std.to(x.device)) + image_mean.to(x.device)

def round_pixel(x):
    """ 
    Round pixel values to nearest integer. 
    Args:
        x: Image tensor with values approx. between [-1,1]
    Returns:
        y: Rounded image tensor with values approx. between [-1,1]
    """
    x_pixel = 255 * unnormalize_img(x)
    y = torch.round(x_pixel).clamp(0, 255)
    y = normalize_img(y/255.0)
    return y

def clamp_pixel(x):
    """ 
    Clamp pixel values to 0 255. 
    Args:
        x: Image tensor with values approx. between [-1,1]
    Returns:
        y: Rounded image tensor with values approx. between [-1,1]
    """
    x_pixel = 255 * unnormalize_img(x)
    y = x_pixel.clamp(0, 255)
    y = normalize_img(y/255.0)
    return y

def project_linf(x, y, radius):
    """ 
    Clamp x so that Linf(x,y)<=radius
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
        radius: Radius of Linf ball for the images in pixel space [0, 255]
     """
    delta = x - y
    delta = 255 * (delta * image_std.to(x.device))
    delta = torch.clamp(delta, -radius, radius)
    delta = (delta / 255.0) / image_std.to(x.device)
    return y + delta

def psnr(x, y):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    delta = x - y
    delta = 255 * (delta * image_std.to(x.device))
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1]) # BxCxHxW
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2, dim=(1,2,3)))  # B
    return psnr

def center_crop(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]

    # left = int(x.size[0]/2-new_edges_size[0]/2)
    # upper = int(x.size[1]/2-new_edges_size[1]/2)
    # right = left + new_edges_size[0]
    # lower = upper + new_edges_size[1]

    # return x.crop((left, upper, right, lower))
    x = functional.center_crop(x, new_edges_size)
    return x

def resize(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.resize(x, new_edges_size)

def rotate(x, angle):
    """ Rotate image by angle
    Args:
        x: image (PIl or tensor)
        angle: angle in degrees
    """
    return functional.rotate(x, angle)

def adjust_brightness(x, brightness_factor):
    """ Adjust brightness of an image
    Args:
        x: PIL image
        brightness_factor: brightness factor
    """
    return normalize_img(functional.adjust_brightness(unnormalize_img(x), brightness_factor))

def adjust_contrast(x, contrast_factor):
    """ Adjust constrast of an image
    Args:
        x: PIL image
        contrast_factor: contrast factor
    """
    return normalize_img(functional.adjust_contrast(unnormalize_img(x), contrast_factor))

def jpeg_compress(x, quality_factor):
    """ Apply jpeg compression to image
    Args:
        x: Tensor image
        quality_factor: quality factor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    x = unnormalize_img(x)
    for ii,img in enumerate(x):
        pil_img = to_pil(img)
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
    return normalize_img(img_aug)

def gaussian_blur(x, sigma=1):
    """ Add gaussian blur to image
    Args:
        x: Tensor image
        sigma: sigma of gaussian kernel
    """
    x = unnormalize_img(x)
    x = functional.gaussian_blur(x, sigma=sigma, kernel_size=21)
    x = normalize_img(x)
    return x

def get_tile(imgs: torch.Tensor,
            imgs_w: torch.Tensor,
            tile_size: int,
            tile_type: str = "grid") -> torch.Tensor:
    """
    Extract a square tile from a batch of images.

    Args
    ----
    imgs: Tensor of shape (B, C, H, W)
    tile_size: edge length (pixels) of the square tile
    tile_type: 'random' | 'random_grid' | 'grid'
        - random       : completely random top-left corner
        - random_grid  : random grid-aligned tile
        - grid         : deterministic top-left tile (row=0, col=0)

    Returns
    -------
    Tensor of shape (B, C, tile_size, tile_size)
    """
    if imgs.ndim != 4:
        raise ValueError("imgs must have shape (B, C, H, W)")
    B, C, H, W = imgs.shape

    if tile_size > H or tile_size > W:
        raise ValueError(
            f"tile_size={tile_size} larger than image size ({H}, {W})"
        )

    device = imgs.device
    dtype  = torch.long

    # ------------------------------------------------------------
    # 1) 计算每张图左上角坐标 (top, left)                       #
    # ------------------------------------------------------------
    if tile_type == "random":
        top  = torch.randint(0, H - tile_size + 1, (B,), device=device, dtype=dtype)
        left = torch.randint(0, W - tile_size + 1, (B,), device=device, dtype=dtype)

    elif tile_type == "random_grid":
        n_rows = H // tile_size          # 可放置多少个整格
        n_cols = W // tile_size
        if n_rows == 0 or n_cols == 0:
                raise ValueError(
                f"Image ({H}x{W}) cannot be divided into grid of size {tile_size}"
            )
        row_idx = torch.randint(0, n_rows, (B,), device=device, dtype=dtype)
        col_idx = torch.randint(0, n_cols, (B,), device=device, dtype=dtype)
        top  = row_idx * tile_size
        left = col_idx * tile_size

    elif tile_type == "grid":
        # 固定左上角 tile
        top  = torch.zeros(B, device=device, dtype=dtype)
        left = torch.zeros(B, device=device, dtype=dtype)

    else:
        raise ValueError(f"Unsupported tile_type '{tile_type}'")

    # ------------------------------------------------------------
    # 2) 根据坐标切片，组装 batch                                #
    # ------------------------------------------------------------
    tiles = torch.empty((B, C, tile_size, tile_size),
                        dtype=imgs.dtype,
                        device=device)
    imgs_tile = torch.empty((B, C, tile_size, tile_size),
                        dtype=imgs.dtype,
                        device=device)
    for b in range(B):
        y0, x0 = top[b].item(), left[b].item()
        imgs_tile[b] = imgs[b, :, y0 : y0 + tile_size, x0 : x0 + tile_size]
        tiles[b] = imgs_w[b, :, y0 : y0 + tile_size, x0 : x0 + tile_size]
        

    return imgs_tile, tiles