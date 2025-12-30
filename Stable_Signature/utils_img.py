# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyright: reportMissingModuleSource=false

import numpy as np
from augly.image import functional as aug_functional
import torch
from torchvision import transforms
from torchvision.transforms import functional
from torchvision.transforms.functional import InterpolationMode

import kornia as K

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize (x - 0.5) / 0.5
unnormalize_vqgan = transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5]) # Unnormalize (x * 0.5) + 0.5
normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize (x - mean) / std
unnormalize_img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]) # Unnormalize (x * std) + mean

def psnr(x, y, img_space='vqgan'):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    if img_space == 'vqgan':
        delta = torch.clamp(unnormalize_vqgan(x), 0, 1) - torch.clamp(unnormalize_vqgan(y), 0, 1)
    elif img_space == 'img':
        delta = torch.clamp(unnormalize_img(x), 0, 1) - torch.clamp(unnormalize_img(y), 0, 1)
    else:
        delta = x - y
    delta = 255 * delta
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
    return functional.center_crop(x, new_edges_size)

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
    """ Adjust contrast of an image
    Args:
        x: PIL image
        contrast_factor: contrast factor
    """
    return normalize_img(functional.adjust_contrast(unnormalize_img(x), contrast_factor))

def adjust_saturation(x, saturation_factor):
    """ Adjust saturation of an image
    Args:
        x: PIL image
        saturation_factor: saturation factor
    """
    return normalize_img(functional.adjust_saturation(unnormalize_img(x), saturation_factor))

def adjust_hue(x, hue_factor):
    """ Adjust hue of an image
    Args:
        x: PIL image
        hue_factor: hue factor
    """
    return normalize_img(functional.adjust_hue(unnormalize_img(x), hue_factor))

def adjust_gamma(x, gamma, gain=1):
    """ Adjust gamma of an image
    Args:
        x: PIL image
        gamma: gamma factor
        gain: gain factor
    """
    return normalize_img(functional.adjust_gamma(unnormalize_img(x), gamma, gain))

def adjust_sharpness(x, sharpness_factor):
    """ Adjust sharpness of an image
    Args:
        x: PIL image
        sharpness_factor: sharpness factor
    """
    return normalize_img(functional.adjust_sharpness(unnormalize_img(x), sharpness_factor))

def overlay_text(x, text='Lorem Ipsum'):
    """ Overlay text on image
    Args:
        x: PIL image
        text: text to overlay
        font_path: path to font
        font_size: font size
        color: text color
        position: text position
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.overlay_text(pil_img, text=text))
    return normalize_img(img_aug)

def jpeg_compress(x, quality_factor):
    """ Apply jpeg compression to image
    Args:
        x: PIL image
        quality_factor: quality factor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
    return normalize_img(img_aug)

def gaussian_blur(x, sigma=1, kernel_size=3):
    x = unnormalize_img(x)
    if sigma is None or sigma <= 0:
        return normalize_img(x)  # skip the blur
    x = functional.gaussian_blur(x, sigma=sigma, kernel_size=kernel_size)
    return normalize_img(x)

def gaussian_noise(x, mean=0.0, var=0.1):
    """Add Gaussian noise (in [0, 1] space) then re-normalize."""
    x = unnormalize_img(x)
    std = float(var) ** 0.5
    noise = torch.randn_like(x) * std + float(mean)
    x = torch.clamp(x + noise, 0.0, 1.0)
    return normalize_img(x)

def bilateral_denoise(x, kernel_size=5, sigma_color=0.1, sigma_space=0.1):
    """Apply bilateral denoising using Kornia on [0, 1] images."""
    x = unnormalize_img(x)
    # Kornia expects BCHW in [0, 1]
    batch = x.size(0)
    sigma_color_tensor = torch.full(
        (batch,), float(sigma_color), device=x.device, dtype=x.dtype
    )
    sigma_space_tensor = torch.full(
        (batch, 2), float(sigma_space), device=x.device, dtype=x.dtype
    )
    denoised = K.filters.bilateral_blur(
        x,
        (kernel_size, kernel_size),
        sigma_color=sigma_color_tensor,
        sigma_space=sigma_space_tensor,
    )
    return normalize_img(torch.clamp(denoised, 0.0, 1.0))

def shear(x: torch.Tensor, shear_x_degrees: float = 15.0, shear_y_degrees: float = 0.0):
    """
    Apply a deterministic shear to the image tensor in ImageNet-normalized space.

    Args:
        x: Tensor of shape [B, 3, H, W], normalized by `normalize_img`.
        shear_x_degrees: Shear factor along the x-axis in degrees.
        shear_y_degrees: Shear factor along the y-axis in degrees.

    Returns:
        Tensor of shape [B, 3, H, W], normalized by `normalize_img`.
    """
    # Unnormalize to [0, 1]
    x_unnorm = unnormalize_img(x)

    # Prepare output tensor
    out = torch.empty_like(x_unnorm)

    # torchvision's affine works on [C, H, W]; apply per image for safety
    shear_param = [float(shear_x_degrees), float(shear_y_degrees)]
    for i in range(x_unnorm.size(0)):
        out[i] = functional.affine(
            x_unnorm[i],
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=shear_param,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
            center=None,
        )

    # Re-normalize to ImageNet stats
    return normalize_img(out)

def kmeans_color_quantize(x: torch.Tensor, k: int = 8, iters: int = 5):
    """
    Simple ML-based attack via color quantization using k-means clustering.

    - Unnormalizes to [0,1], runs k-means in RGB space per image to reduce the
      color palette to `k` colors, then re-normalizes to ImageNet stats.
    - Keeps the implementation lightweight and self-contained (no dependencies).

    Args:
        x: Tensor of shape [B, 3, H, W], normalized by `normalize_img`.
        k: Number of color clusters.
        iters: Number of k-means iterations.

    Returns:
        Tensor of shape [B, 3, H, W], normalized by `normalize_img`.
    """
    B, C, H, W = x.shape
    assert C == 3, "kmeans_color_quantize expects 3-channel RGB input"

    # Work on the same device as input
    device = x.device
    to_pxs = H * W

    out = torch.empty_like(x)

    # Process per-image to keep memory reasonable
    for i in range(B):
        # Unnormalize to [0,1]
        img = unnormalize_img(x[i:i+1])  # [1,3,H,W]
        img = img.clamp(0, 1)

        # Flatten to [N, 3]
        pixels = img[0].permute(1, 2, 0).reshape(-1, 3)  # [H*W,3]

        # Initialize centers by sampling k random pixels
        if pixels.shape[0] < k:
            # Edge case: very small inputs; just skip quantization
            quant = pixels
        else:
            rand_idx = torch.randperm(pixels.shape[0], device=device)[:k]
            centers = pixels[rand_idx].clone()  # [k,3]

            for _ in range(max(1, iters)):
                # Assign step
                # distances: [N,k]
                # Use squared Euclidean distance
                dists = (pixels.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(dim=2)
                assign = dists.argmin(dim=1)  # [N]

                # Update step: recompute centers as means of assigned pixels
                new_centers = torch.zeros_like(centers)
                for c in range(k):
                    mask = (assign == c)
                    if mask.any():
                        new_centers[c] = pixels[mask].mean(dim=0)
                    else:
                        # If a cluster is empty, reinitialize to a random pixel
                        new_centers[c] = pixels[torch.randint(0, pixels.shape[0], (1,), device=device)]
                centers = new_centers

            # Reconstruct quantized image
            quant = centers[assign]  # [N,3]

        # Reshape back to [1,3,H,W]
        quant_img = quant.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0)
        # Re-normalize
        out[i:i+1] = normalize_img(quant_img)

    return out
