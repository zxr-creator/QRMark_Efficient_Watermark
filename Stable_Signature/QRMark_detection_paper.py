# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
from copy import deepcopy
from omegaconf import OmegaConf
from pathlib import Path
from typing import Callable, Iterable
from torch.cuda import nvtx
from torchvision.datasets.folder import is_image_file, default_loader
import torchvision                                                        # add
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.utils import save_image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import triton.language as tl
import triton
import utils
import utils_img
import utils_model
import glob
from optimized_transform import optimized_transform
from PIL import Image

sys.path.append('src')
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddpm import LatentDiffusion
from loss.loss_provider import LossProvider
import torch
from bit_reedsolo import ReedSolomonCodec_BW, UncorrectableError_RS
from typing import Tuple, Union, List, Dict, Any
import time
import csv
import concurrent.futures
from collections import deque
from tradeoff_stream_alloc import (
    allocate_streams_greedy_exact_flow,
    recommend_B_cap,
    choose_Q_under_caps,
    )
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def encode_reed_solomon(original_tensor: torch.Tensor, nbits: int, num_parity_symbols: int, m_bits_per_symbol: int):
    """
    Encode a 1-D tensor of integers (symbols) with Reed-Solomon error correction.

    Args:
        original_tensor: torch.Tensor of shape (k,) containing integer symbols in GF(2^m).
        nbits: Total codeword length (n = k + correct_bits).
        correct_bits: Number of parity symbols (n - k).

    Returns:
        torch.Tensor of shape (n,) containing encoded codeword symbols as torch.float32.

    Raises:
        ValueError: If the GF field size is invalid or encoding fails.
    """
    # Determine message length k and Galois field degree m
    message_symbols = nbits // m_bits_per_symbol - num_parity_symbols
    # Initialize codec
    codec = ReedSolomonCodec_BW(message_symbols, num_parity_symbols, m_bits_per_symbol)

    # Convert tensor to Python list of ints
    print(f"original_tensor:{original_tensor}")
    message_list = original_tensor.to(torch.int64).tolist()
    print(f"message_list:{message_list}")

    # Perform encoding (list[int] -> list[int])
    codeword = codec.encode(message_list[0])
    print(f"codeword:{codeword}")

    # Convert back to torch tensor
    return torch.tensor(codeword, dtype=torch.float32, device=original_tensor.device)
        
def decode_reed_solomon(encoded_tensor: torch.Tensor, nbits: int, num_parity_symbols: int, m_bits_per_symbol: int):
    """
    Batch decode and correct Reed-Solomon codewords.
    If decoding fails, fallback to using the raw message bits.

    Args:
        encoded_tensor: torch.Tensor of shape (batch_size, 768) containing received codeword logits.
        nbits: Total codeword length (n = k + correct_bits).
        num_parity_symbols: Number of parity symbols.
        m_bits_per_symbol: Number of bits per symbol (usually 1 for binary RS code).

    Returns:
        torch.Tensor of shape (batch_size, k) containing decoded message symbols.
    """
    # Determine message length k and Galois field degree m
    num_message_symbols = nbits // m_bits_per_symbol - num_parity_symbols

    # Initialize codec
    codec = ReedSolomonCodec_BW(num_message_symbols, num_parity_symbols, m_bits_per_symbol)

    # If input is 1D, unsqueeze it
    if encoded_tensor.dim() == 1:
        encoded_tensor = encoded_tensor.unsqueeze(0)

    batch_size = encoded_tensor.size(0)

    # Collect decoded messages
    decoded_list = []

    for i in range(batch_size):
        # Get the first `nbits` bits for this sample
        codeword_logits = encoded_tensor[i, :nbits]

        # Convert logits to int symbols (0 or 1)
        codeword_symbols = (codeword_logits > 0).to(torch.int64).tolist()

        if len(codeword_symbols) != nbits:
            raise ValueError(f"Expected {nbits} symbols, got {len(codeword_symbols)}")

        try:
            message, corrected_cw, num_corr = codec.decode(codeword_symbols)
        except UncorrectableError_RS:
            # Fallback: Directly take the first `num_message_symbols`
            message = codeword_symbols[:num_message_symbols * m_bits_per_symbol]

        decoded_list.append(message)

    # Stack all decoded outputs
    decoded_tensor = torch.tensor(decoded_list, dtype=torch.float32, device=encoded_tensor.device)
    return decoded_tensor

def collate_fn(batch):
    """ Collate function for data loader. Allows to have img of different size"""
    return batch
        
class ImageFolder(torch.utils.data.Dataset):
    """ImageFolder that returns (B, 3, H, W) uint8 tensor (PIL already converted)."""
    _EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    def __init__(self, path, transform=None, loader=default_loader, *, num_imgs=None):
        self.samples = [
            p.as_posix()
            for p in Path(path).rglob('*')            
            if p.suffix.lower() in self._EXTS
        ]
        if num_imgs is not None and num_imgs < len(self.samples):
            import random
            self.samples = random.sample(self.samples, num_imgs)

        self.loader = loader
        self.transform = transform

    def __getitem__(self, idx: int):
        nvtx.range_push("load_image_and_transform")
        assert 0 <= idx < len(self)
        img = self.loader(self.samples[idx])  # → PIL.Image.Image
        tensor = TF.pil_to_tensor(img)         # → (3,H,W) uint8 tensor
        nvtx.range_pop()
        return self.transform(tensor) if self.transform else tensor

    def __len__(self):
        return len(self.samples)
    
def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--train_dir", type=str, help="Path to the training data directory", required=True)
    aa("--val_dir", type=str, help="Path to the validation data directory", required=True)

    group = parser.add_argument_group('Model parameters')
    aa("--ldm_config", type=str, default="sd/stable-diffusion-v2-1/v2-1_512-ema-pruned.yaml", help="Path to the configuration file for the LDM model") 
    aa("--ldm_ckpt", type=str, default="sd/stable-diffusion-v2-1/v2-1_512-ema-pruned.ckpt", help="Path to the checkpoint file for the LDM model") 
    aa("--msg_decoder_path", type=str, default= "models/hidden/dec_48b_whit.torchscript.pt", help="Path to the hidden decoder for the watermarking model")
    aa("--num_bits", type=int, default=48, help="Number of bits in the watermark")
    aa("--redundancy", type=int, default=1, help="Number of times the watermark is repeated to increase robustness")
    aa("--decoder_depth", type=int, default=8, help="Depth of the decoder in the watermarking model")
    aa("--decoder_channels", type=int, default=64, help="Number of channels in the decoder of the watermarking model")

    group = parser.add_argument_group('Training parameters')
    aa("--batch_size", type=int, default=4, help="Batch size for training")
    aa("--img_size", type=int, default=256, help="Resize images to this size")
    aa("--loss_i", type=str, default="watson-vgg", help="Type of loss for the image loss. Can be watson-vgg, mse, watson-dft, etc.")
    aa("--loss_w", type=str, default="bce", help="Type of loss for the watermark loss. Can be mse or bce")
    aa("--lambda_i", type=float, default=1.0, help="Weight of the image loss in the total loss")
    aa("--lambda_w", type=float, default=2.0, help="Weight of the watermark loss in the total loss")
    aa("--optimizer", type=str, default="AdamW,lr=5e-4", help="Optimizer and learning rate for training")
    aa("--steps", type=int, default=100, help="Number of steps to train the model for")
    aa("--warmup_steps", type=int, default=20, help="Number of warmup steps for the optimizer")

    group = parser.add_argument_group('Logging and saving freq. parameters')
    aa("--log_freq", type=int, default=20, help="Logging frequency (in steps)")
    aa("--save_img_freq", type=int, default=20, help="Frequency of saving generated images (in steps)")

    group = parser.add_argument_group('Experiments parameters')
    aa("--num_keys", type=int, default=1, help="Number of fine-tuned checkpoints to generate")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--seed", type=int, default=0)
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")
    
    group = parser.add_argument_group('Efficient parameters')
    aa("--tile", type=str, default='none',
       help="Use a fixed top-left tile for extraction, tile strategies include:(random/random_grid/fixed/none)")
    aa("--tile_size", type=int, default=64,
       help="Size of the square tile to use when --tile is enabled")
    aa("--reed_solomon", type=utils.bool_inst, default=False, help="If True, use Reed-Solomon codes for encoding the watermark", required=True)
    aa("--num_parity_symbols", type=int, default=2, help="Length of the correction symbols for Reed-Solomon codes.")
    aa("--m_bits_per_symbol", type=int, default=8, help="bits per symbol for the reed-solomon correction.")
    aa("--transform_fuse", type=utils.bool_inst, default=False, help="Wheather to use optimized transform or not.")
    aa('--wm_dir', type=str, default='dataset/watermark_imgs', help='directory to store water-marked images')
    aa("--val_batch_size", type=int, default=16, help="Batch size for profiling")
    aa('--val_img_num', type=int, default=40503, help='The size of the image workload')
    aa("--num_streams", type=int, default=3, help="Number of pipeline CUDA streams for detection")
    aa("--num_rs_threads", type=int, default=4, help="Number of rs threads for detection")
    aa("--async_rs", type=utils.bool_inst, default=False, help="If True, use async reed-solomon correction for detection")
    aa("--videos_dir", type=str, default="/ssd2/videomme/", help="The path of the video dataset (Default: /ssd2/videomme/)")
    aa("--wm_videos_dir", type=str, default="/ssd2/wm_videomme/", help="The path of the watermarked video dataset (Default: /ssd2/wm_videomme/)")
    aa('--val_videos_num', type=int, default=300, help='The size of the videos workload')
    aa("--workload", type=str, default="images", help="The workload for the QRMark detection (Default: images (images/videos))")
    aa("--adaptive_schedule", type=utils.bool_inst, default=False, help="Wheather to use adaptive scheduling (Default: False)")
    
    return parser


def main(params):

    # Set seeds for reproductibility 
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    
        
    # Print the arguments
    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(json.dumps(vars(params))))
    
    # Create the directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    imgs_dir = os.path.join(params.output_dir, 'imgs')
    params.imgs_dir = imgs_dir
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir, exist_ok=True)
    # Loads LDM auto-encoder models
    print(f'>>> Building LDM model with config {params.ldm_config} and weights from {params.ldm_ckpt}...')
    config = OmegaConf.load(f"{params.ldm_config}")
    ldm_ae: LatentDiffusion = utils_model.load_model_from_config(config, params.ldm_ckpt)
    ldm_ae: AutoencoderKL = ldm_ae.first_stage_model
    ldm_ae.eval()
    ldm_ae.to(device)
    
    # Loads hidden decoder
    print(f'>>> Building hidden decoder with weights from {params.msg_decoder_path}...')
    if 'torchscript' in params.msg_decoder_path: 
        msg_decoder = torch.jit.load(params.msg_decoder_path).to(device)
        # already whitened
        
    else:
        msg_decoder = utils_model.get_hidden_decoder(num_bits=params.num_bits, redundancy=params.redundancy, num_blocks=params.decoder_depth, channels=params.decoder_channels).to(device)
        ckpt = utils_model.get_hidden_decoder_ckpt(params.msg_decoder_path)
        print(msg_decoder.load_state_dict(ckpt, strict=False))
            
        msg_decoder.eval()
        whiten_msg_path = params.msg_decoder_path.replace(".pth", "_whit.torchscript.pt")
        if os.path.exists(whiten_msg_path):
            msg_decoder = torch.jit.load(str(whiten_msg_path)).to(device)
            msg_decoder.eval()
            params.msg_decoder_path = str(whiten_msg_path)
        else:    
            # whitening
            print(f'>>> Whitening...')
            with torch.no_grad():
                # features from the dataset
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                loader = utils.get_dataloader(params.train_dir, transform, batch_size=16, collate_fn=None)
                ys = []
                for i, x in enumerate(loader):
                    x = x.to(device)
                    y = msg_decoder(x)
                    ys.append(y.to('cpu'))
                ys = torch.cat(ys, dim=0)
                nbit = ys.shape[1]
            
                # whitening
                mean = ys.mean(dim=0, keepdim=True) # NxD -> 1xD
                ys_centered = ys - mean # NxD
                cov = ys_centered.T @ ys_centered
                e, v = torch.linalg.eigh(cov)
                L = torch.diag(1.0 / torch.pow(e, exponent=0.5))
                weight = torch.mm(L, v.T)
                bias = -torch.mm(mean, weight.T).squeeze(0)
                linear = nn.Linear(nbit, nbit, bias=True)
                linear.weight.data = np.sqrt(nbit) * weight
                linear.bias.data = np.sqrt(nbit) * bias
                msg_decoder = nn.Sequential(msg_decoder, linear.to(device))
                torchscript_m = torch.jit.script(msg_decoder)
                params.msg_decoder_path = params.msg_decoder_path.replace(".pth", "_whit.torchscript.pt")
                print(f'>>> Creating torchscript at {params.msg_decoder_path}...')
                torch.jit.save(torchscript_m, params.msg_decoder_path)
                
    msg_decoder.eval()
    nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(device)).shape[-1]

    # Freeze LDM and hidden decoder
    for param in [*msg_decoder.parameters(), *ldm_ae.parameters()]:
        param.requires_grad = False

   # Loads the data
    print(f'>>> Loading data from {params.train_dir} and {params.val_dir}...')
    vqgan_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_img,
    ])
    if params.transform_fuse == True:
            vqgan_transform = optimized_transform
            
    # Create losses
    print(f'>>> Creating losses...')
    print(f'Losses: {params.loss_w} and {params.loss_i}...')
    if params.loss_w == 'mse':        
        loss_w = lambda decoded, keys, temp=10.0: torch.mean((decoded*temp - (2*keys-1))**2) # b k - b k
    elif params.loss_w == 'bce':
        loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded*temp, keys, reduction='mean')
    else:
        raise NotImplementedError
    
    if params.loss_i == 'mse':
        loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs)**2)
    elif params.loss_i == 'watson-dft':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif params.loss_i == 'watson-vgg':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif params.loss_i == 'ssim':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    else:
        raise NotImplementedError

    for ii_key in range(params.num_keys):

        # Creating key using Reed-Solomon Codes
        print(f'\n>>> Creating key with {nbit} bits...')
        
        if params.reed_solomon == True:
            print("Now using Reed_Solomon code.")
            message_length = nbit - params.num_parity_symbols * params.m_bits_per_symbol
            msg_key = torch.randint(0, 2, (1, message_length), dtype=torch.float32, device=device)
            key = encode_reed_solomon(msg_key, nbit, params.num_parity_symbols, params.m_bits_per_symbol)
            #print(f"After encoding, key:{key}, message_key:{msg_key}")
            key_str = "".join([ str(int(ii)) for ii in key.tolist()])
            msg_key_str = "".join([str(int(ii)) for ii in msg_key.tolist()[0]])
            print(f"After encoding, key_str:{key_str}, msg_key_str:{msg_key_str}")
        else:
            print("Now not using Reed_Solomon code") 
            key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=device)
            key_str = "".join([ str(int(ii)) for ii in key.tolist()[0]])
            msg_key = key
            msg_key_str = key_str
            print(f"key:{key_str}")
            #print(f"type of the str:{type(key_str)}")
        
        # Copy the LDM decoder and finetune the copy
        ldm_decoder = deepcopy(ldm_ae)
        ldm_decoder.encoder = nn.Identity()
        ldm_decoder.quant_conv = nn.Identity()
        ldm_decoder.to(device)
        for param in ldm_decoder.parameters():
            param.requires_grad = True
        optim_params = utils.parse_params(params.optimizer)
        optimizer = utils.build_optimizer(model_params=ldm_decoder.parameters(), **optim_params)
        
        ## Set transform method
        
        # Detection from saved images 
        
        if params.workload =="images":
            EVAL_IMG_NUM = 1000
            eval_loader = utils.get_dataloader(params.wm_dir, vqgan_transform, batch_size=params.val_batch_size, num_imgs=1000, shuffle=False, num_workers=4,  collate_fn=None)
            eval_loader = DataLoader(
                ImageFolder(params.wm_dir, transform=None, num_imgs=EVAL_IMG_NUM),
                batch_size=params.val_batch_size, shuffle=False, num_workers=4,
                pin_memory=True, drop_last=False, collate_fn=None
            )
            eval_stats = val(eval_loader, ldm_ae, ldm_decoder, msg_decoder, optimized_transform, key, msg_key, params)
            
            nvtx.range_push("E2E_Image_Watermark_Detection")
            print(f'>>> Detecting from saved images...')
            nvtx.range_push("Image Dataset Loading")
            imagefolder = ImageFolder(params.wm_dir, transform=None, num_imgs=params.val_img_num)
            detect_loader = DataLoader(imagefolder, batch_size=params.val_batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, collate_fn=None)
            nvtx.range_pop()
            nvtx.range_push("Core_Pipeline")
            start_e2e = time.perf_counter()
            _ = vqgan_transform(torch.zeros(1, 3, 256, 256, device="cuda"), params.img_size)
            detect_from_images_stats = detect_from_images(detect_loader, ldm_ae, ldm_decoder, msg_decoder, vqgan_transform, key, msg_key, nbit, params)
            torch.cuda.synchronize()
            end_e2e = time.perf_counter() 
            print(f"[E2E] Total wall time = {end_e2e - start_e2e:.4f} s")
            nvtx.range_pop()
            nvtx.range_push("Statistics_and_Logging")
            log_stats = {
                'key': key_str,
                'msg_key': msg_key_str,
                **{f"eval_{k}": v for k, v in eval_stats.items()},
                **{f"detect_from_images_{k}": v for k, v in detect_from_images_stats.items()},
            }

            nvtx.range_pop()
            
            nvtx.range_pop()
            
        '''
        # Detection from saved videos 
        if params.workload =="video":
            nvtx.range_push("E2E_Video_Watermark_Detection")
            print(">>> Detecting from saved videos ...")
            video_ds = VideoFramesDataset(params.wm_videos_dir, transform=None, num_videos=params.val_videos_num)
            detect_loader = DataLoader(video_ds, batch_size=params.val_batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, collate_fn=flatten_video_batch)
            nvtx.range_push("Video Watermark Detection")
            _ = vqgan_to_imnet(torch.zeros(1, 3, 256, 256, device="cuda"))  # warms up fused kernel
            detect_from_video_stats = detect_from_videos(detect_loader, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, msg_key, params)
            nvtx.range_pop()
            log_stats = {
                'key': key_str,
                'msg_key': msg_key_str,
                **{f"detect_from_video_{k}": v for k, v in detect_from_video_stats.items()},
            }
            nvtx.range_pop()
        '''
        
        save_dict = {
            'ldm_decoder': ldm_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        }

# ---------- help function1: for image tiling ----------
def _decode_with_tile(
        imgs_aug: torch.Tensor,
        msg_decoder: nn.Module,
        params: argparse.Namespace
    ):
    """
    Apply decoding based on the tile strategy (random / random_grid / fixed / none).

    imgs_aug : [B, C, H, W]
    """
    B, C, H, W = imgs_aug.shape
    size = params.tile_size
    tile_mode = getattr(params, "tile", "none")

    # ---------- random: pick one (y0, x0) per image ----------
    if tile_mode == "random":
        if size > H or size > W:
            raise ValueError(f"tile_size={size} exceeds image shape ({H}, {W})")
        max_y = H - size
        max_x = W - size

        patches = []
        for img in imgs_aug:
            y0 = torch.randint(0, max_y + 1, ()).item() if max_y > 0 else 0
            x0 = torch.randint(0, max_x + 1, ()).item() if max_x > 0 else 0
            patches.append(img[:, y0:y0 + size, x0:x0 + size])

        patches = torch.stack(patches, dim=0)             # [B, C, S, S]
        return patches

    # ---------- random_grid: divide image into tile_size grid, then choose one ----------
    elif tile_mode == "random_grid":
        n_rows = H // size
        n_cols = W // size
        assert n_rows > 0 and n_cols > 0, "tile_size is larger than the image dimensions"
        patches = []
        for img in imgs_aug:
            row = torch.randint(0, n_rows, ()).item()
            col = torch.randint(0, n_cols, ()).item()
            y0, x0 = row * size, col * size
            patches.append(img[:, y0:y0 + size, x0:x0 + size])
        patches = torch.stack(patches, dim=0)
        return patches

    # ---------- fixed: always take the top-left tile ----------
    elif tile_mode == "fixed":
        patches = imgs_aug[:, :, 0:size, 0:size]
        return patches

    # ---------- default: no tiling ----------
    else:
        return imgs_aug
    
# ---------- help function2: Compute per-stage bytes ----------    
def _bytes_per_sample(params, nbit):
    # Stage-0 output ~ (B,3,H,W) float32
    H = W = params.img_size
    s0 = 3 * H * W * 4
    # Stage-1 output ~ (B,3,S,S) float32  (if no tiling, it equals s0)
    S = params.tile_size if getattr(params, "tile", "none") != "none" else params.img_size
    s1 = 3 * S * S * 4
    # Stage-2 output ~ (B,nbit) float32 logits
    s2 = nbit * 4
    return [s0, s1, s2]






@torch.no_grad()
def detect_from_images(
        data_loader: Iterable,
        ldm_ae, ldm_decoder, msg_decoder,
        vqgan_transform, key, msg_key, nbit, params):

    rank  = torch.cuda.current_device()
    device = torch.device('cuda', rank)
    

    # -------------------- 0) Warm-up --------------------
    w = max(12, 6)  
    evt_beg = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    evt_end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    lat_samples = [[], [], []]

    warm_loader = iter(data_loader)
    for _ in range(w):
        try:
            imgs_cpu = next(warm_loader)
        except StopIteration:
            break
        imgs = imgs_cpu.to(device, non_blocking=True)

        # Stage-0: preprocess
        evt_beg[0].record()
        imgs_aug = (vqgan_transform(imgs, params.img_size)
                    if params.transform_fuse else
                    vqgan_transform(imgs))
        evt_end[0].record()

        # Stage-1: tiling
        evt_beg[1].record()
        patch = _decode_with_tile(imgs_aug, msg_decoder, params)
        evt_end[1].record()

        # Stage-2: decode
        evt_beg[2].record()
        _ = msg_decoder(patch)
        evt_end[2].record()

        torch.cuda.synchronize()
        for k in range(3):
            lat_samples[k].append(evt_beg[k].elapsed_time(evt_end[k]) / 1e3)  # seconds

    def _trimmed_mean(x, trim=0.2):
        if not x:
            return 0.0
        x = sorted(x)
        n = len(x)
        t = int(n * trim)
        xs = x[t:n - t] if n - t > t else x
        return sum(xs) / max(1, len(xs))

    stage_lat = [_trimmed_mean(s, trim=0.2) for s in lat_samples]

    # -------------------- 1) Allocate streams (greedy) & STRICT flow --------------------
    S_max   = max(3, int(params.num_streams))
    mem_ps  = _bytes_per_sample(params, nbit)

    total_hbm   = torch.cuda.get_device_properties(0).total_memory
    M_cap_bytes = int(0.8 * total_hbm)

    B_user = int(getattr(params, "val_batch_size", 0) or 0)
    B_cap  = recommend_B_cap(S_max=S_max,
                             mem_per_item=mem_ps,
                             M_cap_bytes=M_cap_bytes,
                             B_hint=B_user)

    util_ratio = None  # format: [u0,u1,u2]
    s, b, Q = allocate_streams_greedy_exact_flow(
        t_baseline=stage_lat,
        B=B_cap,
        mem_per_item=mem_ps,
        M_cap_bytes=M_cap_bytes,
        S_max=S_max,
        util_ratio=util_ratio,
    )
    if Q > 0:
        B_cap = max(Q, (B_cap // Q) * Q)  

    print(f"[SO1] stage times={stage_lat}, streams={s}, per-stage micro-batch b={b}, Q={Q}, B_cap={B_cap}")
    print(f"[Alloc] CUDA-stream vector  s = {s}")
    print(f"[Alloc] STRICT flow check: {[b[k]*s[k] for k in range(len(s))]}  (all == Q={Q})")
    
    start_detect = time.perf_counter()
    # -------------------- 2) Runtime objects --------------------
    h2d_stream    = torch.cuda.Stream(priority=-1)
    stage_streams = [[torch.cuda.Stream() for _ in range(sk)] for sk in s]  # 0/1/2

    async_rs       = getattr(params, "async_rs", True)
    num_rs_threads = getattr(params, "num_rs_threads", 16)
    rs_pool        = concurrent.futures.ThreadPoolExecutor(max_workers=num_rs_threads) if async_rs else None

    metric_logger  = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.eval()

    inflight0, inflight1, inflight2 = [], [], []
    pool01, pool12 = deque(), deque()

    def _prefetch(batch):
        if batch is None:
            return None
        with torch.cuda.stream(h2d_stream):
            return batch.to(device, non_blocking=True)

    def _drain_inflight(inflight_list, pool):
        keep = []
        for ev, ten in inflight_list:
            if ev.query():
                pool.append(ten)
            else:
                keep.append((ev, ten))
        return keep

    def _cat_from_pool(pool: deque, target_bs: int):
        if target_bs <= 0:
            return None
        need = target_bs
        chunks = []
        while need > 0 and len(pool) > 0:
            t = pool[0]
            n = t.size(0)
            if n <= need:
                chunks.append(t)
                pool.popleft()
                need -= n
            else:
                chunks.append(t[:need])
                pool[0] = t[need:]
                need = 0
        if need == 0:
            return torch.cat(chunks, dim=0)
        return None  

    def _pool_total_size(pool: deque):
        return sum(int(t.size(0)) for t in pool)

    # -------------------- 3) Main loop --------------------
    it = iter(data_loader)
    pending_rs = []
    #batch_idx = 0

    while (batch_cpu := next(it, None)) is not None:
        batch_gpu = _prefetch(batch_cpu)
        torch.cuda.current_stream().wait_stream(h2d_stream)
        n_imgs = batch_gpu.size(0)

        micro_batches0 = list(torch.split(batch_gpu, b[0]))
        q0 = 0

        done_this_batch = 0
        msg_keys_gpu = msg_key.repeat(n_imgs, 1)
        k_ptr = 0

        while (done_this_batch < n_imgs or inflight0 or inflight1 or inflight2
               or pool01 or pool12 or q0 < len(micro_batches0)):

            made_progress = False


            inflight0 = _drain_inflight(inflight0, pool01)
            inflight1 = _drain_inflight(inflight1, pool12)

            # 1) Stage-0：
            for s0 in stage_streams[0]:
                if q0 >= len(micro_batches0):
                    break
                if len(inflight0) >= len(stage_streams[0]):
                    break
                with torch.cuda.stream(s0):
                    nvtx.range_push("VQGAN_to_IMNET")
                    imgs_aug = (vqgan_transform(micro_batches0[q0], params.img_size)
                                if params.transform_fuse else
                                vqgan_transform(micro_batches0[q0]))
                    nvtx.range_pop()
                ev0 = torch.cuda.Event()
                ev0.record(s0)
                inflight0.append((ev0, imgs_aug))
                q0 += 1
                made_progress = True

            # 2) Stage-1：
            for s1 in stage_streams[1]:
                if len(inflight1) >= len(stage_streams[1]):
                    break
                patch_in = _cat_from_pool(pool01, b[1])
                if patch_in is None:
                    break
                with torch.cuda.stream(s1):
                    nvtx.range_push("Tiling")
                    patch = _decode_with_tile(patch_in, msg_decoder, params)
                    nvtx.range_pop()
                ev1 = torch.cuda.Event()
                ev1.record(s1)
                inflight1.append((ev1, patch))
                made_progress = True

            # 3) Stage-2：
            for s2 in stage_streams[2]:
                if len(inflight2) >= len(stage_streams[2]):
                    break
                patch_in2 = _cat_from_pool(pool12, b[2])
                if patch_in2 is None:
                    break
                with torch.cuda.stream(s2):
                    nvtx.range_push("Message_Decode")
                    decoded = msg_decoder(patch_in2)
                    nvtx.range_pop()
                ev2 = torch.cuda.Event()
                ev2.record(s2)
                inflight2.append((ev2, decoded))
                made_progress = True

            # 4)
            keep2 = []
            for ev2, decoded in inflight2:
                if ev2.query():
                    decoded_cpu = decoded.detach().to('cpu')
                    bs2 = decoded_cpu.size(0)
                    msg_keys_cpu = msg_keys_gpu[k_ptr:k_ptr + bs2].detach().to('cpu')
                    k_ptr += bs2
                    done_this_batch += bs2
                    if params.reed_solomon:
                        if async_rs:
                            fut = rs_pool.submit(
                                decode_reed_solomon,
                                decoded_cpu,
                                params.num_bits,
                                params.num_parity_symbols,
                                params.m_bits_per_symbol
                            )
                            pending_rs.append((fut, msg_keys_cpu))
                        else:
                            decoded_corr = decode_reed_solomon(
                                decoded_cpu,
                                params.num_bits,
                                params.num_parity_symbols,
                                params.m_bits_per_symbol
                            )
                            _update_metrics(decoded_corr, msg_keys_cpu, metric_logger)
                    else:
                        _update_metrics(decoded_cpu, msg_keys_cpu, metric_logger)
                    made_progress = True
                else:
                    keep2.append((ev2, decoded))
            inflight2 = keep2

            if (not made_progress) and (q0 >= len(micro_batches0)) \
               and (not inflight0) and (not inflight1) and (not inflight2):

                total_p12 = _pool_total_size(pool12)
                if total_p12 > 0:
                    patch_in2 = []
                    while pool12:
                        patch_in2.append(pool12.popleft())
                    patch_in2 = torch.cat(patch_in2, dim=0)

                    s2 = stage_streams[2][0] 
                    with torch.cuda.stream(s2):
                        nvtx.range_push("Message_Decode_TAIL")
                        decoded_tail = msg_decoder(patch_in2)
                        nvtx.range_pop()
                    torch.cuda.current_stream().wait_stream(s2)

                    decoded_cpu = decoded_tail.detach().to('cpu')
                    bs2 = decoded_cpu.size(0)
                    msg_keys_cpu = msg_keys_gpu[k_ptr:k_ptr + bs2].detach().to('cpu')
                    k_ptr += bs2
                    done_this_batch += bs2

                    if params.reed_solomon:
                        if async_rs:
                            fut = rs_pool.submit(
                                decode_reed_solomon,
                                decoded_cpu,
                                params.num_bits,
                                params.num_parity_symbols,
                                params.m_bits_per_symbol
                            )
                            pending_rs.append((fut, msg_keys_cpu))
                        else:
                            decoded_corr = decode_reed_solomon(
                                decoded_cpu,
                                params.num_bits,
                                params.num_parity_symbols,
                                params.m_bits_per_symbol
                            )
                            _update_metrics(decoded_corr, msg_keys_cpu, metric_logger)
                    else:
                        _update_metrics(decoded_cpu, msg_keys_cpu, metric_logger)

                    continue 

                total_p01 = _pool_total_size(pool01)
                if total_p01 > 0:
                    patch_in = []
                    while pool01:
                        patch_in.append(pool01.popleft())
                    patch_in = torch.cat(patch_in, dim=0)

                    s1 = stage_streams[1][0]
                    s2 = stage_streams[2][0]
                    with torch.cuda.stream(s1):
                        nvtx.range_push("Tiling_TAIL")
                        patch = _decode_with_tile(patch_in, msg_decoder, params)
                        nvtx.range_pop()
                    torch.cuda.current_stream().wait_stream(s1)

                    with torch.cuda.stream(s2):
                        nvtx.range_push("Message_Decode_TAIL")
                        decoded_tail = msg_decoder(patch)
                        nvtx.range_pop()
                    torch.cuda.current_stream().wait_stream(s2)

                    decoded_cpu = decoded_tail.detach().to('cpu')
                    bs2 = decoded_cpu.size(0)
                    msg_keys_cpu = msg_keys_gpu[k_ptr:k_ptr + bs2].detach().to('cpu')
                    k_ptr += bs2
                    done_this_batch += bs2

                    if params.reed_solomon:
                        if async_rs:
                            fut = rs_pool.submit(
                                decode_reed_solomon,
                                decoded_cpu,
                                params.num_bits,
                                params.num_parity_symbols,
                                params.m_bits_per_symbol
                            )
                            pending_rs.append((fut, msg_keys_cpu))
                        else:
                            decoded_corr = decode_reed_solomon(
                                decoded_cpu,
                                params.num_bits,
                                params.num_parity_symbols,
                                params.m_bits_per_symbol
                            )
                            _update_metrics(decoded_corr, msg_keys_cpu, metric_logger)
                    else:
                        _update_metrics(decoded_cpu, msg_keys_cpu, metric_logger)

                    continue 

                break



    # flush async RS
    for fut, k_cpu in pending_rs:
        decoded_corr = fut.result()
        _update_metrics(decoded_corr, k_cpu, metric_logger)
    if rs_pool:
        rs_pool.shutdown(wait=True)

    torch.cuda.synchronize()
    end_detect = time.perf_counter()
    print(f"Optimized:[detect_from_images] Total wall time = {end_detect - start_detect:.4f} s")

    return {k: mtr.global_avg for k, mtr in metric_logger.meters.items()}
# ----------------------------------------------------------------------
def _update_metrics(decoded_cpu, key_cpu, metric_logger):
    diff      = ~(torch.logical_xor(decoded_cpu > 0, key_cpu > 0))
    bit_acc   = diff.sum(dim=-1).float() / diff.size(-1)
    word_acc  = (bit_acc == 1)
    metric_logger.update(bit_acc=bit_acc.mean().item(),
                         word_acc=word_acc.float().mean().item())
# ----------------------------------------------------------------------

@torch.no_grad()
def val(data_loader: Iterable,
        ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL,
        msg_decoder: nn.Module, optimized_transform: nn.Module,
        key: torch.Tensor, msg_key: torch.Tensor,
        params: argparse.Namespace):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    header = 'Eval'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.eval()

    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        # imgs: uint8 [0,255] -> float [-1,1]
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        imgs = imgs * 2.0 - 1.0

        keys = key.repeat(imgs.size(0), 1)
        msg_keys = msg_key.repeat(imgs.size(0), 1)

        # --- encode / decode (reconstruction & watermarked) ---
        imgs_z = ldm_ae.encode(imgs).mode()
        imgs_d0 = ldm_ae.decode(imgs_z)
        imgs_w  = ldm_decoder.decode(imgs_z)

        # --- baseline detection (no attack) via optimized_transform ---
        imgs_aug = (optimized_transform(imgs, params.img_size)
                    if params.transform_fuse else
                    optimized_transform(imgs))
        patch   = _decode_with_tile(imgs_aug, msg_decoder, params)
        decoded = msg_decoder(patch)

        if params.reed_solomon:
            decoded = decode_reed_solomon(decoded,
                                          params.num_bits,
                                          params.num_parity_symbols,
                                          params.m_bits_per_symbol)

        diff = ~torch.logical_xor(decoded > 0, msg_keys > 0)
        bit_accs = diff.sum(dim=-1) / diff.size(-1)
        word_accs = (bit_accs == 1)

        # TPR / FPR for 'none'
        decoded_binary = (decoded > 0).float()
        keys_binary = msg_keys
        TP = torch.sum((decoded_binary == 1) & (keys_binary == 1), dim=-1)
        FP = torch.sum((decoded_binary == 1) & (keys_binary == 0), dim=-1)
        TN = torch.sum((decoded_binary == 0) & (keys_binary == 0), dim=-1)
        FN = torch.sum((decoded_binary == 0) & (keys_binary == 1), dim=-1)
        TPR = TP / (TP + FN + 1e-8)
        FPR = FP / (FP + TN + 1e-8)

        # log base metrics
        metric_logger.update(
            psnr=utils_img.psnr(imgs_w, imgs_d0).mean().item(),
            bit_acc_none=bit_accs.mean().item(),
            word_acc_none=word_accs.float().mean().item(),
            tpr_none=torch.mean(TPR).item(),
            fpr_none=torch.mean(FPR).item(),
        )

        # --- robustness under attacks (apply to imgs_w then transform+decode) ---
        attacks = {
            'none': lambda x: x,
            'crop_01': lambda x: utils_img.center_crop(x, 0.1),
            'crop_05': lambda x: utils_img.center_crop(x, 0.5),
            'rot_25': lambda x: utils_img.rotate(x, 25),
            'rot_90': lambda x: utils_img.rotate(x, 90),
            'resize_03': lambda x: utils_img.resize(x, 0.3),
            'resize_07': lambda x: utils_img.resize(x, 0.7),
            'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
            'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            'jpeg': lambda x: utils_img.jpeg_compress(x, 80),
            'blur': lambda x: utils_img.jpeg_compress(x, 50),
        }
        for name, attack in attacks.items():
            imgs_attacked = attack(imgs_w)
            imgs_aug_a = (optimized_transform(imgs_attacked, params.img_size)
                          if params.transform_fuse else
                          optimized_transform(imgs_attacked))
            patch_a   = _decode_with_tile(imgs_aug_a, msg_decoder, params)
            decoded_a = msg_decoder(patch_a)
            if params.reed_solomon:
                decoded_a = decode_reed_solomon(decoded_a,
                                                params.num_bits,
                                                params.num_parity_symbols,
                                                params.m_bits_per_symbol)
            diff_a = ~torch.logical_xor(decoded_a > 0, msg_keys > 0)
            bit_accs_a = diff_a.sum(dim=-1) / diff_a.size(-1)
            word_accs_a = (bit_accs_a == 1)
            metric_logger.update(**{
                f'bit_acc_{name}': bit_accs_a.mean().item(),
                f'word_acc_{name}': word_accs_a.float().mean().item(),
            })

        # optionally dump a few images
        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs),   0, 1), os.path.join(params.imgs_dir, f'{ii:03}_val_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0), 0, 1), os.path.join(params.imgs_dir, f'{ii:03}_val_d0.png'),   nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w),  0, 1), os.path.join(params.imgs_dir, f'{ii:03}_val_w.png'),    nrow=8)

    print("Averaged Eval stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)