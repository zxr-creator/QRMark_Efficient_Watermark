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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import triton.language as tl
import triton

import utils
import utils_img
import utils_model
import glob

from PIL import Image

sys.path.append('src')
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddpm import LatentDiffusion
from loss.loss_provider import LossProvider
import torch
from bit_reedsolo import ReedSolomonCodec_BW, UncorrectableError_RS
from typing import Tuple, Union
import time
import csv
import concurrent.futures

import torchvision    

try:
    import decord                                                         # add
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False
try:
    import cv2                                                            # add
    _HAS_OPENCV = True
except ImportError:
    _HAS_OPENCV = False

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
        return msg_decoder(patches)

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
        return msg_decoder(patches)

    # ---------- fixed: always take the top-left tile ----------
    elif tile_mode == "fixed":
        patches = imgs_aug[:, :, 0:size, 0:size]
        return msg_decoder(patches)

    # ---------- default: no tiling ----------
    else:
        return msg_decoder(imgs_aug)

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

@torch.no_grad()
def _watermark_one_video(src_path: Path,
                         dst_path: Path,
                         ldm_ae,
                         ldm_decoder,
                         batch_frames: int = 16):
    """
    Read → watermark every frame → write.
    Keeps original H×W, FPS, audio.
    """
    # • read
    video, audio, info = torchvision.io.read_video(str(src_path),
                                                   pts_unit="sec")
    fps = info["video_fps"]
    T   = video.size(0)

    # pre‑allocate output tensor on CPU (uint8)
    out_video = torch.empty_like(video)

    # • process in mini‑batches to fit GPU memory
    for start in range(0, T, batch_frames):
        end   = min(start + batch_frames, T)
        batch = video[start:end].to(device=ldm_ae.device, dtype=torch.float32)  # (B,H,W,3)
        batch = (batch / 255.0).permute(0, 3, 1, 2)                            # → (B,3,H,W) in [0,1]
        batch = utils_img.normalize_vqgan(batch)                               # → [‑1,1]

        # VQGAN → latent → *finetuned* decoder (= embeds the key)
        z      = ldm_ae.encode(batch).mode()
        w_img  = ldm_decoder.decode(z)                                         # still [‑1,1]
        w_img  = utils_img.unnormalize_vqgan(w_img).clamp(0, 1)                # back to [0,1]
        w_img  = (w_img * 255.0 + 0.5).to(torch.uint8).permute(0, 2, 3, 1)     # (B,H,W,3) uint8 CPU
        out_video[start:end].copy_(w_img.cpu())

    # • make sure destination dir exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # • write (with audio if present)
    kwargs = dict(video_codec="libx264",
                  options={"crf": "17", "preset": "slow"})
    if audio.numel():      # preserve audio track if it exists
        torchvision.io.write_video(str(dst_path), out_video, fps,
                                   audio_array=audio, audio_codec="aac",
                                   **kwargs)
    else:
        torchvision.io.write_video(str(dst_path), out_video, fps, **kwargs)

@torch.no_grad()
def wm_videos_and_save_full(ldm_ae,
                            ldm_decoder,
                            params,
                            splits=("short", "medium", "long"),
                            batch_frames: int = 16):
    """
    Watermark every video under params.videos_dir and mirror the structure
    to params.wm_videos_dir.
    """
    src_root = Path(params.videos_dir)
    dst_root = Path(params.wm_videos_dir)

    # (re‑)create destination root, clear existing files
    if dst_root.exists():
        for p in dst_root.rglob("*"):
            if p.is_file():
                p.unlink()
    dst_root.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"[{split}]")
        for vid_idx in range(1, 301):                       # filenames 1.mp4 … 300.mp4
            src = src_root / split / f"{vid_idx}.mp4"
            dst = dst_root / split / f"{vid_idx}.mp4"
            if not src.exists():
                print(f"  ! missing {src}, skipped")
                continue

            _watermark_one_video(src, dst,
                                 ldm_ae=ldm_ae,
                                 ldm_decoder=ldm_decoder,
                                 batch_frames=batch_frames)
            print(f"  ✓ {src.name} → {dst}")

    print("✔ All watermarked videos saved to", dst_root)

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
    aa('--val_img_num', type=int, default=40504, help='The size of the image workload')
    aa("--num_streams", type=int, default=1, help="Number of pipeline CUDA streams for detection")
    aa("--num_rs_threads", type=int, default=4, help="Number of rs threads for detection")
    aa("--async_rs", type=utils.bool_inst, default=False, help="If True, use async reed-solomon correction for detection")
    aa("--videos_dir", type=str, default="/ssd2/videomme/", help="The path of the video dataset (Default: /ssd2/videomme/)")
    aa("--wm_videos_dir", type=str, default="/ssd2/wm_videomme/", help="The path of the watermarked video dataset (Default: /ssd2/wm_videomme/)")
    aa("--workload", type=str, default="images", help="The workload for the QRMark generation default:images (images/videos)")
    aa('--val_videos_num', type=int, default=300, help='The size of the videos workload')
    aa("--batch_frames", type=int, default=16, help="The number of frames in a batch  for the video watermarking generation default:16")
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
        
        
        print(f'>>> Loading data from {params.train_dir} and {params.val_dir}...')
        vqgan_transform = transforms.Compose([
            transforms.Resize(params.img_size),
            transforms.CenterCrop(params.img_size),
            transforms.ToTensor(),
            utils_img.normalize_vqgan,
        ])
        # Training loop
        # Loads the data
        train_loader = utils.get_dataloader(params.train_dir, vqgan_transform, params.batch_size, num_imgs=params.batch_size*params.steps, shuffle=True, num_workers=4, collate_fn=None)
        val_loader = utils.get_dataloader(params.val_dir, vqgan_transform, params.batch_size*4, num_imgs=1000, shuffle=False, num_workers=4, collate_fn=None)
        vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])
            
        print(f'>>> Training...')
        train_stats = train(train_loader, optimizer, loss_w, loss_i, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, params)
        val_stats = val(val_loader, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, msg_key, params)
        if params.workload=="images":
            images_loader = utils.get_dataloader(params.val_dir, vqgan_transform, batch_size=params.val_batch_size, num_imgs=params.val_img_num, shuffle=False, num_workers=4,  collate_fn=None)
            detect_save_stats = wm_imgs_and_save(images_loader, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, msg_key, params)
        elif params.workload=="videos":
            wm_videos_and_save_full(ldm_ae, ldm_decoder, params, batch_frames=params.batch_frames)
        
        if params.workload=="images":
            log_stats = {
                'key': key_str,
                'msg_key': msg_key_str,
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v   for k, v in val_stats.items()},
                **{f"detect_save_image_{k}": v for k, v in detect_save_stats.items()},
            }
        elif params.workload=="videos":
            log_stats = {
                'key': key_str,
                'msg_key': msg_key_str,
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v   for k, v in val_stats.items()}, 
                }
            
        save_dict = {
            'ldm_decoder': ldm_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        }

def train(data_loader: Iterable,
          optimizer: torch.optim.Optimizer,
          loss_w: Callable, loss_i: Callable,
          ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL,
          msg_decoder: nn.Module, vqgan_to_imnet: nn.Module,
          key: torch.Tensor, params: argparse.Namespace):

    header = 'Train'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.train()
    base_lr = optimizer.param_groups[0]["lr"]

    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device, non_blocking=True)
        keys = key.repeat(imgs.size(0), 1)

        utils.adjust_learning_rate(optimizer, ii, params.steps,
                                   params.warmup_steps, base_lr)

        start_evt, end_evt = torch.cuda.Event(True), torch.cuda.Event(True)
        start_evt.record()

        # ▸ 1. Encoder
        imgs_z = ldm_ae.encode(imgs).mode()

        # ▸ 2. diffusion decoder
        imgs_d0 = ldm_ae.decode(imgs_z)

        # ▸ 3. Finetuned decoder
        imgs_w = ldm_decoder.decode(imgs_z)

        # ▸ 4. transform and extract
        imgs_w_trans = vqgan_to_imnet(imgs_w)

        decoded = _decode_with_tile(imgs_w_trans, msg_decoder, params)

        # ▸ 5. loss and back propaganda

        lossw = loss_w(decoded, keys)
        lossi = loss_i(imgs_w, imgs_d0)
        loss = params.lambda_w * lossw + params.lambda_i * lossi


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        end_evt.record()

        # -------- log -------- #
        diff = ~torch.logical_xor(decoded > 0, keys > 0)
        bit_accs = diff.sum(dim=-1) / diff.size(-1)
        word_accs = (bit_accs == 1)

        metric_logger.update(
            loss=loss.item(),
            loss_w=lossw.item(),
            loss_i=lossi.item(),
            psnr=utils_img.psnr(imgs_w, imgs_d0).mean().item(),
            bit_acc_avg=bit_accs.mean().item(),
            word_acc_avg=word_accs.float().mean().item(),
            lr=optimizer.param_groups[0]["lr"],
        )

        if ii % params.log_freq == 0:
            print(json.dumps({
                "iter": ii,
                "loss": loss.item(),
                "loss_w": lossw.item(),
                "loss_i": lossi.item(),
                "bit_acc": bit_accs.mean().item(),
                "word_acc": word_accs.float().mean().item()
            }))

    return {k: m.global_avg for k, m in metric_logger.meters.items()}


@torch.no_grad()
def val(data_loader: Iterable,
        ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL,
        msg_decoder: nn.Module, vqgan_to_imnet: nn.Module,
        key: torch.Tensor, msg_key: torch.Tensor,
        params: argparse.Namespace):

    header = 'Eval'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.eval()

    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device, non_blocking=True)
        keys = key.repeat(imgs.size(0), 1)
        msg_keys = msg_key.repeat(imgs.size(0), 1)

        # --- encode / decode ---
        imgs_z = ldm_ae.encode(imgs).mode()
        imgs_d0 = ldm_ae.decode(imgs_z)
        imgs_w  = ldm_decoder.decode(imgs_z)

        # --- detection & robutstness under various attacks ---
        imgs_aug = vqgan_to_imnet(imgs_w)
        decoded = _decode_with_tile(imgs_aug, msg_decoder, params)

        if params.reed_solomon:
            decoded = decode_reed_solomon(decoded,
                                          params.num_bits,
                                          params.num_parity_symbols,
                                          params.m_bits_per_symbol)

        diff = ~torch.logical_xor(decoded > 0, msg_keys > 0)
        bit_accs = diff.sum(dim=-1) / diff.size(-1)
        word_accs = (bit_accs == 1)

        metric_logger.update(
            psnr=utils_img.psnr(imgs_w, imgs_d0).mean().item(),
            bit_acc_none=bit_accs.mean().item(),
            word_acc_none=word_accs.float().mean().item()
        )

        # Add FPR and TPR calculations here for 'none' attack
        decoded_binary = (decoded > 0).float()  # Convert decoded to binary (0 or 1)
        keys_binary = msg_keys  # Already binary (0 or 1)
        TP = torch.sum((decoded_binary == 1) & (keys_binary == 1), dim=-1)
        FP = torch.sum((decoded_binary == 1) & (keys_binary == 0), dim=-1)
        TN = torch.sum((decoded_binary == 0) & (keys_binary == 0), dim=-1)
        FN = torch.sum((decoded_binary == 0) & (keys_binary == 1), dim=-1)
        TPR = TP / (TP + FN + 1e-8)  # True Positive Rate
        FPR = FP / (FP + TN + 1e-8)  # False Positive Rate
        # Original evaluation loop remains unchanged
        log_stats = {
            "iteration": ii,
            "psnr": utils_img.psnr(imgs_w, imgs_d0).mean().item(),
            "tpr_none": torch.mean(TPR).item(),  # Add TPR for 'none'
            "fpr_none": torch.mean(FPR).item(),  # Add FPR for 'none'
            # "psnr_ori": utils_img.psnr(imgs_w, imgs).mean().item(),
        }
        
        attacks = {
            'none': lambda x: x,
            #    'crop_05': lambda x: utils_img.center_crop(x, 0.5),
            #    'crop_01': lambda x: utils_img.center_crop(x, 0.1),
             #   'resize_07': lambda x: utils_img.resize(x, 0.7),
                # 'resize_05': lambda x: utils_img.resize(x, 0.5),
             #   'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
            #    'blur': lambda x: utils_img.gaussian_blur(x, 1),
             #   'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
             #   'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
             #   'contrast_1p5': lambda x: utils_img.adjust_contrast(x, 1.5),
             #   'contrast_2': lambda x: utils_img.adjust_contrast(x, 2),
             #   'saturation_1p5': lambda x: utils_img.adjust_saturation(x, 1.5),
             #   'saturation_2': lambda x: utils_img.adjust_saturation(x, 2),
             #   'sharpness_1p5': lambda x: utils_img.adjust_sharpness(x, 1.5),
             #   'sharpness_2': lambda x: utils_img.adjust_sharpness(x, 2),
             #   'overlay_text': lambda x: utils_img.overlay_text(x, [76,111,114,101,109,32,73,112,115,117,109]),
        }
        for name, attack in attacks.items():
            imgs_aug = attack(vqgan_to_imnet(imgs_w))
            decoded = _decode_with_tile(imgs_aug, msg_decoder, params)
            if params.reed_solomon:
                decoded = decode_reed_solomon(decoded, params.num_bits, params.num_parity_symbols, params.m_bits_per_symbol)
            diff = (~torch.logical_xor(decoded>0, msg_keys>0)) # b k -> b k
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            word_accs = (bit_accs == 1) # b
            log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
            log_stats[f'word_acc_{name}'] = torch.mean(word_accs.type(torch.float)).item()
                    
        bit_acc_keys = [k for k in log_stats.keys() if k.startswith("bit_acc_")]
        log_stats["distortion_avg"] = float(
            sum(log_stats[k] for k in bit_acc_keys) / len(bit_acc_keys)
        )
        
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})
        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs), 0, 1), os.path.join(params.imgs_dir, f'{ii:03}_val_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0), 0, 1), os.path.join(params.imgs_dir, f'{ii:03}_val_d0.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w), 0, 1), os.path.join(params.imgs_dir, f'{ii:03}_val_w.png'), nrow=8)
    
    print("Averaged {} stats:".format('eval'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------- Detect and Save ---------------- #
@torch.no_grad()
def wm_imgs_and_save(data_loader: Iterable, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, msg_key, params):
    print('Now generate watermarked images and save.')
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.eval()
    os.makedirs(params.wm_dir, exist_ok=True)
    if os.path.exists(params.wm_dir) and os.listdir(params.wm_dir):
        print(f"Directory {params.wm_dir} is not empty. Clearing existing files...")
        for filename in os.listdir(params.wm_dir):
            file_path = os.path.join(params.wm_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                
    total_compute_ms = 0.0
    total_real_s = 0.0
    global_idx = 0

    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, 'DetectSave')):
        imgs = imgs.to(device, non_blocking=True)
        keys = key.repeat(imgs.size(0), 1)
        msg_keys = msg_key.repeat(imgs.size(0), 1)

        start_t = time.perf_counter()
        start_evt, end_evt = torch.cuda.Event(True), torch.cuda.Event(True)
        start_evt.record()

        # ----- Encode -----
        imgs_z = ldm_ae.encode(imgs).mode()
        # ----- Decode -----
        imgs_w = ldm_decoder.decode(imgs_z)

        # ----- VQGAN → IMNET -----
        imgs_aug = vqgan_to_imnet(imgs_w)
        # ----- Message Decode -----
        decoded = _decode_with_tile(imgs_aug, msg_decoder, params)
        # ----- Reed-Solomon Decode (if enabled) -----
        if params.reed_solomon:
            decoded = decode_reed_solomon(
                decoded,
                params.num_bits,
                params.num_parity_symbols,
                params.m_bits_per_symbol
            )

        end_evt.record()
        end_t = time.perf_counter()
        
        total_real_s += end_t - start_t

        # ----- Accuracy Evaluation -----
        diff = ~torch.logical_xor(decoded > 0, msg_keys > 0)
        bit_accs = diff.sum(dim=-1) / diff.size(-1)
        word_accs = (bit_accs == 1)
        metric_logger.update(
            bit_acc=bit_accs.mean().item(),
            word_acc=word_accs.float().mean().item()
        )

        # ----- Save Images -----
        imgs_to_save = torch.clamp(utils_img.unnormalize_vqgan(imgs_w), 0, 1)
        for b in range(imgs_to_save.size(0)):
            path = os.path.join(params.wm_dir, f"{global_idx + b:06d}.png")
            save_image(imgs_to_save[b], path)
        global_idx += imgs_to_save.size(0)

    total_detect_s = total_compute_ms / 1000.0
    print(f"Optimized:[wm_imgs_and_save]  Wall time = {total_real_s:.6f}s")

    metric_logger.update(detect_time_s=total_detect_s)

    return {k: m.global_avg for k, m in metric_logger.meters.items()}


# ---------------- Detect and Save ---------------- #
@torch.no_grad()
def wm_video_and_save(data_loader: Iterable, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, msg_key, params):
    print('Now generate watermarked images and save.')
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.eval()
    os.makedirs(params.wm_videos_dir, exist_ok=True)
    if os.path.exists(params.wm_videos_dir) and os.listdir(params.wm_videos_dir):
        print(f"Directory {params.wm_videos_dir} is not empty. Clearing existing files...")
        for filename in os.listdir(params.wm_videos_dir):
            file_path = os.path.join(params.wm_videos_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                
    total_compute_ms = 0.0
    total_real_s = 0.0
    global_idx = 0
    
    print(f"Now generating watermarked videos")
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, 'DetectSave')):
        imgs = imgs.to(device, non_blocking=True)
        keys = key.repeat(imgs.size(0), 1)
        msg_keys = msg_key.repeat(imgs.size(0), 1)

        start_t = time.perf_counter()
        start_evt, end_evt = torch.cuda.Event(True), torch.cuda.Event(True)
        start_evt.record()

        # ----- Encode -----
        imgs_z = ldm_ae.encode(imgs).mode()
        # ----- Decode -----
        imgs_w = ldm_decoder.decode(imgs_z)

        # ----- VQGAN → IMNET -----
        imgs_aug = vqgan_to_imnet(imgs_w)
        # ----- Message Decode -----
        decoded = _decode_with_tile(imgs_aug, msg_decoder, params)
        # ----- Reed-Solomon Decode (if enabled) -----
        if params.reed_solomon:
            decoded = decode_reed_solomon(
                decoded,
                params.num_bits,
                params.num_parity_symbols,
                params.m_bits_per_symbol
            )

        end_evt.record()
        end_t = time.perf_counter()
        
        total_real_s += end_t - start_t

        # ----- Accuracy Evaluation -----
        diff = ~torch.logical_xor(decoded > 0, msg_keys > 0)
        bit_accs = diff.sum(dim=-1) / diff.size(-1)
        word_accs = (bit_accs == 1)
        metric_logger.update(
            bit_acc=bit_accs.mean().item(),
            word_acc=word_accs.float().mean().item()
        )

        # ----- Save Images -----
        imgs_to_save = torch.clamp(utils_img.unnormalize_vqgan(imgs_w), 0, 1)
        for b in range(imgs_to_save.size(0)):
            path = os.path.join(params.wm_videos_dir, f"{global_idx + b:06d}.png")
            save_image(imgs_to_save[b], path)
        global_idx += imgs_to_save.size(0)

    total_detect_s = total_compute_ms / 1000.0
    print(f"Optimized:[wm_imgs_and_save]  Wall time = {total_real_s:.6f}s")

    metric_logger.update(detect_time_s=total_detect_s)

    return {k: m.global_avg for k, m in metric_logger.meters.items()}

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)