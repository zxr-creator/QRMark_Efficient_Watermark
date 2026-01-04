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
from torchvision.transforms.functional import pil_to_tensor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import triton.language as tl
import triton
import torchvision                                                        # add
import torchvision.transforms.functional as TF

import utils
import utils_img
import utils_model
import glob
from optimized_transform import VQGANTransformFuse

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
    aa("--num_rs_threads", type=int, default=4, help="Number of rs threads for detection")
    aa("--async_rs", type=utils.bool_inst, default=False, help="If True, use async reed-solomon correction for detection")
    aa("--videos_dir", type=str, default="/ssd2/videomme/", help="The path of the video dataset (Default: /ssd2/videomme/)")
    aa("--wm_videos_dir", type=str, default="/ssd2/wm_videomme/", help="The path of the watermarked video dataset (Default: /ssd2/wm_videomme/)")
    aa('--val_videos_num', type=int, default=300, help='The size of the videos workload')
    aa("--workload", type=str, default="images", help="The workload for the QRMark detection (Default: images (images/videos))")
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
        vqgan_transform = transforms.ToTensor()
        vqgan_to_imnet = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        utils_img.normalize_img,
        ])
        
        if params.transform_fuse == True:
            vqgan_to_imnet = VQGANTransformFuse(img_size=params.img_size, device=device)
            
        if params.workload =="images":
            nvtx.range_push("E2E_Image_Watermark_Detection")
            print(f'>>> Detecting from saved images...')
            nvtx.range_push("Image Dataset Loading")
            detect_loader = utils.get_dataloader(params.wm_dir, vqgan_transform, batch_size=params.val_batch_size, num_imgs=params.val_img_num, shuffle=False, num_workers=4,  collate_fn=None)
            nvtx.range_pop()
            nvtx.range_push("Core_Pipeline")
            start_e2e = time.perf_counter()
                         
            detect_from_images_stats = detect_from_images(detect_loader, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, msg_key, params)

            torch.cuda.synchronize()
            end_e2e = time.perf_counter() 
            print(f"[E2E] Total wall time = {end_e2e - start_e2e:.4f} s")
            nvtx.range_pop()
            nvtx.range_push("Statistics_and_Logging")
            log_stats = {
                'key': key_str,
                'msg_key': msg_key_str,
                **{f"detect_from_images_{k}": v for k, v in detect_from_images_stats.items()},
            }
            nvtx.range_pop()
            
            nvtx.range_pop()
            print(f"[E2E] Total wall time = {end_e2e - start_e2e:.4f} s")
            
        '''    
        # Detection from saved videos 
        if params.workload =="video":
            print(">>> Detecting from saved videos ...")
            video_ds = VideoFramesDataset(params.wm_videos_dir, transform=None, num_videos=params.val_videos_num)
            detect_loader = DataLoader(video_ds, batch_size=params.val_batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, collate_fn=flatten_video_batch)
            nvtx.range_push("Video Watermark Detection")
            _ = vqgan_transform(torch.zeros(1, 3, 256, 256, device="cuda"), params.img_size)  # warms up fused kernel
            detect_from_video_stats = detect_from_videos(detect_loader, ldm_ae, ldm_decoder, msg_decoder, vqgan_transform, key, msg_key, params)
            nvtx.range_pop()
            log_stats = {
                'key': key_str,
                'msg_key': msg_key_str,
                **{f"detect_from_video_{k}": v for k, v in detect_from_video_stats.items()},
            }
        '''
        save_dict = {
            'ldm_decoder': ldm_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        }
        


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

@torch.no_grad()
def detect_from_images(
        data_loader: Iterable,
        ldm_ae, ldm_decoder, msg_decoder,
        vqgan_transform, key, msg_key, params):
    """
    Rolled-back detector:
      - No CUDA multi-streams
      - No H2D prefetch
      - Optional CPU RS thread pool (controlled by --async_rs)
    """
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.eval()
    device = torch.device('cuda')

    # ---------- RS Decoder (optional thread pool) ----------
    async_rs = getattr(params, "async_rs", False)
    num_rs_threads = getattr(params, "num_rs_threads", 4)
    rs_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_rs_threads) if async_rs else None
    pending_futures = []
    pending_keys = []

    print(f"[Detect] single-stream (no prefetch)  "
          f"+ RS threadpool={async_rs} (T={num_rs_threads})")
    
    latency_samples = []  
    start_t = time.perf_counter()
    step = 0

    # we pipeline *only* the RS work across batches (CPU overlap with GPU)
    last_decoded_cpu = None
    last_msg_keys_cpu = None

    for imgs in data_loader:
        t_batch_start = time.perf_counter()
        # H2D copy on the default stream
        imgs = imgs.to(device, non_blocking=True)

        # build per-batch keys
        keys = key.repeat(imgs.size(0), 1)
        msg_keys = msg_key.repeat(imgs.size(0), 1)

        # GPU-side work on the default stream
        nvtx.range_push("Vqgan_Transform")
        imgs_aug = vqgan_transform(imgs)
        nvtx.range_pop()

        nvtx.range_push("Tiling")
        patch = _decode_with_tile(imgs_aug, msg_decoder, params)
        nvtx.range_pop()

        nvtx.range_push("Message_Decode")
        decoded = msg_decoder(patch)
        nvtx.range_pop()

        # bring to CPU for RS
        decoded_cpu = decoded.detach().to('cpu')
        msg_keys_cpu = msg_keys.detach().to('cpu')

        # submit RS for the *previous* batch (to overlap with current GPU work)
        if last_decoded_cpu is not None:
            if getattr(params, "reed_solomon", False):
                if async_rs:
                    nvtx.range_push("RS_Correction")
                    fut = rs_pool.submit(
                        decode_reed_solomon,
                        last_decoded_cpu,
                        params.num_bits,
                        params.num_parity_symbols,
                        params.m_bits_per_symbol
                    )
                    nvtx.range_pop()
                    pending_futures.append(fut)
                    pending_keys.append(last_msg_keys_cpu)
                else:
                    nvtx.range_push("RS_Correction")
                    decoded_corrected = decode_reed_solomon(
                        last_decoded_cpu,
                        params.num_bits,
                        params.num_parity_symbols,
                        params.m_bits_per_symbol
                    )
                    nvtx.range_pop()
                    diff = ~torch.logical_xor(decoded_corrected > 0, last_msg_keys_cpu > 0)
                    bit_accs = diff.sum(dim=-1) / diff.size(-1)
                    word_accs = (bit_accs == 1)
                    metric_logger.update(
                        bit_acc=bit_accs.mean().item(),
                        word_acc=word_accs.float().mean().item()
                    )
            else:
                diff = ~torch.logical_xor(last_decoded_cpu > 0, last_msg_keys_cpu > 0)
                bit_accs = diff.sum(dim=-1) / diff.size(-1)
                word_accs = (bit_accs == 1)
                metric_logger.update(
                    bit_acc=bit_accs.mean().item(),
                    word_acc=word_accs.float().mean().item()
                )

        # advance the RS pipeline window
        last_decoded_cpu = decoded_cpu
        last_msg_keys_cpu = msg_keys_cpu
        step += 1
        t_batch_end = time.perf_counter()
        batch_lat = t_batch_end - t_batch_start
        latency_samples.append(batch_lat)

    # handle the final batch’s RS
    if last_decoded_cpu is not None:
        if getattr(params, "reed_solomon", False):
            if async_rs:
                nvtx.range_push("RS_Correction")
                fut = rs_pool.submit(
                    decode_reed_solomon,
                    last_decoded_cpu,
                    params.num_bits,
                    params.num_parity_symbols,
                    params.m_bits_per_symbol
                )
                nvtx.range_pop()
                pending_futures.append(fut)
                pending_keys.append(last_msg_keys_cpu)
            else:
                nvtx.range_push("RS_Correction")
                decoded_corrected = decode_reed_solomon(
                    last_decoded_cpu,
                    params.num_bits,
                    params.num_parity_symbols,
                    params.m_bits_per_symbol
                )
                nvtx.range_pop()
                diff = ~torch.logical_xor(decoded_corrected > 0, last_msg_keys_cpu > 0)
                bit_accs = diff.sum(dim=-1) / diff.size(-1)
                word_accs = (bit_accs == 1)
                metric_logger.update(
                    bit_acc=bit_accs.mean().item(),
                    word_acc=word_accs.float().mean().item()
                )
        else:
            diff = ~torch.logical_xor(last_decoded_cpu > 0, last_msg_keys_cpu > 0)
            bit_accs = diff.sum(dim=-1) / diff.size(-1)
            word_accs = (bit_accs == 1)
            metric_logger.update(
                bit_acc=bit_accs.mean().item(),
                word_acc=word_accs.float().mean().item()
            )
            

    # finalize any async RS
    if async_rs:
        for fut, key_cpu in zip(pending_futures, pending_keys):
            decoded_final = fut.result()
            diff = ~torch.logical_xor(decoded_final > 0, key_cpu > 0)
            bit_accs = diff.sum(dim=-1) / diff.size(-1)
            word_accs = (bit_accs == 1)
            metric_logger.update(
                bit_acc=bit_accs.mean().item(),
                word_acc=word_accs.float().mean().item()
            )
        rs_pool.shutdown(wait=True)
        
    metric_logger.update(
                bit_acc=bit_accs.mean().item(),
                word_acc=word_accs.float().mean().item()
            )   
    end_t = time.perf_counter()
    
    total_wall = end_t - start_t
    if len(latency_samples) > 0:
        try:
            import numpy as _np
            arr = _np.asarray(latency_samples, dtype=float)
            mean = float(arr.mean())
        except Exception:
            arr = sorted(latency_samples)
            n = len(arr)
            mean = sum(arr)/n

        total_imgs = params.val_img_num
        per_img_avg = (sum(latency_samples) / total_imgs) if total_imgs > 0 else float('nan')
        print(f"[Detect][latency] approx per-image latency ≈ {mean:.4f}s  (total_images={total_imgs})")
        
    print(f"[Detect] finished {step} batches "
          f"(rs_threads={num_rs_threads}, rs_threadpool={async_rs})")
    print(f"Original-like:[detect_from_images] wall time = {total_wall:.6f}s")
    bit_avg  = metric_logger.meters['bit_acc'].global_avg  if 'bit_acc'  in metric_logger.meters else float('nan')
    word_avg = metric_logger.meters['word_acc'].global_avg if 'word_acc' in metric_logger.meters else float('nan')
    print(f"[Detect] accuracy: bit_acc={bit_avg:.6f}, word_acc={word_avg:.6f}")

    return {k: m.global_avg for k, m in metric_logger.meters.items()}

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)