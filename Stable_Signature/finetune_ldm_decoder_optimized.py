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
import nvtx

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

import utils
import utils_img
import utils_model

sys.path.append('src')
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddpm import LatentDiffusion
from loss.loss_provider import LossProvider
import torch
from bit_reedsolo import ReedSolomonCodec_BW, UncorrectableError_RS
from typing import Tuple, Union

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Precompute the normalization parameters as tensors for efficient GPU computation
unnorm_mean = torch.tensor([-1.0, -1.0, -1.0], device='cuda').view(1, 3, 1, 1)
unnorm_std = torch.tensor([2.0, 2.0, 2.0], device='cuda').view(1, 3, 1, 1)

norm_mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
norm_std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

@torch.jit.script                       # JIT-compile once, reuse many times
def _vqgan2imnet_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Internal fused kernel:
        •   x is VQGAN-normalised in [-1,1].
        •   return ImageNet-normalised tensor.
    Everything is done in a single CUDA kernel when a GPU is available.
    """
    # constants live on the same device as x (no CPU↔GPU hops)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

    # un-normalise VQGAN   (-1…1) → (0…1)
    y = (x + 1.0) * 0.5
    # normalise to ImageNet statistics
    y = (y - mean) / std
    return y

def optimized_vqgan_to_imnet(
        x: torch.Tensor,
        chunk_size: int = 0,
        use_fp16: bool = False,
        device: torch.device | None = None
    ) -> torch.Tensor:
    """
    Fast drop-in replacement for vqgan_to_imnet.
    Args
    ----
    x          : BCHW tensor in VQGAN space (≈[-1,1]).
    chunk_size : optional - process large batches in smaller chunks to save RAM.
    use_fp16   : optional - cast to float16 for extra speed / lower bandwidth.
    device     : force a device (defaults to CUDA if available, else CPU).

    Returns
    -------
    ImageNet-normalised tensor with shape == x.shape.
    """

    # pick execution device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device, non_blocking=True)

    # optional reduced-precision path
    orig_dtype = x.dtype
    if use_fp16 and x.is_floating_point():
        x = x.half()

    # ▸▸  Vectorised in-place processing; optional chunking to cap memory  ◂◂
    if chunk_size and x.shape[0] > chunk_size:
        out_chunks = []
        for sub in x.split(chunk_size, dim=0):
            out_chunks.append(_vqgan2imnet_kernel(sub))
        y = torch.cat(out_chunks, dim=0)
    else:
        y = _vqgan2imnet_kernel(x)

    # keep interface identical to original (dtype preserved unless fp16 asked)
    return y.to(orig_dtype) if use_fp16 is False else y


def extract_tile(imgs: torch.Tensor, tile_size: int) -> torch.Tensor:
    """Extract the **top-left** tile from a batch of images.

    Args:
        imgs: (B, C, H, W) tensor.
        tile_size: Size of the square tile to return.

    Returns:
        (B, C, tile_size, tile_size) tensor containing the crop.
    """
    return imgs[:, :, :tile_size, :tile_size]

def encode_reed_solomon(original_tensor: torch.Tensor, nbits: int, num_parity_symbols: int, m_bits_per_symbol: int) -> torch.Tensor:
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
        
def decode_reed_solomon(encoded_tensor: torch.Tensor, nbits: int, num_parity_symbols: int, m_bits_per_symbol: int) -> torch.Tensor:
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
    aa("--use_random_msg_decoder", type=utils.bool_inst, default=False, help="If use random msg decoder for profiling")
    aa("--redundancy", type=int, default=1, help="Number of times the watermark is repeated to increase robustness")
    aa("--decoder_depth", type=int, default=8, help="Depth of the decoder in the watermarking model")
    aa("--decoder_channels", type=int, default=64, help="Number of channels in the decoder of the watermarking model")

    group = parser.add_argument_group('Training parameters')
    aa("--batch_size", type=int, default=4, help="Batch size for training")
    aa("--img_size", type=int, default=256, help="Resize images to this size")
    aa("--loss_i", type=str, default="watson-vgg", help="Type of loss for the image loss. Can be watson-vgg, mse, watson-dft, etc.")
    aa("--loss_w", type=str, default="bce", help="Type of loss for the watermark loss. Can be mse or bce")
    aa("--lambda_i", type=float, default=0.2, help="Weight of the image loss in the total loss")
    aa("--lambda_w", type=float, default=1.0, help="Weight of the watermark loss in the total loss")
    aa("--optimizer", type=str, default="AdamW,lr=5e-4", help="Optimizer and learning rate for training")
    aa("--steps", type=int, default=100, help="Number of steps to train the model for")
    aa("--warmup_steps", type=int, default=20, help="Number of warmup steps for the optimizer")
    aa("--reed_solomon", type=utils.bool_inst, default=False, help="If True, use Reed-Solomon codes for encoding the watermark")
    aa("--num_parity_symbols", type=int, default=1, help="Length of the correction bits for Reed-Solomon codes. The correction_length can only be: 1-16,17, 21, 24, 30, 32")
    aa("--m_bits_per_symbol", type=int, default=4, help="Length of the correction bits for Reed-Solomon codes. The correction_length can only be: 1-16,17, 21, 24, 30, 32")

    group = parser.add_argument_group('Logging and saving freq. parameters')
    aa("--log_freq", type=int, default=10, help="Logging frequency (in steps)")
    aa("--save_img_freq", type=int, default=1000, help="Frequency of saving generated images (in steps)")

    group = parser.add_argument_group('Experiments parameters')
    aa("--num_keys", type=int, default=1, help="Number of fine-tuned checkpoints to generate")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--seed", type=int, default=0)
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")
    
    group = parser.add_argument_group('Watermark patch parameters')
    aa("--crop_size", type=int, default=0, help="If > 0, randomly crop a patch of this size (e.g. 32) for watermark extraction")
    aa("--tile", type=utils.bool_inst, default=False,
       help="Use a fixed top-left tile for extraction")
    aa("--tile_size", type=int, default=32,
       help="Size of the square tile to use when --tile is enabled")
    
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
    if 'torchscript' in params.msg_decoder_path and params.use_random_msg_decoder == False: 
        msg_decoder = torch.jit.load(params.msg_decoder_path).to(device)
        # already whitened
        
    else:
        if params.use_random_msg_decoder == False:
            msg_decoder = utils_model.get_hidden_decoder(num_bits=params.num_bits, redundancy=params.redundancy, num_blocks=params.decoder_depth, channels=params.decoder_channels).to(device)
            ckpt = utils_model.get_hidden_decoder_ckpt(params.msg_decoder_path)
            print(msg_decoder.load_state_dict(ckpt, strict=False))
        else:
            msg_decoder = utils_model.get_hidden_decoder(num_bits=params.num_bits, redundancy=params.redundancy, num_blocks=params.decoder_depth, channels=params.decoder_channels).to(device)
            
        msg_decoder.eval()    
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
            params.msg_decoder_path = params.msg_decoder_path.replace(".pth", "_whit.pth")
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
        utils_img.normalize_vqgan,
    ])
    train_loader = utils.get_dataloader(params.train_dir, vqgan_transform, params.batch_size, num_imgs=params.batch_size*params.steps, shuffle=True, num_workers=4, collate_fn=None)
    val_loader = utils.get_dataloader(params.val_dir, vqgan_transform, params.batch_size*4, num_imgs=1000, shuffle=False, num_workers=4, collate_fn=None)
    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])
    
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
        
        # Training loop
        print(f'>>> Training...')
                
        train_stats = train(train_loader, optimizer, loss_w, loss_i, ldm_ae, ldm_decoder, msg_decoder, optimized_vqgan_to_imnet, key, params)
        val_stats = val(val_loader, ldm_ae, ldm_decoder, msg_decoder, optimized_vqgan_to_imnet, key, msg_key, params)
        log_stats = {'key': key_str,
                'msg_key':msg_key_str,
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()},
            }
        save_dict = {
            'ldm_decoder': ldm_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        }

        # Save checkpoint
        torch.save(save_dict, os.path.join(params.output_dir, f"checkpoint_{ii_key:03d}.pth"))
        with (Path(params.output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
        with (Path(params.output_dir) / "keys.txt").open("a") as f:
            f.write(os.path.join(params.output_dir, f"checkpoint_{ii_key:03d}.pth") + "\t" + key_str + "\n")
        print('\n')
        
def _decode_with_tile(imgs_aug: torch.Tensor, msg_decoder: nn.Module,
                          params: argparse.Namespace) -> torch.Tensor:
    """Apply the decoding strategy based on --tile."""
    if params.tile:
        patches = extract_tile(imgs_aug, params.tile_size)
        return msg_decoder(patches)
    return msg_decoder(imgs_aug)

def train(data_loader: Iterable, optimizer: torch.optim.Optimizer, loss_w: Callable, loss_i: Callable, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, msg_decoder: nn.Module, vqgan_to_imnet: nn.Module, key: torch.Tensor, params: argparse.Namespace):
    header = 'Train'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.train()
    base_lr = optimizer.param_groups[0]["lr"]
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device)
        keys = key.repeat(imgs.shape[0], 1)
        
        utils.adjust_learning_rate(optimizer, ii, params.steps, params.warmup_steps, base_lr)

        # Profile Encoder stage
        torch.cuda.nvtx.range_push("Train_Encoder")
        imgs_z = ldm_ae.encode(imgs)  # b c h w -> b z h/f w/f
        imgs_z = imgs_z.mode()
        torch.cuda.nvtx.range_pop()

        # Profile Decoder_Original stage
        torch.cuda.nvtx.range_push("Train_Decoder_Original")
        imgs_d0 = ldm_ae.decode(imgs_z)  # b z h/f w/f -> b c h w
        torch.cuda.nvtx.range_pop()

        # Profile Decoder_Finetuned stage
        torch.cuda.nvtx.range_push("Train_Decoder_Finetuned")
        imgs_w = ldm_decoder.decode(imgs_z)  # b z h/f w/f -> b c h w
        torch.cuda.nvtx.range_pop()

        # Profile ToWatermark and Extraction stages
        torch.cuda.nvtx.range_push("Transform")
        imgs_w_transformed = vqgan_to_imnet(imgs_w)
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push("Train_Extraction")
        decoded = _decode_with_tile(imgs_w_transformed, msg_decoder, params) # b c h w -> b k
        #if params.reed_solomon:
        #    decoded = decode_reed_solomon(decoded, params.num_bits, params.correction_length)
        #    print("After decoding :", decoded)      
        torch.cuda.nvtx.range_pop()

        # Compute loss and optimize (not profiled as per requirement)
        torch.cuda.nvtx.range_push("Train_Loss_and_Optimize")
        #print(f"decoded:{decoded}, keys:{keys}")
        lossw = loss_w(decoded, keys)
        lossi = loss_i(imgs_w, imgs_d0)
        loss = params.lambda_w * lossw + params.lambda_i * lossi
        torch.cuda.nvtx.range_pop()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log stats
        diff = (~torch.logical_xor(decoded > 0, keys > 0))  # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
        word_accs = (bit_accs == 1)  # b
        log_stats = {
            "iteration": ii,
            "loss": loss.item(),
            "loss_w": lossw.item(),
            "loss_i": lossi.item(),
            "psnr": utils_img.psnr(imgs_w, imgs_d0).mean().item(),
            # "psnr_ori": utils_img.psnr(imgs_w, imgs).mean().item(),
            "bit_acc_avg": torch.mean(bit_accs).item(),
            "word_acc_avg": torch.mean(word_accs.type(torch.float)).item(),
            "lr": optimizer.param_groups[0]["lr"],
        }
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})
        if ii % params.log_freq == 0:
            print(json.dumps(log_stats))
        
        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs), 0, 1), os.path.join(params.imgs_dir, f'{ii:03}_train_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0), 0, 1), os.path.join(params.imgs_dir, f'{ii:03}_train_d0.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w), 0, 1), os.path.join(params.imgs_dir, f'{ii:03}_train_w.png'), nrow=8)
    
    print("Averaged {} stats:".format('train'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def val(data_loader: Iterable, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, msg_decoder: nn.Module, vqgan_to_imnet: nn.Module, key: torch.Tensor, msg_key: torch.Tensor, params: argparse.Namespace):
    header = 'Eval'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.eval()
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        
        imgs = imgs.to(device)

        # Profile Encoder stage
        torch.cuda.nvtx.range_push("Val_Encoder")
        imgs_z = ldm_ae.encode(imgs)  # b c h w -> b z h/f w/f
        imgs_z = imgs_z.mode()
        torch.cuda.nvtx.range_pop()

        # Profile Decoder_Original stage
        torch.cuda.nvtx.range_push("Val_Decoder_Original")
        imgs_d0 = ldm_ae.decode(imgs_z)  # b z h/f w/f -> b c h w
        torch.cuda.nvtx.range_pop()

        # Profile Decoder_Finetuned stage
        torch.cuda.nvtx.range_push("Val_Decoder_Finetuned")
        imgs_w = ldm_decoder.decode(imgs_z)  # b z h/f w/f -> b c h w
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("Define_keys")
        # Define keys once per batch
        keys = key.repeat(imgs.shape[0], 1)
        msg_keys=msg_key.repeat(imgs.shape[0], 1)
        torch.cuda.nvtx.range_pop()
        # Profile ToWatermark, Extraction, and Detection for the 'none' attack before the loop
        torch.cuda.nvtx.range_push("Val_Detection_word")
        imgs_aug = vqgan_to_imnet(imgs_w)
        decoded = _decode_with_tile(imgs_aug, msg_decoder, params)
        
        if params.reed_solomon:
            decoded = decode_reed_solomon(decoded, params.num_bits, params.num_parity_symbols, params.m_bits_per_symbol)
        diff = (~torch.logical_xor(decoded > 0, msg_keys > 0))
        #print(f"The current diff:{diff}")
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]
        word_accs = (bit_accs == 1)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("Val_Detection_bit")
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
            'crop_01': lambda x: utils_img.center_crop(x, 0.1),
            'crop_05': lambda x: utils_img.center_crop(x, 0.5),
            'rot_25': lambda x: utils_img.rotate(x, 25),
            'rot_90': lambda x: utils_img.rotate(x, 90),
            'resize_03': lambda x: utils_img.resize(x, 0.3),
            'resize_07': lambda x: utils_img.resize(x, 0.7),
            'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
            'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            'jpeg_80': lambda x: utils_img.jpeg_compress(x, 80),
            'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
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
        
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})
        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs), 0, 1), os.path.join(params.imgs_dir, f'{ii:03}_val_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0), 0, 1), os.path.join(params.imgs_dir, f'{ii:03}_val_d0.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w), 0, 1), os.path.join(params.imgs_dir, f'{ii:03}_val_w.png'), nrow=8)
    
    print("Averaged {} stats:".format('eval'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)