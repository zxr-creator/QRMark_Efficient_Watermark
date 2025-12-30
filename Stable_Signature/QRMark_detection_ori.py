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
from torchvision import transforms
from torchvision.utils import save_image

import utils
import utils_img
import utils_model

sys.path.append('src')
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddpm import LatentDiffusion
from loss.loss_provider import LossProvider
import time
import glob
import csv

from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
    aa("--optimizer", type=str, default="AdamW,lr=5e-4", help="Optimizer and l earning rate for training")
    aa("--steps", type=int, default=100, help="Number of steps to train the model for")
    aa("--warmup_steps", type=int, default=20, help="Number of warmup steps for the optimizer")

    group = parser.add_argument_group('Logging and saving freq. parameters')
    aa("--log_freq", type=int, default=1000000, help="Logging frequency (in steps)")
    aa("--save_img_freq", type=int, default=1000000, help="Frequency of saving generated images (in steps)")

    group = parser.add_argument_group('Experiments parameters')
    aa("--num_keys", type=int, default=1, help="Number of fine-tuned checkpoints to generate")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--seed", type=int, default=0)
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")
    aa("--val_batch_size", type=int, default=16, help="Batch size for profiling")
    aa('--wm_dir', type=str, default='profile_results/watermarked_imgs_original', help='directory to store water-marked images')
    aa('--val_img_num', type=int, default=40000, help='The size of the image workload')
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

        # Creating key
        print(f'\n>>> Creating key with {nbit} bits...')
        key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=device)
        key_str = "".join([ str(int(ii)) for ii in key.tolist()[0]])
        print(f'Key: {key_str}')

        # Copy the LDM decoder and finetune the copy
        ldm_decoder = deepcopy(ldm_ae)
        ldm_decoder.encoder = nn.Identity()
        ldm_decoder.quant_conv = nn.Identity()
        ldm_decoder.to(device)
        for param in ldm_decoder.parameters():
            param.requires_grad = True
        optim_params = utils.parse_params(params.optimizer)
        optimizer = utils.build_optimizer(model_params=ldm_decoder.parameters(), **optim_params)
        
        vqgan_transform = transforms.ToTensor()
        vqgan_to_imnet = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        utils_img.normalize_img,
        ])

        # Detection loop
        print(f'>>> Detecting...')
        start_e2e = time.perf_counter()
        detect_loader = utils.get_dataloader(params.wm_dir, vqgan_transform, batch_size=params.val_batch_size, num_imgs=params.val_img_num, shuffle=False, num_workers=4,  collate_fn=None)
        nvtx.range_push("Image Watermark Detection")
        detect_saved_img_stats = detect_from_saved_images(detect_loader, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, params)
        nvtx.range_pop()
        end_e2e = time.perf_counter() 
        print(f"[E2E] end2end time = {end_e2e - start_e2e:.4f} s")
        '''
        log_stats = {
            'key': key_str,
            **{f"detect_savedimg_{k}": v for k, v in detect_saved_img_stats.items()},
        }
        save_dict = {
            'ldm_decoder': ldm_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        }
        '''
        # Save hecckpoint
        # torch.save(save_dict, os.path.join(params.output_dir, f"checkpoint_{ii_key:03d}.pth"))
        # with (Path(params.output_dir) / "log.txt").open("a") as f:
        #     f.write(json.dumps(log_stats) + "\n")
        #with (Path(params.output_dir) / "keys.txt").open("a") as f:
        #    f.write(os.path.join(params.output_dir, f"checkpoint_{ii_key:03d}.pth") + "\t" + key_str + "\n")
        #print('\n')

@torch.no_grad()
def detect_from_saved_images(
        data_loader: Iterable,
        ldm_ae, ldm_decoder, msg_decoder,
        vqgan_to_imnet, key, params):

    print('Now detecting from saved images.')
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.eval()

    latency_samples = []      # per big-batch (seconds)
    imgs_per_batch  = []      # images count per big-batch

    start_t = time.perf_counter()

    for ii, imgs in enumerate(data_loader):
        t_batch_start = time.perf_counter()

        imgs = imgs.to(device, non_blocking=True)
        keys = key.repeat(imgs.size(0), 1)  # GT bits on GPU

        nvtx.range_push("VQGAN_to_IMNET")
        imgs_aug = vqgan_to_imnet(imgs)
        nvtx.range_pop()

        nvtx.range_push("Message_Decode")
        decoded = msg_decoder(imgs_aug)
        nvtx.range_pop()

        # --- move to CPU once for accuracy computation ---
        decoded_cpu  = decoded.detach().to('cpu')          # [B, nbits] logits
        msg_keys_cpu = keys.detach().to('cpu').long()      # [B, nbits] {0,1}

        # --- use the shared helper for metrics ---
        _update_metrics(decoded_cpu, msg_keys_cpu, metric_logger)

        # per-batch latency end
        t_batch_end = time.perf_counter()
        batch_lat = t_batch_end - t_batch_start
        latency_samples.append(batch_lat)

    end_t = time.perf_counter()

    # ---- latency summary ----
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

    total_wall = end_t - start_t
    print(f"Original:[detect_from_images] wall time = {total_wall:.4f}s")

    # ---- accuracy print (global averages) AFTER wall time ----
    metric_logger.update(detect_time_s=total_wall)
    bit_avg  = metric_logger.meters['bit_acc'].global_avg  if 'bit_acc'  in metric_logger.meters else float('nan')
    word_avg = metric_logger.meters['word_acc'].global_avg if 'word_acc' in metric_logger.meters else float('nan')
    print(f"[Detect] accuracy: bit_acc={bit_avg:.6f}, word_acc={word_avg:.6f}")

    return {k: m.global_avg for k, m in metric_logger.meters.items()}
# ----------------------------------------------------------------------
def _update_metrics(decoded_cpu, key_cpu, metric_logger):
    """
    Update bit_acc and word_acc in the same way as detect_from_saved_images.
    """
    pred_bits = (decoded_cpu > 0).to(torch.int64)  # [B, nbits] {0,1}
    bit_acc_batch = (pred_bits == key_cpu).float().mean().item()
    word_acc_batch = (pred_bits.eq(key_cpu).all(dim=1).float().mean().item())
    metric_logger.update(bit_acc=bit_acc_batch, word_acc=word_acc_batch)
    
if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)