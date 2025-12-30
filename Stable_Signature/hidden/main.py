# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
torchrun --nproc_per_node=2 main.py \
    --local_rank 0 \
    --encoder vit --encoder_depth 12 --encoder_channels 384 --use_tanh True \
    --loss_margin 100 --scaling_w 0.5 \
    --batch_size 16 --eval_freq 10 \
    --attenuation jnd \
    --epochs 100 --optimizer "AdamW,lr=1e-4"
Args Inventory:
    --dist False \
    --encoder vit --encoder_depth 6 --encoder_channels 384 --use_tanh True \
    --batch_size 128 --batch_size_eval 128 --workers 8 \
    --attenuation jnd \
    --num_bits 64 --redundancy 16 \
    --encoder vit --encoder_depth 6 --encoder_channels 384 --use_tanh True \
    --encoder vit --encoder_depth 12 --encoder_channels 384 --use_tanh True \
    --loss_margin 100   --attenuation jnd --batch_size 16 --eval_freq 10 --local_rank 0 \
    --p_crop 0 --p_rot 0 --p_color_jitter 0 --p_blur 0 --p_jpeg 0 --p_res 0 \
"""

import argparse
import datetime
import json
import os
import time
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.transforms import functional
from torchvision.utils import save_image    

import data_augmentation
import utils
import utils_img
import models
import attenuations
from utils import AugParams

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def rs_correction_loss(preds, targets, t, s):
    """
    Reed-Solomon correction-aware loss.
    Penalizes only when the number of bit errors exceeds the correction threshold t.
    
    Args:
        preds: Tensor of logits (before sign), shape (B, k)
        targets: Tensor of target bits in {-1, 1}, shape (B, k)
        t: int, correction threshold (max errors RS can fix)
        s: int, symbol size of the 
    
    Returns:
        Scalar tensor representing the RS-aware loss
    """
    with torch.no_grad():
        pred_bits = torch.sign(preds)  # Convert logits to {-1, 0, 1}
        pred_bits[pred_bits == 0] = 1  # Tie-break 0 to +1Add commentMore actions
        errors = (pred_bits != targets).float()
        E = errors.sum(dim=1)  # total bit errors per sample

    loss = torch.clamp(E - t, min=0.0) ** 2
    return loss.mean()

def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--train_dir", type=str, default="coco/train")
    aa("--val_dir", type=str, default="coco/val")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")

    group = parser.add_argument_group('Marking parameters')
    aa("--num_bits", type=int, default=48, help="Number of bits of the watermark (Default: 32)")
    aa("--redundancy", type=int, default=1, help="Redundancy of the watermark (Default: 1)")
    aa("--img_size", type=int, default=128, help="Image size")

    group = parser.add_argument_group('Encoder parameters')
    aa("--encoder", type=str, default="hidden", help="Encoder type (Default: hidden)")
    aa('--encoder_depth', default=4, type=int, help='Number of blocks in the encoder.')
    aa('--encoder_channels', default=64, type=int, help='Number of channels in the encoder.')
    aa("--use_tanh", type=utils.bool_inst, default=True, help="Use tanh scaling. (Default: True)")

    group = parser.add_argument_group('Decoder parameters')
    aa("--decoder", type=str, default="hidden", help="Decoder type (Default: hidden)")
    aa("--decoder_depth", type=int, default=8, help="Number of blocks in the decoder (Default: 4)")
    aa("--decoder_channels", type=int, default=64, help="Number of blocks in the decoder (Default: 4)")

    group = parser.add_argument_group('Training parameters')
    aa("--bn_momentum", type=float, default=0.01, help="Momentum of the batch normalization layer. (Default: 0.1)")
    aa('--eval_freq', default=1, type=int)
    aa('--saveckp_freq', default=10000, type=int)
    aa('--saveimg_freq', default=10, type=int)
    aa('--resume_from', default=None, type=str, help='Checkpoint path to resume from.')
    aa("--scaling_w", type=float, default=1.0, help="Scaling of the watermark signal. (Default: 1.0)")
    aa("--scaling_i", type=float, default=1.0, help="Scaling of the original image. (Default: 1.0)")

    group = parser.add_argument_group('Optimization parameters')
    aa("--epochs", type=int, default=400, help="Number of epochs for optimization. (Default: 100)")
    aa("--optimizer", type=str, default="Adam", help="Optimizer to use. (Default: Adam)")
    aa("--scheduler", type=str, default=None, help="Scheduler to use. (Default: None)")
    aa("--lambda_w", type=float, default=1.0, help="Weight of the watermark loss. (Default: 1.0)")
    aa("--lambda_i", type=float, default=0.0, help="Weight of the image loss. (Default: 0.0)")
    aa("--loss_margin", type=float, default=1, help="Margin of the Hinge loss or temperature of the sigmoid of the BCE loss. (Default: 1.0)")
    aa("--loss_i_type", type=str, default='mse', help="Loss type. 'mse' for mean squared error, 'l1' for l1 loss (Default: mse)")
    aa("--loss_w_type", type=str, default='bce', help="Loss type. 'bce' for binary cross entropy, 'cossim' for cosine similarity (Default: bce)")
    aa("--loss_lambda", type=float, default=0, help="Balances the bit errors beyond the RS correction capacity and the normal bit errors  (Default: 0)")
    aa("--rs_correction_t", type=int, default=2, help="Maximum correctable symbol of the RS correction (Default: 1)")
    aa("--rs_correction_s", type=int, default=8, help="symbol size of the RS correction (Default: 4)")

    group = parser.add_argument_group('Loader parameters')
    aa("--batch_size", type=int, default=16, help="Batch size. (Default: 16)")
    aa("--batch_size_eval", type=int, default=64, help="Batch size. (Default: 128)")
    aa("--workers", type=int, default=8, help="Number of workers for data loading. (Default: 8)")

    group = parser.add_argument_group('Attenuation parameters')
    aa("--attenuation", type=str, default=None, help="Attenuation type. (Default: jnd)")
    aa("--scale_channels", type=utils.bool_inst, default=True, help="Use channel scaling. (Default: True)")

    group = parser.add_argument_group('DA parameters')
    aa("--data_augmentation", type=str, default="combined", help="Type of data augmentation to use at marking time. (Default: combined)")
    aa("--p_crop", type=float, default=0.5, help="Probability of the crop augmentation. (Default: 0.5)")
    aa("--p_res", type=float, default=0.5, help="Probability of the crop augmentation. (Default: 0.5)")
    aa("--p_blur", type=float, default=0.5, help="Probability of the blur augmentation. (Default: 0.5)")
    aa("--p_jpeg", type=float, default=0.5, help="Probability of the diff JPEG augmentation. (Default: 0.5)")
    aa("--p_rot", type=float, default=0.5, help="Probability of the rotation augmentation. (Default: 0.5)")
    aa("--p_color_jitter", type=float, default=0.5, help="Probability of the color jitter augmentation. (Default: 0.5)")

    group = parser.add_argument_group('Distributed training parameters')
    aa('--debug_slurm', action='store_true')
    aa('--local_rank', default=-1, type=int)
    aa('--master_port', default=-1, type=int)
    aa('--dist', type=utils.bool_inst, default=True, help='Enabling distributed training')

    group = parser.add_argument_group('Misc')
    aa('--seed', default=0, type=int, help='Random seed')
    
    group = parser.add_argument_group('Tiling parameters')
    aa('--tile', type=str, default="none", help='Enable tiling and choose tiling type.  both/decode/none (Default: none)')
    aa('--tile_size', type=int, default=32, help='Tile size of the encoderdecoder. (Default: 32)')
    aa('--tile_type', type=str,default="random",  help='Tile type of the encoderdecoder. Options: random, random_grid, grid (Default: random)')
    

    return parser



def main(params):
    # Distributed mode
    if params.dist:
        local_rank = int(os.environ.get('LOCAL_RANK', params.local_rank))
        torch.cuda.set_device(local_rank)  # Explicitly set GPU device
        utils.init_distributed_mode(params)  # Initialize process group
        device = torch.device(f'cuda:{local_rank}')  # Use specific GPU
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # cudnn.benchmark = False
        # cudnn.deterministic = True

    # Set seeds for reproductibility 
    seed = params.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Print the arguments
    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(json.dumps(vars(params))))

    # handle params that are "none"
    if params.attenuation is not None:
        if params.attenuation.lower() == 'none':
            params.attenuation = None
    if params.scheduler is not None:
        if params.scheduler.lower() == 'none':
            params.scheduler = None

    # Build encoder
    print('building encoder...')
    if params.encoder == 'hidden':
        encoder = models.HiddenEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits, channels=params.encoder_channels, last_tanh=params.use_tanh)
    elif params.encoder == 'dvmark':
        encoder = models.DvmarkEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits, channels=params.encoder_channels, last_tanh=params.use_tanh)
    elif params.encoder == 'vit':
        encoder = models.VitEncoder(
            img_size=params.img_size, patch_size=16, init_values=None,
            embed_dim=params.encoder_channels, depth=params.encoder_depth, 
            num_bits=params.num_bits, last_tanh=params.use_tanh
            )
    else:
        raise ValueError('Unknown encoder type')
    print('\nencoder: \n%s'% encoder)
    print('total parameters: %d'%sum(p.numel() for p in encoder.parameters()))

    # Build decoder
    print('building decoder...')
    if params.decoder == 'hidden':
        decoder = models.HiddenDecoder(num_blocks=params.decoder_depth, num_bits=params.num_bits*params.redundancy, channels=params.decoder_channels)
    else:
        raise ValueError('Unknown decoder type')
    print('\ndecoder: \n%s'% decoder)
    print('total parameters: %d'%sum(p.numel() for p in decoder.parameters()))
    
    # Adapt bn momentum
    for module in [*decoder.modules(), *encoder.modules()]:
        if type(module) == torch.nn.BatchNorm2d:
            module.momentum = params.bn_momentum if params.bn_momentum != -1 else None

    # Construct attenuation
    if params.attenuation == 'jnd':
        attenuation = attenuations.JND(preprocess = lambda x: utils_img.unnormalize_rgb(x)).to(device)
    else:
        attenuation = None

    # Construct data augmentation seen at train time
    if params.data_augmentation == 'combined':
        data_aug = data_augmentation.HiddenAug(params.img_size, params.p_crop, params.p_blur,  params.p_jpeg, params.p_rot,  params.p_color_jitter, params.p_res).to(device)
        data_aug_tile = data_augmentation.HiddenAug(params.tile_size, params.p_crop, params.p_blur,  params.p_jpeg, params.p_rot,  params.p_color_jitter, params.p_res).to(device)
        #data_aug = data_augmentation.HiddenAug(params.img_size, params.tile_size, 0, 0, params.p_crop, params.p_blur,  params.p_jpeg, params.p_rot,  params.p_color_jitter, params.p_res).to(device)
    elif params.data_augmentation == 'kornia':
        data_aug = data_augmentation.KorniaAug().to(device)
    elif params.data_augmentation == 'none':
        data_aug, data_aug_tile = nn.Identity().to(device)
    else:
        raise ValueError('Unknown data augmentation type')
    print('data augmentation: %s'%data_aug)
    
    aug_params = AugParams(
        img_size=params.img_size,
        tile_size=params.tile_size,
        p_crop=params.p_crop,
        p_blur=params.p_blur,
        p_jpeg=params.p_jpeg,
        p_rot=params.p_rot,
        p_color_jitter=params.p_color_jitter,
        p_res=params.p_res
    )
    
    if params.tile != 'none':
        print('Set Tiling parameters')
        print(f'tile type: {params.tile_type}')
        if params.tile_type != 'random' and params.tile_type != 'random_grid' and params.tile_type != 'grid':
            raise ValueError('Unknown tile type')
        encoder_decoder = models.EncoderDecoder(encoder, attenuation, data_aug, decoder, 
        params.scale_channels, params.scaling_i, params.scaling_w, params.num_bits, params.redundancy, data_aug_tile, params.tile, params.tile_size, params.tile_type)
    else:
        # Create encoder/decoder
        encoder_decoder = models.EncoderDecoder(encoder, attenuation, data_aug, decoder, 
        params.scale_channels, params.scaling_i, params.scaling_w, params.num_bits, params.redundancy)
        
    if params.dist:
        # Set the device for this rank
        local_rank = int(os.environ.get('LOCAL_RANK', params.local_rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    
        # Move the model to the correct GPU before DDP wrapping
        encoder_decoder = encoder_decoder.to(device)
        
        # Convert to SyncBatchNorm (after moving to device, if needed)
        encoder_decoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder_decoder)
    
        # Wrap with DDP
        encoder_decoder = nn.parallel.DistributedDataParallel(
            encoder_decoder,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    else:
        # For non-distributed mode, use the default device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder_decoder = encoder_decoder.to(device)


    # Build optimizer and scheduler
    optim_params = utils.parse_params(params.optimizer)
    lr_mult = params.batch_size * utils.get_world_size() / 512.0
    optim_params['lr'] = lr_mult * optim_params['lr'] if 'lr' in optim_params else lr_mult * 1e-3
    to_optim = [*encoder.parameters(), *decoder.parameters()]
    optimizer = utils.build_optimizer(model_params=to_optim, **optim_params)
    scheduler = utils.build_lr_scheduler(optimizer=optimizer, **utils.parse_params(params.scheduler)) if params.scheduler is not None else None
    print('optimizer: %s'%optimizer)
    print('scheduler: %s'%scheduler)

    # Data loaders
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(params.img_size),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        utils_img.normalize_rgb,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_rgb,
        ])
    train_loader = utils.get_dataloader(params.train_dir, transform=train_transform, batch_size=params.batch_size, num_workers=params.workers, shuffle=True)
    val_loader = utils.get_dataloader(params.val_dir, transform=val_transform,  batch_size=params.batch_size_eval, num_workers=params.workers, shuffle=False)

    # optionally resume training 
    if params.resume_from is not None: 
        utils.restart_from_checkpoint(
            params.resume_from,
            encoder_decoder=encoder_decoder
        )
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(params.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        encoder_decoder=encoder_decoder,
        optimizer=optimizer
    )
    start_epoch = to_restore["epoch"]
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']

    # create output dir
    os.makedirs(params.output_dir, exist_ok=True)

    print('training...')
    start_time = time.time()
    best_bit_acc = 0
    for epoch in range(start_epoch + 1, params.epochs + 1):
        
        if params.dist:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(encoder_decoder, train_loader, optimizer, scheduler, epoch, params)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        
        if epoch % params.eval_freq == 0:
            val_stats = eval_one_epoch(encoder_decoder, val_loader, epoch, params)
            log_stats = {**log_stats, **{f'val_{k}': v for k, v in val_stats.items()}}
    
        save_dict = {
            'encoder_decoder': encoder_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'params': params,
        }
        utils.save_on_master(save_dict, os.path.join(params.output_dir, 'checkpoint.pth'))
        if params.saveckp_freq and epoch % params.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(params.output_dir, f'checkpoint{epoch:03}.pth'))
        if utils.is_main_process():
            with (Path(params.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if params.dist:
        torch.distributed.destroy_process_group()
        
def extract_ref_tiles(imgs, coords, tile_size):
    if coords is None:
        return imgs
    B, C, H, W = imgs.shape
    tiles = torch.empty(B, C, tile_size, tile_size,
                        device=imgs.device, dtype=imgs.dtype)
    for b in range(B):
        y0 = int(coords[b, 0])
        x0 = int(coords[b, 1])
        y0 = min(max(y0, 0), H - tile_size)
        x0 = min(max(x0, 0), W - tile_size)
        tiles[b] = imgs[b, :, y0:y0 + tile_size, x0:x0 + tile_size]
    return tiles        

def fused_loss(logits, targets, rs_correction_t, rs_correction_s):
    # logits, targets: [B, K]
    B, K = logits.shape
    P = 1 << ((K - 1).bit_length())  # next power of 2 ≥ K
    out = torch.empty(B, device=logits.device, dtype=logits.dtype)
    grid = lambda meta: (B,)
    
    # Note: the Triton kernel must now accept 5 constexprs: K, P, T, S
    triton_rs_bce.rs_bce_kernel[grid](
        logits, targets, out,
        K, P,                     # <- pass both explicitly
        rs_correction_t,
        rs_correction_s
    )
    return out.mean()

def message_loss(fts, targets, m, loss_type='mse', loss_lambda=0, rs_correction_t=0, rs_correction_s=0):
    """
    Compute the message loss
    Args:
        dot products (b k*r): the dot products between the carriers and the feature
        targets (KxD): boolean message vectors or gaussian vectors
        m: margin of the Hinge loss or temperature of the sigmoid of the BCE loss
    """
    
    if loss_type == 'bce':
        logits = fts / m
        targets_rescaled = 0.5 * (targets + 1)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets_rescaled, reduction='mean')

        if loss_lambda != 0 and rs_correction_t > 0:
            rs_loss = rs_correction_loss(logits, targets, rs_correction_t, rs_correction_s)
            return bce_loss + loss_lambda * rs_loss

        return bce_loss

    elif loss_type == 'cossim':
        return -torch.mean(torch.cosine_similarity(fts, targets, dim=-1))

    elif loss_type == 'mse':
        return F.mse_loss(fts, targets, reduction='mean')

    else:
        raise ValueError('Unknown loss type')
    
def image_loss(imgs, imgs_ori, loss_type='mse'):
    """
    Compute the image loss
    Args:
        imgs (BxCxHxW): the reconstructed images
        imgs_ori (BxCxHxW): the original images
        loss_type: the type of loss
    """
    if loss_type == 'mse':
        return F.mse_loss(imgs, imgs_ori, reduction='mean')
    if loss_type == 'l1':
        return F.l1_loss(imgs, imgs_ori, reduction='mean')
    else:
        raise ValueError('Unknown loss type')

def train_one_epoch(encoder_decoder: models.EncoderDecoder, loader, optimizer, scheduler, epoch, params):
    """
    One epoch of training.
    """
    if params.scheduler is not None:
        scheduler.step(epoch)
    encoder_decoder.train()
    header = 'Train - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")

    for it, (imgs, _) in enumerate(metric_logger.log_every(loader, 10, header)):
        imgs = imgs.to(device, non_blocking=True) # b c h w

        msgs_ori = torch.rand((imgs.shape[0],params.num_bits)) > 0.5 # b k
        msgs = 2 * msgs_ori.type(torch.float).to(device) - 1 # b k

        fts, (imgs_w, imgs_aug, coords) = encoder_decoder(imgs, msgs)
        if params.tile =='both' :
            imgs_ref = extract_ref_tiles(imgs, coords, params.tile_size)
        else:
            imgs_ref = imgs

        loss_w = message_loss(fts, msgs, m=params.loss_margin, loss_type=params.loss_w_type, loss_lambda=params.loss_lambda, rs_correction_t=params.rs_correction_t, rs_correction_s=params.rs_correction_s) # b k -> 1
        loss_i = image_loss(imgs_w, imgs_ref, loss_type=params.loss_i_type) # b c h w -> 1
        
        #print(f"Rank {utils.get_rank()}: Input device {imgs.device}, Model device {next(encoder_decoder.parameters()).device}")
        loss = params.lambda_w*loss_w + params.lambda_i*loss_i

        # gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # img stats
        psnrs = utils_img.psnr(imgs_w, imgs_ref) # b 1
        # msg stats
        ori_msgs = torch.sign(msgs) > 0
        decoded_msgs = torch.sign(fts) > 0 # b k -> b k
        diff = (~torch.logical_xor(ori_msgs, decoded_msgs)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        word_accs = (bit_accs == 1) # b
        norm = torch.norm(fts, dim=-1, keepdim=True) # b d -> b 1
        
        errors = (~torch.logical_xor(ori_msgs, decoded_msgs)).float()  # shape (B, k): 1 for correct, 0 for error
        E = (1 - errors).sum(dim=-1)  # number of bit‐errors per sample
        
        residual = torch.clamp(E - params.rs_correction_t, min=0.0)
        
        rs_bit_accs = (fts.shape[1] - residual) / fts.shape[1]
        rs_word_accs = (residual == 0).float()
        
        log_stats = {
            'loss_w': loss_w.item(),
            'loss_i': loss_i.item(),
            'loss': loss.item(),
            'psnr_avg': torch.mean(psnrs).item(),
            'lr': optimizer.param_groups[0]['lr'],
            'bit_acc_avg': torch.mean(bit_accs).item(),
            'word_acc_avg': torch.mean(word_accs.type(torch.float)).item(),
            'norm_avg': torch.mean(norm).item(),
            'rs_bit_acc_avg': rs_bit_accs.mean().item(),
            'rs_word_acc_avg': rs_word_accs.mean().item(),
        }
        
        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})
        
        # if epoch % 1 == 0 and it % 10 == 0 and utils.is_main_process():
        if epoch % params.saveimg_freq == 0 and it == 0 and utils.is_main_process():
            print(f"[Epoch {epoch} it {it}] imgs      shape: {imgs.shape}")      
            print(f"[Epoch {epoch} it {it}] imgs_w    shape: {imgs_w.shape}")
            print(f"[Epoch {epoch} it {it}] imgs_aug  shape: {imgs_aug.shape}")
            save_image(utils_img.unnormalize_img(imgs), os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_ori_shape{imgs.shape}.png'), nrow=8)
            save_image(utils_img.unnormalize_img(imgs_w), os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_w_shape{imgs_w.shape}.png'), nrow=8)
            save_image(utils_img.unnormalize_img(imgs_aug), os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_aug_shape{imgs_aug.shape}.png'), nrow=8)


    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('train'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_one_epoch(encoder_decoder: models.EncoderDecoder, loader, epoch, params):
    """
    One epoch of eval.
    """
    encoder_decoder.eval()
    header = 'Eval - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")
    for it, (imgs, _) in enumerate(metric_logger.log_every(loader, 10, header)):
        imgs = imgs.to(device, non_blocking=True) # b c h w

        msgs_ori = torch.rand((imgs.shape[0],params.num_bits)) > 0.5 # b k
        msgs = 2 * msgs_ori.type(torch.float).to(device) - 1 # b k
        
        
        fts, (imgs_w, imgs_aug, coords) = encoder_decoder(imgs, msgs, eval_mode='eval')
        
        if params.tile =='both' :
            imgs_ref = extract_ref_tiles(imgs, coords, params.tile_size)
        else:
            imgs_ref = imgs
        
        loss_w = message_loss(fts, msgs, m=params.loss_margin, loss_type=params.loss_w_type, loss_lambda=params.loss_lambda, rs_correction_t=params.rs_correction_t, rs_correction_s=params.rs_correction_s) # b -> 1
        loss_i = image_loss(imgs_w, imgs_ref, loss_type=params.loss_i_type) # b c h w -> 1
        
        loss = params.lambda_w*loss_w + params.lambda_i*loss_i

        # img stats
        psnrs = utils_img.psnr(imgs_w, imgs_ref)# b 1
        # msg stats
        ori_msgs = torch.sign(msgs) > 0
        decoded_msgs = torch.sign(fts) > 0 # b k -> b k
        diff = (~torch.logical_xor(ori_msgs, decoded_msgs)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        word_accs = (bit_accs == 1) # b
        norm = torch.norm(fts, dim=-1, keepdim=True) # b d -> b 1
        
        errors = (~torch.logical_xor(ori_msgs, decoded_msgs)).float()  # shape (B, k): 1 for correct, 0 for error
        E = (1 - errors).sum(dim=-1)  # number of bit‐errors per sample
        
        residual = torch.clamp(E - params.rs_correction_t, min=0.0)
        
        rs_bit_accs = (fts.shape[1] - residual) / fts.shape[1]
        rs_word_accs = (residual == 0).float()
        
        log_stats = {
            'loss_w': loss_w.item(),
            'loss_i': loss_i.item(),
            'loss': loss.item(),
            'psnr_avg': torch.mean(psnrs).item(),
            'bit_acc_avg': torch.mean(bit_accs).item(),
            'word_acc_avg': torch.mean(word_accs.type(torch.float)).item(),
            'norm_avg': torch.mean(norm).item(),
            'rs_bit_acc_avg': rs_bit_accs.mean().item(),
            'rs_word_acc_avg': rs_word_accs.mean().item(),
        }
        attacks = {
            'none': lambda x: x,
            'crop_01': lambda x: utils_img.center_crop(x, 0.1),
            'crop_05': lambda x: utils_img.center_crop(x, 0.5),
            # 'resize_03': lambda x: utils_img.resize(x, 0.3),
            'resize_05': lambda x: utils_img.resize(x, 0.5),
            'rot_25': lambda x: utils_img.rotate(x, 25),
            'rot_90': lambda x: utils_img.rotate(x, 90),
            'blur': lambda x: utils_img.gaussian_blur(x, sigma=2.0),
            'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
            }
        for name, attack in attacks.items():
            fts, (_) = encoder_decoder(imgs, msgs, eval_mode='attack', eval_aug=attack)
            decoded_msgs = torch.sign(fts) > 0 # b k -> b k
            diff = (~torch.logical_xor(ori_msgs, decoded_msgs)) # b k -> b k
            log_stats[f'bit_acc_{name}'] = diff.float().mean().item()

        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})
        
        if epoch % params.saveimg_freq == 0 and it == 0 and utils.is_main_process():
            save_image(utils_img.unnormalize_img(imgs), os.path.join(params.output_dir, f'{epoch:03}_{it:03}_val_ori.png'), nrow=8)
            save_image(utils_img.unnormalize_img(imgs_w), os.path.join(params.output_dir, f'{epoch:03}_{it:03}_val_w.png'), nrow=8)

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('eval'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)