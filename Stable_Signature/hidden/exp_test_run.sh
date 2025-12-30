#!/bin/bash
set -e

EXPERIMENT_NAME="new_random_grid"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TILE_SIZE=64  # or your preferred default
export CUDA_VISIBLE_DEVICES=0,1,2

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_PORT=$((29500 + RANDOM % (50000 - 29500)))
torchrun --nproc_per_node=3 --master_port=$MASTER_PORT main.py \
      --val_dir ../dataset/COCO/val \
      --train_dir ../dataset/COCO/train \
      --output_dir "output/${EXPERIMENT_NAME}_tile${TILE_SIZE}_shorttest_${TIMESTAMP}" \
      --eval_freq 10 \
      --num_bits 48 --img_size 256 --batch_size 16 --epochs 400 \
      --scheduler CosineLRScheduler,lr_min=5e-7,t_initial=400,warmup_lr_init=1e-6,warmup_t=5 \
      --optimizer Lamb,lr=2e-2 \
      --p_color_jitter 0.0 --p_blur 1.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
      --scaling_w 0.3 --scale_channels False --attenuation none \
      --loss_w_type bce --loss_margin 1 \
      --tile both --tile_size 64 --tile_type random_grid