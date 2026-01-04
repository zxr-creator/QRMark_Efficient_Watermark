#!/bin/bash
# Using 4 GPUs to run the tile size experiments for Stable Signature
set -e

# Get a timestamp (e.g. 20250514_153022)
EXPIREMENT_NAME="exp_tile_size"
# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Tile sizes to iterate over
TILE_SIZES=(16 24 32 40 48 56 64)

for TILE_SIZE in "${TILE_SIZES[@]}"; do
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  MASTER_PORT=$((29500 + RANDOM % (50000 - 29500)))
  torchrun --nproc_per_node=4 --master_port=$MASTER_PORT main.py \
    --val_dir ../dataset/COCO/val \
    --train_dir ../dataset/COCO/train \
    --output_dir "output/${EXPIREMENT_NAME}_${TILE_SIZE}_${TIMESTAMP}" \
    --eval_freq 10 \
    --num_bits 48 --img_size 256 --batch_size 16 --epochs 400 \
    --scheduler CosineLRScheduler,lr_min=5e-7,t_initial=400,warmup_lr_init=1e-6,warmup_t=5 \
    --optimizer Lamb,lr=2e-2 \
    --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
    --scaling_w 0.3 --scale_channels False --attenuation none \
    --loss_w_type bce --loss_margin 1 \
    --tile both --tile_size $TILE_SIZE --tile_type random_grid
done
