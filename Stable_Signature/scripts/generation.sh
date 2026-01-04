#!/usr/bin/env bash
# ------------------------------------------------------------
# Ablation study: measure detection‑stage wall‑time when using
# different batch sizes.  Results are written to a CSV.
# Usage:  bash detect_ablation_streams.sh
# ------------------------------------------------------------
set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=5 # choose your GPU

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_PATH="./profile_results/paper"
LOG_PATH="${RESULT_PATH}/logs"
mkdir -p "${LOG_PATH}"
VAL_IMG_NUM=40504               # images to process in detection
WM_ORI_DIR="dataset/watermark_imgs_ori"
WM_DIR="dataset/watermark_imgs_tile"


# ---------- generate watermarked images ----------

experiment_name1="exp_gen_original"
LOGFILE_GEN1="$LOG_PATH/${experiment_name1}-${TIMESTAMP}.log"
OUTPUT_GEN_DIR="$RESULT_PATH/gen/${experiment_name1}_${TIMESTAMP}"
python QRMark_generate.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size 16 \
    --train_dir dataset/COCO/train/ \
    --val_dir dataset/COCO/val/ \
    --msg_decoder_path models/dec_48b.pth \
    --reed_solomon False \
    --num_parity_symbols 2 \
    --m_bits_per_symbol 8 \
    --tile False \
    --tile_size 64 \
    --transform_fuse False \
    --val_img_num $VAL_IMG_NUM \
    --output_dir "$OUTPUT_GEN_DIR" \
    --workload images \
    --wm_dir  "$WM_ORI_DIR" \
> "${LOGFILE_GEN1}" 2>&1

experiment_name2="exp_gen_tile"
LOGFILE_GEN2="$LOG_PATH/${experiment_name2}-${TIMESTAMP}.log"
OUTPUT_GEN_DIR="$RESULT_PATH/gen/${experiment_name2}_${TIMESTAMP}"
python QRMark_generate.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size 16 \
    --train_dir dataset/COCO/train/ \
    --img_size 256 \
    --val_dir dataset/COCO/val/ \
    --msg_decoder_path models/random_grid_64_48_new.pth \
    --reed_solomon True \
    --num_parity_symbols 2 \
    --m_bits_per_symbol 8 \
    --tile True \
    --tile_size 64 \
    --transform_fuse False \
    --val_img_num $VAL_IMG_NUM \
    --output_dir "$OUTPUT_GEN_DIR" \
    --workload images \
    --wm_dir  "dataset/watermark_imgs_tile_256" \
> "${LOGFILE_GEN2}" 2>&1