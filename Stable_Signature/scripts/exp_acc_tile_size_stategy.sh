#!/usr/bin/env bash
# ------------------------------------------------------------
# Sweep QRMark_generate.py across multiple tile sizes
# Usage:
#   bash run_generate_tilesize_sweep.sh
# ------------------------------------------------------------
set -euo pipefail
set -x

# ----------------- Config -----------------
export CUDA_VISIBLE_DEVICES=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_PATH="./profile_results/acc"
LOG_PATH="${RESULT_PATH}/logs"
mkdir -p "${LOG_PATH}"

VAL_IMG_NUM=1000                      # images to generate
WM_DIR="dataset/watermark_tile_size_acc" # watermark images folder
TRAIN_DIR="dataset/COCO/train/"
VAL_DIR="dataset/COCO/val/"

BITS=48
VAL_BATCH_SIZE=16                     # fixed in your snippet
TILE_SIZES=(64)

# ----------------- Sweep -----------------
experiment_name="exp_gen_random"
MSG_DECODER_PATH="models/exp_tile_random.pth"
OUTPUT_DIR="${RESULT_PATH}/gen/${experiment_name}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
LOGFILE1="${LOG_PATH}/${experiment_name}-${TIMESTAMP}.log"
python QRMark_generate.py \
    --num_keys 1 \
    --num_bits "${BITS}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --train_dir "${TRAIN_DIR}" \
    --val_dir "${VAL_DIR}" \
    --msg_decoder_path "${MSG_DECODER_PATH}" \
    --reed_solomon False \
    --num_parity_symbols 2 \
    --m_bits_per_symbol 8 \
    --tile random \
    --tile_size 64 \
    --transform_fuse False \
    --val_img_num "${VAL_IMG_NUM}" \
    --output_dir "${OUTPUT_DIR}" \
    --workload images \
    --wm_dir "${WM_DIR}" \
      > "${LOGFILE1}" 2>&1

experiment_name="exp_gen_random_grid"
MSG_DECODER_PATH="models/random_grid_tile_64_48.pth"
OUTPUT_DIR="${RESULT_PATH}/gen/${experiment_name}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
LOGFILE2="${LOG_PATH}/${experiment_name}-${TIMESTAMP}.log"
python QRMark_generate.py \
    --num_keys 1 \
    --num_bits "${BITS}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --train_dir "${TRAIN_DIR}" \
    --val_dir "${VAL_DIR}" \
    --msg_decoder_path "${MSG_DECODER_PATH}" \
    --reed_solomon False \
    --num_parity_symbols 2 \
    --m_bits_per_symbol 8 \
    --tile random_grid \
    --tile_size 64 \
    --transform_fuse False \
    --val_img_num "${VAL_IMG_NUM}" \
    --output_dir "${OUTPUT_DIR}" \
    --workload images \
    --wm_dir "${WM_DIR}" \
      > "${LOGFILE2}" 2>&1

experiment_name="exp_gen_fixed"
MSG_DECODER_PATH="models/exp_tile_grid.pth"
OUTPUT_DIR="${RESULT_PATH}/gen/${experiment_name}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
LOGFILE3="${LOG_PATH}/${experiment_name}-${TIMESTAMP}.log"
python QRMark_generate.py \
    --num_keys 1 \
    --num_bits "${BITS}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --train_dir "${TRAIN_DIR}" \
    --val_dir "${VAL_DIR}" \
    --msg_decoder_path "${MSG_DECODER_PATH}" \
    --reed_solomon False \
    --num_parity_symbols 2 \
    --m_bits_per_symbol 8 \
    --tile fixed \
    --tile_size 64 \
    --transform_fuse False \
    --val_img_num "${VAL_IMG_NUM}" \
    --output_dir "${OUTPUT_DIR}" \
    --workload images \
    --wm_dir "${WM_DIR}" \
      > "${LOGFILE3}" 2>&1

echo "All generation runs finished. Logs are in: ${LOG_PATH}"
