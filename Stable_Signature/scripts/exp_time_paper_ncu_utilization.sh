#!/usr/bin/env bash
# ------------------------------------------------------------
# Ablation study: measure detection‑stage wall‑time when using
# different batch sizes.  Results are written to a CSV.
# Usage:  bash detect_ablation_streams.sh
# ------------------------------------------------------------
set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=4 # choose your GPU

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_PATH="./profile_results/paper"
LOG_PATH="${RESULT_PATH}/logs"
mkdir -p "${LOG_PATH}"
VAL_IMG_NUM=40504               # images to process in detection
WM_DIR="dataset/watermark_imgs_tile"
# ---------- CSV init ----------
CSV_PATH1="${RESULT_PATH}/ablation_nbatch_size_${TIMESTAMP}.csv"
echo "num_streams,wall_time_s" > "${CSV_PATH1}"

# ---------- helper: run one configuration ----------
run_opt1() {
  local NBATCH_SIZE="$1"               # 8-512

  local EXP_NAME="exp_paper_no_schedule_${NBATCH_SIZE}"
  local OUTPUT_DIR="${RESULT_PATH}/${EXP_NAME}_${TIMESTAMP}"
  mkdir -p "${OUTPUT_DIR}"

  local LOGFILE1="${LOG_PATH}/${EXP_NAME}_${TIMESTAMP}.log"

  # ---- launch python ----
  python QRMark_detection_paper.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size "$NBATCH_SIZE" \
    --train_dir dataset/COCO/train/ \
    --val_dir dataset/COCO/val/ \
    --msg_decoder_path models/random_grid_tile_64_48.pth \
    --reed_solomon True \
    --num_parity_symbols 2 \
    --m_bits_per_symbol 8 \
    --tile random_grid \
    --tile_size 64 \
    --transform_fuse True \
    --num_rs_threads 64 \
    --num_streams  12 \
    --output_dir "$OUTPUT_DIR"  \
    --val_img_num $VAL_IMG_NUM \
    --adaptive_schedule False \
    --workload images \
    --wm_dir  "$WM_DIR" \
    >  "${LOGFILE1}" 2>&1

  # ---- extract wall‑time ----
}

# ---------- helper: run one configuration ----------
run_opt2() {
  local NBATCH_SIZE="$1"               # 8-512

  local EXP_NAME="exp_paper_schedule_${NBATCH_SIZE}"
  local OUTPUT_DIR="${RESULT_PATH}/${EXP_NAME}_${TIMESTAMP}"
  mkdir -p "${OUTPUT_DIR}"

  local LOGFILE2="${LOG_PATH}/${EXP_NAME}_${TIMESTAMP}.log"

  # ---- launch python ----
  python QRMark_detection_paper.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size "$NBATCH_SIZE" \
    --train_dir dataset/COCO/train/ \
    --val_dir dataset/COCO/val/ \
    --msg_decoder_path models/random_grid_tile_64_48.pth \
    --reed_solomon True \
    --num_parity_symbols 2 \
    --m_bits_per_symbol 8 \
    --tile random_grid \
    --tile_size 64 \
    --transform_fuse True \
    --num_rs_threads 64 \
    --num_streams 10 \
    --output_dir "$OUTPUT_DIR"  \
    --val_img_num $VAL_IMG_NUM \
    --adaptive_schedule True \
    --workload images \
    --wm_dir  "$WM_DIR" \
    >  "${LOGFILE2}" 2>&1

  # ---- extract wall‑time ----
}

# ---------- sweep nstreams = 1 … 5 ----------

for s in 64; do
  run_opt1 "${s}"
done

#for s in 64; do
#  run_opt2 "${s}"
#done