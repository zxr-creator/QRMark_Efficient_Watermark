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
RESULT_PATH="./profile_results/batch"
LOG_PATH="${RESULT_PATH}/logs"
mkdir -p "${LOG_PATH}"
VAL_IMG_NUM=40504               # images to process in detection
WM_DIR="dataset/watermark_imgs_tile"
# ---------- CSV init ----------
CSV_PATH1="${RESULT_PATH}/ablation_ori_nbatch_size_${TIMESTAMP}.csv"
echo "num_streams,wall_time_s" > "${CSV_PATH1}"
CSV_PATH2="${RESULT_PATH}/ablation_nbatch_size_${TIMESTAMP}.csv"
echo "num_streams,wall_time_s" > "${CSV_PATH2}"

# ---------- helper: run one configuration ----------
run_opt() {
  local NBATCH_SIZE="$1"               # 8-512

  local EXP_NAME="exp_nbatch_size${NBATCH_SIZE}"
  local OUTPUT_DIR="${RESULT_PATH}/${EXP_NAME}_${TIMESTAMP}"
  mkdir -p "${OUTPUT_DIR}"

  local LOGFILE2="${LOG_PATH}/${EXP_NAME}_${TIMESTAMP}.log"

  # ---- launch python ----
  python QRMark_detection.py \
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
    --num_streams 2 \
    --num_rs_threads 128 \
    --output_dir "$OUTPUT_DIR"  \
    --val_img_num $VAL_IMG_NUM \
    --async_rs True \
    --workload images \
    --wm_dir  "$WM_DIR" \
    >  "${LOGFILE2}" 2>&1

  # ---- extract wall‑time ----
  local time_line
  time_line=$(grep -E "wall time =" "${LOGFILE2}" | tail -n 1 || true)
  local time_val="N/A"
  if [[ -n "${time_line}" ]]; then
    time_val=$(echo "${time_line}" | awk -F'= ' '{print $2}' | tr -d 's')
  fi
  echo "${NBATCH_SIZE},${time_val}" >> "${CSV_PATH2}"
}


for s in 4 8 16 32 64 128 256 512 1024 2048 4096 9192; do
  run_opt "${s}"
done

echo "nbatch_size Ablation summary saved to ${CSV_PATH2}"