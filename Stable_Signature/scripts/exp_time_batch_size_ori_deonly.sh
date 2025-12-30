#!/usr/bin/env bash
# ------------------------------------------------------------
# Ablation study: measure detection-stage wall-time for
# different batch sizes. Results are written to a CSV.
# Usage:  bash exp_time_batch_size_ori_deonly.sh
# ------------------------------------------------------------
set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=5

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_PATH="./profile_results/batch"
LOG_PATH="${RESULT_PATH}/logs"
mkdir -p "${LOG_PATH}"
VAL_IMG_NUM=1000
WM_DIR="dataset/watermark_imgs_ori"

# ---------- CSV init ----------
CSV_PATH1="${RESULT_PATH}/ablation_ori_nbatch_size_${TIMESTAMP}.csv"
echo "batch_size,wall_time_s" > "${CSV_PATH1}"

# ---------- helper: run one configuration ----------
run_ori() {
  local NBATCH_SIZE="$1"                # 8-1024
  local EXP_NAME="exp_ori_nbatch_size${NBATCH_SIZE}"
  local OUTPUT_DIR="${RESULT_PATH}/${EXP_NAME}_${TIMESTAMP}"
  mkdir -p "${OUTPUT_DIR}"

  local LOGFILE1="${LOG_PATH}/${EXP_NAME}_${TIMESTAMP}.log"

  # Build args as an array to avoid quoting/continuation issues
  local -a args=(
    "QRMark_detection_ori.py"
    --num_keys 1
    --num_bits 48
    --val_batch_size "${NBATCH_SIZE}"
    --train_dir "dataset/COCO/train/"
    --val_dir "dataset/COCO/val/"
    --msg_decoder_path "models/dec_48b.pth"
    --output_dir "${OUTPUT_DIR}"
    --val_img_num "${VAL_IMG_NUM}"
    --workload images
    --wm_dir "${WM_DIR}"
    --img_size 256
  )

  python "${args[@]}" > "${LOGFILE1}" 2>&1

  # ---- extract wall-time ----
  local time_line time_val="N/A"
  time_line="$(grep -E 'wall time =' "${LOGFILE1}" | tail -n 1 || true)"
  if [[ -n "${time_line}" ]]; then
    time_val="$(echo "${time_line}" | awk -F'= ' '{print $2}' | tr -d 's')"
  fi
  echo "${NBATCH_SIZE},${time_val}" >> "${CSV_PATH1}"
}

# ---------- sweep ----------
for s in 4 8 16 32 64 128 256 512 1024; do
  run_ori "${s}"
done

echo "ori_nbatch_size ablation summary saved to ${CSV_PATH1}"