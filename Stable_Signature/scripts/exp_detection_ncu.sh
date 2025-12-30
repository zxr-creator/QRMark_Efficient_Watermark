#!/usr/bin/env bash
set -euo pipefail

# ------------ Config ------------
export CUDA_VISIBLE_DEVICES=1
RESULT_PATH="./profile_results/ncu"
LOG_PATH="$RESULT_PATH/logs"
mkdir -p "$RESULT_PATH" "$LOG_PATH"

timestamp=$(date +%Y%m%d_%H%M%S)

val_img_num=40504
mkdir -p "$RESULT_PATH/logs"


experiment_name="exp_ori_ncu"
OUTPUT_DIR="$RESULT_PATH/${experiment_name}_${timestamp}"
mkdir -p "$OUTPUT_DIR"
LOGFILE1="$LOG_PATH/${experiment_name}-${timestamp}.log"
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=system-wide \
  --output="$OUTPUT_DIR" \
  --force-overwrite=true \
  --gpu-metrics-devices cuda-visible \
  python QRMark_detection_ori.py \
        --num_keys 1 \
        --num_bits 48 \
        --val_batch_size 32 \
        --train_dir "dataset/COCO/train/" \
        --val_dir "dataset/COCO/val/" \
        --msg_decoder_path "models/dec_48b.pth" \
        --output_dir "$OUTPUT_DIR" \
        --val_img_num "$val_img_num" \
        --workload images \
        --wm_dir "dataset/watermark_imgs_ori" \
        --img_size 256 \
  > "${LOGFILE1}" 2>&1    
nsys stats "$OUTPUT_DIR.nsys-rep" \
  --report=summary,gpukernsum,gpumemsumm \
  --format=csv \
  > "$OUTPUT_DIR.csv"

experiment_name="exp_opt1_ncu"
OUTPUT_DIR="$RESULT_PATH/${experiment_name}_${timestamp}"
mkdir -p "$OUTPUT_DIR"
LOGFILE2="$LOG_PATH/${experiment_name}-${timestamp}.log"
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=system-wide \
  --output="$OUTPUT_DIR" \
  --force-overwrite=true \
  --gpu-metrics-devices cuda-visible \
   python QRMark_detection.py \
        --num_keys 1 \
        --num_bits 48 \
        --val_batch_size 32 \
        --train_dir dataset/COCO/train/ \
        --val_dir dataset/COCO/val/ \
        --msg_decoder_path models/random_grid_64_48_new.pth \
        --reed_solomon True \
        --num_parity_symbols 2 \
        --m_bits_per_symbol 8 \
        --tile random_grid \
        --tile_size 64 \
        --transform_fuse True \
        --num_rs_threads 32 \
        --output_dir "$OUTPUT_DIR"  \
        --val_img_num "$val_img_num" \
        --async_rs True \
        --workload images \
        --wm_dir dataset/watermark_imgs_tile \
  > "${LOGFILE2}" 2>&1    
nsys stats "$OUTPUT_DIR.nsys-rep" \
  --report=summary,gpukernsum,gpumemsumm \
  --format=csv \
  > "$OUTPUT_DIR.csv"

experiment_name="exp_opt2_ncu"
OUTPUT_DIR="$RESULT_PATH/${experiment_name}_${timestamp}"
mkdir -p "$OUTPUT_DIR"
LOGFILE3="$LOG_PATH/${experiment_name}-${timestamp}.log"
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=system-wide \
  --output="$OUTPUT_DIR" \
  --force-overwrite=true \
  --gpu-metrics-devices cuda-visible \
    python QRMark_detection_paper_reserve.py \
        --num_keys 1 \
        --num_bits 48 \
        --val_batch_size 32 \
        --train_dir dataset/COCO/train/ \
        --val_dir dataset/COCO/val/ \
        --msg_decoder_path models/random_grid_64_48_new.pth \
        --reed_solomon True \
        --num_parity_symbols 2 \
        --m_bits_per_symbol 8 \
        --tile "${TILE_METHOD:-random_grid}" \
        --tile_size "${TILE_SIZE:-64}" \
        --transform_fuse True \
        --num_rs_threads 32 \
        --num_streams  12  \
        --output_dir "$OUTPUT_DIR"  \
        --val_img_num "$val_img_num" \
        --async_rs False \
        --adaptive_schedule False \
        --workload images \
        --wm_dir dataset/watermark_imgs_tile \
        --img_size "${IMG_SIZE:-256}"
  > "${LOGFILE3}" 2>&1    
nsys stats "$OUTPUT_DIR.nsys-rep" \
  --report=summary,gpukernsum,gpumemsumm \
  --format=csv \
  > "$OUTPUT_DIR.csv"


# ==============================================================
#  --------  Extract wall-time and write CSV summary  ----------
# ==============================================================

CSV_BASE="exp_ncu"
csv_idx=1
while [ -f "${RESULT_PATH}/${CSV_BASE}${csv_idx}.csv" ]; do
  csv_idx=$((csv_idx+1))
done
CSV_PATH="${RESULT_PATH}/${CSV_BASE}${csv_idx}.csv"

echo "experiment_name,wall_time_s" > "${CSV_PATH}"

extract_time () {
  local logfile="$1"
  local expname="$2"
  # grep the last occurrence of "wall time =" and strip trailing 's'
  local line
  line=$(grep -E "wall time =" "${logfile}" | tail -n 1)
  if [[ -n "${line}" ]]; then
    local time_val
    time_val=$(echo "${line}" | awk -F'= ' '{print $2}' | tr -d 's')
    echo "${expname},${time_val}" >> "${CSV_PATH}"
  else
    echo "${expname},N/A" >> "${CSV_PATH}"
  fi
}

extract_time "${LOGFILE1}" "exp_ori_ncu"
extract_time "${LOGFILE2}"  "exp_opt1_ncu"
extract_time "${LOGFILE3}" "exp_opt2_ncu"

echo "Summary written to ${CSV_PATH}"
    