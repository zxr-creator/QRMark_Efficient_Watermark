#!/bin/bash
# bash /workspace/Cogvideo_project/scripts/run_nsight_systems_profile_nsys.sh
set -x
export PATH=$PATH:/opt/nvidia/nsight-systems/2025.3.1/bin
export CUDA_VISIBLE_DEVICES=3

timestamp=$(date +%Y%m%d_%H%M%S)
RESULT_PATH="./profile_results/speedup"
LOG_PATH="$RESULT_PATH/logs" 
mkdir -p "$LOG_PATH"
val_img_num=40504
WM_ORI_DIR="dataset/watermark_ori_imgs_speedup"
WM_DIR="dataset/watermark_imgs_speedup"

# ==========================
# Profile finetune_ldm_decoder.py
# ==========================
experiment_name="exp_gen_original"
LOGFILE_GEN1="$LOG_PATH/${experiment_name}-${timestamp}.log"
OUTPUT_GEN_DIR="$RESULT_PATH/gen/${experiment_name}_${timestamp}"
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
    --val_img_num $val_img_num \
    --output_dir "$OUTPUT_GEN_DIR" \
    --workload images \
    --wm_dir  "$WM_ORI_DIR" \
> "${LOGFILE_GEN1}" 2>&1

experiment_name="exp_nsys_original"
OUTPUT_DIR="$RESULT_PATH/${experiment_name}_${timestamp}"
mkdir -p "$OUTPUT_DIR"
LOGFILE_ORI1="$RESULT_PATH/logs/${experiment_name}-${timestamp}.log"
python QRMark_detection_ori.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size 16 \
    --train_dir dataset/COCO/train/ \
    --val_dir dataset/COCO/val/ \
    --msg_decoder_path models/dec_48b.pth \
    --output_dir "$OUTPUT_DIR"  \
    --val_img_num $val_img_num \
    --workload images \
    --wm_dir  "$WM_ORI_DIR" \
  > "${LOGFILE_ORI1}" 2>&1    

experiment_name="exp_nsys_original_batch"
OUTPUT_DIR="$RESULT_PATH/${experiment_name}_${timestamp}"
mkdir -p "$OUTPUT_DIR"
LOGFILE_ORI2="$RESULT_PATH/logs/${experiment_name}-${timestamp}.log"
python QRMark_detection_ori.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size 16 \
    --train_dir dataset/COCO/train/ \
    --val_dir dataset/COCO/val/ \
    --msg_decoder_path models/dec_48b.pth \
    --output_dir "$OUTPUT_DIR"  \
    --val_img_num $val_img_num \
    --workload images \
    --wm_dir  "$WM_ORI_DIR" \
  > "${LOGFILE_ORI2}" 2>&1    

# ==========================
# Profile QRMark_detection.py tile
# ==========================
experiment_name="exp_gen_tile"
LOGFILE_GEN2="$LOG_PATH/${experiment_name}-${timestamp}.log"
OUTPUT_GEN_DIR="$RESULT_PATH/gen/${experiment_name}_${timestamp}"
python QRMark_generate.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size 16 \
    --train_dir dataset/COCO/train/ \
    --val_dir dataset/COCO/val/ \
    --msg_decoder_path models/random_grid_tile_64_48.pth \
    --reed_solomon True \
    --num_parity_symbols 2 \
    --m_bits_per_symbol 8 \
    --tile True \
    --tile_size 64 \
    --transform_fuse False \
    --val_img_num $val_img_num \
    --output_dir "$OUTPUT_GEN_DIR" \
    --workload images \
    --wm_dir  "$WM_DIR" \
> "${LOGFILE_GEN2}" 2>&1

experiment_name="exp_nsys_tile"
OUTPUT_DIR="$RESULT_PATH/${experiment_name}_${timestamp}"
mkdir -p "$OUTPUT_DIR"
LOGFILE_OPT1="$RESULT_PATH/logs/${experiment_name}-${timestamp}.log"
python QRMark_detection.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size 16 \
    --train_dir dataset/COCO/train/ \
    --val_dir dataset/COCO/val/ \
    --msg_decoder_path models/random_grid_tile_64_48.pth \
    --reed_solomon True \
    --num_parity_symbols 2 \
    --m_bits_per_symbol 8 \
    --tile random_grid \
    --tile_size 64 \
    --transform_fuse Flase \
    --num_rs_threads 8 \
    --output_dir "$OUTPUT_DIR"  \
    --val_img_num $val_img_num \
    --async_rs False \
    --workload images \
    --wm_dir  "$WM_DIR" \
  > "${LOGFILE_OPT1}" 2>&1    
# ==========================
# Profile QRMark_detection.py (transform_fuse=True)
# ==========================
experiment_name="exp_nsys_preprocess"
OUTPUT_DIR="$RESULT_PATH/${experiment_name}_${timestamp}"
mkdir -p "$OUTPUT_DIR"
LOGFILE_OPT2="$RESULT_PATH/logs/${experiment_name}-${timestamp}.log"
python QRMark_detection.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size 16 \
    --train_dir dataset/COCO/train/ \
    --val_dir dataset/COCO/val/ \
    --msg_decoder_path models/random_grid_tile_64_48.pth \
    --reed_solomon True \
    --num_parity_symbols 2 \
    --m_bits_per_symbol 8 \
    --tile random_grid \
    --tile_size 64 \
    --transform_fuse True \
    --num_rs_threads 8 \
    --output_dir "$OUTPUT_DIR"  \
    --val_img_num $val_img_num \
    --async_rs False \
    --workload images \
    --wm_dir  "$WM_DIR" \
  > "${LOGFILE_OPT2}" 2>&1    

# ==========================
# Profile QRMark_detection.py (async_rs=True)
# ==========================

experiment_name="exp_nsys_batch"
OUTPUT_DIR="$RESULT_PATH/${experiment_name}_${timestamp}"
mkdir -p "$OUTPUT_DIR"
LOGFILE_OPT3="$RESULT_PATH/logs/${experiment_name}-${timestamp}.log"
python QRMark_detection.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size 128 \
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
    --output_dir "$OUTPUT_DIR"  \
    --val_img_num $val_img_num \
    --async_rs False \
    --workload images \
    --wm_dir  "$WM_DIR" \
  > "${LOGFILE_OPT3}" 2>&1    

# ==========================
# Profile QRMark_detection.py 
# ==========================
experiment_name="exp_nsys_aync"
OUTPUT_DIR="$RESULT_PATH/${experiment_name}_${timestamp}"
mkdir -p "$OUTPUT_DIR"
LOGFILE_OPT4="$RESULT_PATH/logs/${experiment_name}-${timestamp}.log"
python QRMark_detection.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size 128 \
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
    --output_dir "$OUTPUT_DIR"  \
    --val_img_num $val_img_num \
    --async_rs True \
    --workload images \
    --wm_dir  "$WM_DIR" \
  > "${LOGFILE_OPT4}" 2>&1

experiment_name="exp_nsys_opt1"
OUTPUT_DIR="$RESULT_PATH/${experiment_name}_${timestamp}"
mkdir -p "$OUTPUT_DIR"
LOGFILE_OPT5="$RESULT_PATH/logs/${experiment_name}-${timestamp}.log"
python QRMark_detection_paper.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size 128 \
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
    --num_streams  6 \
    --output_dir "$OUTPUT_DIR"  \
    --val_img_num $val_img_num \
    --adaptive_schedule False \
    --workload images \
    --wm_dir  "$WM_DIR" \
    >  "${LOGFILE_OPT5}" 2>&1

# ==============================================================
#  --------  Extract wall-time and write CSV summary  ----------
# ==============================================================

CSV_BASE="exp_finetuning_sys"
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

extract_time "${LOGFILE_ORI1}"  "exp_nsys_original"
extract_time "${LOGFILE_ORI2}"  "exp_nsys_original_batch"
extract_time "${LOGFILE_OPT1}" "exp_nsys_tile"
extract_time "${LOGFILE_OPT2}" "exp_nsys_preprocess"
extract_time "${LOGFILE_OPT3}" "exp_nsys_batch"
extract_time "${LOGFILE_OPT4}" "exp_nsys_aync"
extract_time "${LOGFILE_OPT5}" "exp_nsys_opt1"


echo "Summary written to ${CSV_PATH}"
