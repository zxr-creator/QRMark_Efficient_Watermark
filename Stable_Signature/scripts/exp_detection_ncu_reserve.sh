#!/bin/bash
# bash /workspace/Cogvideo_project/scripts/run_nsight_compute_profile.sh
set -x

export PATH=$PATH:/opt/nvidia/nsight-systems/2025.3.1/bin
export CUDA_VISIBLE_DEVICES=3

timestamp=$(date +%Y%m%d_%H%M%S)
RESULT_PATH="./profile_results/ncu"
LOG_PATH="$RESULT_PATH/logs"
LOGFILE="$LOG_PATH/profile-ncu-debug-${timestamp}.log"
mkdir -p "$LOG_PATH"
val_img_num=40504

mkdir -p "$RESULT_PATH/logs"

experiment_name="exp_finetuning_ncu"
OUTPUT_DIR="$RESULT_PATH/${experiment_name}_${timestamp}"
mkdir -p "$OUTPUT_DIR"
LOGFILE_OPT1="$LOG_PATH/${experiment_name}-${timestamp}.log"
ncu --replay-mode kernel \
    --export $OUTPUT_DIR/${experiment_name}_${timestamp} \
    --kernel-name "regex:(sm90_xmma_fprop_implicit_gemm_f32f32|sm90_xmma_fprop_implicit_gemm_f32f32|RowwiseMomentsCUDAKernel|RowwiseMomentsCUDAKernel|nchwToNhwcKernel|nhwcToNchwKernel|elementwise_kernel|c|bn_fw_inf_1C11_kernel_NCHW|vectorized_elementwise_kernel)" \
    --nvtx  \
    --nvtx-include "regex:.*" \
    -c 20 \
    python finetune_ldm_decoder_optimized.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size 16 \
    --train_dir dataset/COCO/train/ \
    --val_dir dataset/COCO/val/ \
    --use_random_msg_decoder False \
    --msg_decoder_path models/random_tile_64_48.pth \
    --reed_solomon True \
    --num_parity_symbols 2 \
    --m_bits_per_symbol 4 \
    --tile random \
    --tile_size 64 \
    --transform_fuse False \
    --output_dir "$OUTPUT_DIR" \
    --val_img_num $val_img_num \
    > ${LOGFILE_OPT1} 2>&1

experiment_name="exp_finetuning_ncu_original"
OUTPUT_DIR="$RESULT_PATH/${experiment_name}_${timestamp}"
mkdir -p "$OUTPUT_DIR"
LOGFILE_ORI="$LOG_PATH/${experiment_name}-${timestamp}.log"
ncu --replay-mode kernel \
    --export $OUTPUT_DIR/${experiment_name}_${timestamp}.ncu-rep \
    --kernel-name "regex:(sm90_xmma_fprop_implicit_gemm_f32f32|sm90_xmma_fprop_implicit_gemm_f32f32|RowwiseMomentsCUDAKernel|RowwiseMomentsCUDAKernel|nchwToNhwcKernel|nhwcToNchwKernel|elementwise_kernel|c|bn_fw_inf_1C11_kernel_NCHW|vectorized_elementwise_kernel)" \
    --nvtx --target-processes all \
    --nvtx-include 'regex:.*' \
    -c 20 \
    python finetune_ldm_decoder.py \
      --num_keys 1 \
      --num_bits 48 \
      --val_batch_size 16 \
      --train_dir dataset/COCO/train/ \
      --val_dir dataset/COCO/val/ \
      --msg_decoder_path models/dec_48b.pth \
      --output_dir "$OUTPUT_DIR"  \
      --val_img_num $val_img_num \
    > ${LOGFILE_ORI} 2>&1



# ==========================
# Profile finetune_ldm_decoder_optimized.py
# ==========================
experiment_name="exp_finetuning_ncu_prepocess"
OUTPUT_DIR="$RESULT_PATH/${experiment_name}_${timestamp}"
mkdir -p "$OUTPUT_DIR"
LOGFILE_OPT2="$LOG_PATH/${experiment_name}-${timestamp}.log"
ncu --replay-mode kernel \
    --export $RESULT_PATH/${experiment_name}_${timestamp}.ncu-rep \
    --kernel-name "regex:(sm90_xmma_fprop_implicit_gemm_f32f32|sm90_xmma_fprop_implicit_gemm_f32f32|RowwiseMomentsCUDAKernel|RowwiseMomentsCUDAKernel|nchwToNhwcKernel|nhwcToNchwKernel|elementwise_kernel|c|bn_fw_inf_1C11_kernel_NCHW|vectorized_elementwise_kernel)" \
    --nvtx --target-processes all \
    --nvtx-include 'regex:.*' \
    -c 20 \
  python finetune_ldm_decoder_optimized.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size 16 \
    --train_dir dataset/COCO/train/ \
    --val_dir dataset/COCO/val/ \
    --use_random_msg_decoder False \
    --msg_decoder_path models/random_tile_64_48.pth \
    --reed_solomon True \
    --num_parity_symbols 2 \
    --m_bits_per_symbol 4 \
    --tile random \
    --tile_size 64 \
    --transform_fuse True \
    --output_dir "$OUTPUT_DIR"  \
    --val_img_num $val_img_num \
  > "${LOGFILE_OPT2}" 2>&1    


# ==============================================================
#  --------  Extract wall-time and write CSV summary  ----------
# ==============================================================

CSV_BASE="exp_finetuning_ncu"
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

extract_time "${LOGFILE_OPT1}" "exp_ncu"
extract_time "${LOGFILE_ORI}"  "exp_ncu_original"
extract_time "${LOGFILE_OPT2}" "exp_ncu_preprocess"

echo "Summary written to ${CSV_PATH}"
    