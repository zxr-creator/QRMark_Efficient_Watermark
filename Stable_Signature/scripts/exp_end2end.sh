#!/usr/bin/env bash
# ------------------------------------------------------------
# End-to-end experiment:
# Sweep batch sizes and compare QRMark_detection_paper vs _ori.
# Results: logs + a CSV summary including wall time & per-image latency.
# ------------------------------------------------------------
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7 # choose your GPU
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

RESULT_PATH="./profile_results/end2end"
LOG_PATH="${RESULT_PATH}/logs"
mkdir -p "${LOG_PATH}"

VAL_IMG_NUM="${VAL_IMG_NUM:-40504}"
WM_DIR="${WM_DIR:-dataset/watermark_imgs_tile}"
WM_DIR_ORI="${WM_DIR_ORI:-dataset/watermark_imgs_ori}"

PAPER_DECODER_PATH="${PAPER_DECODER_PATH:-models/random_grid_tile_64_48.pth}"
ORI_DECODER_PATH="${ORI_DECODER_PATH:-models/dec_48b.pth}"

TILE_METHOD="${TILE_METHOD:-random_grid}"
TILE_SIZE="${TILE_SIZE:-64}"
NUM_RS_THREADS="${NUM_RS_THREADS:-32}"
NUM_STREAMS="${NUM_STREAMS:-10}"
IMG_SIZE="${IMG_SIZE:-256}"

BATCH_LIST=(1 4 8 16 32 64 128 256 512)

CSV_PATH="${RESULT_PATH}/end2end.csv"
# CSV header: add per-image latency column
echo "timestamp,exp,script,batch_size,wall_time_s,per_img_latency_s,bit_acc,word_acc,imgs_per_s,val_img_num,logfile" > "${CSV_PATH}"

# ----------- parse helpers -----------
parse_wall_time () {
  local log="$1"
  local line
  line=$(grep -E "wall time" -a "$log" | tail -n1 || true)
  if [[ -n "$line" ]]; then
    echo "$line" | grep -Eo '[0-9]+\.[0-9]+' | tail -n1
  else
    echo ""
  fi
}

parse_per_image_latency () {
  local log="$1"
  local line
  line=$(grep -E "\[Detect\]\[latency\] approx per-image latency" -a "$log" | tail -n1 || true)
  if [[ -n "$line" ]]; then
    # 取 ≈ 后面的秒数字
    echo "$line" | sed -E 's/.*≈ ([0-9]*\.[0-9]+)s.*/\1/'
  else
    echo ""
  fi
}

parse_accuracy () {
  local log="$1"
  local line
  line=$(grep -E "^\[Detect\] accuracy: bit_acc=" -a "$log" | tail -n1 || true)
  if [[ -z "$line" ]]; then
    echo ","
    return 0
  fi
  local bit word
  bit=$(echo "$line" | sed -E 's/.*bit_acc=([0-9]*\.[0-9]+).*/\1/' )
  word=$(echo "$line" | sed -E 's/.*word_acc=([0-9]*\.[0-9]+).*/\1/' )
  echo "${bit},${word}"
}

append_row () {
  local exp="$1" script="$2" bs="$3" wall="$4" perimg="$5" bit="$6" word="$7" log="$8"

  # imgs/s: prefer 1 / per-image-latency; fallback to VAL_IMG_NUM / wall_time
  local ips=""
  if [[ -n "$perimg" ]]; then
    ips=$(awk -v x="$perimg" 'BEGIN{ if (x+0>0) printf "%.4f", 1.0/x }')
  elif [[ -n "$wall" ]]; then
    ips=$(awk -v n="$VAL_IMG_NUM" -v w="$wall" 'BEGIN{ if (w+0>0) printf "%.4f", n/w }')
  fi

  echo "${TIMESTAMP},${exp},${script},${bs},${wall},${perimg},${bit},${word},${ips},${VAL_IMG_NUM},${log}" >> "${CSV_PATH}"
}

# ----------- runners -----------
run_paper () {
  local bs="$1"
  local EXP_NAME="end2end_paper_bs${bs}"
  local OUTPUT_DIR="${RESULT_PATH}/${EXP_NAME}_${TIMESTAMP}"
  mkdir -p "${OUTPUT_DIR}"
  local LOGFILE="${LOG_PATH}/${EXP_NAME}_${TIMESTAMP}.log"

  echo "[RUN] paper bs=${bs} -> ${LOGFILE}"
  python QRMark_detection_reserve.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size "${bs}" \
    --train_dir dataset/COCO/train/ \
    --val_dir dataset/COCO/val/ \
    --msg_decoder_path "${PAPER_DECODER_PATH}" \
    --reed_solomon False \
    --num_parity_symbols 2 \
    --m_bits_per_symbol 8 \
    --tile "${TILE_METHOD}" \
    --tile_size "${TILE_SIZE}" \
    --transform_fuse True \
    --num_rs_threads "${NUM_RS_THREADS}" \
    --num_streams  "${NUM_STREAMS}"  \
    --output_dir "${OUTPUT_DIR}"  \
    --val_img_num "${VAL_IMG_NUM}" \
    --async_rs False \
    --adaptive_schedule False \
    --workload images \
    --wm_dir  "${WM_DIR}" \
    --img_size "${IMG_SIZE}" \
    > "${LOGFILE}" 2>&1

  local WALL; WALL=$(parse_wall_time "${LOGFILE}")
  local PERIMG; PERIMG=$(parse_per_image_latency "${LOGFILE}")
  local ACC; ACC=$(parse_accuracy "${LOGFILE}")
  local BIT="${ACC%,*}"; local WORD="${ACC#*,}"
  append_row "paper" "QRMark_detection_paper.py" "${bs}" "${WALL}" "${PERIMG}" "${BIT}" "${WORD}" "${LOGFILE}"
}

run_ori () {
  local bs="$1"
  local EXP_NAME="end2end_ori_bs${bs}"
  local OUTPUT_DIR="${RESULT_PATH}/${EXP_NAME}_${TIMESTAMP}"
  mkdir -p "${OUTPUT_DIR}"
  local LOGFILE="${LOG_PATH}/${EXP_NAME}_${TIMESTAMP}.log"

  echo "[RUN] ori   bs=${bs} -> ${LOGFILE}"
  python QRMark_detection_ori.py \
    --num_keys 1 \
    --num_bits 48 \
    --val_batch_size "${bs}" \
    --train_dir dataset/COCO/train/ \
    --val_dir dataset/COCO/val/ \
    --msg_decoder_path "${ORI_DECODER_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --val_img_num "${VAL_IMG_NUM}" \
    --workload images \
    --wm_dir  "${WM_DIR_ORI}" \
    --img_size "${IMG_SIZE}" \
    > "${LOGFILE}" 2>&1

  local WALL; WALL=$(parse_wall_time "${LOGFILE}")
  local PERIMG; PERIMG=$(parse_per_image_latency "${LOGFILE}")
  local ACC; ACC=$(parse_accuracy "${LOGFILE}")
  local BIT="${ACC%,*}"; local WORD="${ACC#*,}"
  append_row "ori" "QRMark_detection_ori.py" "${bs}" "${WALL}" "${PERIMG}" "${BIT}" "${WORD}" "${LOGFILE}"
}

# ----------- main sweep -----------
echo "[INFO] Results dir: ${RESULT_PATH}"
echo "[INFO] Logs dir   : ${LOG_PATH}"
echo "[INFO] CSV        : ${CSV_PATH}"
echo "[INFO] Batch list : ${BATCH_LIST[*]}"
echo

for bs in "${BATCH_LIST[@]}"; do
  run_paper "${bs}"
  run_ori   "${bs}"
done

echo
echo "[DONE] Summary at: ${CSV_PATH}"
