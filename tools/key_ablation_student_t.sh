#!/bin/bash
set -euo pipefail

CONFIG="configs/vitb_384_OVRSIS95K.yaml"
GPUS=4
BASE_OUTPUT="output/ablation_student_t"

mkdir -p "${BASE_OUTPUT}"

run_exp () {
  EXP_NAME="$1"
  shift
  EXP_DIR="${BASE_OUTPUT}/${EXP_NAME}"
  mkdir -p "${EXP_DIR}"

  # python train_net.py \
  #   --config-file "${CONFIG}" \
  #   --num-gpus "${GPUS}" \
  #   OUTPUT_DIR "${EXP_DIR}" \
  #   "$@" \
  #   2>&1 | tee "${EXP_DIR}/train_log.txt"
  # sh tools/eval_OVRSIS.sh "${CONFIG}" "${GPUS}" "${EXP_DIR}"
  sh tools/eval_Building_Extraction.sh "${CONFIG}" "${GPUS}" "${EXP_DIR}"
  sh tools/eval_Flood_Detection.sh "${CONFIG}" "${GPUS}" "${EXP_DIR}"
  sh tools/eval_Road_Extraction.sh "${CONFIG}" "${GPUS}" "${EXP_DIR}"
  # bash tools/run.sh "${CONFIG}" "${GPUS}" "${EXP_DIR}"
}

DFS=(10.0)

for df in "${DFS[@]}"; do
  df_tag="${df/./p}"

  # student_t on image side only
  run_exp "img_student_t_df_${df_tag}_txt_gaussian" \
    MODEL.PINI.ENABLED True \
    MODEL.PINI.IMAGE_VPN_ENABLED True \
    MODEL.PINI.TEXT_VPN_ENABLED True \
    MODEL.PINI.REDUCTION 1 \
    MODEL.PINI.TEXT_NOISE_STD 0.02 \
    MODEL.PINI.IMAGE_NOISE_TYPE "student_t" \
    MODEL.PINI.IMAGE_STUDENT_T_DF "${df}" \
    MODEL.PINI.TEXT_NOISE_TYPE "gaussian" \
    MODEL.PINI.TEXT_STUDENT_T_DF 3.0

  # # student_t on text side only
  # run_exp "img_gaussian_txt_student_t_df_${df_tag}" \
  #   MODEL.PINI.ENABLED True \
  #   MODEL.PINI.IMAGE_VPN_ENABLED True \
  #   MODEL.PINI.TEXT_VPN_ENABLED True \
  #   MODEL.PINI.REDUCTION 1 \
  #   MODEL.PINI.TEXT_NOISE_STD 0.02 \
  #   MODEL.PINI.IMAGE_NOISE_TYPE "gaussian" \
  #   MODEL.PINI.IMAGE_STUDENT_T_DF 3.0 \
  #   MODEL.PINI.TEXT_NOISE_TYPE "student_t" \
  #   MODEL.PINI.TEXT_STUDENT_T_DF "${df}"
done