#!/bin/bash
set -euo pipefail

CONFIG="configs/vitl_336_OVRSIS95K.yaml"
GPUS=4
BASE_OUTPUT="output/ablation_pini_agg_l"

echo "============================================"
echo "PiNI Ablation Experiments"
echo "============================================"
echo "CONFIG: ${CONFIG}"
echo "GPUS: ${GPUS}"
echo "BASE_OUTPUT: ${BASE_OUTPUT}"
echo "============================================"

mkdir -p "${BASE_OUTPUT}"

run_exp () {
  EXP_NAME="$1"
  shift

  EXP_DIR="${BASE_OUTPUT}/${EXP_NAME}"
  mkdir -p "${EXP_DIR}"

  echo ""
  echo "--------------------------------------------"
  echo "Running: ${EXP_NAME}"
  echo "Output:  ${EXP_DIR}"
  echo "--------------------------------------------"

  python train_net.py \
    --config-file "${CONFIG}" \
    --num-gpus "${GPUS}" \
    OUTPUT_DIR "${EXP_DIR}" \
    "$@" \
    2>&1 | tee "${EXP_DIR}/train_log.txt"

  # 如果你自己的 tools/run.sh 是做测试/汇总，这里保留
  bash tools/run.sh "${CONFIG}" "${GPUS}" "${EXP_DIR}"
}

# echo ""
# echo "[Stage 0] Baselines"

# # 0.1 Baseline: no PiNI
# run_exp "baseline_no_pini" \
#   MODEL.PINI.ENABLED False

# # 0.2 Full PiNI default
# run_exp "full_pini_default" \
#   MODEL.PINI.ENABLED True \
#   MODEL.PINI.IMAGE_VPN_ENABLED True \
#   MODEL.PINI.TEXT_VPN_ENABLED True

# --------------------------------------------------
# Stage 1: Noise type combinations
# --------------------------------------------------
# echo ""
# echo "[Stage 1] Noise type combinations"

# IMAGE_NOISE_TYPES=("gaussian" "laplace" "uniform")
# TEXT_NOISE_TYPES=("gaussian" "laplace" "uniform")

# for img_noise in "${IMAGE_NOISE_TYPES[@]}"; do
#   for txt_noise in "${TEXT_NOISE_TYPES[@]}"; do
#     EXP_NAME="noise_img_${img_noise}_txt_${txt_noise}"
#     run_exp "${EXP_NAME}" \
#       MODEL.PINI.ENABLED True \
#       MODEL.PINI.IMAGE_VPN_ENABLED True \
#       MODEL.PINI.TEXT_VPN_ENABLED True \
#       MODEL.PINI.REDUCTION 1 \
#       MODEL.PINI.TEXT_NOISE_STD 0.02 \
#       MODEL.PINI.IMAGE_NOISE_TYPE "${img_noise}" \
#       MODEL.PINI.TEXT_NOISE_TYPE "${txt_noise}" \
#       MODEL.PINI.IMAGE_STUDENT_T_DF 3.0 \
#       MODEL.PINI.TEXT_STUDENT_T_DF 3.0
#   done
# done


echo ""
echo "[Stage 1] Noise type combinations"

# IMAGE_NOISE_TYPES=("gaussian" "laplace" "uniform")
IMAGE_NOISE_TYPES=("student_t")
TEXT_NOISE_TYPES=("laplace" "uniform" "student_t")

for img_noise in "${IMAGE_NOISE_TYPES[@]}"; do
  for txt_noise in "${TEXT_NOISE_TYPES[@]}"; do
    EXP_NAME="noise_img_${img_noise}_txt_${txt_noise}"
    run_exp "${EXP_NAME}" \
      MODEL.PINI.ENABLED True \
      MODEL.PINI.IMAGE_VPN_ENABLED True \
      MODEL.PINI.TEXT_VPN_ENABLED True \
      MODEL.PINI.REDUCTION 1 \
      MODEL.PINI.TEXT_NOISE_STD 0.02 \
      MODEL.PINI.IMAGE_NOISE_TYPE "${img_noise}" \
      MODEL.PINI.TEXT_NOISE_TYPE "${txt_noise}" \
      MODEL.PINI.IMAGE_STUDENT_T_DF 3.0 \
      MODEL.PINI.TEXT_STUDENT_T_DF 3.0
  done
done


echo ""
echo "[Stage 2] Noise type combinations"

IMAGE_NOISE_TYPES=("gaussian" "laplace" "uniform")
TEXT_NOISE_TYPES=("student_t")

for img_noise in "${IMAGE_NOISE_TYPES[@]}"; do
  for txt_noise in "${TEXT_NOISE_TYPES[@]}"; do
    EXP_NAME="noise_img_${img_noise}_txt_${txt_noise}"
    run_exp "${EXP_NAME}" \
      MODEL.PINI.ENABLED True \
      MODEL.PINI.IMAGE_VPN_ENABLED True \
      MODEL.PINI.TEXT_VPN_ENABLED True \
      MODEL.PINI.REDUCTION 1 \
      MODEL.PINI.TEXT_NOISE_STD 0.02 \
      MODEL.PINI.IMAGE_NOISE_TYPE "${img_noise}" \
      MODEL.PINI.TEXT_NOISE_TYPE "${txt_noise}" \
      MODEL.PINI.IMAGE_STUDENT_T_DF 3.0 \
      MODEL.PINI.TEXT_STUDENT_T_DF 3.0
  done
done

# # --------------------------------------------------
# # Stage 2: REDUCTION ablation
# # --------------------------------------------------
# echo ""
# echo "[Stage 2] REDUCTION ablation"

# REDUCTIONS=(1 2 4 8)

# for red in "${REDUCTIONS[@]}"; do
#   EXP_NAME="reduction_${red}"
#   run_exp "${EXP_NAME}" \
#     MODEL.PINI.ENABLED True \
#     MODEL.PINI.IMAGE_VPN_ENABLED True \
#     MODEL.PINI.TEXT_VPN_ENABLED True \
#     MODEL.PINI.REDUCTION "${red}" \
#     MODEL.PINI.TEXT_NOISE_STD 0.02 \
#     MODEL.PINI.IMAGE_NOISE_TYPE "gaussian" \
#     MODEL.PINI.TEXT_NOISE_TYPE "gaussian" \
#     MODEL.PINI.IMAGE_STUDENT_T_DF 3.0 \
#     MODEL.PINI.TEXT_STUDENT_T_DF 3.0
# done

# # --------------------------------------------------
# # Stage 3: TEXT_NOISE_STD ablation
# # --------------------------------------------------
# echo ""
# echo "[Stage 3] TEXT_NOISE_STD ablation"

# TEXT_NOISE_STDS=(0.0 0.005 0.01 0.02 0.05)

# for std in "${TEXT_NOISE_STDS[@]}"; do
#   std_tag="${std/./p}"   # 0.02 -> 0p02
#   EXP_NAME="text_noise_std_${std_tag}"
#   run_exp "${EXP_NAME}" \
#     MODEL.PINI.ENABLED True \
#     MODEL.PINI.IMAGE_VPN_ENABLED True \
#     MODEL.PINI.TEXT_VPN_ENABLED True \
#     MODEL.PINI.REDUCTION 1 \
#     MODEL.PINI.TEXT_NOISE_STD "${std}" \
#     MODEL.PINI.IMAGE_NOISE_TYPE "gaussian" \
#     MODEL.PINI.TEXT_NOISE_TYPE "gaussian" \
#     MODEL.PINI.IMAGE_STUDENT_T_DF 3.0 \
#     MODEL.PINI.TEXT_STUDENT_T_DF 3.0
# done

echo ""
echo "============================================"
echo "All experiments finished."
echo "============================================"

echo ""
echo "Quick summary:"
for dir in "${BASE_OUTPUT}"/*; do
  if [ -d "${dir}" ] && [ -f "${dir}/train_log.txt" ]; then
    echo "--------------------------------------------"
    echo "$(basename "${dir}")"
    grep copypaste "${dir}/train_log.txt" | tail -1 || true
  fi
done