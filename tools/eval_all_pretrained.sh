#!/bin/sh

# ============================================================
# Master evaluation script for Pi-Seg / RSKT-Seg pretrained model
#
# It calls:
#   tools/eval_OVRSIS.sh
#   tools/eval_Building_Extraction.sh
#   tools/eval_Flood_Detection.sh
#   tools/eval_Road_Extraction.sh
#
# Usage:
#   sh tools/eval_all_pretrained.sh \
#      configs/vitb_384_OVRSIS95K.yaml \
#      4 \
#      output_eval_vitb_noise \
#      /lby_data01/zhaozy/lby/OVRSIS/RSKT-Seg_and_Pi-Seg-Pi-Seg/pretrained_weights/VIT-B_noise_img_gaussian_txt_gaussian.pth
#
# Optional extra opts:
#   sh tools/eval_all_pretrained.sh CONFIG GPUS OUTPUT WEIGHT MODEL.SOME_KEY value
# ============================================================

config=$1
gpus=$2
output=$3
pretrained_weight=$4

if [ -z "$config" ]; then
    echo "No config file found!"
    echo "Usage: sh tools/eval_all_pretrained.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [PRETRAINED_WEIGHT] [OPTS]"
    exit 1
fi

if [ -z "$gpus" ]; then
    echo "Number of GPUs not specified!"
    echo "Usage: sh tools/eval_all_pretrained.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [PRETRAINED_WEIGHT] [OPTS]"
    exit 1
fi

if [ -z "$output" ]; then
    echo "No output directory found!"
    echo "Usage: sh tools/eval_all_pretrained.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [PRETRAINED_WEIGHT] [OPTS]"
    exit 1
fi

if [ -z "$pretrained_weight" ]; then
    echo "No pretrained weight specified!"
    echo "Usage: sh tools/eval_all_pretrained.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [PRETRAINED_WEIGHT] [OPTS]"
    exit 1
fi

if [ ! -f "$pretrained_weight" ]; then
    echo "[ERROR] Pretrained weight not found:"
    echo "$pretrained_weight"
    exit 1
fi

shift 4
extra_opts=${@}

mkdir -p "$output"

echo "============================================================"
echo "  Pi-Seg / RSKT-Seg Master Evaluation"
echo "============================================================"
echo "  Config            : $config"
echo "  GPUs              : $gpus"
echo "  Output dir        : $output"
echo "  Pretrained weight : $pretrained_weight"
echo "  Extra opts        : $extra_opts"
echo "============================================================"

# 关键：用后置 MODEL.WEIGHTS 覆盖各子脚本里的 $output/model_final.pth
weight_opts="MODEL.WEIGHTS $pretrained_weight"

# ============================================================
# 1. OVRSIS
# ============================================================
echo ""
echo "============================================================"
echo "  [1/4] Evaluating OVRSIS"
echo "============================================================"
sh tools/eval_OVRSIS.sh \
  "$config" \
  "$gpus" \
  "$output" \
  $weight_opts \
  $extra_opts

# ============================================================
# 2. Building Extraction
# ============================================================
echo ""
echo "============================================================"
echo "  [2/4] Evaluating Building Extraction"
echo "============================================================"
sh tools/eval_Building_Extraction.sh \
  "$config" \
  "$gpus" \
  "$output" \
  $weight_opts \
  $extra_opts

# ============================================================
# 3. Flood Detection
# ============================================================
echo ""
echo "============================================================"
echo "  [3/4] Evaluating Flood Detection"
echo "============================================================"
sh tools/eval_Flood_Detection.sh \
  "$config" \
  "$gpus" \
  "$output" \
  $weight_opts \
  $extra_opts

# ============================================================
# 4. Road Extraction
# ============================================================
echo ""
echo "============================================================"
echo "  [4/4] Evaluating Road Extraction"
echo "============================================================"
sh tools/eval_Road_Extraction.sh \
  "$config" \
  "$gpus" \
  "$output" \
  $weight_opts \
  $extra_opts

echo ""
echo "============================================================"
echo "  All evaluations finished."
echo "  Output dir: $output"
echo "============================================================"