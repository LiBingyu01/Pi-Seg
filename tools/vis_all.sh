#!/bin/bash

# ==========================================
# 1. 核心配置 (请确保路径与你服务器一致)
# ==========================================
CONFIG="/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/CAT-Seg-OVRSIS-PI/configs/vitb_384_OVRSIS95K.yaml"
MODEL_WEIGHTS="/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/CAT-Seg-OVRSIS-PI/output_vitb_384_OVRSIS95K_time_1/model_final.pth"

# 路径配置
OUTPUT_BASE="/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/CAT-Seg-OVRSIS-PI/vis_rs/PISeg_UABG"
SAMPLED_BASE="/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/CAT-Seg-OVRSIS-PI/vis_rs/vis_images"

# 数据集配置 (注意：我已经帮你去掉了行尾的逗号)
DATASETS=(
    "LoveDA|png|LoveDA_all_sem_seg|/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/LoveDA.json"
    "OpenEarthMap|tif|OpenEarthMap_all_sem_seg|/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/openearthmap.json"
    "uavid|png|uavid_all_sem_seg|/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/uavid.json"
    "UDD5|JPG|UDD5_all_sem_seg|/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/UDD5.json"
    "VDD|JPG|VDD_all_sem_seg|/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/VDD.json"
)

# 指定单卡运行使用的 GPU ID (默认使用 0 号卡)
GPU_ID=1

# ==========================================
# 2. 单卡串行推理
# ==========================================
for dataset_info in "${DATASETS[@]}"; do
    # 这里将第四个变量名改为 json_path 更贴切
    IFS='|' read -r vis_folder ext dataset_name json_path <<< "$dataset_info"

    INPUT_DIR="$SAMPLED_BASE/$vis_folder"
    OUTPUT_DIR="$OUTPUT_BASE/preds_${vis_folder}"

    if [ ! -d "$INPUT_DIR" ]; then
        echo "⚠️ 跳过：输入目录不存在 $INPUT_DIR"
        continue
    fi

    mkdir -p "$OUTPUT_DIR"

    echo "🔥 [GPU $GPU_ID] 启动 $dataset_name"

    # 前台串行运行
    CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="$(pwd)" python demo/demo.py \
        --config-file "$CONFIG" \
        --output "$OUTPUT_DIR" \
        --input "$INPUT_DIR/*.$ext" \
        --confidence-threshold 0.1 \
        --opts MODEL.WEIGHTS "$MODEL_WEIGHTS" \
             DATASETS.TEST "('${dataset_name}',)" \
             MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "$json_path" \
             MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON "$json_path" \
        > "$OUTPUT_DIR/log.txt" 2>&1

    echo "✅ $dataset_name 处理完成！"
    echo "--------------------------------------------------------"
done

echo ""
echo "🎉 所有数据集推理完成！"