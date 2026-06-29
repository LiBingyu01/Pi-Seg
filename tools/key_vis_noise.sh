

python train_net.py \
  --config-file configs/vitb_384_OVRSIS95K.yaml \
  --num-gpus 1 \
  --dist-url "auto" \
  --eval-only \
  SOLVER.IMS_PER_BATCH 1 \
  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/OVRSIS95K.json" \
  DATASETS.TEST \(\"OVRSIS95K_val_sem_seg\"\,\) \
  TEST.SLIDING_WINDOW "False" \
  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
  MODEL.WEIGHTS output_vitb_384_OVRSIS95K/model_final.pth \
  OUTPUT_DIR output_vitb_384_OVRSIS95K \
  MODEL.PINI.ENABLED True \
  MODEL.PINI.IMAGE_VPN_ENABLED True \
  MODEL.PINI.TEXT_VPN_ENABLED True \
  MODEL.PINI.IS_VIS True
  
# python train_net.py --config $config \
#  --num-gpus $gpus \
#  --dist-url "auto" \
#  --eval-only \
#  OUTPUT_DIR $output/Road_Extraction/eval-CHN6_CUG \
#  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/road_extraction.json" \
#  DATASETS.TEST \(\"Road_Extraction_CHN6_CUG_sem_seg\"\,\) \
#  TEST.SLIDING_WINDOW "True" \
#  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
#  MODEL.WEIGHTS $output/model_final.pth \
#  $opts

# bash tools/run.sh  configs/vitb_384_OVRSIS95K.yaml 4 output/ablation_pini_agg/full_pini
