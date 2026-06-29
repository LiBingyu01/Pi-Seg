# !/bin/sh

config=$1
gpus=$2
output=$3

if [ -z $config ]
then
    echo "No config file found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

shift 3
opts=${@}

# Inria
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/Building_Extraction/eval-Inria \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/building_extraction.json" \
 DATASETS.TEST \(\"Building_Extraction_Inria_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# WHU_BD
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/Building_Extraction/eval-WHU_BD \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/building_extraction.json" \
 DATASETS.TEST \(\"Building_Extraction_WHU_BD_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# WHU_SAT
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/Building_Extraction/eval-WHU_SAT \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/building_extraction.json" \
 DATASETS.TEST \(\"Building_Extraction_WHU_SAT_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# xBD
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/Building_Extraction/eval-xBD \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/building_extraction.json" \
 DATASETS.TEST \(\"Building_Extraction_xBD_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

cat $output/Building_Extraction/eval-Inria/log.txt | grep copypaste
cat $output/Building_Extraction/eval-WHU_BD/log.txt | grep copypaste
cat $output/Building_Extraction/eval-WHU_SAT/log.txt | grep copypaste
cat $output/Building_Extraction/eval-xBD/log.txt | grep copypaste