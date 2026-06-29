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

# WBS_SI_val
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/Flood_Detection/eval-WBS_SI_val \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/flood_detection.json" \
 DATASETS.TEST \(\"Flood_Detection_WBS_SI_val_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts


cat $output/Flood_Detection/eval-WBS_SI_val/log.txt | grep copypaste