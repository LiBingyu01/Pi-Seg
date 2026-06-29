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

# CHN6_CUG
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/Road_Extraction/eval-CHN6_CUG \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/road_extraction.json" \
 DATASETS.TEST \(\"Road_Extraction_CHN6_CUG_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# DeepGlobe
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/Road_Extraction/eval-DeepGlobe \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/road_extraction.json" \
 DATASETS.TEST \(\"Road_Extraction_DeepGlobe_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# Massachusetts
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/Road_Extraction/eval-Massachusetts \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/road_extraction.json" \
 DATASETS.TEST \(\"Road_Extraction_Massachusetts_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# SpaceNet
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/Road_Extraction/eval-SpaceNet \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/road_extraction.json" \
 DATASETS.TEST \(\"Road_Extraction_SpaceNet_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

cat $output/Road_Extraction/eval-CHN6_CUG/log.txt | grep copypaste
cat $output/Road_Extraction/eval-DeepGlobe/log.txt | grep copypaste
cat $output/Road_Extraction/eval-Massachusetts/log.txt | grep copypaste
cat $output/Road_Extraction/eval-SpaceNet/log.txt | grep copypaste