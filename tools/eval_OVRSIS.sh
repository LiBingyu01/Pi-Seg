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

# iSAID_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/OVRSIS/eval-iSAID \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/iSAID.json" \
 DATASETS.TEST \(\"iSAID_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#DLRSD_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/OVRSIS/eval-DLRSD \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/DLRSD.json" \
 DATASETS.TEST \(\"DLRSD_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts


#Postdam_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/OVRSIS/eval-Potsdam \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/Potsdam.json" \
 DATASETS.TEST \(\"Potsdam_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#Vaihingen_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/OVRSIS/eval-Vaihingen \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/Vaihingen.json" \
 DATASETS.TEST \(\"Vaihingen_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# UDD5_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/OVRSIS/eval-UDD5 \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/UDD5.json" \
 DATASETS.TEST \(\"UDD5_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# LoveDA_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/OVRSIS/eval-LoveDA \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/LoveDA.json" \
 DATASETS.TEST \(\"LoveDA_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# uavid_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/OVRSIS/eval-uavid \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/uavid.json" \
 DATASETS.TEST \(\"uavid_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts
 
# VDD_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/OVRSIS/eval-VDD \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/VDD.json" \
 DATASETS.TEST \(\"VDD_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# FLAIR_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/OVRSIS/eval-flair \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/flair.json" \
 DATASETS.TEST \(\"FLAIR_test_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts
 
# OpenEarthMap_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/OVRSIS/eval-openearthmap \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/openearthmap.json" \
 DATASETS.TEST \(\"OpenEarthMap_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

cat $output/OVRSIS/eval-iSAID/log.txt | grep copypaste
cat $output/OVRSIS/eval-DLRSD/log.txt | grep copypaste
cat $output/OVRSIS/eval-Potsdam/log.txt | grep copypaste
cat $output/OVRSIS/eval-Vaihingen/log.txt | grep copypaste
cat $output/OVRSIS/eval-UDD5/log.txt | grep copypaste
cat $output/OVRSIS/eval-LoveDA/log.txt | grep copypaste
cat $output/OVRSIS/eval-uavid/log.txt | grep copypaste
cat $output/OVRSIS/eval-VDD/log.txt | grep copypaste
cat $output/OVRSIS/eval-flair/log.txt | grep copypaste
cat $output/OVRSIS/eval-OpenEarthMap/log.txt | grep copypaste