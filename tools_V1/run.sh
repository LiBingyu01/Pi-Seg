#!/bin/sh

config=$1
gpus=$2
output=$3


# CONFIG="configs/vitb_384_OVRSIS95K.yaml"
# GPUS=4
# BASE_OUTPUT="output/ablation_pini"

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

python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --resume \
 OUTPUT_DIR $output \
 $opts

sh tools_V1/eval_v1.sh $config $gpus $output $opts
sh tools_V1/eval_v1_slide.sh $config $gpus $output $opts