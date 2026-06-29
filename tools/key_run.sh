# =================================
# .sh file to train your own Pi-Seg
# =================================

# =================================
# python train_net.py --config $config \
#  --num-gpus $gpus \
#  --dist-url "auto" \
#  --resume \
#  OUTPUT_DIR $output \
#  $opts
# sh eval.sh $config $gpus $output $opts
# =================================

# =================================
# Train 
# =================================

sh tools/run.sh configs/vitb_384_OVRSIS95K.yaml 4 output_vitb_384_OVRSIS95K/
sh tools/run.sh configs/vitl_336_OVRSIS95K.yaml 4 output_vitl_336_OVRSIS95K/