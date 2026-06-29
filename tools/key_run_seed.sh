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

for seed in 0 1 2 3 4 5 6 7 8 9
do
  sh tools/run.sh \
    configs/vitb_384_OVRSIS95K.yaml \
    4 \
    output_vitb_384_OVRSIS95K_time_${seed}
done
