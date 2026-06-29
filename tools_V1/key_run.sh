# =================================
# .sh file to train your own RSKT-Seg
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
  sh tools_V1/run.sh configs/vitb_384_DLRSD.yaml 4 output_vitb_384_DLRSD/${seed}/
  sh tools_V1/run.sh configs/vitb_384_iSAID.yaml 4 output_vitb_384_iSAID/${seed}/
  sh tools_V1/run.sh configs/vitl_336_DLRSD.yaml 4 output_vitl_336_DLRSD/${seed}/
  sh tools_V1/run.sh configs/vitl_336_iSAID.yaml 4 output_vitl_336_iSAID/${seed}/
done


# sh tools_V1/run.sh configs/vitb_384_DLRSD.yaml 4 output_vitb_384_DLRSD_2/
# sh tools_V1/run.sh configs/vitb_384_iSAID.yaml 4 output_vitb_384_iSAID_2/
# sh tools_V1/run.sh configs/vitl_336_DLRSD.yaml 4 output_vitl_336_DLRSD_2/
# sh tools/run.sh configs/vitl_336_iSAID.yaml 4 output_vitl_336_iSAID_2/