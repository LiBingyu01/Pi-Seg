# ==================================================================================
#!/bin/bash
CONFIG="configs/vitb_384_OVRSIS95K.yaml"
GPUS=4
BASE_OUTPUT="output/ablation_pini_baseline_image_text_b"

echo "============================================"
echo "PiNI Ablation Experiments"
echo "============================================"

# Exp1: Full PiNI (Image + Text VPN)
EXP_NAME="full_pini"
echo ""
echo "[1/4] Running: ${EXP_NAME}"
python train_net.py --config-file $CONFIG --num-gpus $GPUS \
  OUTPUT_DIR ${BASE_OUTPUT}/${EXP_NAME} \
  MODEL.PINI.ENABLED True \
  MODEL.PINI.IMAGE_VPN_ENABLED True \
  MODEL.PINI.TEXT_VPN_ENABLED True \
  2>&1 | tee ${BASE_OUTPUT}/${EXP_NAME}/train_log.txt

# Exp2: Image VPN only
EXP_NAME="image_vpn_only"
echo ""
echo "[2/4] Running: ${EXP_NAME}"
python train_net.py --config-file $CONFIG --num-gpus $GPUS \
  OUTPUT_DIR ${BASE_OUTPUT}/${EXP_NAME} \
  MODEL.PINI.ENABLED True \
  MODEL.PINI.IMAGE_VPN_ENABLED True \
  MODEL.PINI.TEXT_VPN_ENABLED False \
  2>&1 | tee ${BASE_OUTPUT}/${EXP_NAME}/train_log.txt

# Exp3: Text VPN only
EXP_NAME="text_vpn_only"
echo ""
echo "[3/4] Running: ${EXP_NAME}"
python train_net.py --config-file $CONFIG --num-gpus $GPUS \
  OUTPUT_DIR ${BASE_OUTPUT}/${EXP_NAME} \
  MODEL.PINI.ENABLED True \
  MODEL.PINI.IMAGE_VPN_ENABLED False \
  MODEL.PINI.TEXT_VPN_ENABLED True \
  2>&1 | tee ${BASE_OUTPUT}/${EXP_NAME}/train_log.txt

# Exp4: Baseline (PiNI off)
EXP_NAME="baseline_no_pini"
echo ""
echo "[4/4] Running: ${EXP_NAME}"
python train_net.py --config-file $CONFIG --num-gpus $GPUS \
  OUTPUT_DIR ${BASE_OUTPUT}/${EXP_NAME} \
  MODEL.PINI.ENABLED False \
  2>&1 | tee ${BASE_OUTPUT}/${EXP_NAME}/train_log.txt

# Summary
echo ""
echo "============================================"
echo "All experiments done. Results:"
echo "============================================"
echo "--- [1] Full PiNI ---"
cat ${BASE_OUTPUT}/full_pini/train_log.txt | grep copypaste | tail -1
echo "--- [2] Image VPN only ---"
cat ${BASE_OUTPUT}/image_vpn_only/train_log.txt | grep copypaste | tail -1
echo "--- [3] Text VPN only ---"
cat ${BASE_OUTPUT}/text_vpn_only/train_log.txt | grep copypaste | tail -1
echo "--- [4] Baseline (no PiNI) ---"
cat ${BASE_OUTPUT}/baseline_no_pini/train_log.txt | grep copypaste | tail -1
