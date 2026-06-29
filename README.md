<div align="center">

# Pi-Seg for OVRSISBenchV2
## Perturbation-Injected Open-Vocabulary Remote Sensing Segmentation

**Official implementation of Pi-Seg on CAT-Seg framework for OVRSISBenchV2**

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red?style=flat-square)](https://arxiv.org/html/2604.15652v1)
[![Benchmark](https://img.shields.io/badge/Benchmark-OVRSISBenchV2-blue?style=flat-square)](#Pi-Seg)
[![Dataset](https://img.shields.io/badge/Dataset-OVRSIS95K-green?style=flat-square)](#ovrsis95k)
[![Model](https://img.shields.io/badge/Model-Pi--Seg-orange?style=flat-square)](#pi-seg)


[Paper](https://arxiv.org/html/2604.15652v1)

[Hugging Face: OVRSIS95K](https://huggingface.co/datasets/kkk2026/OVRSIS95K) &nbsp;&nbsp;&nbsp;
[Hugging Face: OVRSISBenchV2_OVRSIS](https://huggingface.co/datasets/kkk2026/OVRSISBenchtest) &nbsp;&nbsp;&nbsp;
[Hugging Face: OVRSISBenchV2_other3task](https://huggingface.co/datasets/kkk2026/OVRSISBenchV2o3)

[Hugging Face: Pi-Seg weights for OVRSISBenchV1](https://huggingface.co/kkk2026/Pi-Seg_for_OVRSISBenchV1)&nbsp;&nbsp;&nbsp;
[Hugging Face: Pi-Seg weights for OVRSISBenchV2](https://huggingface.co/kkk2026/Pi-Seg_for_OVRSISBenchV2)

</div>

---

## Overview

This repository implements **Pi-Seg (Perturbation-Injected Segmentation)** on the CAT-Seg framework for open-vocabulary remote sensing image segmentation. Pi-Seg introduces a positive-incentive noise learning mechanism that improves model generalization to unseen categories and novel domains through semantically guided perturbations in both visual and textual feature spaces.

### Key Features

- **Lightweight Architecture**: Achieves strong performance without heavy multi-encoder transfer frameworks
- **Positive-Incentive Noise Injection**: Stochastic training strategy for smoother decision boundaries
- **High-Resolution Support**: Efficient design enables processing of large remote sensing images
- **Strong Generalization**: Robust performance on OVRSISBenchV1, OVRSISBenchV2, and downstream tasks

![](assets/pic_pi_seg_01.png)

---

## News

- **2026/05**: Released OVRSISBenchV2 pretrained-model evaluation scripts, including `eval_all_pretrained.sh` and `eval_all_pretrained_start.sh`.
- **2026/04**: Released Pi-Seg checkpoints for OVRSISBenchV1 and OVRSISBenchV2 on Hugging Face.
- **2026/04**: Released the official Pi-Seg implementation based on the CAT-Seg framework.
- **2026/03**: Released the OVRSIS95K training dataset on Hugging Face.
- **2026/03**: Released OVRSISBenchV2 benchmark resources, including the OVRSIS test set and three downstream-task evaluation sets.
---

## What is OVRSISBenchV2?

**OVRSISBenchV2** is a large-scale, multi-domain benchmark for open-vocabulary remote sensing segmentation that extends beyond standard semantic segmentation to practical downstream applications.

### Benchmark Highlights

- **170K+ annotated images** across diverse remote sensing scenes
- **128 semantic categories** covering satellite and UAV imagery
- **Training on OVRSIS95K**: 95K balanced image-mask pairs with 35 categories
- **Multi-task evaluation**: Standard OVRSIS + 3 downstream tasks
  - Building Extraction
  - Road Extraction
  - Flood Detection

![](assets/pic_fig_2_dataset_01.png)

### Why OVRSISBenchV2?

Remote sensing segmentation faces unique challenges:
- Large domain gaps from natural-image pretraining
- Arbitrary object orientations and severe scale variation
- Small targets with dense layouts
- Limited scene diversity and long-tail distributions

OVRSISBenchV2 addresses these by providing a realistic, application-oriented platform for evaluating semantic generalization, cross-domain transfer, and deployment robustness.

---

## OVRSIS95K Dataset

**OVRSIS95K** is the foundation training dataset for OVRSISBenchV2, featuring:

- **~95K image-mask pairs** with high-quality annotations
- **35 semantic categories** with balanced class distribution
- **5 scene domains**: town, industrial, forest, waterfront, wasteland
- **Semi-automated annotation pipeline** for scalability and quality

![](assets/pic_ovrsis95k_01.png)

### Dataset Access

- **OVRSIS95K**: [Hugging Face](https://huggingface.co/datasets/kkk2026/OVRSIS95K)
- **OVRSISBenchV2 (OVRSIS test)**: [Hugging Face](https://huggingface.co/datasets/kkk2026/OVRSISBenchtest)
- **OVRSISBenchV2 (3 downstream tasks)**: [Hugging Face](https://huggingface.co/datasets/kkk2026/OVRSISBenchV2o3)

---

## Pi-Seg: Perturbation-Injected Segmentation

Pi-Seg improves open-vocabulary segmentation through a novel training strategy that injects semantically guided perturbations into feature spaces.

### Core Innovation

Unlike methods relying on multiple pretrained encoders for domain adaptation, Pi-Seg learns a broader, more transferable feature space by:

1. **Visual Perturbation Network (VPN)**: Injects controlled noise into image features
2. **Text Perturbation Network (TPN)**: Adds semantic-aware perturbations to text embeddings
3. **Positive-Incentive Learning**: Encourages the model to maintain correct predictions under perturbations

### Advantages

- ✅ Lighter than multi-encoder frameworks
- ✅ Lower memory and computational cost
- ✅ Better high-resolution image support
- ✅ Stronger cross-domain generalization
- ✅ Improved robustness to unseen categories

---

## Installation

### Requirements

- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.13 and matching torchvision
- CUDA 11.7 or higher (for GPU training)
- Detectron2

### Step-by-step Installation

```bash
# Clone repository
git clone https://github.com/YourRepo/CAT-Seg-OVRSIS-PI.git
cd CAT-Seg-OVRSIS-PI

# Create conda environment
conda create -n piseg python=3.8
conda activate piseg

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt

# Install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

---

## Data Preparation

### Download Datasets

1. **OVRSIS95K** (training): Download from [Hugging Face](https://huggingface.co/datasets/kkk2026/OVRSIS95K)
2. **OVRSISBenchV2 testing** (evaluation): Download test sets from Hugging Face 
OVRSISBenchV2 (OVRSIS test): [Hugging Face](https://huggingface.co/datasets/kkk2026/OVRSISBenchtest). 
OVRSISBenchV2 (3 downstream tasks): [Hugging Face](https://huggingface.co/datasets/kkk2026/OVRSISBenchV2o3)


### Dataset Structure

Organize your datasets as follows:

```
datasets/
├── OVRSIS95K/
│   ├── train/
│   │   ├── images/
│   │   └── annotations/
│   └── val/
│       ├── images/
│       └── annotations/
├── OVRSISBench_test/
│   ├── OpenEarthMap/
│   ├── Potsdam/
│   ├── WHU_Building/
│   └── ...
└── OVRSISBenchV2_other3task/
    ├── Building_Extraction/
    ├── Road_Extraction/
    └── Flood_Detection/
```

Update dataset paths in config files (`configs/*.yaml`) accordingly.

---

## Training

We provide configuration files for different model variants:

- `vitb_384_OVRSIS95K.yaml`: ViT-B/16 backbone
- `vitl_336_OVRSIS95K.yaml`: ViT-L/14 backbone

### Training Commands

V1 training: in `tools_V1/key_run.sh`, `tools_V1/run.sh`

V2 training: in `tools/key_run.sh`, `tools/run.sh`


### root config

```bash
for every dataset in cat_seg/data/datasets

replace with your dataset path
root = "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/OVRSIS"
to replace with your dataset path like this:

datasets/
├── OVRSIS95K/
│   ├── train/
│   │   ├── images/
│   │   └── annotations/
│   └── val/
│       ├── images/
│       └── annotations/
├── OVRSISBench_test/
│   ├── OpenEarthMap/
│   ├── Potsdam/
│   ├── WHU_Building/
│   └── ...
└── OVRSISBenchV2_other3task/
    ├── Building_Extraction/
    ├── Road_Extraction/
    └── Flood_Detection/
    
```
### Multi-GPU Training on V2

For distributed training across multiple nodes:

```bash
# run main script for V2
bash tools/key_run.sh

# run abalation
bash tools/key_ablation_baseline_image_text.sh
bash tools/key_ablation_noise_parameter.sh
bash tools/key_ablation_parameter_l.sh
bash tools/key_ablation_parameter.sh
bash tools/key_ablation_pini_baseline_image_text_b.sh
bash tools/key_ablation_pini_baseline_image_text_l.sh
bash tools/key_ablation_student_t.sh
bash tools/key_run_seed_l.sh
bash tools/key_run_seed.sh
```

### Multi-GPU Training on V1

For distributed training across multiple nodes:

```bash
# run main script for V2
bash tools_V1/key_run.sh
```

---

## Evaluation

### Evaluate on OVRSISBenchV2

```bash
# Evaluate on all OVRSISBenchV2 datasets
bash tools/vis_all.sh

# Evaluate on individual tasks (in tools/run.sh)
sh tools/eval_OVRSIS.sh $config $gpus $output $opts
sh tools/eval_Building_Extraction.sh $config $gpus $output $opts
sh tools/eval_Flood_Detection.sh $config $gpus $output $opts
sh tools/eval_Road_Extraction.sh $config $gpus $output $opts

# Or evaluate on specific datasets
python train_net.py \
    --config-file configs/vitl_336_OVRSIS95K.yaml \
    --eval-only \
    MODEL.WEIGHTS output/piseg_vitl/model_final.pth \
    DATASETS.TEST '("OpenEarthMap_sem_seg",)'

# getting results in excel
python tools/get_results.py
```

### Evaluate on OVRSISBenchV1
```bash
# Evaluate on individual tasks (in tools_V1/run.sh)
sh tools_V1/eval_v1.sh $config $gpus $output $opts
sh tools_V1/eval_v1_slide.sh $config $gpus $output $opts

# getting results in excel
python tools_V1/get_results.py
```

### Supported Evaluation Datasets

**Standard OVRSIS:**
- OpenEarthMap
- Potsdam
- WHU Building Dataset
- WHU Satellite Dataset
- Inria Aerial
- xBD (Disaster Assessment)

**Downstream Tasks:**
- Building Extraction
- Road Extraction
- Flood Detection

---

## Visualization (V2)

Visualize model predictions on test images:

```bash
# Visualize predictions
bash tools/vis_all.sh

# visualize noise cost map
bash tools/key_vis_noise.sh

# Or run visualization script directly
python demo/demo.py \
    --config-file configs/vitl_336_OVRSIS95K.yaml \
    --input path/to/images/*.jpg \
    --output vis_output/ \
    --opts MODEL.WEIGHTS output/piseg_vitl/model_final.pth

# visualize the training process 
python tools/vis_delta.py
```

More visualization examples in `vis_examples`

---

## Model Zoo

We provide pretrained weights for Pi-Seg models trained on OVRSIS95K, DLRSD, iSAID:

[Hugging Face: Pi-Seg weights for OVRSISBenchV1](https://huggingface.co/kkk2026/Pi-Seg_for_OVRSISBenchV1)&nbsp;&nbsp;&nbsp;
[Hugging Face: Pi-Seg weights for OVRSISBenchV2](https://huggingface.co/kkk2026/Pi-Seg_for_OVRSISBenchV2)


*Note: Checkpoints are not same as those in the paper. (better performance)*

---

## Configuration

Key configuration options in `configs/vitl_336_OVRSIS95K.yaml`:

```yaml
MODEL:
  META_ARCHITECTURE: "CATSeg"
  SEM_SEG_HEAD:
    NUM_CLASSES: 35  # OVRSIS95K categories
    CLIP_PRETRAINED: "ViT-L/14@336px"
  PINI:  # Pi-Seg settings
    ENABLED: True
    IMAGE_VPN_ENABLED: True  # Visual perturbation
    TEXT_VPN_ENABLED: True   # Text perturbation
    TEXT_NOISE_STD: 0.02     # Noise strength

DATASETS:
  TRAIN: ("OVRSIS95K_train_sem_seg",)
  TEST: ("OVRSIS95K_val_sem_seg",)

SOLVER:
  BASE_LR: 0.0002
  MAX_ITER: 40000
  IMS_PER_BATCH: 8
```

---

## Project Structure

```
CAT-Seg-OVRSIS-PI/
├── cat_seg/              # Core model implementation
│   ├── clip/            # CLIP encoder with perturbation
│   ├── data/            # Dataset registration
│   └── modeling/        # Pi-Seg architecture
├── configs/             # Configuration files
├── demo/                # Visualization scripts
├── tools/               # Utility scripts
├── train_net.py         # Training entry point
└── requirements.txt     # Python dependencies
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2026exploring,
  title={Exploring efficient open-vocabulary segmentation in the remote sensing},
  author={Li, Bingyu and Dong, Haocheng and Zhang, Da and Zhao, Zhiyuan and Sun, Hao and Gao, Junyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={8},
  pages={5982--5991},
  year={2026}
}

@article{li2026towards,
  title={Towards Realistic Open-Vocabulary Remote Sensing Segmentation: Benchmark and Baseline},
  author={Li, Bingyu and Huo, Tao and Dong, Haocheng and Zhang, Da and Zhao, Zhiyuan and Gao, Junyu and Li, Xuelong},
  journal={arXiv preprint arXiv:2604.15652},
  year={2026}
}
```

---

## Acknowledgements

This project builds upon excellent prior work:

- [CAT-Seg](https://github.com/KU-CVLAB/CAT-Seg) - Cost aggregation framework
- [Detectron2](https://github.com/facebookresearch/detectron2) - Detection framework
- [CLIP](https://github.com/openai/CLIP) - Vision-language pretraining
- [OVRS](https://github.com/caoql98/OVRS) - Open-vocabulary RS baseline
- [GSNet](https://github.com/yecy749/GSNet) - Generalized segmentation

We thank the authors for their valuable contributions to the community.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

For questions, issues, or collaboration:

- **Email**: libingyu0205@mail.ustc.edu.cn
- **Issues**: Please open an issue on GitHub

We welcome contributions and feedback from the community!

---


## install tips

```
detectron2 installation - No module named 'torch'

solutions:
https://www.bing.com/search?q=no%20module%20named%20torch%20%E5%AE%89%E8%A3%85detectron2&qs=n&form=QBRE&sp=-1&lq=0&pq=no%20module%20named%20torch%20%E5%AE%89%E8%A3%85detectron2&sc=2-34&sk=&cvid=6C7F8A032616473185B41BBD6A95B6D4


rm -rf build/ **/*.so
python setup.py build develop
```

<div align="center">

**Star this repo if you find it useful! ⭐**

</div>
