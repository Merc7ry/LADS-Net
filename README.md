# LADS-Net

**低空场景下双重偏移自校正与加权引导监督的轻量化语义分割方法**

Official PyTorch implementation of **LADS-Net** for UAV / aerial semantic segmentation.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Encoder](https://img.shields.io/badge/Encoder-PVTv2--B2-0A66C2)

We present **LADS-Net**, a lightweight semantic segmentation network tailored for resource-constrained low-altitude UAV scenes. It combines a multi-scale attention decoder (**MSAIR** / **MSIR** + **GAG**), **DOSC** (Dual Offset Self-Correction) for input-adaptive class–feature matching, and **WGS** (Weighted Guided Supervision) for cross-scale training — removed at inference with zero extra cost.

![LADS-Net Architecture](doc/arch.png)

## 📢 News

- **2026-09-07:** Code of LADS-Net is released.



## 🖼️ Qualitative Results



### Cross-domain generalization (ISPRS)

把泛化可视化图放到：`doc/generalization.png`

![Cross-domain generalization on ISPRS Vaihingen / Potsdam](doc/generalization.png)

## 🛠️ Setup



### 💻 Dependencies

```bash
conda create -n ladsnet python=3.10 -y
conda activate ladsnet
pip install -r requirements.txt
```



### ⬇ Pretrained encoder

从 [PVTv2](https://github.com/whai362/PVT) 下载 **PVTv2-B2**，放到：

```text
pretrained_pth/pvt/pvt_v2_b2.pth
```



### 📂 Data layout

```text
data/<dataset>/
├── train/{images,masks}/
└── val/{images,masks}/
```



## 📊 Datasets & Splits

主实验用 5 个低空无人机数据集；ISPRS 用于跨平台泛化。


| Dataset              | `--dataset`      | #Classes | Default crop |
| -------------------- | ---------------- | -------- | ------------ |
| UAVid                | `uavid`          | 8        | 512×512      |
| VDD                  | `vdd`            | 7        | 400          |
| AeroScapes           | `aeroscapes`     | 12       | 512×512      |
| UDD5                 | `udd5`           | 5        | 512×512      |
| Semantic Drone (SDD) | `semantic_drone` | 24       | 512×512      |
| ISPRS Vaihingen      | `vaihingen`      | 6        | 256×256      |
| ISPRS Potsdam        | `potsdam`        | 6        | 256×256      |




## ⚙️ Experiment Configuration

除特别说明外，主实验采用统一协议。

### Hardware & framework


| Item         | Setting                          |
| ------------ | -------------------------------- |
| Framework    | PyTorch                          |
| GPU          | 1× NVIDIA L40S                   |
| Encoder init | ImageNet-pretrained **PVTv2-B2** |




### Optimization


| Item                     | Setting                   |
| ------------------------ | ------------------------- |
| Optimizer                | AdamW                     |
| Initial learning rate    | `1e-4`                    |
| Weight decay             | `1e-4`                    |
| LR schedule              | Cosine annealing → `1e-6` |
| Epochs                   | **150**                   |
| Batch size               | **4**                     |
| Input resolution (UAV)   | **512×512**               |
| Input resolution (ISPRS) | **256×256**               |




## 🏷️ Evaluating

```bash
python val.py \
  --data_dir [YOUR_DATA_DIR] \
  --dataset {dataset} \
  --model LADS_Net \
  --model_path {model weights}
```



## 🧪 Training

```bash
python train.py \
  --data_dir [YOUR_DATA_DIR] \
  --dataset {dataset} \
  --model LADS_Net \
  --encoder pvt_v2_b2 \
  --encoder_pretrain_path ./pretrained_pth/pvt/pvt_v2_b2.pth \
  --batch_size 4 \
  --epochs 150 \
  --lr 1e-4 \
  --optimizer adamw \
  --scheduler cosine \
  --input_w 512 \
  --input_h 512 \
  --name {dataset}_LADSNet
```

ISPRS（256×256）：

```bash
python train.py \
  --data_dir [YOUR_DATA_DIR] \
  --dataset {dataset} \
  --model LADS_Net \
  --encoder pvt_v2_b2 \
  --encoder_pretrain_path ./pretrained_pth/pvt/pvt_v2_b2.pth \
  --batch_size 4 \
  --epochs 150 \
  --input_w 256 \
  --input_h 256 \
  --name vaihingen_LADSNet
```

