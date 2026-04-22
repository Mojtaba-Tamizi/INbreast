# Skin Lesion Segmentation on ISIC2018

This repository provides a **modular, experiment-friendly pipeline** for **skin lesion segmentation** on the **ISIC2018 Task 1** dataset.

The project is designed to support:

- **Patch-based training**
- **Sliding-window validation/inference**
- **Boundary supervision**
- **Config-driven experiments**
- **Ablation studies** on augmentation, loss design, and model variants
- Clean logging, checkpointing, evaluation, and visualization

The current implementation includes a full baseline pipeline with:

- dataset indexing
- boundary mask generation
- patch index generation
- data sanity checking
- patch/full-image datasets
- YAML-driven transforms
- tiling for full-image inference
- modular losses and metrics
- a baseline UNet
- training, validation, inference, and log plotting scripts

---

# 1. Project goal

The main task in this repository is:

**Skin lesion segmentation on ISIC2018**

Given a dermoscopic RGB image, the model predicts a **binary lesion mask**.  
In addition to the lesion mask, this repository also supports **boundary supervision**, because lesion contours are often difficult and clinically important.

The training/validation design is:

- **Training:** patch-based
- **Validation:** full-image sliding window
- **Inference:** full-image sliding window

This design was chosen because ISIC images can be too large to fit efficiently into GPU memory during end-to-end full-resolution training.

---

# 2. Dataset assumption

This repository assumes the ISIC2018 Task 1 dataset is arranged like this:

```bash
ISIC2018/
├── ISIC2018_Task1-2_Training_Input/
├── ISIC2018_Task1_Training_GroundTruth/
├── ISIC2018_Task1-2_Validation_Input/
├── ISIC2018_Task1_Validation_GroundTruth/
├── ISIC2018_Task1-2_Test_Input/
└── ISIC2018_Task1_Test_GroundTruth/
```

Mask file naming is assumed to follow the standard ISIC pattern:

```text
<image_id>_segmentation.png
```

Example:

```text
ISIC_0000000.jpg
ISIC_0000000_segmentation.png
```

---

# 3. Repository structure

```text
skin_lesion_segmentation/
├── configs/
│   ├── dataset.yaml
│   ├── train.yaml
│   └── model.yaml
│
├── data/
│   └── processed/
│       ├── indices/
│       ├── boundaries/
│       ├── patch_indices/
│       └── stats/
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── predictions/
│   └── visualizations/
│
├── scripts/
│   ├── prepare_data.py
│   ├── sanity_check_dataloader.py
│   ├── train.py
│   ├── validate.py
│   ├── infer.py
│   └── plot_logs.py
│
├── src/
│   ├── data/
│   │   ├── build_dataset_index.py
│   │   ├── build_boundary_masks.py
│   │   ├── build_patch_index.py
│   │   ├── validate_prepared_data.py
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── tiling.py
│   │
│   ├── engine/
│   │   ├── train.py
│   │   └── validate.py
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   └── losses.py
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   └── metrics.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   └── unet.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── mask_utils.py
│       └── patch_utils.py
│
├── requirements.txt
└── README.md
```

# 4. Typical workflow for a new experiment

A recommended experiment workflow is:

1. update configs
2. run `prepare_data.py`
3. run `sanity_check_dataloader.py`
4. run a smoke-test training
5. inspect saved outputs and plots
6. run full training
7. run `validate.py` on best checkpoint
8. run `infer.py` for final prediction outputs
9. plot logs and compare experiments


# 5. Common commands

## Prepare data
```bash
python scripts/prepare_data.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml
```

## Sanity check train loader
```bash
python scripts/sanity_check_dataloader.py --split train
```

## Sanity check validation loader
```bash
python scripts/sanity_check_dataloader.py --split val
```

## Train
```bash
python scripts/train.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml --model-config configs/model.yaml
```

## Validate checkpoint
```bash
python scripts/validate.py --checkpoint outputs/baseline_unet/20260422_123456/checkpoints/best.pth --split val
```

## Infer on test
```bash
python scripts/infer.py --checkpoint outputs/baseline_unet/20260422_123456/checkpoints/best.pth --split test
```

## Plot logs
```bash
python scripts/plot_logs.py --run-dir outputs/baseline_unet/20260422_123456
```

---

