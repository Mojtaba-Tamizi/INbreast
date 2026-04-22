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

# 2. Main design ideas

This repository follows a few important principles:

## 2.1 Patch-based training
Instead of training directly on full images, we:

- generate a **patch index**
- keep patch metadata in CSV
- crop image/mask/boundary on the fly during training

This is much more flexible than saving patch images to disk.

## 2.2 Sliding-window validation and inference
Validation and test prediction are done on **full images** using sliding-window tiling with overlap and reconstruction.

This ensures:

- full-image evaluation
- better consistency with final deployment-style prediction
- less bias than patch-only validation

## 2.3 Boundary-aware learning
For each lesion mask, a corresponding **boundary mask** is generated and stored.  
This allows the model to learn:

- lesion region
- lesion contour

## 2.4 Config-driven experimentation
Most important settings are controlled from YAML:

- dataset paths
- patch size / stride
- augmentation mode
- sampling ratios
- loss terms and weights
- metric selection
- optimizer / scheduler
- model type and model params

This is intentionally designed to make **ablation studies** easier.

---

# 3. Dataset assumption

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

# 4. Repository structure

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

---

# 5. What each directory/file does

## 5.1 `configs/`
All experiment settings live here.

### `configs/dataset.yaml`
Controls:

- raw dataset path
- processed output paths
- split folder names
- image/mask extensions
- boundary generation settings

Important: the raw dataset can live **outside the repo**, for example on another drive.

### `configs/train.yaml`
Controls:

- batch size
- workers
- patch size / stride
- patch sampling ratios
- transforms
- normalization
- training hyperparameters
- optimizer
- scheduler
- validation tiling settings
- inference save options
- checkpointing / logging

### `configs/model.yaml`
Controls:

- model name
- model parameters
- auxiliary heads such as boundary head

---

## 5.2 `data/processed/`
This contains all files generated from the raw dataset.

### `data/processed/indices/`
Stores CSV tables that match images and masks:

- `train_index.csv`
- `val_index.csv`
- `test_index.csv`
- `all_index.csv`

These files contain image paths, mask paths, and boundary paths.

### `data/processed/boundaries/`
Stores generated boundary masks for each split.

### `data/processed/patch_indices/`
Stores CSV files describing patches.  
These files contain metadata only, not patch images.

### `data/processed/stats/`
Stores patch statistics such as:

- number of total patches
- number of foreground/background patches
- number of boundary patches

---

## 5.3 `src/data/`
Core data pipeline code.

### `build_dataset_index.py`
Builds image-mask pairing CSV files.

### `build_boundary_masks.py`
Generates a binary lesion boundary mask from each lesion mask.

### `build_patch_index.py`
Creates sliding-window patch metadata for training.

### `validate_prepared_data.py`
Runs consistency checks on all generated files.

### `dataset.py`
Defines the PyTorch datasets:

- patch dataset for training
- full-image dataset for validation/inference

### `transforms.py`
Builds augmentation pipelines from YAML.

### `tiling.py`
Handles sliding-window tiling and reconstruction for full-image prediction.

---

## 5.4 `src/losses/`
Loss system.

### `losses.py`
Defines atomic losses, currently including:

- BCE
- Dice

### `builder.py`
Builds a composite loss from YAML config and applies per-loss weights.

---

## 5.5 `src/metrics/`
Metric system.

### `metrics.py`
Defines segmentation metrics, currently including:

- Dice
- IoU
- Sensitivity
- Specificity
- HD95

### `builder.py`
Builds metric functions from YAML and aggregates them over an epoch.

---

## 5.6 `src/models/`
Model system.

### `unet.py`
Baseline UNet implementation.

### `builder.py`
Loads a model from config by name.

The system is designed so new models can be added later and selected from YAML.

---

## 5.7 `src/engine/`
Training/validation logic.

### `train.py`
Contains the patch-based training loop.

### `validate.py`
Contains full-image validation logic using sliding-window inference.

---

## 5.8 `scripts/`
User-facing executable scripts.

### `prepare_data.py`
Runs the full preprocessing pipeline:

- build image-mask indices
- generate boundary masks
- build patch indices
- validate processed data

### `sanity_check_dataloader.py`
Loads datasets/dataloaders and saves visual samples to confirm:

- transforms are correct
- masks align with images
- boundary masks are aligned
- patch sampling behaves as expected

### `train.py`
Runs full model training.

### `validate.py`
Evaluates a saved checkpoint on val/test set.

### `infer.py`
Runs prediction and saves masks, probability maps, and overlays.

### `plot_logs.py`
Plots training curves from `history.csv`.

---

## 5.9 `outputs/`
Stores all experiment outputs.

Typical structure:

```text
outputs/<experiment_name>/<timestamp>/
├── checkpoints/
├── configs/
├── logs/
└── plots/
```

Inference and evaluation outputs are also stored inside `outputs/`.

---

# 6. Installation

## 6.1 Create environment
You can use either `venv`, `conda`, or your preferred Python environment manager.

Example with conda:

```bash
conda create -n isic_seg python=3.10 -y
conda activate isic_seg
```

## 6.2 Install dependencies

```bash
pip install -r requirements.txt
```

Typical packages used in this repo:

- torch
- torchvision
- numpy
- pandas
- PyYAML
- opencv-python
- albumentations
- scipy
- matplotlib

If OpenCV installation fails in your setup, you may also install with conda:

```bash
conda install -c conda-forge opencv
```

---

# 7. Configuration overview

Before running anything, check these three files carefully:

- `configs/dataset.yaml`
- `configs/train.yaml`
- `configs/model.yaml`

## 7.1 `dataset.yaml`
Most important field:

```yaml
paths:
  raw_root: "F:/Datasets/Segmentation/ISIC2018"
```

This must point to your raw ISIC2018 dataset location.

This repo intentionally supports raw data being on a **different drive** than the code.

---

## 7.2 `train.yaml`
Important sections:

- `patching`
- `sampling`
- `transforms`
- `loss`
- `metrics`
- `training`
- `optimizer`
- `scheduler`
- `validation`
- `inference`
- `checkpoint`

Examples of things controlled from here:

- patch size and stride
- whether to downsample background patches
- augmentation mode
- normalization mode
- loss weights
- metric thresholds
- training epochs and AMP usage

---

## 7.3 `model.yaml`
Important fields:

- model name
- number of channels
- UNet channel sizes
- whether the boundary head is enabled

---

# 8. Full step-by-step user guide

This section is the recommended workflow for a new user.

---

## Step 1 — Clone the repository

```bash
git clone <your_repo_url>
cd skin_lesion_segmentation
```

---

## Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3 — Set dataset path

Open:

```text
configs/dataset.yaml
```

and update:

```yaml
paths:
  raw_root: "F:/Datasets/Segmentation/ISIC2018"
```

to your real dataset location.

Do not copy the raw dataset into the repository if you do not have enough SSD space.  
This repository is designed so the raw data can live anywhere.

---

## Step 4 — Review core experiment settings

Before preparing data, review:

### `configs/train.yaml`
Check at least:

- patch size
- stride
- sampling ratios
- transform mode
- normalization mode

### `configs/model.yaml`
Check at least:

- model name
- UNet channels
- whether boundary head is enabled

---

## Step 5 — Prepare the data

Run:

```bash
python scripts/prepare_data.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml
```

This script performs the following pipeline:

1. builds train/val/test image-mask indices
2. generates boundary masks
3. generates patch index CSV for selected splits
4. validates all generated files

After this step, you should have:

```text
data/processed/indices/
data/processed/boundaries/
data/processed/patch_indices/
data/processed/stats/
```

---

## Step 6 — Check that preparation succeeded

Look at the console output from `prepare_data.py`.  
It should show successful messages for:

- split indices
- boundary masks
- patch CSV generation
- sanity checks

Also check these files manually:

```text
data/processed/indices/train_index.csv
data/processed/patch_indices/train_patches.csv
data/processed/stats/patch_stats_train.json
```

If these do not exist, do not continue to training yet.

---

## Step 7 — Run dataloader sanity check

This is highly recommended before training.

### Check training patches

```bash
python scripts/sanity_check_dataloader.py --split train
```

### Check validation images

```bash
python scripts/sanity_check_dataloader.py --split val
```

The script saves visual panels that show:

- image
- lesion mask
- boundary mask
- overlay

These are saved under:

```text
outputs/visualizations/sanity_check/
```

Inspect these images manually and confirm:

- masks align with lesions
- boundaries align with lesion edges
- augmentations are reasonable
- nothing is rotated/flipped incorrectly
- normalization is not corrupting the image

Do not skip this step.

---

## Step 8 — Decide augmentation setting for your experiment

This repository supports three augmentation modes:

- `none`
- `general`
- `general_artifact_aware`

You select them in `configs/train.yaml`.

Example:

```yaml
transforms:
  train_mode: "general"
```

Recommended ablation sequence:

1. `none`
2. `general`
3. `general_artifact_aware`

Validation and test should usually remain:

```yaml
val_mode: "none"
test_mode: "none"
```

---

## Step 9 — Decide normalization setting

Supported normalization modes:

- `none`
- `imagenet`
- `dataset`

Recommendation for early experiments:

```yaml
normalization:
  mode: "none"
```

If you later use a pretrained encoder, you may want:

```yaml
normalization:
  mode: "imagenet"
```

If using `dataset`, you must provide dataset-specific mean/std in the config.

---

## Step 10 — Start with a smoke test

Before full training, reduce the experiment size in `configs/train.yaml`:

- set a small number of epochs
- keep batch size moderate
- keep validation enabled

Then run:

```bash
python scripts/train.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml --model-config configs/model.yaml
```

The goal of the smoke test is only to confirm:

- training starts
- validation starts
- checkpoints are saved
- logs are written
- no tensor/key mismatch exists

---

## Step 11 — Inspect training outputs

A training run creates a folder like:

```text
outputs/baseline_unet/20260422_123456/
```

Inside it you will find:

### `checkpoints/`
- `last.pth`
- `best.pth`
- optional epoch checkpoints

### `logs/`
- `history.csv`
- `summary.json`

### `configs/`
A snapshot of the exact config files used for this run.

This is important for reproducibility.

---

## Step 12 — Plot training curves

After training, plot logs with:

```bash
python scripts/plot_logs.py --run-dir outputs/baseline_unet/20260422_123456
```

This creates figures such as:

- loss curves
- dice curves
- IoU curves
- sensitivity/specificity curves
- learning-rate plot

These plots are saved under:

```text
outputs/baseline_unet/20260422_123456/plots/
```

---

## Step 13 — Evaluate a saved checkpoint

To validate a trained checkpoint again later, run:

```bash
python scripts/validate.py --checkpoint outputs/baseline_unet/20260422_123456/checkpoints/best.pth --split val
```

or on the test split:

```bash
python scripts/validate.py --checkpoint outputs/baseline_unet/20260422_123456/checkpoints/best.pth --split test
```

This is useful when:

- comparing multiple checkpoints
- re-evaluating after changing threshold settings
- generating final evaluation logs separately from training

Evaluation results are saved under:

```text
outputs/<experiment_name>/evaluations/
```

---

## Step 14 — Run inference

To generate prediction outputs without evaluation logic, run:

```bash
python scripts/infer.py --checkpoint outputs/baseline_unet/20260422_123456/checkpoints/best.pth --split test
```

This saves:

- probability maps
- binary masks
- overlays
- optional boundary outputs

Inference outputs are saved under:

```text
outputs/<experiment_name>/inference/
```

---

# 9. Sanity checks you should always do

Before serious training, the following checks are strongly recommended.

## 9.1 Dataset indexing sanity
Confirm:

- every row in `train_index.csv` has a real image path
- every image has a matching mask
- every boundary file exists

## 9.2 Patch sanity
Confirm:

- patch coordinates are valid
- patch sizes are correct
- fg/bg/boundary counts make sense
- `patch_type` looks reasonable

The repository already checks these automatically during data preparation.

## 9.3 Visual sanity
Use `sanity_check_dataloader.py` and inspect saved visual samples manually.

Look for:

- mask/image misalignment
- wrong color/normalization
- bad augmentations
- empty or broken boundary masks

## 9.4 Tiny overfit test
Before long training, it is a good idea to test if the model can overfit a tiny subset.

If the model cannot overfit 1–4 samples, something is likely wrong in:

- labels
- loss
- transforms
- model output keys
- training loop

---

# 10. Training pipeline summary

The full pipeline is:

## Data preparation
1. raw dataset
2. image-mask indexing
3. boundary generation
4. patch index generation
5. processed data validation

## Training
1. patch CSV is loaded
2. patches are sampled using configured ratios
3. image/mask/boundary are cropped on-the-fly
4. train transforms are applied
5. model predicts mask and optionally boundary
6. composite loss is computed
7. metrics are logged

## Validation
1. full image is loaded
2. sliding-window patches are extracted
3. tiled predictions are stitched together
4. full-image metrics are computed

## Inference
1. full image is loaded
2. sliding-window predictions are stitched
3. final outputs are saved

---

# 11. Loss design

The loss system is modular and YAML-driven.

Current planned baseline loss:

- Dice loss on lesion mask
- BCE loss on lesion mask
- BCE loss on boundary map

Each term has:

- `name`
- `weight`
- `pred_key`
- `target_key`
- `params`

This allows easy future extension.

Example:

```yaml
loss:
  terms:
    - name: "bce"
      weight: 1.0
      pred_key: "mask"
      target_key: "mask"
      params:
        from_logits: true

    - name: "dice"
      weight: 1.0
      pred_key: "mask"
      target_key: "mask"
      params:
        from_logits: true
        smooth: 1.0

    - name: "bce"
      weight: 0.5
      pred_key: "boundary"
      target_key: "boundary"
      params:
        from_logits: true
```

---

# 12. Metrics

Current metrics include:

- Dice
- IoU
- Sensitivity
- Specificity
- HD95

These are also config-driven.

Important note for HD95:

- if both masks are empty, it is set to `0.0`
- if only one is empty, it becomes `NaN` by default and is ignored in mean aggregation

---

# 13. Model system

The repository currently includes a **baseline UNet**.

The model system is registry-based so new models can be added later without changing the rest of the pipeline.

Expected model output format:

```python
{
    "mask": mask_logits,
    "boundary": boundary_logits,
}
```

If only one output is present, it is treated as:

```python
{"mask": output}
```

The boundary loss and boundary inference logic will work only if the model provides a boundary output.

---

# 14. Typical workflow for a new experiment

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

---

# 15. Suggested ablation studies

This repository was intentionally designed for ablation experiments.  
Good first ablations include:

## 15.1 Transform ablation
- no augmentation
- general augmentation
- general + artifact-aware augmentation

## 15.2 Boundary supervision ablation
- mask-only loss
- mask + boundary loss

## 15.3 Sampling ablation
- full patch set
- downsampled background
- oversampled boundary-rich patches

## 15.4 Model ablation
- baseline UNet
- future improved architectures

## 15.5 Normalization ablation
- no normalization
- ImageNet normalization
- dataset-specific normalization

---

# 16. Common commands

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

# 17. Troubleshooting

## Problem: `ModuleNotFoundError: No module named 'src'`
Some scripts require the repository root to be on Python path.  
Use the current versions of the scripts in this repo, which already handle this.

---

## Problem: `ModuleNotFoundError: No module named 'cv2'`
Install OpenCV:

```bash
pip install opencv-python
```

or:

```bash
conda install -c conda-forge opencv
```

---

## Problem: dataloader sanity images look wrong
Check:

- transform mode
- normalization mode
- mask thresholding
- whether mask and boundary align
- whether augmentation is too strong

---

## Problem: training starts but metrics do not improve
Check:

- tiny overfit test
- patch sampling ratios
- whether too many background patches remain
- whether boundary loss is too strong
- whether learning rate is reasonable

---

## Problem: boundary loss crashes
Check that the model really returns:

```python
{
    "mask": ...,
    "boundary": ...
}
```

If the model only returns `"mask"`, any boundary loss term must be disabled.

---

## Problem: HD95 returns NaN
This can happen when one mask is empty and the other is not empty.  
That behavior is intentional by default.

You can change it in metric config using `one_empty_value`.

---

# 18. Reproducibility notes

To make experiments reproducible:

- keep config files versioned
- never overwrite old outputs
- keep the copied config snapshot inside each run folder
- use fixed seeds
- record model name and run timestamp
- compare runs using their saved `history.csv` and evaluation logs

---

# 19. Current status of the repository

At the current stage, this repo provides a **complete baseline end-to-end training pipeline** for ISIC2018 skin lesion segmentation.

What is already ready:

- data preparation
- patch generation
- sanity checks
- transform system
- training/validation/inference
- baseline UNet
- plotting and experiment logging

What can be improved later:

- stronger model architectures
- richer boundary losses
- additional augmentations
- dataset mean/std computation
- threshold search
- test-time augmentation
- more advanced stitching/blending

---

# 20. Final note

This repository is built not just to run one experiment, but to support a **research workflow**:

- reliable preprocessing
- clear sanity checks
- reproducible configs
- modular experimentation
- clean evaluation

If you are a new user, do not jump directly into long training.  
Follow the step-by-step order in this README, especially:

1. prepare data
2. run sanity checks
3. run a small smoke test
4. inspect outputs
5. then begin full training
