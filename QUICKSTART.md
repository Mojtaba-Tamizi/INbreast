# QUICKSTART.md

# Day-to-day usage guide

This file is the **practical run-order guide** for the repository.

Use this when you want to know:

- what to run
- in what order
- which steps are mandatory
- which steps are optional
- what a normal experiment workflow looks like

This is intentionally shorter and more operational than the main `README.md`.

---

# 1. One-time setup

## 1.1 Install dependencies

```bash
pip install -r requirements.txt
```

## 1.2 Set your dataset path

Open:

```text
configs/dataset.yaml
```

and update:

```yaml
paths:
  raw_root: "F:/Datasets/Segmentation/ISIC2018"
```

to the real location of your ISIC2018 raw dataset.

Important:
- the raw dataset can live on another drive
- it does **not** need to be inside the repo

---

# 2. Normal workflow order

For normal usage, run the project in this order:

1. configure paths and experiment settings
2. prepare data
3. run sanity checks
4. train
5. plot logs
6. validate best checkpoint
7. run inference if needed

That is the standard order.

---

# 3. Step-by-step commands

## Step 1 — Check configs

Before running anything, review these files:

```text
configs/dataset.yaml
configs/train.yaml
configs/model.yaml
```

Main things to check:

### `dataset.yaml`
- raw dataset path
- processed data paths

### `train.yaml`
- patch size / stride
- sampling ratios
- transform mode
- normalization mode
- epochs
- batch size
- optimizer / scheduler
- checkpoint monitor metric

### `model.yaml`
- model name
- UNet channels
- whether boundary head is enabled

---

## Step 2 — Prepare processed data

Run:

```bash
python scripts/prepare_data.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml
```

This will:

- build image/mask index CSVs
- generate boundary masks
- generate patch CSVs
- validate the processed data

You usually run this:
- once at the beginning
- again if you change patch settings
- again if you change boundary generation settings

---

## Step 3 — Run sanity checks

### 3.1 Check training patches

```bash
python scripts/sanity_check_dataloader.py --split train
```

### 3.2 Check validation full images

```bash
python scripts/sanity_check_dataloader.py --split val
```

This step is strongly recommended before training.

You should inspect the saved images and confirm:

- image is correct
- lesion mask aligns correctly
- boundary mask aligns correctly
- augmentations look reasonable
- normalization did not corrupt the image

Saved outputs are usually under:

```text
outputs/visualizations/sanity_check/
```

---

## Step 4 — Train the model

Run:

```bash
python scripts/train.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml --model-config configs/model.yaml
```

This will:

- build train and validation dataloaders
- build model, loss, metrics, optimizer, scheduler
- train for the configured number of epochs
- validate during training
- save checkpoints
- save logs

A training run will create a folder like:

```text
outputs/<experiment_name>/<timestamp>/
```

Inside that run folder, the most important files are:

```text
checkpoints/best.pth
checkpoints/last.pth
logs/history.csv
logs/summary.json
```

---

## Step 5 — Plot training curves

After training, plot the logs:

```bash
python scripts/plot_logs.py --run-dir outputs/<experiment_name>/<timestamp>
```

Example:

```bash
python scripts/plot_logs.py --run-dir outputs/baseline_unet/20260422_123456
```

This will create plots such as:

- loss
- learning rate
- Dice
- IoU
- Sensitivity
- Specificity
- HD95

---

## Step 6 — Validate a saved checkpoint

To evaluate a trained checkpoint again, run:

```bash
python scripts/validate.py --checkpoint outputs/<experiment_name>/<timestamp>/checkpoints/best.pth --split val
```

Example:

```bash
python scripts/validate.py --checkpoint outputs/baseline_unet/20260422_123456/checkpoints/best.pth --split val
```

Use this when you want to:

- re-evaluate the best checkpoint
- compare multiple checkpoints
- run validation separately from training
- save clean evaluation logs

You can also validate on test if you have ground-truth labels available:

```bash
python scripts/validate.py --checkpoint outputs/baseline_unet/20260422_123456/checkpoints/best.pth --split test
```

---

## Step 7 — Run inference

To generate predictions only, run:

```bash
python scripts/infer.py --checkpoint outputs/<experiment_name>/<timestamp>/checkpoints/best.pth --split test
```

Example:

```bash
python scripts/infer.py --checkpoint outputs/baseline_unet/20260422_123456/checkpoints/best.pth --split test
```

This saves:

- probability maps
- binary masks
- overlays
- optional boundary predictions

Use inference when you want outputs, not evaluation.

---

# 4. What to run for a brand new experiment

This is the recommended order for a fresh experiment:

## 4.1 Configure the experiment
Edit:

```text
configs/train.yaml
configs/model.yaml
```

Typical things you may change:
- augmentation mode
- loss weights
- sampling ratios
- epochs
- learning rate
- model channels

## 4.2 Prepare data
```bash
python scripts/prepare_data.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml
```

## 4.3 Run dataloader sanity checks
```bash
python scripts/sanity_check_dataloader.py --split train
python scripts/sanity_check_dataloader.py --split val
```

## 4.4 Run a smoke test training
Use a small epoch count first, then:

```bash
python scripts/train.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml --model-config configs/model.yaml
```

## 4.5 Inspect logs and plots
```bash
python scripts/plot_logs.py --run-dir outputs/<experiment_name>/<timestamp>
```

## 4.6 Validate best checkpoint
```bash
python scripts/validate.py --checkpoint outputs/<experiment_name>/<timestamp>/checkpoints/best.pth --split val
```

## 4.7 Run inference if needed
```bash
python scripts/infer.py --checkpoint outputs/<experiment_name>/<timestamp>/checkpoints/best.pth --split test
```

---

# 5. What to run for daily work

For day-to-day usage, you usually do **not** run everything every time.

## Case A — only changed model or training settings
If you changed:
- model config
- loss config
- optimizer
- scheduler
- transforms
- training epochs

then usually run only:

```bash
python scripts/train.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml --model-config configs/model.yaml
```

Then:

```bash
python scripts/plot_logs.py --run-dir outputs/<experiment_name>/<timestamp>
python scripts/validate.py --checkpoint outputs/<experiment_name>/<timestamp>/checkpoints/best.pth --split val
```

You do **not** need to re-run `prepare_data.py` unless data-processing settings changed.

---

## Case B — changed patch size or stride
If you changed:
- patch size
- stride
- build splits

then re-run:

```bash
python scripts/prepare_data.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml
```

because the patch index must be rebuilt.

Then run sanity checks again:

```bash
python scripts/sanity_check_dataloader.py --split train
```

Then train again.

---

## Case C — changed boundary generation settings
If you changed:
- boundary mode
- boundary kernel size
- boundary iterations

then re-run:

```bash
python scripts/prepare_data.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml
```

because boundary masks must be regenerated.

Then run sanity checks again.

---

## Case D — only want to evaluate an existing checkpoint
Run only:

```bash
python scripts/validate.py --checkpoint <path_to_checkpoint> --split val
```

---

## Case E — only want predictions
Run only:

```bash
python scripts/infer.py --checkpoint <path_to_checkpoint> --split test
```

---

# 6. Recommended quick command checklist

## First time
```bash
pip install -r requirements.txt
python scripts/prepare_data.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml
python scripts/sanity_check_dataloader.py --split train
python scripts/sanity_check_dataloader.py --split val
python scripts/train.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml --model-config configs/model.yaml
python scripts/plot_logs.py --run-dir outputs/<experiment_name>/<timestamp>
python scripts/validate.py --checkpoint outputs/<experiment_name>/<timestamp>/checkpoints/best.pth --split val
```

## Later, for a new experiment with same processed data
```bash
python scripts/train.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml --model-config configs/model.yaml
python scripts/plot_logs.py --run-dir outputs/<experiment_name>/<timestamp>
python scripts/validate.py --checkpoint outputs/<experiment_name>/<timestamp>/checkpoints/best.pth --split val
```

## Later, for prediction only
```bash
python scripts/infer.py --checkpoint outputs/<experiment_name>/<timestamp>/checkpoints/best.pth --split test
```

---

# 7. When you must re-run `prepare_data.py`

You should re-run data preparation if you changed any of these:

- raw dataset path
- boundary generation settings
- patch size
- stride
- which splits to build patches for

You usually do **not** need to re-run it if you changed only:

- model architecture
- optimizer
- scheduler
- loss weights
- training epochs
- augmentation probabilities
- checkpoint settings

---

# 8. Recommended experiment order for this project

A good practical sequence is:

## Experiment 1
- baseline UNet
- no augmentation
- simple loss

## Experiment 2
- general augmentation

## Experiment 3
- general + artifact-aware augmentation

## Experiment 4
- tune loss weights
- tune boundary weight
- tune patch sampling ratios

## Experiment 5
- stronger model design

This repo is built so those changes mainly happen in YAML.

---

# 9. Most important files to check after training

After training, always inspect:

## 9.1 `logs/history.csv`
Used for:
- plotting curves
- comparing experiments
- checking loss/metric behavior

## 9.2 `logs/summary.json`
Used for:
- quick run summary
- best score info
- best checkpoint path

## 9.3 `checkpoints/best.pth`
Used for:
- validation
- inference
- future comparisons

---

# 10. Very short daily workflow

If you just want the shortest practical sequence, use this:

## Data prep
```bash
python scripts/prepare_data.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml
```

## Sanity check
```bash
python scripts/sanity_check_dataloader.py --split train
python scripts/sanity_check_dataloader.py --split val
```

## Train
```bash
python scripts/train.py --dataset-config configs/dataset.yaml --train-config configs/train.yaml --model-config configs/model.yaml
```

## Plot
```bash
python scripts/plot_logs.py --run-dir outputs/<experiment_name>/<timestamp>
```

## Validate
```bash
python scripts/validate.py --checkpoint outputs/<experiment_name>/<timestamp>/checkpoints/best.pth --split val
```

## Infer
```bash
python scripts/infer.py --checkpoint outputs/<experiment_name>/<timestamp>/checkpoints/best.pth --split test
```

---

# 11. Final advice

For daily work, the safest order is always:

1. make config changes
2. prepare data only if needed
3. run sanity checks
4. run a short smoke test
5. run full training
6. validate best checkpoint
7. run inference
8. plot logs and compare runs

Do not skip sanity checks after changing patching, boundaries, or transforms.
