from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ISICFullImageDataset
from src.data.transforms import build_transforms_from_config
from src.engine import build_validation_params_from_config, validate_one_epoch
from src.losses import build_loss_from_config
from src.metrics import build_metrics_from_config
from src.models import build_model_from_config


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        return ckpt

    if isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
        return {}

    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")


def build_eval_loader(
    split: str,
    dataset_config_path: str,
    train_config_path: str,
):
    dataset_cfg = load_yaml(dataset_config_path)
    train_cfg = load_yaml(train_config_path)
    transforms = build_transforms_from_config(train_config_path)

    indices_dir = Path(dataset_cfg["paths"]["indices_dir"])
    index_csv = indices_dir / f"{split}_index.csv"

    dataset = ISICFullImageDataset(
        index_csv=str(index_csv),
        transform=transforms["val"] if split == "val" else transforms["test"],
        return_meta=True,
    )

    num_workers = int(train_cfg["training"]["num_workers"])
    pin_memory = bool(train_cfg["training"]["pin_memory"])

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return loader, transforms


def make_eval_dir(train_cfg: dict, split: str, checkpoint_path: str) -> Path:
    output_root = Path(train_cfg["logging"]["output_root"])
    exp_name = train_cfg["logging"]["experiment_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = Path(checkpoint_path).stem

    eval_dir = output_root / exp_name / "evaluations" / f"{split}_{ckpt_name}_{timestamp}"
    ensure_dir(eval_dir)
    ensure_dir(eval_dir / "configs")
    ensure_dir(eval_dir / "logs")
    return eval_dir


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main(
    checkpoint_path: str,
    split: str,
    dataset_config_path: str,
    train_config_path: str,
    model_config_path: str,
    device_str: str | None = None,
) -> None:
    train_cfg = load_yaml(train_config_path)
    device_cfg = train_cfg["training"].get("device", "auto")
    device = resolve_device(device_str if device_str is not None else device_cfg)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Split: {split}")

    eval_dir = make_eval_dir(train_cfg, split, checkpoint_path)
    print(f"[INFO] Evaluation directory: {eval_dir}")

    shutil.copy2(dataset_config_path, eval_dir / "configs" / "dataset.yaml")
    shutil.copy2(train_config_path, eval_dir / "configs" / "train.yaml")
    shutil.copy2(model_config_path, eval_dir / "configs" / "model.yaml")

    loader, transforms = build_eval_loader(
        split=split,
        dataset_config_path=dataset_config_path,
        train_config_path=train_config_path,
    )

    model = build_model_from_config(model_config_path).to(device)
    ckpt = load_checkpoint_weights(model, checkpoint_path, device)

    criterion = build_loss_from_config(train_config_path)
    metrics_fn = build_metrics_from_config(train_config_path)

    val_params = build_validation_params_from_config(train_config_path)

    results = validate_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        metrics_fn=metrics_fn,
        device=device,
        patch_size=val_params["patch_size"],
        stride=val_params["stride"],
        tile_batch_size=val_params["tile_batch_size"],
        tile_transform=transforms["val"] if split == "val" else transforms["test"],
        log_interval=10,
    )

    results_row = {"split": split}
    results_row.update(results)

    pd.DataFrame([results_row]).to_csv(eval_dir / "logs" / "metrics.csv", index=False)

    summary = {
        "split": split,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "device": str(device),
        "epoch_from_checkpoint": ckpt.get("epoch", None),
        "best_score_from_checkpoint": ckpt.get("best_score", None),
        "metrics": results,
    }
    save_json(summary, eval_dir / "logs" / "summary.json")

    print("[DONE] Validation finished.")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a saved checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (.pth)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset.yaml",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/train.yaml",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config",
    )
    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        split=args.split,
        dataset_config_path=args.dataset_config,
        train_config_path=args.train_config,
        model_config_path=args.model_config,
        device_str=args.device,
    )