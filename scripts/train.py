from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ISICFullImageDataset, ISICPatchDataset
from src.data.transforms import build_transforms_from_config
from src.engine import build_validation_params_from_config, train_one_epoch, validate_one_epoch
from src.losses import build_loss_from_config
from src.metrics import build_metrics_from_config
from src.models import build_model_from_config


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
    }


def count_parameters_in_module(module: torch.nn.Module) -> dict[str, int]:
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
    }


def get_model_module_parameter_summary(model: torch.nn.Module) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}

    module_names = [
        "encoder",
        "dec3",
        "dec2",
        "dec1",
        "final_up_proj",
        "final_up_refine",
        "fusion",
        "post_no_fusion",
        "boundary_head",
        "seg_head",
    ]

    for name in module_names:
        if hasattr(model, name):
            module = getattr(model, name)
            if module is not None:
                summary[name] = count_parameters_in_module(module)

    return summary


def print_model_parameter_summary(model: torch.nn.Module) -> None:
    overall = count_parameters(model)

    print(
        "[INFO] Model params: "
        f"total={overall['total']:,} "
        f"trainable={overall['trainable']:,} "
        f"non_trainable={overall['non_trainable']:,}"
    )
    print(
        "[INFO] Model params (M): "
        f"total={overall['total'] / 1e6:.3f}M "
        f"trainable={overall['trainable'] / 1e6:.3f}M "
        f"non_trainable={overall['non_trainable'] / 1e6:.3f}M"
    )

    module_summary = get_model_module_parameter_summary(model)

    if len(module_summary) > 0:
        print("[INFO] Parameter count by module:")
        for name, stats in module_summary.items():
            print(
                f"  - {name}: "
                f"total={stats['total']:,} "
                f"trainable={stats['trainable']:,} "
                f"non_trainable={stats['non_trainable']:,} "
                f"(total={stats['total'] / 1e6:.3f}M)"
            )

def build_optimizer(model: torch.nn.Module, cfg: dict):
    name = cfg["optimizer"]["name"].lower()
    lr = float(cfg["optimizer"]["lr"])
    weight_decay = float(cfg["optimizer"].get("weight_decay", 0.0))

    if name == "adam":
        betas = tuple(cfg["optimizer"].get("betas", [0.9, 0.999]))
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    if name == "adamw":
        betas = tuple(cfg["optimizer"].get("betas", [0.9, 0.999]))
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    if name == "sgd":
        momentum = float(cfg["optimizer"].get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, cfg: dict):
    sched_cfg = cfg.get("scheduler", {})
    if not sched_cfg.get("enabled", False):
        return None

    name = sched_cfg["name"].lower()
    params = sched_cfg.get("params", {})

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)

    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)

    if name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)

    raise ValueError(f"Unsupported scheduler: {name}")


def make_run_dir(train_cfg: dict) -> Path:
    output_root = Path(train_cfg["logging"]["output_root"])
    exp_name = train_cfg["logging"]["experiment_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / exp_name / timestamp

    ensure_dir(run_dir)
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "configs")

    return run_dir


def copy_configs_to_run_dir(
    run_dir: Path,
    dataset_config_path: str,
    train_config_path: str,
    model_config_path: str,
) -> None:
    shutil.copy2(dataset_config_path, run_dir / "configs" / "dataset.yaml")
    shutil.copy2(train_config_path, run_dir / "configs" / "train.yaml")
    shutil.copy2(model_config_path, run_dir / "configs" / "model.yaml")


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    scaler,
    best_score: float | None,
    history: list[dict],
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_score": best_score,
        "history": history,
    }
    torch.save(ckpt, path)


def is_better(current: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > best
    if mode == "min":
        return current < best
    raise ValueError(f"Unsupported checkpoint mode: {mode}")


def prefix_dict(d: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def print_epoch_log(epoch: int, total_epochs: int, row: dict[str, float]) -> None:
    msg = (
        f"[EPOCH {epoch}/{total_epochs}] "
        f"train_loss={row.get('train_total_loss', float('nan')):.4f} "
        f"val_loss={row.get('val_total_loss', float('nan')):.4f} "
        f"val_dice={row.get('val_mask_dice', float('nan')):.4f} "
        f"lr={row.get('lr', float('nan')):.6f}"
    )
    print(msg)


def build_dataloaders(
    dataset_config_path: str,
    train_config_path: str,
):
    dataset_cfg = load_yaml(dataset_config_path)
    train_cfg = load_yaml(train_config_path)
    transforms = build_transforms_from_config(train_config_path)

    indices_dir = Path(dataset_cfg["paths"]["indices_dir"])
    patch_indices_dir = Path(dataset_cfg["paths"]["patch_indices_dir"])

    train_dataset = ISICPatchDataset.from_config(
        patch_csv=str(patch_indices_dir / "train_patches.csv"),
        train_config_path=train_config_path,
        transform=transforms["train"],
        return_meta=True,
    )

    val_dataset = ISICFullImageDataset(
        index_csv=str(indices_dir / "val_index.csv"),
        transform=transforms["val"],
        return_meta=True,
    )

    batch_size = int(train_cfg["training"]["batch_size"])
    num_workers = int(train_cfg["training"]["num_workers"])
    pin_memory = bool(train_cfg["training"]["pin_memory"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, transforms


def main(
    dataset_config_path: str,
    train_config_path: str,
    model_config_path: str,
) -> None:
    train_cfg = load_yaml(train_config_path)
    model_cfg = load_yaml(model_config_path)

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device = resolve_device(train_cfg["training"].get("device", "auto"))
    print(f"[INFO] Using device: {device}")

    run_dir = make_run_dir(train_cfg)
    copy_configs_to_run_dir(
        run_dir=run_dir,
        dataset_config_path=dataset_config_path,
        train_config_path=train_config_path,
        model_config_path=model_config_path,
    )
    print(f"[INFO] Run directory: {run_dir}")

    train_loader, val_loader, transforms = build_dataloaders(
        dataset_config_path=dataset_config_path,
        train_config_path=train_config_path,
    )

    model = build_model_from_config(model_config_path).to(device)
    param_info = count_parameters(model)
    print_model_parameter_summary(model)
    criterion = build_loss_from_config(train_config_path)
    metrics_fn = build_metrics_from_config(train_config_path)

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    use_amp = bool(train_cfg["training"].get("use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    val_params = build_validation_params_from_config(train_config_path)

    total_epochs = int(train_cfg["training"]["epochs"])
    val_every = int(train_cfg["training"].get("val_every", 1))
    max_grad_norm = train_cfg["training"].get("max_grad_norm", None)
    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)

    save_every = int(train_cfg["logging"].get("save_checkpoint_every", 1))
    monitor_key = train_cfg["checkpoint"]["monitor"]
    monitor_mode = train_cfg["checkpoint"]["mode"]

    history: list[dict] = []
    best_score = None
    best_epoch = None

    summary_path = run_dir / "logs" / "summary.json"
    history_csv_path = run_dir / "logs" / "history.csv"
    last_ckpt_path = run_dir / "checkpoints" / "last.pth"
    best_ckpt_path = run_dir / "checkpoints" / "best.pth"

    for epoch in range(1, total_epochs + 1):
        train_results = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            metrics_fn=metrics_fn,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
            scheduler=None,
            scheduler_step_on_batch=False,
            log_interval=100,
        )

        row = {"epoch": epoch}
        row.update(prefix_dict(train_results, "train_"))
        row["lr"] = float(optimizer.param_groups[0]["lr"])

        do_validate = (epoch % val_every == 0)

        if do_validate:
            val_results = validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                metrics_fn=metrics_fn,
                device=device,
                patch_size=val_params["patch_size"],
                stride=val_params["stride"],
                tile_batch_size=val_params["tile_batch_size"],
                tile_transform=transforms["val"],
                log_interval=10,
            )
            row.update(prefix_dict(val_results, "val_"))

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if monitor_key not in row:
                        raise KeyError(f"Monitor key '{monitor_key}' not found in epoch results")
                    scheduler.step(row[monitor_key])
                else:
                    scheduler.step()
        else:
            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

        history.append(row)
        pd.DataFrame(history).to_csv(history_csv_path, index=False)

        save_checkpoint(
            path=last_ckpt_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_score=best_score,
            history=history,
        )

        if epoch % save_every == 0:
            save_checkpoint(
                path=run_dir / "checkpoints" / f"epoch_{epoch:03d}.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_score=best_score,
                history=history,
            )

        if do_validate:
            if monitor_key not in row:
                raise KeyError(f"Monitor key '{monitor_key}' not found in epoch results")

            current_score = float(row[monitor_key])
            if is_better(current_score, best_score, monitor_mode):
                best_score = current_score
                best_epoch = epoch

                save_checkpoint(
                    path=best_ckpt_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_score=best_score,
                    history=history,
                )

        print_epoch_log(epoch, total_epochs, row)

        summary = {
            "run_dir": str(run_dir.resolve()),
            "device": str(device),
            "model_name": model_cfg["model"]["name"],
            "best_score": best_score,
            "best_epoch": best_epoch,
            "monitor_key": monitor_key,
            "monitor_mode": monitor_mode,
            "history_csv": str(history_csv_path.resolve()),
            "last_checkpoint": str(last_ckpt_path.resolve()),
            "best_checkpoint": str(best_ckpt_path.resolve()),
        }
        save_json(summary, summary_path)

    print("[DONE] Training finished.")
    print(f"[INFO] History CSV: {history_csv_path}")
    print(f"[INFO] Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train skin lesion segmentation model.")
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
    args = parser.parse_args()

    main(
        dataset_config_path=args.dataset_config,
        train_config_path=args.train_config,
        model_config_path=args.model_config,
    )