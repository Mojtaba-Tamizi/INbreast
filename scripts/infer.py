from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import build_transforms_from_config
from src.data.tiling import read_rgb_image, save_binary_mask, save_probability_map
from src.engine.validate import sliding_window_predict_outputs
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


def sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_overlay(
    image_rgb: np.ndarray,
    mask_u8: np.ndarray,
    boundary_u8: np.ndarray | None = None,
) -> np.ndarray:
    overlay = image_rgb.copy()

    mask_bool = mask_u8 > 0
    overlay[mask_bool] = (
        0.6 * overlay[mask_bool] + 0.4 * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)

    if boundary_u8 is not None:
        boundary_bool = boundary_u8 > 0
        overlay[boundary_bool] = np.array([255, 0, 0], dtype=np.uint8)

    return overlay


def build_infer_params_from_config(train_config_path: str) -> dict:
    cfg = load_yaml(train_config_path)

    val_cfg = cfg["validation"]["sliding_window"]
    infer_cfg = cfg["inference"]

    return {
        "patch_size": int(val_cfg["patch_size"]),
        "stride": int(val_cfg["stride"]),
        "tile_batch_size": int(val_cfg["tile_batch_size"]),
        "threshold": float(infer_cfg.get("threshold", 0.5)),
        "save_probability_maps": bool(infer_cfg.get("save_probability_maps", True)),
        "save_binary_masks": bool(infer_cfg.get("save_binary_masks", True)),
        "save_overlays": bool(infer_cfg.get("save_overlays", True)),
        "save_boundary_probability_maps": bool(infer_cfg.get("save_boundary_probability_maps", True)),
        "save_boundary_masks": bool(infer_cfg.get("save_boundary_masks", True)),
    }


def make_infer_dir(train_cfg: dict, split: str, checkpoint_path: str) -> Path:
    output_root = Path(train_cfg["logging"]["output_root"])
    exp_name = train_cfg["logging"]["experiment_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = Path(checkpoint_path).stem

    infer_dir = output_root / exp_name / "inference" / f"{split}_{ckpt_name}_{timestamp}"

    ensure_dir(infer_dir)
    ensure_dir(infer_dir / "configs")
    ensure_dir(infer_dir / "logs")
    ensure_dir(infer_dir / "predictions")
    ensure_dir(infer_dir / "predictions" / "masks")
    ensure_dir(infer_dir / "predictions" / "probability_maps")
    ensure_dir(infer_dir / "predictions" / "overlays")
    ensure_dir(infer_dir / "predictions" / "boundary_masks")
    ensure_dir(infer_dir / "predictions" / "boundary_probability_maps")

    return infer_dir


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
    dataset_cfg = load_yaml(dataset_config_path)
    train_cfg = load_yaml(train_config_path)

    device_cfg = train_cfg["training"].get("device", "auto")
    device = resolve_device(device_str if device_str is not None else device_cfg)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Split: {split}")

    infer_dir = make_infer_dir(train_cfg, split, checkpoint_path)
    print(f"[INFO] Inference directory: {infer_dir}")

    shutil.copy2(dataset_config_path, infer_dir / "configs" / "dataset.yaml")
    shutil.copy2(train_config_path, infer_dir / "configs" / "train.yaml")
    shutil.copy2(model_config_path, infer_dir / "configs" / "model.yaml")

    indices_dir = Path(dataset_cfg["paths"]["indices_dir"])
    index_csv = indices_dir / f"{split}_index.csv"
    df = pd.read_csv(index_csv)

    transforms = build_transforms_from_config(train_config_path)
    tile_transform = transforms["val"] if split == "val" else transforms["test"]

    infer_params = build_infer_params_from_config(train_config_path)
    threshold = infer_params["threshold"]

    model = build_model_from_config(model_config_path).to(device)
    ckpt = load_checkpoint_weights(model, checkpoint_path, device)
    model.eval()

    rows = []

    for idx, row in df.iterrows():
        image_id = row["image_id"]
        image_path = row["image_path"]

        print(f"[INFER] {idx + 1}/{len(df)} image_id={image_id}")

        raw_image = read_rgb_image(image_path)

        stitched_outputs = sliding_window_predict_outputs(
            model=model,
            image=raw_image,
            patch_size=infer_params["patch_size"],
            stride=infer_params["stride"],
            device=device,
            batch_size=infer_params["tile_batch_size"],
            transform=tile_transform,
            output_keys=None,
        )

        if "mask" not in stitched_outputs:
            raise KeyError("Model output does not contain 'mask'")

        mask_logits = stitched_outputs["mask"][0]
        mask_prob = sigmoid_numpy(mask_logits).astype(np.float32)
        pred_mask = (mask_prob >= threshold).astype(np.uint8)

        mask_prob_path = ""
        mask_bin_path = ""
        overlay_path = ""
        boundary_prob_path = ""
        boundary_bin_path = ""

        if infer_params["save_probability_maps"]:
            mask_prob_path = str(
                (infer_dir / "predictions" / "probability_maps" / f"{image_id}_prob.png").resolve()
            )
            save_probability_map(mask_prob, mask_prob_path)

        if infer_params["save_binary_masks"]:
            mask_bin_path = str(
                (infer_dir / "predictions" / "masks" / f"{image_id}_mask.png").resolve()
            )
            save_binary_mask(pred_mask, mask_bin_path)

        boundary_u8_for_overlay = None

        if "boundary" in stitched_outputs:
            boundary_logits = stitched_outputs["boundary"][0]
            boundary_prob = sigmoid_numpy(boundary_logits).astype(np.float32)
            boundary_mask = (boundary_prob >= threshold).astype(np.uint8)
            boundary_u8_for_overlay = boundary_mask * 255

            if infer_params["save_boundary_probability_maps"]:
                boundary_prob_path = str(
                    (infer_dir / "predictions" / "boundary_probability_maps" / f"{image_id}_boundary_prob.png").resolve()
                )
                save_probability_map(boundary_prob, boundary_prob_path)

            if infer_params["save_boundary_masks"]:
                boundary_bin_path = str(
                    (infer_dir / "predictions" / "boundary_masks" / f"{image_id}_boundary.png").resolve()
                )
                save_binary_mask(boundary_mask, boundary_bin_path)

        if infer_params["save_overlays"]:
            overlay = make_overlay(
                image_rgb=raw_image,
                mask_u8=(pred_mask * 255).astype(np.uint8),
                boundary_u8=boundary_u8_for_overlay,
            )
            overlay_path = str(
                (infer_dir / "predictions" / "overlays" / f"{image_id}_overlay.png").resolve()
            )
            ok = cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if not ok:
                raise IOError(f"Could not save overlay to: {overlay_path}")

        rows.append(
            {
                "image_id": image_id,
                "split": split,
                "image_path": str(Path(image_path).resolve()),
                "mask_probability_map_path": mask_prob_path,
                "mask_binary_path": mask_bin_path,
                "overlay_path": overlay_path,
                "boundary_probability_map_path": boundary_prob_path,
                "boundary_binary_path": boundary_bin_path,
            }
        )

    pd.DataFrame(rows).to_csv(infer_dir / "logs" / "predictions.csv", index=False)

    summary = {
        "split": split,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "device": str(device),
        "epoch_from_checkpoint": ckpt.get("epoch", None),
        "best_score_from_checkpoint": ckpt.get("best_score", None),
        "num_images": len(rows),
        "output_dir": str(infer_dir.resolve()),
    }
    save_json(summary, infer_dir / "logs" / "summary.json")

    print("[DONE] Inference finished.")
    print(f"[INFO] Predictions CSV: {infer_dir / 'logs' / 'predictions.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a saved checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (.pth)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
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