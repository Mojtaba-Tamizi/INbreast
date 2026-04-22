from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import ISICFullImageDataset, ISICPatchDataset
from src.data.transforms import build_transforms_from_config

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_normalization_cfg(train_config_path: str) -> dict:
    cfg = load_yaml(train_config_path)
    return cfg["transforms"]["normalization"]


def denormalize_image(image_chw: np.ndarray, normalization_cfg: dict) -> np.ndarray:
    mode = normalization_cfg.get("mode", "none")

    image = np.transpose(image_chw, (1, 2, 0)).astype(np.float32)

    if mode == "none":
        image = np.clip(image, 0.0, 1.0)

    elif mode == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = image * std + mean
        image = np.clip(image, 0.0, 1.0)

    elif mode == "dataset":
        mean = normalization_cfg["mean"]
        std = normalization_cfg["std"]
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        image = image * std + mean
        image = np.clip(image, 0.0, 1.0)

    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    image = (image * 255.0).astype(np.uint8)
    return image


def tensor_to_mask_uint8(mask_chw: np.ndarray) -> np.ndarray:
    mask = mask_chw[0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask


def make_overlay(image_rgb: np.ndarray, mask_u8: np.ndarray, boundary_u8: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy()

    mask_bool = mask_u8 > 0
    boundary_bool = boundary_u8 > 0

    overlay[mask_bool] = (
        0.6 * overlay[mask_bool] + 0.4 * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)

    overlay[boundary_bool] = np.array([255, 0, 0], dtype=np.uint8)
    return overlay


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    canvas = image.copy()
    cv2.putText(
        canvas,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def make_panel(image_t: torch.Tensor, mask_t: torch.Tensor, boundary_t: torch.Tensor, normalization_cfg: dict) -> np.ndarray:
    image_np = image_t.detach().cpu().numpy()
    mask_np = mask_t.detach().cpu().numpy()
    boundary_np = boundary_t.detach().cpu().numpy()

    image_rgb = denormalize_image(image_np, normalization_cfg)
    mask_u8 = tensor_to_mask_uint8(mask_np)
    boundary_u8 = tensor_to_mask_uint8(boundary_np)

    mask_vis = np.stack([mask_u8, mask_u8, mask_u8], axis=-1)
    boundary_vis = np.stack([boundary_u8, boundary_u8, boundary_u8], axis=-1)
    overlay = make_overlay(image_rgb, mask_u8, boundary_u8)

    p1 = add_title(image_rgb, "image")
    p2 = add_title(mask_vis, "mask")
    p3 = add_title(boundary_vis, "boundary")
    p4 = add_title(overlay, "overlay")

    panel = np.concatenate([p1, p2, p3, p4], axis=1)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    return panel_bgr


def validate_batch(batch: dict[str, Any], split: str) -> None:
    images = batch["image"]
    masks = batch["mask"]
    boundaries = batch["boundary"]

    assert isinstance(images, torch.Tensor), "image must be a torch.Tensor"
    assert isinstance(masks, torch.Tensor), "mask must be a torch.Tensor"
    assert isinstance(boundaries, torch.Tensor), "boundary must be a torch.Tensor"

    assert images.ndim == 4, f"Expected image shape [B,C,H,W], got {images.shape}"
    assert masks.ndim == 4, f"Expected mask shape [B,1,H,W], got {masks.shape}"
    assert boundaries.ndim == 4, f"Expected boundary shape [B,1,H,W], got {boundaries.shape}"

    assert images.shape[0] == masks.shape[0] == boundaries.shape[0], "Batch size mismatch"
    assert images.shape[2:] == masks.shape[2:] == boundaries.shape[2:], "Spatial size mismatch"

    assert torch.isfinite(images).all(), "Image tensor has non-finite values"
    assert torch.isfinite(masks).all(), "Mask tensor has non-finite values"
    assert torch.isfinite(boundaries).all(), "Boundary tensor has non-finite values"

    assert masks.min().item() >= 0.0 and masks.max().item() <= 1.0, "Mask values out of range [0,1]"
    assert boundaries.min().item() >= 0.0 and boundaries.max().item() <= 1.0, "Boundary values out of range [0,1]"

    print(
        f"[BATCH] split={split} "
        f"image={tuple(images.shape)} "
        f"mask={tuple(masks.shape)} "
        f"boundary={tuple(boundaries.shape)} "
        f"image_range=({images.min().item():.4f}, {images.max().item():.4f})"
    )


def build_dataset(split: str, dataset_config_path: str, train_config_path: str):
    dataset_cfg = load_yaml(dataset_config_path)
    transforms = build_transforms_from_config(train_config_path)

    patch_dir = Path(dataset_cfg["paths"]["patch_indices_dir"])
    index_dir = Path(dataset_cfg["paths"]["indices_dir"])

    if split == "train":
        dataset = ISICPatchDataset.from_config(
            patch_csv=str(patch_dir / "train_patches.csv"),
            train_config_path=train_config_path,
            transform=transforms["train"],
            return_meta=True,
        )
    elif split == "val":
        dataset = ISICFullImageDataset(
            index_csv=str(index_dir / "val_index.csv"),
            transform=transforms["val"],
            return_meta=True,
        )
    elif split == "test":
        dataset = ISICFullImageDataset(
            index_csv=str(index_dir / "test_index.csv"),
            transform=transforms["test"],
            return_meta=True,
        )
    else:
        raise ValueError(f"Unsupported split: {split}")

    return dataset


def print_dataset_summary(dataset, split: str) -> None:
    print(f"[DATASET] split={split} length={len(dataset)}")

    if hasattr(dataset, "df"):
        df = dataset.df
        if "patch_type" in df.columns:
            counts = df["patch_type"].value_counts().to_dict()
            print(f"[DATASET] patch_type_counts={counts}")


def main(
    dataset_config: str,
    train_config: str,
    split: str,
    num_save: int,
    output_dir: str,
    batch_size: int | None,
    num_workers: int,
) -> None:
    dataset = build_dataset(
        split=split,
        dataset_config_path=dataset_config,
        train_config_path=train_config,
    )
    print_dataset_summary(dataset, split)

    if batch_size is None:
        batch_size = 4 if split == "train" else 1

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=False,
    )

    normalization_cfg = get_normalization_cfg(train_config)
    save_root = Path(output_dir) / split
    ensure_dir(save_root)

    saved = 0

    for batch_idx, batch in enumerate(loader):
        validate_batch(batch, split)

        bs = batch["image"].shape[0]
        meta = batch.get("meta", None)

        for i in range(bs):
            panel = make_panel(
                image_t=batch["image"][i],
                mask_t=batch["mask"][i],
                boundary_t=batch["boundary"][i],
                normalization_cfg=normalization_cfg,
            )

            if meta is not None and isinstance(meta, dict) and "image_id" in meta:
                image_id = meta["image_id"][i]
            else:
                image_id = f"{batch_idx:03d}_{i:02d}"

            save_path = save_root / f"{saved:03d}_{image_id}.png"
            ok = cv2.imwrite(str(save_path), panel)
            if not ok:
                raise IOError(f"Could not save visualization: {save_path}")

            print(f"[SAVE] {save_path}")
            saved += 1

            if saved >= num_save:
                print("[DONE] sanity check completed.")
                return

    print("[DONE] sanity check completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check dataloader and save sample visualizations.")
    parser.add_argument("--dataset-config", type=str, default="configs/dataset.yaml")
    parser.add_argument("--train-config", type=str, default="configs/train.yaml")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--num-save", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs/visualizations/sanity_check")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    main(
        dataset_config=args.dataset_config,
        train_config=args.train_config,
        split=args.split,
        num_save=args.num_save,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )