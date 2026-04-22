from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.utils.mask_utils import load_binary_mask
from src.utils.patch_utils import generate_sliding_window_boxes


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def classify_patch(fg_pixels: int, boundary_pixels: int) -> str:
    if boundary_pixels > 0:
        return "boundary"
    if fg_pixels > 0:
        return "positive"
    return "negative"


def build_rows_for_one_image(
    row: pd.Series,
    patch_size: int,
    stride: int,
) -> list[dict]:
    image_id = row["image_id"]
    split = row["split"]
    image_path = row["image_path"]
    mask_path = row["mask_path"]
    boundary_path = row["boundary_path"]

    mask = load_binary_mask(mask_path)
    boundary = load_binary_mask(boundary_path)

    image_h, image_w = mask.shape

    boxes = generate_sliding_window_boxes(
        image_h=image_h,
        image_w=image_w,
        patch_size=patch_size,
        stride=stride,
    )

    rows = []

    for patch_idx, (x0, y0, x1, y1) in enumerate(boxes):
        mask_patch = mask[y0:y1, x0:x1]
        boundary_patch = boundary[y0:y1, x0:x1]

        patch_h, patch_w = mask_patch.shape
        total_pixels = int(patch_h * patch_w)

        fg_pixels = int(mask_patch.sum())
        boundary_pixels = int(boundary_patch.sum())
        bg_pixels = int(total_pixels - fg_pixels)

        fg_ratio = fg_pixels / total_pixels
        bg_ratio = bg_pixels / total_pixels
        boundary_ratio = boundary_pixels / total_pixels

        has_fg = int(fg_pixels > 0)
        has_boundary = int(boundary_pixels > 0)
        patch_type = classify_patch(fg_pixels, boundary_pixels)

        rows.append(
            {
                "image_id": image_id,
                "split": split,
                "image_path": image_path,
                "mask_path": mask_path,
                "boundary_path": boundary_path,
                "image_h": image_h,
                "image_w": image_w,
                "patch_id": f"{image_id}_{patch_idx:04d}",
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "patch_h": patch_h,
                "patch_w": patch_w,
                "fg_pixels": fg_pixels,
                "bg_pixels": bg_pixels,
                "boundary_pixels": boundary_pixels,
                "fg_ratio": fg_ratio,
                "bg_ratio": bg_ratio,
                "boundary_ratio": boundary_ratio,
                "has_fg": has_fg,
                "has_boundary": has_boundary,
                "patch_type": patch_type,
            }
        )

    return rows


def build_patch_dataframe(
    split_df: pd.DataFrame,
    patch_size: int,
    stride: int,
) -> pd.DataFrame:
    all_rows = []

    for _, row in split_df.iterrows():
        all_rows.extend(
            build_rows_for_one_image(
                row=row,
                patch_size=patch_size,
                stride=stride,
            )
        )

    return pd.DataFrame(all_rows)


def build_stats(
    patch_df: pd.DataFrame,
    split_name: str,
    patch_size: int,
    stride: int,
) -> dict:
    stats = {
        "split": split_name,
        "patch_size": int(patch_size),
        "stride": int(stride),
        "total_patches": int(len(patch_df)),
        "positive_patches": int((patch_df["has_fg"] == 1).sum()),
        "negative_patches": int((patch_df["has_fg"] == 0).sum()),
        "boundary_patches": int((patch_df["has_boundary"] == 1).sum()),
        "total_fg_pixels": int(patch_df["fg_pixels"].sum()),
        "total_bg_pixels": int(patch_df["bg_pixels"].sum()),
        "total_boundary_pixels": int(patch_df["boundary_pixels"].sum()),
        "patch_type_counts": {
            k: int(v) for k, v in patch_df["patch_type"].value_counts().to_dict().items()
        },
    }
    return stats


def main(dataset_config_path: str, train_config_path: str) -> None:
    dataset_cfg = load_yaml(dataset_config_path)
    train_cfg = load_yaml(train_config_path)

    indices_dir = Path(dataset_cfg["paths"]["indices_dir"])
    patch_indices_dir = Path(dataset_cfg["paths"]["patch_indices_dir"])
    stats_dir = Path(dataset_cfg["paths"]["stats_dir"])

    ensure_dir(patch_indices_dir)
    ensure_dir(stats_dir)

    patch_cfg = train_cfg["patching"]
    build_splits = patch_cfg.get("build_splits", ["train"])
    patch_size = int(patch_cfg["patch_size"])
    stride = int(patch_cfg["stride"])

    for split_name in build_splits:
        split_index_path = indices_dir / f"{split_name}_index.csv"
        if not split_index_path.exists():
            raise FileNotFoundError(f"Missing split index: {split_index_path}")

        split_df = pd.read_csv(split_index_path)

        required_cols = {"image_id", "split", "image_path", "mask_path", "boundary_path"}
        missing_cols = required_cols - set(split_df.columns)
        if missing_cols:
            raise ValueError(
                f"{split_index_path} is missing columns: {sorted(missing_cols)}"
            )

        patch_df = build_patch_dataframe(
            split_df=split_df,
            patch_size=patch_size,
            stride=stride,
        )

        patch_csv_path = patch_indices_dir / f"{split_name}_patches.csv"
        patch_df.to_csv(patch_csv_path, index=False)

        stats = build_stats(
            patch_df=patch_df,
            split_name=split_name,
            patch_size=patch_size,
            stride=stride,
        )
        stats_path = stats_dir / f"patch_stats_{split_name}.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        print(f"[INFO] {split_name}: {len(patch_df)} patches saved to {patch_csv_path}")
        print(f"[INFO] {split_name}: stats saved to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build patch index.")
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
    args = parser.parse_args()

    main(
        dataset_config_path=args.dataset_config,
        train_config_path=args.train_config,
    )