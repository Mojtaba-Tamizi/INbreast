from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_image_id(file_path: Path) -> str:
    return file_path.stem


def build_split_index(
    split_name: str,
    raw_root: Path,
    image_dir_name: str,
    mask_dir_name: str,
    image_ext: str,
    mask_ext: str,
) -> pd.DataFrame:
    image_dir = raw_root / image_dir_name
    mask_dir = raw_root / mask_dir_name

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_paths = sorted(image_dir.glob(f"*{image_ext}"))
    if len(image_paths) == 0:
        raise RuntimeError(f"No image files found in: {image_dir}")

    rows: List[Dict] = []
    missing_masks: List[str] = []

    for image_path in image_paths:
        image_id = extract_image_id(image_path)
        mask_path = mask_dir / f"{image_id}_segmentation{mask_ext}"

        if not mask_path.exists():
            missing_masks.append(image_id)
            continue

        rows.append(
            {
                "image_id": image_id,
                "split": split_name,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
            }
        )

    if missing_masks:
        print(f"[WARNING] {split_name}: {len(missing_masks)} masks were missing.")

    df = pd.DataFrame(rows)
    return df


def save_index(df: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)


def main(config_path: str) -> None:
    cfg = load_yaml(config_path)

    raw_root = Path(cfg["paths"]["raw_root"])
    indices_dir = Path(cfg["paths"]["indices_dir"])

    dataset_cfg = cfg["dataset"]
    image_ext = dataset_cfg["image_ext"]
    mask_ext = dataset_cfg["mask_ext"]

    split_to_cfg = {
        "train": dataset_cfg["train"],
        "val": dataset_cfg["val"],
        "test": dataset_cfg["test"],
    }

    all_dfs = []

    for split_name, split_cfg in split_to_cfg.items():
        df = build_split_index(
            split_name=split_name,
            raw_root=raw_root,
            image_dir_name=split_cfg["image_dir"],
            mask_dir_name=split_cfg["mask_dir"],
            image_ext=image_ext,
            mask_ext=mask_ext,
        )

        output_path = indices_dir / f"{split_name}_index.csv"
        save_index(df, output_path)
        all_dfs.append(df)

        print(f"[INFO] {split_name}: {len(df)} pairs saved to {output_path}")

    all_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    save_index(all_df, indices_dir / "all_index.csv")

    print(f"[INFO] all: {len(all_df)} pairs saved to {indices_dir / 'all_index.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ISIC2018 dataset index.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset yaml config.",
    )
    args = parser.parse_args()
    main(args.config)