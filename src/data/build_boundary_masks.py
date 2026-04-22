from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.utils.mask_utils import load_binary_mask, make_boundary_mask, save_mask


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_boundaries_for_split(
    split_df: pd.DataFrame,
    split_name: str,
    boundaries_root: Path,
    boundary_cfg: dict,
) -> pd.DataFrame:
    split_boundary_dir = boundaries_root / split_name
    ensure_dir(split_boundary_dir)

    mode = boundary_cfg["mode"]
    kernel_size = int(boundary_cfg["kernel_size"])
    iterations = int(boundary_cfg["iterations"])
    suffix = boundary_cfg.get("suffix", "_boundary")

    boundary_paths = []

    for _, row in split_df.iterrows():
        image_id = row["image_id"]
        mask_path = row["mask_path"]

        binary_mask = load_binary_mask(mask_path)
        boundary_mask = make_boundary_mask(
            binary_mask=binary_mask,
            mode=mode,
            kernel_size=kernel_size,
            iterations=iterations,
        )

        save_path = split_boundary_dir / f"{image_id}{suffix}.png"
        save_mask(boundary_mask, str(save_path))
        boundary_paths.append(str(save_path.resolve()))

    out_df = split_df.copy()
    out_df["boundary_path"] = boundary_paths
    return out_df


def main(config_path: str) -> None:
    cfg = load_yaml(config_path)

    indices_dir = Path(cfg["paths"]["indices_dir"])
    boundaries_root = Path(cfg["paths"]["boundaries_dir"])
    ensure_dir(boundaries_root)

    boundary_cfg = cfg["boundary"]

    split_files = {
        "train": indices_dir / "train_index.csv",
        "val": indices_dir / "val_index.csv",
        "test": indices_dir / "test_index.csv",
    }

    updated_dfs = []

    for split_name, split_file in split_files.items():
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split index: {split_file}")

        split_df = pd.read_csv(split_file)
        split_df = build_boundaries_for_split(
            split_df=split_df,
            split_name=split_name,
            boundaries_root=boundaries_root,
            boundary_cfg=boundary_cfg,
        )

        split_df.to_csv(split_file, index=False)
        updated_dfs.append(split_df)

        print(f"[INFO] {split_name}: {len(split_df)} boundary masks saved.")

    all_df = pd.concat(updated_dfs, axis=0, ignore_index=True)
    all_df.to_csv(indices_dir / "all_index.csv", index=False)

    print(f"[INFO] all: {len(all_df)} index updated with boundary_path.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build boundary masks for ISIC2018.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset yaml config.",
    )
    args = parser.parse_args()
    main(args.config)