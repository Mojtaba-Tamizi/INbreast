from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_image_shape(image_path: str) -> tuple[int, int]:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    return h, w


def read_mask_shape(mask_path: str) -> tuple[int, int]:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    h, w = mask.shape[:2]
    return h, w


def validate_index_file(index_path: Path, split_name: str) -> pd.DataFrame:
    assert_true(index_path.exists(), f"Missing index file: {index_path}")

    df = pd.read_csv(index_path)

    required_cols = {"image_id", "split", "image_path", "mask_path", "boundary_path"}
    missing = required_cols - set(df.columns)
    assert_true(not missing, f"{index_path} missing columns: {sorted(missing)}")

    assert_true(len(df) > 0, f"{index_path} is empty")
    assert_true(df["image_id"].is_unique, f"{index_path} has duplicate image_id values")
    assert_true((df["split"] == split_name).all(), f"{index_path} has wrong split values")

    for _, row in df.iterrows():
        image_id = row["image_id"]
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        boundary_path = Path(row["boundary_path"])

        assert_true(image_path.exists(), f"Missing image file for {image_id}: {image_path}")
        assert_true(mask_path.exists(), f"Missing mask file for {image_id}: {mask_path}")
        assert_true(boundary_path.exists(), f"Missing boundary file for {image_id}: {boundary_path}")

        img_h, img_w = read_image_shape(str(image_path))
        mask_h, mask_w = read_mask_shape(str(mask_path))
        boundary_h, boundary_w = read_mask_shape(str(boundary_path))

        assert_true(
            (img_h, img_w) == (mask_h, mask_w),
            f"Image/mask shape mismatch for {image_id}: "
            f"image={(img_h, img_w)}, mask={(mask_h, mask_w)}",
        )
        assert_true(
            (mask_h, mask_w) == (boundary_h, boundary_w),
            f"Mask/boundary shape mismatch for {image_id}: "
            f"mask={(mask_h, mask_w)}, boundary={(boundary_h, boundary_w)}",
        )

    print(f"[CHECK] {split_name}_index.csv passed.")
    return df


def validate_all_index(
    all_index_path: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    assert_true(all_index_path.exists(), f"Missing all_index file: {all_index_path}")

    all_df = pd.read_csv(all_index_path)

    expected_cols = {"image_id", "split", "image_path", "mask_path", "boundary_path"}
    missing = expected_cols - set(all_df.columns)
    assert_true(not missing, f"{all_index_path} missing columns: {sorted(missing)}")

    expected_total = len(train_df) + len(val_df) + len(test_df)
    assert_true(
        len(all_df) == expected_total,
        f"all_index row count mismatch: got {len(all_df)}, expected {expected_total}",
    )

    expected_pairs = set(
        pd.concat([train_df, val_df, test_df], ignore_index=True)[["image_id", "split"]]
        .apply(tuple, axis=1)
        .tolist()
    )
    got_pairs = set(all_df[["image_id", "split"]].apply(tuple, axis=1).tolist())

    assert_true(got_pairs == expected_pairs, "all_index contents do not match split indices")

    print("[CHECK] all_index.csv passed.")


def validate_patch_file(
    patch_csv_path: Path,
    stats_path: Path,
    split_df: pd.DataFrame,
    split_name: str,
) -> None:
    assert_true(patch_csv_path.exists(), f"Missing patch file: {patch_csv_path}")
    assert_true(stats_path.exists(), f"Missing stats file: {stats_path}")

    patch_df = pd.read_csv(patch_csv_path)

    required_cols = {
        "image_id",
        "split",
        "image_path",
        "mask_path",
        "boundary_path",
        "image_h",
        "image_w",
        "patch_id",
        "x0",
        "y0",
        "x1",
        "y1",
        "patch_h",
        "patch_w",
        "fg_pixels",
        "bg_pixels",
        "boundary_pixels",
        "fg_ratio",
        "bg_ratio",
        "boundary_ratio",
        "has_fg",
        "has_boundary",
        "patch_type",
    }
    missing = required_cols - set(patch_df.columns)
    assert_true(not missing, f"{patch_csv_path} missing columns: {sorted(missing)}")

    assert_true(len(patch_df) > 0, f"{patch_csv_path} is empty")
    assert_true(patch_df["patch_id"].is_unique, f"{patch_csv_path} has duplicate patch_id values")
    assert_true((patch_df["split"] == split_name).all(), f"{patch_csv_path} has wrong split values")

    valid_types = {"negative", "positive", "boundary"}
    bad_types = set(patch_df["patch_type"].unique()) - valid_types
    assert_true(not bad_types, f"{patch_csv_path} has invalid patch_type values: {sorted(bad_types)}")

    valid_image_ids = set(split_df["image_id"].tolist())

    for _, row in patch_df.iterrows():
        image_id = row["image_id"]
        assert_true(image_id in valid_image_ids, f"Unknown image_id in patch file: {image_id}")

        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        boundary_path = Path(row["boundary_path"])

        assert_true(image_path.exists(), f"Missing image file in patch row: {image_path}")
        assert_true(mask_path.exists(), f"Missing mask file in patch row: {mask_path}")
        assert_true(boundary_path.exists(), f"Missing boundary file in patch row: {boundary_path}")

        image_h = int(row["image_h"])
        image_w = int(row["image_w"])
        x0 = int(row["x0"])
        y0 = int(row["y0"])
        x1 = int(row["x1"])
        y1 = int(row["y1"])
        patch_h = int(row["patch_h"])
        patch_w = int(row["patch_w"])
        fg_pixels = int(row["fg_pixels"])
        bg_pixels = int(row["bg_pixels"])
        boundary_pixels = int(row["boundary_pixels"])
        has_fg = int(row["has_fg"])
        has_boundary = int(row["has_boundary"])
        patch_type = row["patch_type"]

        assert_true(0 <= x0 < x1 <= image_w, f"Bad x coords in patch {row['patch_id']}")
        assert_true(0 <= y0 < y1 <= image_h, f"Bad y coords in patch {row['patch_id']}")
        assert_true((x1 - x0) == patch_w, f"patch_w mismatch in patch {row['patch_id']}")
        assert_true((y1 - y0) == patch_h, f"patch_h mismatch in patch {row['patch_id']}")

        total_pixels = patch_h * patch_w
        assert_true(fg_pixels >= 0, f"Negative fg_pixels in patch {row['patch_id']}")
        assert_true(bg_pixels >= 0, f"Negative bg_pixels in patch {row['patch_id']}")
        assert_true(boundary_pixels >= 0, f"Negative boundary_pixels in patch {row['patch_id']}")
        assert_true(fg_pixels + bg_pixels == total_pixels, f"fg/bg mismatch in patch {row['patch_id']}")
        assert_true(boundary_pixels <= total_pixels, f"boundary_pixels too large in patch {row['patch_id']}")

        fg_ratio = float(row["fg_ratio"])
        bg_ratio = float(row["bg_ratio"])
        boundary_ratio = float(row["boundary_ratio"])

        assert_true(abs(fg_ratio - (fg_pixels / total_pixels)) < 1e-6, f"fg_ratio mismatch in {row['patch_id']}")
        assert_true(abs(bg_ratio - (bg_pixels / total_pixels)) < 1e-6, f"bg_ratio mismatch in {row['patch_id']}")
        assert_true(
            abs(boundary_ratio - (boundary_pixels / total_pixels)) < 1e-6,
            f"boundary_ratio mismatch in {row['patch_id']}",
        )

        assert_true(has_fg in {0, 1}, f"Invalid has_fg in patch {row['patch_id']}")
        assert_true(has_boundary in {0, 1}, f"Invalid has_boundary in patch {row['patch_id']}")
        assert_true(has_fg == int(fg_pixels > 0), f"has_fg mismatch in patch {row['patch_id']}")
        assert_true(has_boundary == int(boundary_pixels > 0), f"has_boundary mismatch in patch {row['patch_id']}")

        if patch_type == "negative":
            assert_true(fg_pixels == 0 and boundary_pixels == 0, f"patch_type mismatch in {row['patch_id']}")
        elif patch_type == "positive":
            assert_true(fg_pixels > 0 and boundary_pixels == 0, f"patch_type mismatch in {row['patch_id']}")
        elif patch_type == "boundary":
            assert_true(boundary_pixels > 0, f"patch_type mismatch in {row['patch_id']}")

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    assert_true(stats["split"] == split_name, f"Stats split mismatch in {stats_path}")
    assert_true(stats["total_patches"] == len(patch_df), f"Stats total_patches mismatch in {stats_path}")
    assert_true(
        stats["positive_patches"] == int((patch_df["has_fg"] == 1).sum()),
        f"Stats positive_patches mismatch in {stats_path}",
    )
    assert_true(
        stats["negative_patches"] == int((patch_df["has_fg"] == 0).sum()),
        f"Stats negative_patches mismatch in {stats_path}",
    )
    assert_true(
        stats["boundary_patches"] == int((patch_df["has_boundary"] == 1).sum()),
        f"Stats boundary_patches mismatch in {stats_path}",
    )
    assert_true(
        stats["total_fg_pixels"] == int(patch_df["fg_pixels"].sum()),
        f"Stats total_fg_pixels mismatch in {stats_path}",
    )
    assert_true(
        stats["total_bg_pixels"] == int(patch_df["bg_pixels"].sum()),
        f"Stats total_bg_pixels mismatch in {stats_path}",
    )
    assert_true(
        stats["total_boundary_pixels"] == int(patch_df["boundary_pixels"].sum()),
        f"Stats total_boundary_pixels mismatch in {stats_path}",
    )

    expected_type_counts = {
        k: int(v) for k, v in patch_df["patch_type"].value_counts().to_dict().items()
    }
    got_type_counts = {k: int(v) for k, v in stats["patch_type_counts"].items()}
    assert_true(got_type_counts == expected_type_counts, f"patch_type_counts mismatch in {stats_path}")

    print(f"[CHECK] {split_name}_patches.csv passed.")
    print(
        f"[SUMMARY] {split_name}: patches={len(patch_df)}, "
        f"positive={(patch_df['has_fg'] == 1).sum()}, "
        f"negative={(patch_df['has_fg'] == 0).sum()}, "
        f"boundary={(patch_df['has_boundary'] == 1).sum()}"
    )


def main(dataset_config_path: str, train_config_path: str) -> None:
    dataset_cfg = load_yaml(dataset_config_path)
    train_cfg = load_yaml(train_config_path)

    indices_dir = Path(dataset_cfg["paths"]["indices_dir"])
    patch_indices_dir = Path(dataset_cfg["paths"]["patch_indices_dir"])
    stats_dir = Path(dataset_cfg["paths"]["stats_dir"])

    train_df = validate_index_file(indices_dir / "train_index.csv", "train")
    val_df = validate_index_file(indices_dir / "val_index.csv", "val")
    test_df = validate_index_file(indices_dir / "test_index.csv", "test")

    validate_all_index(
        all_index_path=indices_dir / "all_index.csv",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    build_splits = train_cfg["patching"].get("build_splits", ["train"])
    split_to_df = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for split_name in build_splits:
        validate_patch_file(
            patch_csv_path=patch_indices_dir / f"{split_name}_patches.csv",
            stats_path=stats_dir / f"patch_stats_{split_name}.json",
            split_df=split_to_df[split_name],
            split_name=split_name,
        )

    print("[CHECK] all sanity checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate prepared dataset and patches.")
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