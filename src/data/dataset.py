from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_rgb_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_binary_mask(mask_path: str) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return (mask > 127).astype(np.uint8)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    orig_dtype = image.dtype
    image = image.astype(np.float32)

    if np.issubdtype(orig_dtype, np.integer):
        image = image / 255.0

    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float()


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=0)
    return torch.from_numpy(mask).float()


class ISICPatchDataset(Dataset):
    def __init__(
        self,
        patch_csv: str,
        transform: Any | None = None,
        filter_query: str | None = None,
        return_meta: bool = True,
        sampling_ratios: dict[str, float] | None = None,
        total_samples: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.patch_csv = patch_csv
        self.transform = transform
        self.return_meta = return_meta

        df = pd.read_csv(patch_csv)

        if filter_query is not None:
            df = df.query(filter_query).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No samples found in patch csv: {patch_csv}")

        required_cols = {
            "image_id",
            "split",
            "image_path",
            "mask_path",
            "boundary_path",
            "patch_id",
            "x0",
            "y0",
            "x1",
            "y1",
            "has_fg",
            "has_boundary",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{patch_csv} is missing columns: {sorted(missing)}")

        if sampling_ratios is not None:
            df = self._apply_ratio_sampling(
                df=df,
                sampling_ratios=sampling_ratios,
                total_samples=total_samples,
                random_state=random_state,
            )

        self.df = df.reset_index(drop=True)

    @staticmethod
    def _apply_ratio_sampling(
        df: pd.DataFrame,
        sampling_ratios: dict[str, float],
        total_samples: int | None,
        random_state: int,
    ) -> pd.DataFrame:
        boundary_df = df[df["has_boundary"] == 1]
        positive_df = df[(df["has_fg"] == 1) & (df["has_boundary"] == 0)]
        negative_df = df[df["has_fg"] == 0]

        groups = {
            "boundary": boundary_df,
            "positive": positive_df,
            "negative": negative_df,
        }

        required_keys = {"boundary", "positive", "negative"}
        if set(sampling_ratios.keys()) != required_keys:
            raise ValueError(
                f"sampling_ratios must have exactly these keys: {sorted(required_keys)}"
            )

        ratio_sum = sum(float(v) for v in sampling_ratios.values())
        if ratio_sum <= 0:
            raise ValueError("Sum of sampling ratios must be > 0")

        normalized_ratios = {
            k: float(v) / ratio_sum for k, v in sampling_ratios.items()
        }

        if total_samples is None:
            total_samples = len(df)

        sampled_parts = []

        for group_name, ratio in normalized_ratios.items():
            group_df = groups[group_name]
            target_n = int(round(total_samples * ratio))

            if len(group_df) == 0 or target_n == 0:
                continue

            replace = target_n > len(group_df)
            sampled_group = group_df.sample(
                n=target_n,
                replace=replace,
                random_state=random_state,
            )
            sampled_parts.append(sampled_group)

        if len(sampled_parts) == 0:
            raise ValueError("Sampling produced an empty dataset")

        out_df = pd.concat(sampled_parts, axis=0)
        out_df = out_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        return out_df

    @classmethod
    def from_config(
        cls,
        patch_csv: str,
        train_config_path: str,
        transform: Any | None = None,
        filter_query: str | None = None,
        return_meta: bool = True,
    ) -> "ISICPatchDataset":
        cfg = load_yaml(train_config_path)
        sampling_cfg = cfg.get("sampling", {})

        if sampling_cfg.get("enabled", False):
            sampling_ratios = sampling_cfg["ratios"]
            total_samples = sampling_cfg.get("total_samples", None)
            random_state = int(sampling_cfg.get("random_state", 42))
        else:
            sampling_ratios = None
            total_samples = None
            random_state = 42

        return cls(
            patch_csv=patch_csv,
            transform=transform,
            filter_query=filter_query,
            return_meta=return_meta,
            sampling_ratios=sampling_ratios,
            total_samples=total_samples,
            random_state=random_state,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]

        image = read_rgb_image(row["image_path"])
        mask = read_binary_mask(row["mask_path"])
        boundary = read_binary_mask(row["boundary_path"])

        x0, y0 = int(row["x0"]), int(row["y0"])
        x1, y1 = int(row["x1"]), int(row["y1"])

        image = image[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]
        boundary = boundary[y0:y1, x0:x1]

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask, boundary=boundary)
            image = transformed["image"]
            mask = transformed["mask"]
            boundary = transformed["boundary"]

        sample = {
            "image": image_to_tensor(image),
            "mask": mask_to_tensor(mask),
            "boundary": mask_to_tensor(boundary),
        }

        if self.return_meta:
            sample["meta"] = {
                "image_id": row["image_id"],
                "patch_id": row["patch_id"],
                "split": row["split"],
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "patch_type": row.get("patch_type", None),
                "has_fg": int(row.get("has_fg", 0)),
                "has_boundary": int(row.get("has_boundary", 0)),
            }

        return sample
    

class ISICFullImageDataset(Dataset):
    def __init__(
        self,
        index_csv: str,
        transform: Any | None = None,
        return_meta: bool = True,
    ) -> None:
        self.index_csv = index_csv
        self.transform = transform
        self.return_meta = return_meta

        df = pd.read_csv(index_csv).reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No samples found in index csv: {index_csv}")

        required_cols = {
            "image_id",
            "split",
            "image_path",
            "mask_path",
            "boundary_path",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{index_csv} is missing columns: {sorted(missing)}")

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]

        image = read_rgb_image(row["image_path"])
        mask = read_binary_mask(row["mask_path"])
        boundary = read_binary_mask(row["boundary_path"])

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask, boundary=boundary)
            image = transformed["image"]
            mask = transformed["mask"]
            boundary = transformed["boundary"]

        sample = {
            "image": image_to_tensor(image),
            "mask": mask_to_tensor(mask),
            "boundary": mask_to_tensor(boundary),
        }

        if self.return_meta:
            h, w = mask.shape
            sample["meta"] = {
                "image_id": row["image_id"],
                "split": row["split"],
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "boundary_path": row["boundary_path"],
                "height": h,
                "width": w,
            }

        return sample