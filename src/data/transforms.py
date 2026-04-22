from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
import numpy as np
import yaml


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RandomSyntheticHair(A.ImageOnlyTransform):
    def __init__(
        self,
        min_hairs: int = 1,
        max_hairs: int = 6,
        max_thickness: int = 2,
        p: float = 0.3,
    ) -> None:
        super().__init__(p=p)
        self.min_hairs = min_hairs
        self.max_hairs = max_hairs
        self.max_thickness = max_thickness

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]

        num_hairs = np.random.randint(self.min_hairs, self.max_hairs + 1)

        for _ in range(num_hairs):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            x2 = np.random.randint(0, w)
            y2 = np.random.randint(0, h)

            color_value = int(np.random.randint(0, 60))
            color = (color_value, color_value, color_value)
            thickness = int(np.random.randint(1, self.max_thickness + 1))

            cv2.line(out, (x1, y1), (x2, y2), color=color, thickness=thickness)

        return out

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("min_hairs", "max_hairs", "max_thickness")


def _build_normalization(normalization_cfg: dict) -> A.BasicTransform:
    mode = normalization_cfg.get("mode", "none")

    if mode == "none":
        return A.ToFloat(max_value=255.0)

    if mode == "imagenet":
        return A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0,
        )

    if mode == "dataset":
        mean = normalization_cfg.get("mean", None)
        std = normalization_cfg.get("std", None)

        if mean is None or std is None:
            raise ValueError("For normalization.mode='dataset', mean and std must be set.")

        if len(mean) != 3 or len(std) != 3:
            raise ValueError("Dataset normalization mean/std must each have length 3.")

        return A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0,
        )

    raise ValueError(f"Unsupported normalization mode: {mode}")


def _build_general_transforms(general_cfg: dict) -> list[A.BasicTransform]:
    return [
        A.HorizontalFlip(p=float(general_cfg.get("hflip_p", 0.5))),
        A.VerticalFlip(p=float(general_cfg.get("vflip_p", 0.5))),
        A.RandomRotate90(p=float(general_cfg.get("rotate90_p", 0.5))),
        A.ShiftScaleRotate(
            shift_limit=float(general_cfg.get("shift_limit", 0.0625)),
            scale_limit=float(general_cfg.get("scale_limit", 0.10)),
            rotate_limit=int(general_cfg.get("rotate_limit", 20)),
            border_mode=cv2.BORDER_REFLECT_101,
            p=float(general_cfg.get("shift_scale_rotate_p", 0.5)),
        ),
        A.RandomBrightnessContrast(
            brightness_limit=float(general_cfg.get("brightness_limit", 0.20)),
            contrast_limit=float(general_cfg.get("contrast_limit", 0.20)),
            p=float(general_cfg.get("brightness_contrast_p", 0.3)),
        ),
        A.RandomGamma(
            gamma_limit=tuple(general_cfg.get("gamma_limit", [80, 120])),
            p=float(general_cfg.get("gamma_p", 0.3)),
        ),
    ]


def _build_artifact_aware_transforms(artifact_cfg: dict) -> list[A.BasicTransform]:
    artifact_block = A.OneOf(
        [
            RandomSyntheticHair(
                min_hairs=int(artifact_cfg.get("min_hairs", 1)),
                max_hairs=int(artifact_cfg.get("max_hairs", 6)),
                max_thickness=int(artifact_cfg.get("max_hair_thickness", 2)),
                p=float(artifact_cfg.get("hair_p", 0.3)),
            ),
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=float(artifact_cfg.get("blur_p", 0.2)),
            ),
            A.GaussNoise(
                p=float(artifact_cfg.get("noise_p", 0.15)),
            ),
            A.RandomShadow(
                p=float(artifact_cfg.get("shadow_p", 0.2)),
            ),
        ],
        p=float(artifact_cfg.get("artifact_block_p", 0.4)),
    )

    return [artifact_block]


def build_transform(
    mode: str,
    normalization_cfg: dict,
    general_cfg: dict,
    artifact_cfg: dict,
) -> A.Compose:
    transforms: list[A.BasicTransform] = []

    if mode == "none":
        pass
    elif mode == "general":
        transforms.extend(_build_general_transforms(general_cfg))
    elif mode == "general_artifact_aware":
        transforms.extend(_build_general_transforms(general_cfg))
        transforms.extend(_build_artifact_aware_transforms(artifact_cfg))
    else:
        raise ValueError(f"Unsupported transform mode: {mode}")

    transforms.append(_build_normalization(normalization_cfg))

    return A.Compose(
        transforms,
        additional_targets={"boundary": "mask"},
    )


def build_transforms_from_config(train_config_path: str) -> dict[str, A.Compose]:
    cfg = load_yaml(train_config_path)
    tf_cfg = cfg["transforms"]

    normalization_cfg = tf_cfg["normalization"]
    general_cfg = tf_cfg["general"]
    artifact_cfg = tf_cfg["artifact_aware"]

    train_transform = build_transform(
        mode=tf_cfg.get("train_mode", "none"),
        normalization_cfg=normalization_cfg,
        general_cfg=general_cfg,
        artifact_cfg=artifact_cfg,
    )

    val_transform = build_transform(
        mode=tf_cfg.get("val_mode", "none"),
        normalization_cfg=normalization_cfg,
        general_cfg=general_cfg,
        artifact_cfg=artifact_cfg,
    )

    test_transform = build_transform(
        mode=tf_cfg.get("test_mode", "none"),
        normalization_cfg=normalization_cfg,
        general_cfg=general_cfg,
        artifact_cfg=artifact_cfg,
    )

    return {
        "train": train_transform,
        "val": val_transform,
        "test": test_transform,
    }