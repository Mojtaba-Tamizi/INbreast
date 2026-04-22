from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_erosion, distance_transform_edt


EPS = 1e-7


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _prepare_binary_predictions(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    target = (target > 0.5).float()

    if from_logits:
        pred = torch.sigmoid(pred)

    pred = (pred >= threshold).float()

    pred_np = _to_numpy(pred).astype(np.uint8)
    target_np = _to_numpy(target).astype(np.uint8)

    return pred_np, target_np


def _flatten_binary_batch(
    pred_np: np.ndarray,
    target_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_flat = pred_np.reshape(pred_np.shape[0], -1)
    target_flat = target_np.reshape(target_np.shape[0], -1)
    return pred_flat, target_flat


def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return np.array([], dtype=np.float32)

    structure = np.ones((3, 3), dtype=bool)

    border_a = mask_a ^ binary_erosion(mask_a, structure=structure, border_value=0)
    border_b = mask_b ^ binary_erosion(mask_b, structure=structure, border_value=0)

    dt_b = distance_transform_edt(~border_b)
    distances_a_to_b = dt_b[border_a]

    return distances_a_to_b.astype(np.float32)


def _compute_hd95_single(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    both_empty_value: float = 0.0,
    one_empty_value: float | None = None,
) -> float:
    pred_mask = pred_mask.astype(bool)
    target_mask = target_mask.astype(bool)

    pred_empty = pred_mask.sum() == 0
    target_empty = target_mask.sum() == 0

    if pred_empty and target_empty:
        return float(both_empty_value)

    if pred_empty != target_empty:
        return np.nan if one_empty_value is None else float(one_empty_value)

    d1 = _surface_distances(pred_mask, target_mask)
    d2 = _surface_distances(target_mask, pred_mask)

    all_distances = np.concatenate([d1, d2], axis=0)
    if len(all_distances) == 0:
        return np.nan if one_empty_value is None else float(one_empty_value)

    return float(np.percentile(all_distances, 95))


class DiceMetric(nn.Module):
    def __init__(self, from_logits: bool = True, threshold: float = 0.5) -> None:
        super().__init__()
        self.from_logits = from_logits
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_np, target_np = _prepare_binary_predictions(
            pred=pred,
            target=target,
            from_logits=self.from_logits,
            threshold=self.threshold,
        )
        pred_flat, target_flat = _flatten_binary_batch(pred_np, target_np)

        intersection = (pred_flat * target_flat).sum(axis=1)
        denom = pred_flat.sum(axis=1) + target_flat.sum(axis=1)

        dice = (2.0 * intersection + EPS) / (denom + EPS)
        return float(dice.mean())


class IoUMetric(nn.Module):
    def __init__(self, from_logits: bool = True, threshold: float = 0.5) -> None:
        super().__init__()
        self.from_logits = from_logits
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_np, target_np = _prepare_binary_predictions(
            pred=pred,
            target=target,
            from_logits=self.from_logits,
            threshold=self.threshold,
        )
        pred_flat, target_flat = _flatten_binary_batch(pred_np, target_np)

        intersection = (pred_flat * target_flat).sum(axis=1)
        union = pred_flat.sum(axis=1) + target_flat.sum(axis=1) - intersection

        iou = (intersection + EPS) / (union + EPS)
        return float(iou.mean())


class SensitivityMetric(nn.Module):
    def __init__(self, from_logits: bool = True, threshold: float = 0.5) -> None:
        super().__init__()
        self.from_logits = from_logits
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_np, target_np = _prepare_binary_predictions(
            pred=pred,
            target=target,
            from_logits=self.from_logits,
            threshold=self.threshold,
        )
        pred_flat, target_flat = _flatten_binary_batch(pred_np, target_np)

        tp = (pred_flat * target_flat).sum(axis=1)
        fn = ((1 - pred_flat) * target_flat).sum(axis=1)

        sensitivity = (tp + EPS) / (tp + fn + EPS)
        return float(sensitivity.mean())


class SpecificityMetric(nn.Module):
    def __init__(self, from_logits: bool = True, threshold: float = 0.5) -> None:
        super().__init__()
        self.from_logits = from_logits
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_np, target_np = _prepare_binary_predictions(
            pred=pred,
            target=target,
            from_logits=self.from_logits,
            threshold=self.threshold,
        )
        pred_flat, target_flat = _flatten_binary_batch(pred_np, target_np)

        tn = ((1 - pred_flat) * (1 - target_flat)).sum(axis=1)
        fp = (pred_flat * (1 - target_flat)).sum(axis=1)

        specificity = (tn + EPS) / (tn + fp + EPS)
        return float(specificity.mean())


class HD95Metric(nn.Module):
    def __init__(
        self,
        from_logits: bool = True,
        threshold: float = 0.5,
        both_empty_value: float = 0.0,
        one_empty_value: float | None = None,
    ) -> None:
        super().__init__()
        self.from_logits = from_logits
        self.threshold = threshold
        self.both_empty_value = both_empty_value
        self.one_empty_value = one_empty_value

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_np, target_np = _prepare_binary_predictions(
            pred=pred,
            target=target,
            from_logits=self.from_logits,
            threshold=self.threshold,
        )

        values = []
        for i in range(pred_np.shape[0]):
            pred_i = pred_np[i, 0]
            target_i = target_np[i, 0]
            value = _compute_hd95_single(
                pred_mask=pred_i,
                target_mask=target_i,
                both_empty_value=self.both_empty_value,
                one_empty_value=self.one_empty_value,
            )
            values.append(value)

        return float(np.nanmean(values))


METRIC_REGISTRY: dict[str, type[nn.Module]] = {
    "dice": DiceMetric,
    "iou": IoUMetric,
    "sensitivity": SensitivityMetric,
    "specificity": SpecificityMetric,
    "hd95": HD95Metric,
}