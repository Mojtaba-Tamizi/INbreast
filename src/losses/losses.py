from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self, from_logits: bool = True) -> None:
        super().__init__()
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()

        if self.from_logits:
            return F.binary_cross_entropy_with_logits(pred, target)

        pred = pred.float()
        return F.binary_cross_entropy(pred, target)


class DiceLoss(nn.Module):
    def __init__(self, from_logits: bool = True, smooth: float = 1.0) -> None:
        super().__init__()
        self.from_logits = from_logits
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()

        if self.from_logits:
            pred = torch.sigmoid(pred)

        pred = pred.float()

        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        denominator = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        loss = 1.0 - dice
        return loss.mean()


LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "bce": BCELoss,
    "dice": DiceLoss,
}