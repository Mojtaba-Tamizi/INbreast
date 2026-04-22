from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import yaml

from src.losses.losses import LOSS_REGISTRY


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_model_outputs(outputs: Any) -> dict[str, torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        return {"mask": outputs}

    if isinstance(outputs, dict):
        return outputs

    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 0:
            raise ValueError("Model outputs list/tuple is empty")
        return {"mask": outputs[0]}

    raise TypeError(f"Unsupported model output type: {type(outputs)}")


class CompositeLoss(nn.Module):
    def __init__(self, terms_cfg: list[dict[str, Any]]) -> None:
        super().__init__()

        if len(terms_cfg) == 0:
            raise ValueError("loss.terms cannot be empty")

        self.terms_cfg = terms_cfg
        self.loss_modules = nn.ModuleList()

        for term in terms_cfg:
            loss_name = term["name"]
            params = term.get("params", {})

            if loss_name not in LOSS_REGISTRY:
                raise ValueError(f"Unsupported loss name: {loss_name}")

            loss_module = LOSS_REGISTRY[loss_name](**params)
            self.loss_modules.append(loss_module)

    def forward(
        self,
        outputs: Any,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        outputs_dict = normalize_model_outputs(outputs)

        total_loss = 0.0
        log_dict: dict[str, float] = {}

        for term_cfg, loss_module in zip(self.terms_cfg, self.loss_modules):
            loss_name = term_cfg["name"]
            weight = float(term_cfg.get("weight", 1.0))
            pred_key = term_cfg["pred_key"]
            target_key = term_cfg["target_key"]

            if pred_key not in outputs_dict:
                raise KeyError(f"Prediction key '{pred_key}' not found in model outputs")
            if target_key not in batch:
                raise KeyError(f"Target key '{target_key}' not found in batch")

            pred = outputs_dict[pred_key]
            target = batch[target_key]

            loss_value = loss_module(pred, target)
            weighted_loss = weight * loss_value
            total_loss = total_loss + weighted_loss

            log_name = f"{pred_key}_{loss_name}"
            log_dict[log_name] = float(loss_value.detach().item())
            log_dict[f"{log_name}_weighted"] = float(weighted_loss.detach().item())

        log_dict["total_loss"] = float(total_loss.detach().item())
        return total_loss, log_dict


def build_loss_from_config(train_config_path: str) -> CompositeLoss:
    cfg = load_yaml(train_config_path)
    loss_cfg = cfg["loss"]
    terms_cfg = loss_cfg["terms"]
    return CompositeLoss(terms_cfg=terms_cfg)