from __future__ import annotations

from typing import Any

import numpy as np
import torch
import yaml

from src.losses.builder import normalize_model_outputs
from src.metrics.metrics import METRIC_REGISTRY


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CompositeMetrics:
    def __init__(self, terms_cfg: list[dict[str, Any]]) -> None:
        if len(terms_cfg) == 0:
            raise ValueError("metrics.terms cannot be empty")

        self.terms_cfg = terms_cfg
        self.metric_modules = []

        for term in terms_cfg:
            metric_name = term["name"]
            params = term.get("params", {})

            if metric_name not in METRIC_REGISTRY:
                raise ValueError(f"Unsupported metric name: {metric_name}")

            metric_module = METRIC_REGISTRY[metric_name](**params)
            self.metric_modules.append(metric_module)

    def __call__(
        self,
        outputs: Any,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        outputs_dict = normalize_model_outputs(outputs)
        results: dict[str, float] = {}

        for term_cfg, metric_module in zip(self.terms_cfg, self.metric_modules):
            metric_name = term_cfg["name"]
            pred_key = term_cfg["pred_key"]
            target_key = term_cfg["target_key"]

            if pred_key not in outputs_dict:
                raise KeyError(f"Prediction key '{pred_key}' not found in model outputs")
            if target_key not in batch:
                raise KeyError(f"Target key '{target_key}' not found in batch")

            pred = outputs_dict[pred_key]
            target = batch[target_key]

            value = metric_module(pred, target)
            results[f"{pred_key}_{metric_name}"] = float(value)

        return results


class MetricMeter:
    def __init__(self) -> None:
        self.storage: dict[str, list[float]] = {}

    def update(self, metric_dict: dict[str, float]) -> None:
        for key, value in metric_dict.items():
            if key not in self.storage:
                self.storage[key] = []
            self.storage[key].append(float(value))

    def compute(self) -> dict[str, float]:
        results = {}
        for key, values in self.storage.items():
            values_np = np.array(values, dtype=np.float32)
            results[key] = float(np.nanmean(values_np))
        return results

    def reset(self) -> None:
        self.storage = {}


def build_metrics_from_config(train_config_path: str) -> CompositeMetrics:
    cfg = load_yaml(train_config_path)
    metrics_cfg = cfg["metrics"]
    terms_cfg = metrics_cfg["terms"]
    return CompositeMetrics(terms_cfg=terms_cfg)