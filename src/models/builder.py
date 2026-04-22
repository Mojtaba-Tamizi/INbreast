from __future__ import annotations

import yaml

from src.models.unet import UNet


MODEL_REGISTRY = {
    "unet": UNet,
}


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_from_config(model_config_path: str):
    cfg = load_yaml(model_config_path)
    model_cfg = cfg["model"]

    model_name = model_cfg["name"]
    model_params = model_cfg.get("params", {})

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model name: {model_name}")

    model = MODEL_REGISTRY[model_name](**model_params)
    return model