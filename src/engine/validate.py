from __future__ import annotations

from typing import Any

import numpy as np
import torch
import yaml

from src.data.tiling import build_tile_batch, read_rgb_image
from src.engine.train import RunningAverageDict, move_batch_to_device
from src.utils.patch_utils import generate_sliding_window_boxes


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _pad_image_to_min_size(
    image: np.ndarray,
    min_h: int,
    min_w: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    orig_h, orig_w = image.shape[:2]

    pad_h = max(0, min_h - orig_h)
    pad_w = max(0, min_w - orig_w)

    if pad_h == 0 and pad_w == 0:
        return image, (orig_h, orig_w)

    padded = np.pad(
        image,
        pad_width=((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect",
    )
    return padded, (orig_h, orig_w)

def _normalize_model_outputs(outputs: Any) -> dict[str, torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        return {"mask": outputs}

    if isinstance(outputs, dict):
        return outputs

    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 0:
            raise ValueError("Model returned an empty list/tuple")
        return {"mask": outputs[0]}

    raise TypeError(f"Unsupported model output type: {type(outputs)}")


def _meta_value(meta: dict[str, Any], key: str, index: int):
    value = meta[key]
    if torch.is_tensor(value):
        v = value[index]
        return v.item() if v.ndim == 0 else v
    return value[index]


def _slice_tensor_batch(batch: dict[str, Any], index: int) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if key == "meta":
            continue
        if torch.is_tensor(value):
            out[key] = value[index:index + 1]
    return out


def _infer_required_output_keys(criterion=None, metrics_fn=None) -> list[str]:
    keys = set()

    if criterion is not None and hasattr(criterion, "terms_cfg"):
        keys.update(term["pred_key"] for term in criterion.terms_cfg)

    if metrics_fn is not None and hasattr(metrics_fn, "terms_cfg"):
        keys.update(term["pred_key"] for term in metrics_fn.terms_cfg)

    if len(keys) == 0:
        keys.add("mask")

    return sorted(keys)


# @torch.no_grad()
# def sliding_window_predict_outputs(
#     model: torch.nn.Module,
#     image: np.ndarray,
#     patch_size: int,
#     stride: int,
#     device: torch.device | str,
#     batch_size: int = 4,
#     transform: Any | None = None,
#     output_keys: list[str] | None = None,
# ) -> dict[str, np.ndarray]:
#     model.eval()

#     image_h, image_w = image.shape[:2]
#     boxes = generate_sliding_window_boxes(
#         image_h=image_h,
#         image_w=image_w,
#         patch_size=patch_size,
#         stride=stride,
#     )

#     pred_sums: dict[str, np.ndarray] = {}
#     count_map = np.zeros((1, image_h, image_w), dtype=np.float32)

#     for start_idx in range(0, len(boxes), batch_size):
#         batch_tensor, batch_boxes = build_tile_batch(
#             image=image,
#             boxes=boxes,
#             start_idx=start_idx,
#             batch_size=batch_size,
#             transform=transform,
#         )
#         batch_tensor = batch_tensor.to(device, non_blocking=True)

#         outputs = _normalize_model_outputs(model(batch_tensor))

#         if output_keys is None:
#             current_keys = list(outputs.keys())
#         else:
#             current_keys = output_keys

#         preds_np_dict = {
#             key: outputs[key].detach().cpu().numpy()
#             for key in current_keys
#         }

#         for key, preds_np in preds_np_dict.items():
#             if key not in pred_sums:
#                 pred_sums[key] = np.zeros(
#                     (preds_np.shape[1], image_h, image_w),
#                     dtype=np.float32,
#                 )

#             for pred, (x0, y0, x1, y1) in zip(preds_np, batch_boxes):
#                 pred_sums[key][:, y0:y1, x0:x1] += pred

#         for x0, y0, x1, y1 in batch_boxes:
#             count_map[:, y0:y1, x0:x1] += 1.0

#     count_map = np.clip(count_map, a_min=1e-8, a_max=None)
#     stitched = {key: pred_sum / count_map for key, pred_sum in pred_sums.items()}
#     return stitched
@torch.no_grad()
def sliding_window_predict_outputs(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: int,
    stride: int,
    device: torch.device | str,
    batch_size: int = 4,
    transform: Any | None = None,
    output_keys: list[str] | None = None,
) -> dict[str, np.ndarray]:
    model.eval()

    # Pad small images so every tile fed to the model matches patch_size
    image, (orig_h, orig_w) = _pad_image_to_min_size(
        image=image,
        min_h=patch_size,
        min_w=patch_size,
    )

    image_h, image_w = image.shape[:2]
    boxes = generate_sliding_window_boxes(
        image_h=image_h,
        image_w=image_w,
        patch_size=patch_size,
        stride=stride,
    )

    pred_sums: dict[str, np.ndarray] = {}
    count_map = np.zeros((1, image_h, image_w), dtype=np.float32)

    for start_idx in range(0, len(boxes), batch_size):
        batch_tensor, batch_boxes = build_tile_batch(
            image=image,
            boxes=boxes,
            start_idx=start_idx,
            batch_size=batch_size,
            transform=transform,
        )
        batch_tensor = batch_tensor.to(device, non_blocking=True)

        outputs = _normalize_model_outputs(model(batch_tensor))

        if output_keys is None:
            current_keys = list(outputs.keys())
        else:
            current_keys = output_keys

        preds_np_dict = {
            key: outputs[key].detach().cpu().numpy()
            for key in current_keys
        }

        for key, preds_np in preds_np_dict.items():
            if key not in pred_sums:
                pred_sums[key] = np.zeros(
                    (preds_np.shape[1], image_h, image_w),
                    dtype=np.float32,
                )

            for pred, (x0, y0, x1, y1) in zip(preds_np, batch_boxes):
                pred_sums[key][:, y0:y1, x0:x1] += pred

        for x0, y0, x1, y1 in batch_boxes:
            count_map[:, y0:y1, x0:x1] += 1.0

    count_map = np.clip(count_map, a_min=1e-8, a_max=None)

    stitched = {}
    for key, pred_sum in pred_sums.items():
        pred = pred_sum / count_map
        pred = pred[:, :orig_h, :orig_w]   # crop back to original image size
        stitched[key] = pred

    return stitched

def build_validation_params_from_config(train_config_path: str) -> dict[str, Any]:
    cfg = load_yaml(train_config_path)
    val_cfg = cfg["validation"]

    return {
        "patch_size": int(val_cfg["sliding_window"]["patch_size"]),
        "stride": int(val_cfg["sliding_window"]["stride"]),
        "tile_batch_size": int(val_cfg["sliding_window"]["tile_batch_size"]),
        "threshold": float(val_cfg.get("threshold", 0.5)),
    }


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    metrics_fn=None,
    device: torch.device | str = "cpu",
    patch_size: int = 256,
    stride: int = 128,
    tile_batch_size: int = 4,
    tile_transform: Any | None = None,
    log_interval: int = 20,
) -> dict[str, float]:
    model.eval()

    meter = RunningAverageDict()
    required_output_keys = _infer_required_output_keys(
        criterion=criterion,
        metrics_fn=metrics_fn,
    )

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        if "meta" not in batch:
            raise KeyError("Validation batch must contain 'meta' with image_path")

        meta = batch["meta"]
        batch_size = int(batch["image"].shape[0])

        for i in range(batch_size):
            image_path = _meta_value(meta, "image_path", i)
            raw_image = read_rgb_image(image_path)

            stitched_outputs_np = sliding_window_predict_outputs(
                model=model,
                image=raw_image,
                patch_size=patch_size,
                stride=stride,
                device=device,
                batch_size=tile_batch_size,
                transform=tile_transform,
                output_keys=required_output_keys,
            )

            stitched_outputs = {
                key: torch.from_numpy(value).unsqueeze(0).to(device)
                for key, value in stitched_outputs_np.items()
            }

            sample_batch = _slice_tensor_batch(batch, i)

            loss, loss_dict = criterion(stitched_outputs, sample_batch)

            if metrics_fn is not None:
                metric_dict = metrics_fn(stitched_outputs, sample_batch)
            else:
                metric_dict = {}

            sample_dict = {}
            sample_dict.update(loss_dict)
            sample_dict.update(metric_dict)

            meter.update(sample_dict, n=1)

        if log_interval > 0 and (step + 1) % log_interval == 0:
            current = meter.compute()
            print(
                f"[VAL] step={step + 1}/{len(loader)} "
                f"loss={current.get('total_loss', 0.0):.4f}"
            )

    return meter.compute()