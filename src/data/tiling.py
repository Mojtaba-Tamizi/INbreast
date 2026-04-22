from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from src.utils.patch_utils import generate_sliding_window_boxes


def read_rgb_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    orig_dtype = image.dtype
    image = image.astype(np.float32)

    if np.issubdtype(orig_dtype, np.integer):
        image = image / 255.0

    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float()


def apply_transform_to_tile(
    tile: np.ndarray,
    transform: Any | None = None,
) -> np.ndarray:
    if transform is None:
        return tile

    transformed = transform(image=tile)
    if not isinstance(transformed, dict) or "image" not in transformed:
        raise ValueError("Transform must return a dict containing 'image'")
    return transformed["image"]


def build_tile_batch(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    start_idx: int,
    batch_size: int,
    transform: Any | None = None,
) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
    batch_boxes = boxes[start_idx : start_idx + batch_size]
    batch_tensors = []

    for x0, y0, x1, y1 in batch_boxes:
        tile = image[y0:y1, x0:x1]
        tile = apply_transform_to_tile(tile, transform=transform)
        tile_tensor = image_to_tensor(tile)
        batch_tensors.append(tile_tensor)

    batch_tensor = torch.stack(batch_tensors, dim=0)
    return batch_tensor, batch_boxes


def _allocate_output_maps(
    image_h: int,
    image_w: int,
    num_output_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred_sum = np.zeros((num_output_channels, image_h, image_w), dtype=np.float32)
    count_map = np.zeros((1, image_h, image_w), dtype=np.float32)
    return pred_sum, count_map


def _update_output_maps(
    pred_sum: np.ndarray,
    count_map: np.ndarray,
    batch_preds: np.ndarray,
    batch_boxes: list[tuple[int, int, int, int]],
) -> None:
    for pred, (x0, y0, x1, y1) in zip(batch_preds, batch_boxes):
        pred_sum[:, y0:y1, x0:x1] += pred
        count_map[:, y0:y1, x0:x1] += 1.0


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: int,
    stride: int,
    device: torch.device | str,
    batch_size: int = 4,
    transform: Any | None = None,
    apply_sigmoid: bool = True,
    apply_softmax: bool = False,
) -> np.ndarray:
    """
    Predict full-image segmentation map with sliding-window inference.

    Args:
        model:
            Segmentation model. Output shape must be [B, C, H, W].
        image:
            RGB image as numpy array [H, W, 3].
        patch_size:
            Sliding window patch size.
        stride:
            Sliding window stride.
        device:
            Torch device.
        batch_size:
            Number of tiles per forward pass.
        transform:
            Image-only preprocessing transform for inference tiles.
        apply_sigmoid:
            Apply sigmoid to model output. Use for binary segmentation.
        apply_softmax:
            Apply softmax to model output. Use for multi-class segmentation.

    Returns:
        full_pred:
            Stitched prediction map as numpy array [C, H, W].
    """
    if apply_sigmoid and apply_softmax:
        raise ValueError("Only one of apply_sigmoid/apply_softmax can be True.")

    model.eval()

    image_h, image_w = image.shape[:2]
    boxes = generate_sliding_window_boxes(
        image_h=image_h,
        image_w=image_w,
        patch_size=patch_size,
        stride=stride,
    )

    pred_sum = None
    count_map = None

    for start_idx in range(0, len(boxes), batch_size):
        batch_tensor, batch_boxes = build_tile_batch(
            image=image,
            boxes=boxes,
            start_idx=start_idx,
            batch_size=batch_size,
            transform=transform,
        )
        batch_tensor = batch_tensor.to(device, non_blocking=True)

        logits = model(batch_tensor)

        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        if logits.ndim != 4:
            raise ValueError(f"Model output must have shape [B, C, H, W], got {logits.shape}")

        if apply_sigmoid:
            preds = torch.sigmoid(logits)
        elif apply_softmax:
            preds = torch.softmax(logits, dim=1)
        else:
            preds = logits

        preds_np = preds.detach().cpu().numpy()

        if pred_sum is None or count_map is None:
            num_output_channels = preds_np.shape[1]
            pred_sum, count_map = _allocate_output_maps(
                image_h=image_h,
                image_w=image_w,
                num_output_channels=num_output_channels,
            )

        _update_output_maps(
            pred_sum=pred_sum,
            count_map=count_map,
            batch_preds=preds_np,
            batch_boxes=batch_boxes,
        )

    if pred_sum is None or count_map is None:
        raise RuntimeError("No predictions were generated.")

    count_map = np.clip(count_map, a_min=1e-8, a_max=None)
    full_pred = pred_sum / count_map
    return full_pred.astype(np.float32)


def predict_binary_mask(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: int,
    stride: int,
    device: torch.device | str,
    batch_size: int = 4,
    transform: Any | None = None,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        prob_map: [H, W] float32
        bin_mask: [H, W] uint8 in {0,1}
    """
    full_pred = sliding_window_predict(
        model=model,
        image=image,
        patch_size=patch_size,
        stride=stride,
        device=device,
        batch_size=batch_size,
        transform=transform,
        apply_sigmoid=True,
        apply_softmax=False,
    )

    if full_pred.shape[0] != 1:
        raise ValueError(
            f"Binary prediction expects output channels=1, got {full_pred.shape[0]}"
        )

    prob_map = full_pred[0]
    bin_mask = (prob_map >= threshold).astype(np.uint8)
    return prob_map, bin_mask


def predict_multiclass_mask(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: int,
    stride: int,
    device: torch.device | str,
    batch_size: int = 4,
    transform: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        prob_map: [C, H, W] float32
        label_map: [H, W] int64
    """
    full_pred = sliding_window_predict(
        model=model,
        image=image,
        patch_size=patch_size,
        stride=stride,
        device=device,
        batch_size=batch_size,
        transform=transform,
        apply_sigmoid=False,
        apply_softmax=True,
    )

    label_map = np.argmax(full_pred, axis=0).astype(np.int64)
    return full_pred, label_map


def save_probability_map(prob_map: np.ndarray, save_path: str) -> None:
    prob_u8 = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(save_path, prob_u8)
    if not ok:
        raise IOError(f"Could not save probability map to: {save_path}")


def save_binary_mask(mask: np.ndarray, save_path: str) -> None:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    ok = cv2.imwrite(save_path, mask_u8)
    if not ok:
        raise IOError(f"Could not save binary mask to: {save_path}")