from __future__ import annotations

import cv2
import numpy as np


def load_binary_mask(mask_path: str) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    mask = (mask > 127).astype(np.uint8)
    return mask


def make_boundary_mask(
    binary_mask: np.ndarray,
    mode: str = "xor_erode",
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    if mode == "xor_erode":
        eroded = cv2.erode(binary_mask, kernel, iterations=iterations)
        boundary = cv2.bitwise_xor(binary_mask, eroded)

    elif mode == "morph_gradient":
        dilated = cv2.dilate(binary_mask, kernel, iterations=iterations)
        eroded = cv2.erode(binary_mask, kernel, iterations=iterations)
        boundary = dilated - eroded
        boundary = (boundary > 0).astype(np.uint8)

    else:
        raise ValueError(f"Unsupported boundary mode: {mode}")

    return boundary.astype(np.uint8)


def save_mask(mask: np.ndarray, save_path: str) -> None:
    mask_to_save = (mask * 255).astype(np.uint8)
    ok = cv2.imwrite(save_path, mask_to_save)
    if not ok:
        raise IOError(f"Could not save mask to: {save_path}")