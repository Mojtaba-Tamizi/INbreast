from __future__ import annotations


def compute_start_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    if length <= patch_size:
        return [0]

    positions = list(range(0, length - patch_size + 1, stride))
    last_start = length - patch_size

    if positions[-1] != last_start:
        positions.append(last_start)

    return positions


def generate_sliding_window_boxes(
    image_h: int,
    image_w: int,
    patch_size: int,
    stride: int,
) -> list[tuple[int, int, int, int]]:
    y_starts = compute_start_positions(image_h, patch_size, stride)
    x_starts = compute_start_positions(image_w, patch_size, stride)

    boxes = []
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + patch_size, image_h)
            x1 = min(x0 + patch_size, image_w)
            boxes.append((x0, y0, x1, y1))

    return boxes