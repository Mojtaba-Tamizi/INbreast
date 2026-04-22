from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device | str,
) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


class RunningAverageDict:
    def __init__(self) -> None:
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, values: dict[str, float], n: int = 1) -> None:
        for key, value in values.items():
            self.sums[key] += float(value) * n
            self.counts[key] += n

    def compute(self) -> dict[str, float]:
        out = {}
        for key in self.sums:
            count = max(self.counts[key], 1)
            out[key] = self.sums[key] / count
        return out


def _get_device_type(device: torch.device | str) -> str:
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        return device.split(":")[0]
    return "cpu"


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    metrics_fn=None,
    device: torch.device | str = "cpu",
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
    max_grad_norm: float | None = None,
    scheduler=None,
    scheduler_step_on_batch: bool = False,
    log_interval: int = 50,
) -> dict[str, float]:
    model.train()

    meter = RunningAverageDict()
    device_type = _get_device_type(device)
    amp_enabled = use_amp and device_type == "cuda"

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device_type, enabled=amp_enabled):
            outputs = model(batch["image"])
            loss, loss_dict = criterion(outputs, batch)

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()

            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        if scheduler is not None and scheduler_step_on_batch:
            scheduler.step()

        with torch.no_grad():
            metric_dict = metrics_fn(outputs, batch) if metrics_fn is not None else {}

        step_dict = {}
        step_dict.update(loss_dict)
        step_dict.update(metric_dict)

        batch_size = int(batch["image"].shape[0])
        meter.update(step_dict, n=batch_size)

        if log_interval > 0 and (step + 1) % log_interval == 0:
            current = meter.compute()
            print(
                f"[TRAIN] step={step + 1}/{len(loader)} "
                f"loss={current.get('total_loss', 0.0):.4f}"
            )

    epoch_results = meter.compute()
    epoch_results["lr"] = float(optimizer.param_groups[0]["lr"])
    return epoch_results