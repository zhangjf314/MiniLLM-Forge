from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch


@contextmanager
def measure_efficiency(device: torch.device) -> Iterator[dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    metrics: dict[str, Any] = {}
    try:
        yield metrics
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            metrics["peak_vram_mb"] = None
        metrics["elapsed_seconds"] = time.perf_counter() - started


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
