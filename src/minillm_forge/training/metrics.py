from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn


def global_parameter_norm(model: nn.Module) -> float:
    squares = [parameter.detach().float().norm(2).pow(2) for parameter in model.parameters()]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


def gradients_are_finite(model: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def tensor_statistics(tensor: torch.Tensor, max_elements: int = 65_536) -> dict[str, float]:
    values = tensor.detach().float().reshape(-1)
    if values.numel() > max_elements:
        stride = max(values.numel() // max_elements, 1)
        values = values[::stride][:max_elements]
    return {
        "activation_mean": float(values.mean()),
        "activation_std": float(values.std()),
        "activation_max_abs": float(values.abs().max()),
    }


class JsonlMetricLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started_at = time.perf_counter()

    def log(self, values: dict[str, Any]) -> None:
        record = {"elapsed_seconds": time.perf_counter() - self.started_at, **values}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def safe_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))
