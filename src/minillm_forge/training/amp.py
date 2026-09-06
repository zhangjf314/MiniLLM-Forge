from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AMPController:
    precision: str
    device_type: str

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        self.enabled = self.precision != "fp32" and self.device_type in {"cuda", "cpu"}
        self.dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        scaler_enabled = self.enabled and self.precision == "fp16" and self.device_type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    def autocast(self) -> Any:
        if not self.enabled:
            return nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.scaler.step(optimizer)
        self.scaler.update()
