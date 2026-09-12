from __future__ import annotations

import torch
from torch import nn


def build_adamw(
    model: nn.Module,
    *,
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    eps: float = 1e-8,
    foreach: bool | None = None,
    fused: bool | None = None,
) -> torch.optim.AdamW:
    """Create AdamW groups so norms and biases are not weight-decayed."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=lr, betas=tuple(betas), eps=eps, foreach=foreach, fused=fused
    )


def build_adamw_8bit(
    model: nn.Module,
    *,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Build bitsandbytes AdamW with quantized optimizer state, not quantized weights."""
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError("8-bit AdamW requires the qlora extra (bitsandbytes)") from exc
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return bnb.optim.AdamW8bit(groups, lr=lr, betas=betas, eps=eps)
