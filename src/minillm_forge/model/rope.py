from __future__ import annotations

import torch
from torch import nn


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE) for tensors shaped [B, H, T, D]."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def cos_sin(
        self,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have shape [batch, sequence]")
        if position_ids.numel() and int(position_ids.max()) >= self.max_seq_len:
            raise ValueError(f"position exceeds max_seq_len={self.max_seq_len}")
        frequencies = position_ids.float().unsqueeze(-1) * self.inv_freq.float()
        angles = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(1)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin(position_ids, dtype=query.dtype)
        return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin
