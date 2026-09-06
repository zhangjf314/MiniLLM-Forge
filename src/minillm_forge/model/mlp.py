from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from minillm_forge.model.config import MiniLLMConfig


class SwiGLU(nn.Module):
    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        activated = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.dropout(self.down_proj(activated))
