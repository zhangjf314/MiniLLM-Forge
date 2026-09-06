from __future__ import annotations

import torch
from torch import nn

from minillm_forge.model.attention import GroupedQueryAttention
from minillm_forge.model.config import MiniLLMConfig
from minillm_forge.model.mlp import SwiGLU
from minillm_forge.model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_norm(hidden_states), attention_mask, position_ids
        )
        return hidden_states + self.mlp(self.post_attention_norm(hidden_states))
