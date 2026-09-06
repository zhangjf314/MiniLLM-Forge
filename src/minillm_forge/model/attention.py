from __future__ import annotations

import torch
from torch import nn

from minillm_forge.model.config import MiniLLMConfig
from minillm_forge.model.rope import RotaryEmbedding


def repeat_key_value(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Expand KV heads to query heads while preserving group ordering."""
    if repeats == 1:
        return hidden_states
    return hidden_states.repeat_interleave(repeats, dim=1)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)
        self.rope = (
            RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
            if config.use_rope
            else None
        )

    def _shape(self, values: torch.Tensor, heads: int) -> torch.Tensor:
        batch, seq_len, _ = values.shape
        return values.view(batch, seq_len, heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}"
            )

        query = self._shape(self.q_proj(hidden_states), self.num_heads)
        key = self._shape(self.k_proj(hidden_states), self.num_kv_heads)
        value = self._shape(self.v_proj(hidden_states), self.num_kv_heads)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).expand(batch, -1)
        if self.rope is not None:
            query, key = self.rope(query, key, position_ids)

        key = repeat_key_value(key, self.config.kv_groups)
        value = repeat_key_value(value, self.config.kv_groups)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        causal = torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool).tril()
        allowed = causal.view(1, 1, seq_len, seq_len)
        if attention_mask is not None:
            if attention_mask.shape != (batch, seq_len):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            allowed = allowed & attention_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        probabilities = self.attention_dropout(probabilities)
        context = torch.matmul(probabilities, value)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.residual_dropout(self.o_proj(context))


class CausalSelfAttention(GroupedQueryAttention):
    """MHA specialization retained as an explicit architecture ablation."""

    def __init__(self, config: MiniLLMConfig) -> None:
        if config.num_key_value_heads != config.num_attention_heads:
            config = MiniLLMConfig.from_dict(
                {**config.to_dict(), "num_key_value_heads": config.num_attention_heads}
            )
        super().__init__(config)
