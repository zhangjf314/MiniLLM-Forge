from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from minillm_forge.model.block import TransformerBlock
from minillm_forge.model.config import MiniLLMConfig
from minillm_forge.model.rmsnorm import RMSNorm


@dataclass
class MiniLLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Apply the next-token label shift exactly once."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must share batch and sequence dimensions")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


class MiniLLM(nn.Module):
    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.apply(self._initialize_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> MiniLLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        hidden_states = self.token_embedding(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, attention_mask, position_ids)
        logits = self.lm_head(self.norm(hidden_states))
        loss = causal_lm_loss(logits, labels) if labels is not None else None
        return MiniLLMOutput(logits=logits, loss=loss)

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_seq_len :]
            logits = self(context).logits[:, -1]
            if temperature > 0:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if bool((next_token == self.config.eos_token_id).all()):
                break
        self.train(was_training)
        return generated
