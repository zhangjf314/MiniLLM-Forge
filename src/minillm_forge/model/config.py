from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MiniLLMConfig:
    vocab_size: int = 24_000
    hidden_size: int = 512
    num_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    intermediate_size: int = 1_536
    max_seq_len: int = 1_024
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    initializer_range: float = 0.02
    use_rope: bool = True
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    attention_backend: str = "manual"

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_seq_len": self.max_seq_len,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if not 0.0 <= self.residual_dropout < 1.0:
            raise ValueError("residual_dropout must be in [0, 1)")
        if self.attention_backend not in {
            "manual",
            "sdpa_math",
            "sdpa_auto",
            "sdpa_flash",
        }:
            raise ValueError(f"unsupported attention_backend={self.attention_backend!r}")
        for token_id in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
            if not 0 <= token_id < self.vocab_size:
                raise ValueError("special token ids must be inside the vocabulary")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> MiniLLMConfig:
        return cls(**values)
