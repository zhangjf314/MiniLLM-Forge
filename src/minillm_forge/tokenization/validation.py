from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from minillm_forge.tokenization.train_tokenizer import SPECIAL_TOKENS, normalize_text


@dataclass(frozen=True)
class TokenizerStatistics:
    vocab_size: int
    document_count: int
    average_tokens_per_document: float
    unk_ratio: float
    compression_ratio: float
    roundtrip_success_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_tokenizer(tokenizer: Any, texts: list[str]) -> TokenizerStatistics:
    if not texts:
        raise ValueError("at least one validation text is required")
    missing = [token for token in SPECIAL_TOKENS if tokenizer.token_to_id(token) is None]
    if missing:
        raise ValueError(f"tokenizer is missing special tokens: {missing}")
    unk_id = tokenizer.token_to_id("<unk>")
    total_tokens = 0
    unk_tokens = 0
    total_characters = 0
    roundtrip_matches = 0
    for text in texts:
        normalized = normalize_text(text)
        encoding = tokenizer.encode(normalized, add_special_tokens=False)
        total_tokens += len(encoding.ids)
        unk_tokens += sum(token_id == unk_id for token_id in encoding.ids)
        total_characters += len(normalized)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=True)
        roundtrip_matches += normalize_text(decoded) == normalized
    return TokenizerStatistics(
        vocab_size=tokenizer.get_vocab_size(),
        document_count=len(texts),
        average_tokens_per_document=total_tokens / len(texts),
        unk_ratio=unk_tokens / max(total_tokens, 1),
        compression_ratio=total_characters / max(total_tokens, 1),
        roundtrip_success_rate=roundtrip_matches / len(texts),
    )
