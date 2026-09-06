from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from minillm_forge.data.packing import pack_token_sequences
from minillm_forge.tokenization.train_tokenizer import normalize_text


class CausalLMDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, sequences: Sequence[Sequence[int]], pad_token_id: int = 0) -> None:
        if not sequences:
            raise ValueError("sequences must not be empty")
        widths = {len(sequence) for sequence in sequences}
        if len(widths) != 1:
            raise ValueError("all token sequences must have the same length")
        self.tokens = torch.tensor(sequences, dtype=torch.long)
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return self.tokens.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = self.tokens[index]
        attention_mask = input_ids.ne(self.pad_token_id)
        labels = input_ids.clone()
        labels[~attention_mask] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def iter_local_documents(paths: Iterable[str | Path], text_field: str = "text") -> Iterator[str]:
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() == ".jsonl":
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = normalize_text(str(row[text_field]))
                    if text:
                        yield text
            else:
                for line in handle:
                    text = normalize_text(line)
                    if text:
                        yield text


def build_packed_dataset(
    texts: Iterable[str],
    tokenizer: Any,
    sequence_length: int,
    *,
    pad_token_id: int,
    eos_token_id: int,
    drop_remainder: bool = True,
) -> CausalLMDataset:
    def encode(text: str) -> list[int]:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        values = encoded.ids if hasattr(encoded, "ids") else encoded
        return [int(token_id) for token_id in values]

    tokenized = (encode(text) for text in texts)
    sequences = list(
        pack_token_sequences(
            tokenized,
            sequence_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            drop_remainder=drop_remainder,
        )
    )
    return CausalLMDataset(sequences, pad_token_id=pad_token_id)
