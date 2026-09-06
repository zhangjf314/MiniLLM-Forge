from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_SYSTEM_PROMPT = "You are a mathematical reasoning assistant."


def _as_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    if isinstance(tokenized, torch.Tensor):
        tokenized = tokenized.tolist()
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return [int(value) for value in tokenized]


def encode_sft_example(
    tokenizer: Any,
    *,
    user: str,
    assistant: str,
    system: str = DEFAULT_SYSTEM_PROMPT,
    max_length: int = 1_024,
) -> dict[str, list[int]]:
    """Tokenize a chat sample and train only assistant tokens.

    Prefix tokenization uses the model's own chat template, which avoids assuming
    any Qwen-specific control-token ids.
    """
    if not user.strip() or not assistant.strip():
        raise ValueError("user and assistant content must not be empty")
    prefix_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    all_messages = [*prefix_messages, {"role": "assistant", "content": assistant}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prefix_ids = _as_ids(
            tokenizer.apply_chat_template(
                prefix_messages, tokenize=True, add_generation_prompt=True
            )
        )
        input_ids = _as_ids(
            tokenizer.apply_chat_template(all_messages, tokenize=True, add_generation_prompt=False)
        )
    else:
        prefix = f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"
        full_text = prefix + assistant
        prefix_ids = _as_ids(tokenizer(prefix, add_special_tokens=True)["input_ids"])
        input_ids = _as_ids(tokenizer(full_text, add_special_tokens=True)["input_ids"])

    if input_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("chat template produced inconsistent prefix and full-message tokens")
    input_ids = input_ids[:max_length]
    assistant_start = min(len(prefix_ids), len(input_ids))
    labels = [-100] * assistant_start + input_ids[assistant_start:]
    if all(label == -100 for label in labels):
        raise ValueError("max_length truncates every assistant token")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


@dataclass
class SFTDataCollator:
    pad_token_id: int
    label_pad_token_id: int = -100
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("features must not be empty")
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple
        input_ids, attention_masks, labels = [], [], []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(
                feature.get("attention_mask", [1] * len(feature["input_ids"])) + [0] * padding
            )
            labels.append(feature["labels"] + [self.label_pad_token_id] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def is_high_quality_math_sample(
    sample: dict[str, Any],
    *,
    max_characters: int = 12_000,
) -> bool:
    problem = str(sample.get("problem") or sample.get("question") or "").strip()
    solution = str(sample.get("solution") or sample.get("answer") or "").strip()
    if not problem or not solution or len(solution) > max_characters:
        return False
    if "final" not in solution.lower() and "\\boxed" not in solution:
        return False
    return "\x00" not in problem and "\x00" not in solution
