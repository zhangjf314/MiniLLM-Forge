from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def stream_huggingface_texts(
    dataset_name: str,
    *,
    subset: str | None = None,
    split: str = "train",
    text_field: str = "text",
    revision: str | None = None,
    max_samples: int | None = None,
) -> Iterator[str]:
    """Stream a pinned dataset revision without downloading the full corpus."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the 'datasets' dependency to stream Hugging Face data") from exc
    dataset = load_dataset(
        dataset_name,
        subset,
        split=split,
        revision=revision,
        streaming=True,
    )
    for index, row in enumerate(dataset):
        if max_samples is not None and index >= max_samples:
            break
        value: Any = row.get(text_field)
        if value:
            yield str(value)
