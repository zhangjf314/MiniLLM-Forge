from __future__ import annotations

import hashlib
import json
from array import array
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from minillm_forge.data.contamination import character_ngrams
from minillm_forge.data.dedup import text_hash
from minillm_forge.tokenization.train_tokenizer import normalize_text


@dataclass
class FrozenTextPartition:
    documents: list[str]
    tokens: torch.Tensor
    raw_text_bytes: int
    duplicate_removals: int = 0
    exact_contamination_removals: int = 0
    near_contamination_removals: int = 0


class ProbeContaminationFilter:
    """Bounded exact/near n-gram filter against frozen evaluation probes."""

    def __init__(self, probes: Iterable[str], *, ngram_size: int, threshold: float) -> None:
        self.ngram_size = ngram_size
        self.threshold = threshold
        self.probe_hashes: set[str] = set()
        self.probe_ngrams: list[set[str]] = []
        self.inverted: dict[str, set[int]] = {}
        for probe in probes:
            normalized = normalize_text(probe)
            if not normalized:
                continue
            self.probe_hashes.add(text_hash(normalized))
            grams = character_ngrams(normalized, ngram_size)
            probe_index = len(self.probe_ngrams)
            self.probe_ngrams.append(grams)
            for gram in grams:
                self.inverted.setdefault(gram, set()).add(probe_index)

    def classify(self, text: str) -> tuple[str | None, float]:
        if text_hash(text) in self.probe_hashes:
            return "normalized_exact", 1.0
        grams = character_ngrams(text, self.ngram_size)
        candidates: set[int] = set()
        for gram in grams:
            candidates.update(self.inverted.get(gram, ()))
        best = 0.0
        for probe_index in candidates:
            probe_grams = self.probe_ngrams[probe_index]
            similarity = len(grams & probe_grams) / max(len(grams | probe_grams), 1)
            best = max(best, similarity)
        return ("ngram", best) if best >= self.threshold else (None, best)


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    values = encoded.ids if hasattr(encoded, "ids") else encoded
    return [int(value) for value in values]


def _append_document_tokens(target: array[int], tokenizer: Any, text: str, eos_id: int) -> int:
    values = _encode(tokenizer, text)
    target.extend(values)
    target.append(eos_id)
    return len(values) + 1


def freeze_math_partitions(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    text_field: str,
    math_validation_tokens: int,
    training_tokens: int,
    benchmark_texts: Iterable[str],
    ngram_size: int = 8,
    near_threshold: float = 0.8,
) -> tuple[FrozenTextPartition, FrozenTextPartition, dict[str, Any]]:
    """Select deterministic first-range math validation and disjoint CPT train data."""
    if min(math_validation_tokens, training_tokens) <= 0:
        raise ValueError("token budgets must be positive")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    iterator: Iterator[Mapping[str, Any]] = iter(rows)
    validation_documents: list[str] = []
    validation_values: array[int] = array("I")
    validation_bytes = 0
    validation_seen: set[str] = set()
    validation_duplicates = 0
    while len(validation_values) < math_validation_tokens:
        try:
            row = next(iterator)
        except StopIteration as exc:
            raise ValueError("source exhausted before math validation budget") from exc
        text = normalize_text(str(row.get(text_field, "")))
        if not text:
            continue
        digest = text_hash(text)
        if digest in validation_seen:
            validation_duplicates += 1
            continue
        validation_seen.add(digest)
        validation_documents.append(text)
        validation_bytes += len(text.encode("utf-8"))
        _append_document_tokens(validation_values, tokenizer, text, eos_id)

    probes = [*validation_documents, *(normalize_text(text) for text in benchmark_texts)]
    contamination_filter = ProbeContaminationFilter(
        probes, ngram_size=ngram_size, threshold=near_threshold
    )
    training_documents: list[str] = []
    training_values: array[int] = array("I")
    training_bytes = 0
    training_seen: set[str] = set()
    duplicates = 0
    removed_exact = 0
    removed_near = 0
    collision_examples: list[dict[str, Any]] = []
    before_count = 0
    while len(training_values) < training_tokens:
        try:
            row = next(iterator)
        except StopIteration as exc:
            raise ValueError("source exhausted before CPT training budget") from exc
        text = normalize_text(str(row.get(text_field, "")))
        if not text:
            continue
        before_count += 1
        digest = text_hash(text)
        if digest in training_seen:
            duplicates += 1
            continue
        collision_type, similarity = contamination_filter.classify(text)
        if collision_type == "normalized_exact":
            removed_exact += 1
        elif collision_type == "ngram":
            removed_near += 1
        if collision_type is not None:
            if len(collision_examples) < 20:
                collision_examples.append(
                    {
                        "type": collision_type,
                        "source_index": before_count - 1,
                        "similarity": round(similarity, 6),
                        "text_sha256": digest,
                    }
                )
            continue
        training_seen.add(digest)
        training_documents.append(text)
        training_bytes += len(text.encode("utf-8"))
        _append_document_tokens(training_values, tokenizer, text, eos_id)

    math_validation = FrozenTextPartition(
        documents=validation_documents,
        tokens=torch.tensor(validation_values[:math_validation_tokens], dtype=torch.long),
        raw_text_bytes=validation_bytes,
        duplicate_removals=validation_duplicates,
    )
    training = FrozenTextPartition(
        documents=training_documents,
        tokens=torch.tensor(training_values[:training_tokens], dtype=torch.long),
        raw_text_bytes=training_bytes,
        duplicate_removals=duplicates,
        exact_contamination_removals=removed_exact,
        near_contamination_removals=removed_near,
    )
    audit = {
        "before_count": before_count,
        "removed_exact": removed_exact,
        "removed_near": removed_near,
        "duplicate_removals": duplicates,
        "after_count": len(training_documents),
        "ngram_size": ngram_size,
        "near_threshold": near_threshold,
        "probe_documents": len(probes),
        "collision_examples": collision_examples,
        "scope_limitation": (
            "This local audit cannot rule out contamination in Qwen's upstream pretraining."
        ),
    }
    return training, math_validation, audit


def freeze_general_partition(
    texts: Iterable[str], tokenizer: Any, *, token_budget: int
) -> FrozenTextPartition:
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    documents: list[str] = []
    values: array[int] = array("I")
    raw_bytes = 0
    seen: set[str] = set()
    duplicates = 0
    for raw_text in texts:
        text = normalize_text(raw_text)
        if not text:
            continue
        digest = text_hash(text)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        documents.append(text)
        raw_bytes += len(text.encode("utf-8"))
        _append_document_tokens(values, tokenizer, text, eos_id)
        if len(values) >= token_budget:
            break
    if len(values) < token_budget:
        raise ValueError("source exhausted before general validation budget")
    return FrozenTextPartition(
        documents=documents,
        tokens=torch.tensor(values[:token_budget], dtype=torch.long),
        raw_text_bytes=raw_bytes,
        duplicate_removals=duplicates,
    )


def tensor_digest(tokens: torch.Tensor) -> str:
    values = tokens.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def write_jsonl(path: str | Path, texts: Iterable[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
