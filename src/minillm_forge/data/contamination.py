from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minillm_forge.data.dedup import normalize_for_match, text_hash


def character_ngrams(text: str, n: int = 8) -> set[str]:
    normalized = normalize_for_match(text)
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def ngram_similarity(left: str, right: str, n: int = 8) -> float:
    left_grams, right_grams = character_ngrams(left, n), character_ngrams(right, n)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def audit_contamination(
    training_texts: list[str],
    benchmark_texts: list[str],
    *,
    threshold: float = 0.8,
    ngram_size: int = 8,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    train_hashes: dict[str, list[int]] = {}
    train_ngrams: list[set[str]] = []
    inverted_ngrams: dict[str, set[int]] = {}
    for index, text in enumerate(training_texts):
        train_hashes.setdefault(text_hash(text), []).append(index)
        grams = character_ngrams(text, ngram_size)
        train_ngrams.append(grams)
        for gram in grams:
            inverted_ngrams.setdefault(gram, set()).add(index)
    collisions: list[dict[str, Any]] = []
    for benchmark_index, benchmark in enumerate(benchmark_texts):
        digest = text_hash(benchmark)
        for train_index in train_hashes.get(digest, []):
            collisions.append(
                {
                    "type": "normalized_exact",
                    "train_index": train_index,
                    "benchmark_index": benchmark_index,
                    "similarity": 1.0,
                }
            )
        if digest in train_hashes:
            continue
        benchmark_grams = character_ngrams(benchmark, ngram_size)
        candidate_indices: set[int] = set()
        for gram in benchmark_grams:
            candidate_indices.update(inverted_ngrams.get(gram, ()))
        for train_index in candidate_indices:
            union = train_ngrams[train_index] | benchmark_grams
            similarity = len(train_ngrams[train_index] & benchmark_grams) / max(len(union), 1)
            if similarity >= threshold:
                collisions.append(
                    {
                        "type": "ngram",
                        "train_index": train_index,
                        "benchmark_index": benchmark_index,
                        "similarity": round(similarity, 6),
                    }
                )
    report = {
        "training_samples": len(training_texts),
        "benchmark_samples": len(benchmark_texts),
        "threshold": threshold,
        "ngram_size": ngram_size,
        "collision_count": len(collisions),
        "collisions": collisions,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
