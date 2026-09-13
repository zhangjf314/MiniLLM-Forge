from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from minillm_forge.cli.common import read_jsonl, write_json
from minillm_forge.data.sft import DEFAULT_SYSTEM_PROMPT, _as_ids, encode_sft_example
from minillm_forge.evaluation.math_eval import extract_final_answer, normalize_answer
from minillm_forge.execution.gpu2b import file_sha256, load_frozen_tokenizer

STAGE = "STAGE_GPU_2D_R0"
CONTEXTS = (512, 768, 1024)
TRAIN = Path("data/processed/gpu2b/sft_math_v1_train.jsonl")
VALIDATION = Path("data/processed/gpu2b/sft_math_v1_validation.jsonl")
OUTPUT = Path("artifacts/gpu2d/context_retention_analysis.json")


def _nearest_rank(values: list[int], percentile: int) -> int:
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile / 100) * len(ordered)))
    return ordered[rank - 1]


def _distribution(values: list[int]) -> dict[str, int | float]:
    return {
        "count": len(values),
        "p50": _nearest_rank(values, 50),
        "p75": _nearest_rank(values, 75),
        "p90": _nearest_rank(values, 90),
        "p95": _nearest_rank(values, 95),
        "p99": _nearest_rank(values, 99),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _dataset_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["uuid"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["problem"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["solution"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _analyze_partition(tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_lengths: list[int] = []
    prefix_lengths: list[int] = []
    assistant_lengths: list[int] = []
    contexts: dict[str, dict[str, Any]] = {
        str(context): {
            "context": context,
            "examples": len(rows),
            "examples_unaffected": 0,
            "examples_truncated": 0,
            "examples_losing_assistant_tokens": 0,
            "examples_losing_all_assistant_tokens": 0,
            "examples_eligible_for_training": 0,
            "assistant_tokens_retained": 0,
            "input_tokens_retained": 0,
            "final_answer_retained": 0,
            "reference_answer_available": 0,
        }
        for context in CONTEXTS
    }
    full_reference_matches = 0
    serialization_errors: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prefix_messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": str(row["problem"])},
        ]
        all_messages = [
            *prefix_messages,
            {"role": "assistant", "content": str(row["solution"])},
        ]
        prefix_ids = _as_ids(
            tokenizer.apply_chat_template(
                prefix_messages, tokenize=True, add_generation_prompt=True
            )
        )
        full_ids = _as_ids(
            tokenizer.apply_chat_template(all_messages, tokenize=True, add_generation_prompt=False)
        )
        if full_ids[: len(prefix_ids)] != prefix_ids:
            serialization_errors.append(
                {"index": index, "uuid": row["uuid"], "error": "prefix_mismatch"}
            )
            continue
        total_length = len(full_ids)
        prefix_length = len(prefix_ids)
        assistant_length = total_length - prefix_length
        total_lengths.append(total_length)
        prefix_lengths.append(prefix_length)
        assistant_lengths.append(assistant_length)
        full_answer = normalize_answer(extract_final_answer(str(row["solution"])))
        reference_answer = normalize_answer(extract_final_answer(str(row["reference_answer"])))
        full_reference_matches += full_answer == reference_answer

        for context in CONTEXTS:
            item = contexts[str(context)]
            encoded_error = None
            try:
                encoded = encode_sft_example(
                    tokenizer,
                    user=str(row["problem"]),
                    assistant=str(row["solution"]),
                    system=DEFAULT_SYSTEM_PROMPT,
                    max_length=context,
                )
                production_ids = encoded["input_ids"]
                retained_assistant = sum(label != -100 for label in encoded["labels"])
            except ValueError as exc:
                encoded_error = str(exc)
                production_ids = full_ids[:context]
                retained_assistant = 0
            expected_retained = max(min(total_length, context) - prefix_length, 0)
            if production_ids != full_ids[:context] or retained_assistant != expected_retained:
                serialization_errors.append(
                    {
                        "index": index,
                        "uuid": row["uuid"],
                        "context": context,
                        "error": "production_semantics_mismatch",
                    }
                )
                continue
            if encoded_error and expected_retained:
                serialization_errors.append(
                    {
                        "index": index,
                        "uuid": row["uuid"],
                        "context": context,
                        "error": encoded_error,
                    }
                )
                continue
            item["input_tokens_retained"] += len(production_ids)
            item["assistant_tokens_retained"] += retained_assistant
            item["examples_unaffected"] += total_length <= context
            item["examples_truncated"] += total_length > context
            item["examples_losing_assistant_tokens"] += retained_assistant < assistant_length
            item["examples_losing_all_assistant_tokens"] += retained_assistant == 0
            item["examples_eligible_for_training"] += retained_assistant > 0
            retained_text = tokenizer.decode(
                production_ids[prefix_length:], skip_special_tokens=True
            )
            retained_answer = normalize_answer(extract_final_answer(retained_text))
            item["final_answer_retained"] += retained_answer == full_answer
            item["reference_answer_available"] += retained_answer == reference_answer

    assistant_total = sum(assistant_lengths)
    input_total = sum(total_lengths)
    for item in contexts.values():
        item["assistant_token_retention_rate"] = item["assistant_tokens_retained"] / assistant_total
        item["input_token_retention_rate"] = item["input_tokens_retained"] / input_total
        item["final_answer_retention_rate"] = item["final_answer_retained"] / len(rows)
        item["reference_answer_availability_rate"] = item["reference_answer_available"] / len(rows)
    return {
        "examples": len(rows),
        "dataset_digest": _dataset_digest(rows),
        "total_tokens": _distribution(total_lengths),
        "prefix_tokens": _distribution(prefix_lengths),
        "assistant_tokens": _distribution(assistant_lengths),
        "untruncated_input_tokens": input_total,
        "untruncated_assistant_tokens": assistant_total,
        "full_solution_reference_matches": full_reference_matches,
        "full_solution_reference_match_rate": full_reference_matches / len(rows),
        "contexts": contexts,
        "serialization_errors": serialization_errors,
    }


def analyze_context_retention() -> dict[str, Any]:
    tokenizer = load_frozen_tokenizer()
    train_rows = read_jsonl(TRAIN)
    validation_rows = read_jsonl(VALIDATION)
    partitions = {
        "train": _analyze_partition(tokenizer, train_rows),
        "validation": _analyze_partition(tokenizer, validation_rows),
    }
    train_1024 = partitions["train"]["contexts"]["1024"]["assistant_tokens_retained"]
    for context in CONTEXTS:
        item = partitions["train"]["contexts"][str(context)]
        item["assistant_token_retention_relative_to_1024"] = (
            item["assistant_tokens_retained"] / train_1024
        )
    errors = [
        error for partition in partitions.values() for error in partition["serialization_errors"]
    ]
    result = {
        "stage": STAGE,
        "classification": (
            "GPU2D_CONTEXT_RETENTION_ANALYSIS_COMPLETE"
            if not errors
            else "GPU2D_CONTEXT_RETENTION_ANALYSIS_FAILED"
        ),
        "status": "PASS" if not errors else "FAIL",
        "method": {
            "serialization": "production encode_sft_example and pinned Qwen chat template",
            "truncation": "right truncation of complete serialized message",
            "percentiles": "nearest-rank",
            "final_answer_retention": (
                "normalized current-scorer extraction after truncation equals normalized "
                "current-scorer extraction from the untruncated solution"
            ),
            "reference_answer_availability": (
                "normalized current-scorer extraction after truncation equals the frozen "
                "reference_answer"
            ),
        },
        "contexts": list(CONTEXTS),
        "source_files": {
            "train": {"path": str(TRAIN), "sha256": file_sha256(TRAIN)},
            "validation": {
                "path": str(VALIDATION),
                "sha256": file_sha256(VALIDATION),
            },
        },
        "partitions": partitions,
        "historical_artifacts_modified": False,
        "formal_training_launches": 0,
        "benchmark_launches": 0,
    }
    write_json(OUTPUT, result)
    return result
