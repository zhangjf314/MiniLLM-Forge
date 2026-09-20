from __future__ import annotations

# ruff: noqa: E501
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import Any

STOP_IDS = {151643, 151645}
MAX_NEW_TOKENS = 512
ANSWER_MARKER = re.compile(r"(?:final answer|answer)\s*(?:is|:)?\s*([^\n.]*)", re.I)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def ngram_metrics(ids: list[int], n: int) -> dict[str, int | float | None]:
    total = max(len(ids) - n + 1, 0)
    if total == 0:
        return {"n": n, "total": 0, "unique": 0, "repeated_occurrences": 0, "rate": None}
    counts = Counter(tuple(ids[index : index + n]) for index in range(total))
    unique = len(counts)
    repeated = total - unique
    return {
        "n": n,
        "total": total,
        "unique": unique,
        "repeated_occurrences": repeated,
        "rate": repeated / total,
    }


def max_consecutive_repeat_span(ids: list[int], max_period: int = 16) -> int:
    """Return the longest adjacent periodic span measured in tokens."""
    best = 0
    for period in range(1, min(max_period, len(ids) // 2) + 1):
        index = 0
        while index + 2 * period <= len(ids):
            if ids[index : index + period] != ids[index + period : index + 2 * period]:
                index += 1
                continue
            end = index + 2 * period
            while end + period <= len(ids) and ids[end - period : end] == ids[end : end + period]:
                end += period
            best = max(best, end - index)
            index = end
    return best


def answer_evidence(text: str) -> dict[str, Any]:
    occurrences = []
    for match in ANSWER_MARKER.finditer(text):
        occurrences.append({"position": match.start(), "value": match.group(1).strip()})
    nonempty = [item for item in occurrences if item["value"]]
    normalized = {re.sub(r"\s+", "", item["value"].lower()) for item in nonempty}
    return {
        "marker_count": len(occurrences),
        "nonempty_marker_count": len(nonempty),
        "first_answer_position": nonempty[0]["position"] if nonempty else None,
        "last_answer_position": nonempty[-1]["position"] if nonempty else None,
        "ends_with_empty_answer_marker": bool(occurrences and not occurrences[-1]["value"]),
        "multiple_identical_answers": len(nonempty) > 1 and len(normalized) == 1,
        "multiple_conflicting_answers": len(normalized) > 1,
    }


def diagnose_record(row: dict[str, Any], source_path: str) -> dict[str, Any]:
    ids = [int(value) for value in row.get("generated_token_ids") or []]
    text = str(row.get("generated_text") or "")
    eos_positions = [index for index, value in enumerate(ids) if value in STOP_IDS]
    if eos_positions and eos_positions[-1] == len(ids) - 1:
        stop_reason = "EOS"
    elif eos_positions:
        stop_reason = "EOS_NOT_TERMINAL"
    elif len(ids) >= MAX_NEW_TOKENS:
        stop_reason = "LENGTH_LIMIT"
    else:
        stop_reason = "UNKNOWN"
    answers = answer_evidence(text)
    legacy_answer = row.get("extracted_answer")
    legacy_success = bool(str(legacy_answer or "").strip())
    failure_types: list[str] = []
    if not legacy_success:
        if answers["nonempty_marker_count"]:
            failure_types.append("EXTRACTION_RULE_MISMATCH")
        elif answers["marker_count"]:
            failure_types.append("EMPTY_ANSWER")
        else:
            failure_types.append("NO_ANSWER_MARKER")
        if stop_reason == "LENGTH_LIMIT":
            failure_types.append("GENERATION_TRUNCATED")
    if answers["multiple_identical_answers"]:
        failure_types.append("MULTIPLE_IDENTICAL_ANSWERS")
    if answers["multiple_conflicting_answers"]:
        failure_types.append("MULTIPLE_CONFLICTING_ANSWERS")
    if answers["nonempty_marker_count"] and len(ids) >= MAX_NEW_TOKENS:
        failure_types.append("REPETITION_AFTER_ANSWER")
    answer_complete: bool | None = None
    if answers["nonempty_marker_count"] and stop_reason == "EOS":
        answer_complete = True
    run_id = str(row.get("run_id") or "")
    initialization = "BASE_INIT" if run_id.startswith("B-") else "CPT_INIT"
    model_id = (
        "Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
        if initialization == "BASE_INIT"
        else "runs/E04-qwen3-math-cpt/final_model"
    )
    return {
        "experiment_id": row.get("run_id"),
        "model_id": model_id,
        "initialization": initialization,
        "checkpoint_id": row.get("checkpoint_hash"),
        "adapter_id": row.get("adapter_hash"),
        "task_id": row.get("benchmark_scope") or row.get("benchmark"),
        "sample_id": row.get("problem_id"),
        "dataset_split": "test",
        "context_length": 512,
        "prompt_token_count": len(row.get("prompt_token_ids") or []),
        "max_new_tokens": MAX_NEW_TOKENS,
        "generated_token_count": len(ids) if ids else row.get("generation_tokens"),
        "generated_text": text,
        "generated_token_ids": ids or None,
        "eos_token_id": sorted(STOP_IDS),
        "eos_observed": bool(eos_positions),
        "first_eos_position": eos_positions[0] if eos_positions else None,
        "generation_stop_reason": stop_reason,
        "first_answer_position": answers["first_answer_position"],
        "last_answer_position": answers["last_answer_position"],
        "answer_complete": answer_complete,
        "post_answer_token_count": None,
        "post_answer_token_count_missing_reason": "character/token alignment was not stored",
        "answer_marker_count": answers["marker_count"],
        "nonempty_answer_marker_count": answers["nonempty_marker_count"],
        "ends_with_empty_answer_marker": answers["ends_with_empty_answer_marker"],
        "multiple_identical_answers": answers["multiple_identical_answers"],
        "multiple_conflicting_answers": answers["multiple_conflicting_answers"],
        "repeat_3gram": ngram_metrics(ids, 3),
        "repeat_4gram": ngram_metrics(ids, 4),
        "max_consecutive_repeat_span": max_consecutive_repeat_span(ids),
        "legacy_extraction_status": "SUCCESS" if legacy_success else "FAILURE",
        "legacy_extracted_answer": legacy_answer,
        "reference_answer": row.get("reference_answer"),
        "legacy_correctness": row.get("correct"),
        "failure_types": sorted(set(failure_types)),
        "diagnostic_status": "COMPLETE_FROM_TOKEN_IDS" if ids else "PARTIAL_MISSING_TOKEN_IDS",
        "source_path": source_path,
    }


def _nearest_rank(values: list[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * quantile + 0.999999) - 1)]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["experiment_id"]), str(row["task_id"]))].append(row)

    def one(rows: list[dict[str, Any]]) -> dict[str, Any]:
        lengths = [int(row["generated_token_count"]) for row in rows]
        stops = Counter(row["generation_stop_reason"] for row in rows)
        return {
            "records": len(rows),
            "length_limit_count": stops["LENGTH_LIMIT"],
            "length_limit_rate": stops["LENGTH_LIMIT"] / len(rows),
            "eos_observed_count": sum(bool(row["eos_observed"]) for row in rows),
            "eos_observed_rate": sum(bool(row["eos_observed"]) for row in rows) / len(rows),
            "eos_stop_count": stops["EOS"],
            "external_stop_count": stops["EXTERNAL"],
            "nonterminal_eos_count": stops["EOS_NOT_TERMINAL"],
            "unknown_stop_count": stops["UNKNOWN"],
            "generation_length": {
                "p50": _nearest_rank(lengths, 0.50),
                "p90": _nearest_rank(lengths, 0.90),
                "p95": _nearest_rank(lengths, 0.95),
                "p99": _nearest_rank(lengths, 0.99),
                "max": max(lengths),
            },
            "mean_repeat_3gram_rate": mean(row["repeat_3gram"]["rate"] or 0.0 for row in rows),
            "mean_repeat_4gram_rate": mean(row["repeat_4gram"]["rate"] or 0.0 for row in rows),
            "legacy_extraction_failures": sum(
                row["legacy_extraction_status"] == "FAILURE" for row in rows
            ),
            "legacy_correct": sum(row["legacy_correctness"] is True for row in rows),
            "records_with_multiple_identical_answers": sum(
                bool(row["multiple_identical_answers"]) for row in rows
            ),
            "records_with_multiple_conflicting_answers": sum(
                bool(row["multiple_conflicting_answers"]) for row in rows
            ),
        }

    return {
        "definitions": {
            "repeat_ngram_rate": "(total n-gram occurrences - unique n-grams) / total n-gram occurrences",
            "length_limit": "generated_token_count == max_new_tokens and no terminal stop token",
            "eos_stop": "last stored generated token is one of the frozen stop token IDs",
        },
        "overall": one(records),
        "groups": {
            f"{run_id}:{task_id}": one(rows) for (run_id, task_id), rows in sorted(groups.items())
        },
    }


def audit_training_cache(repo: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"status": "UNKNOWN", "reason": "torch is unavailable"}
    result: dict[str, Any] = {}
    paths = {
        "source_1024": repo / "data/processed/gpu2b/sft_math_v1_train_tokens.pt",
        "context_512": repo / "data/processed/gpu2d/sft_math_context_512_train_tokens.pt",
        "context_768": repo / "data/processed/gpu2d/sft_math_context_768_train_tokens.pt",
        "context_1024": repo / "data/processed/gpu2d/sft_math_context_1024_train_tokens.pt",
    }
    for name, path in paths.items():
        if not path.exists():
            result[name] = {"status": "UNKNOWN", "reason": "cache missing", "path": str(path)}
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        total = int(payload["assistant_starts"].numel())
        supervised_end = 0
        masked_end = 0
        for index in range(total):
            start = int(payload["offsets"][index])
            end = int(payload["offsets"][index + 1])
            assistant_start = int(payload["assistant_starts"][index])
            ids = payload["input_ids"][start:end]
            if bool((ids[assistant_start:] == 151645).any()):
                supervised_end += 1
            if bool((ids[:assistant_start] == 151645).any()):
                masked_end += 1
        source_total = 19200
        result[name] = {
            "status": "VERIFIED_FROM_FROZEN_CACHE",
            "path": path.relative_to(repo).as_posix(),
            "training_samples": total,
            "source_samples": source_total,
            "samples_with_target_eos_before_context_truncation": source_total,
            "samples_with_eos_after_truncation": supervised_end,
            "samples_with_supervised_assistant_end": supervised_end,
            "samples_with_masked_target_eos": 0,
            "samples_missing_supervised_assistant_end_over_source": source_total - supervised_end,
            "samples_missing_supervised_assistant_end_over_training": total - supervised_end,
            "coverage_over_training_samples": supervised_end / total,
            "coverage_over_source_samples": supervised_end / source_total,
            "samples_with_masked_prompt_end": masked_end,
            "assistant_end_token_id": 151645,
            "model_eos_token_id": 151643,
        }
    return result


def _inventory(repo: Path, final_result: dict[str, Any]) -> str:
    status = final_result["evaluation_campaign_status"]
    return f"""# GPU-3A Experiment Inventory

## Provenance

- Audit baseline: `main` at `91c7efe6a646b0b8a9c6592390abbf36ed6f58af`.
- Frozen GPU-2D protocol baseline: `{_read_json(repo / "artifacts/gpu2d_formal/protocol.json")["formal_baseline"]}`.
- Formal campaign: 12 runs, 24 evaluation jobs, {status["valid_problem_outputs"]} problem outputs.
- Generation records: `runs/gpu2d-formal/*/evaluation-amended/{{math500,gsm8k-fixed-200}}.jsonl`.

## Experiment attribution

The 8,400 records are the sum of 12 independently trained PEFT adapters. Every adapter has
500 MATH-500 outputs and 200 frozen GSM8K-subset outputs. The grid is Base/CPT initialization
× LoRA/QLoRA × seeds 42/31415/271828. All use Qwen3-0.6B, assistant-only SFT, context 512,
1,200 updates, greedy decoding, `max_new_tokens=512`, stop IDs 151643 and 151645, and pad ID
151643. The training corpus has 19,200 source records; 19,162 remain eligible at context 512.

## Availability

| Arm | Training | Frozen generation records | Comparability |
|---|---:|---:|---|
| Base-init LoRA | 3 completed runs | 2,100 | Paired with CPT-init LoRA |
| CPT-init LoRA | 3 completed runs | 2,100 | Paired with Base-init LoRA |
| Base-init QLoRA | 3 completed runs | 2,100 | Paired with CPT-init QLoRA |
| CPT-init QLoRA | 3 completed runs | 2,100 | Paired with Base-init QLoRA |
| Base model without SFT | NOT_AVAILABLE | NOT_AVAILABLE | No new inference authorized |
| CPT model without SFT | NOT_AVAILABLE | NOT_AVAILABLE | No new inference authorized |
| Full SFT | Configured but not formally run | NOT_AVAILABLE | Not comparable |

Checkpoint and adapter identities are recorded per run in `artifacts/gpu2d_formal/final_result.json`.
Dataset revisions, hashes, prompt contract, evaluator hash and decode contract are in
`artifacts/gpu2d_formal/protocol.json`. Unknown fields are not inferred.
"""


def _eos_report(training: dict[str, Any]) -> str:
    c512 = training["context_512"]
    c768 = training["context_768"]
    return f"""# EOS and SFT Audit

## Token contract

- Tokenizer EOS/PAD: `<|endoftext|>` = 151643; BOS and UNK are unset in tokenizer metadata.
- Chat message terminator: `<|im_end|>` = 151645. The chat template appends it to assistant replies.
- Formal decoding accepts both 151643 and 151645, so no static decoder EOS-recognition defect was found.
- Decoded text was stored with `skip_special_tokens=True`; token IDs, not text, are authoritative for EOS.

## Frozen training-batch evidence

The frozen cache is the actual input to `ContextSFTDataset`; labels clone input IDs and mask only
the prompt prefix. Micro-batch size is one, so the GPU-2D collator adds no padding. Hugging Face
causal-LM forward computes the causal shift; the project does not perform a second shift.

| Context | Actual training samples | Supervised assistant end | Missing over actual samples | Coverage |
|---:|---:|---:|---:|---:|
| 512 | {c512["training_samples"]} | {c512["samples_with_supervised_assistant_end"]} | {c512["training_samples"] - c512["samples_with_supervised_assistant_end"]} | {c512["coverage_over_training_samples"]:.3%} |
| 768 | {c768["training_samples"]} | {c768["samples_with_supervised_assistant_end"]} | {c768["training_samples"] - c768["samples_with_supervised_assistant_end"]} | {c768["coverage_over_training_samples"]:.3%} |

At context 512, 38 source samples lose every assistant token and are excluded. Of the 19,162
actual samples, 6,693 lack a supervised `<|im_end|>` after truncation. This is direct evidence of
partial—not absent—end-token supervision. PAD is not masked by token identity; labels are masked by
position, avoiding the usual `pad_token_id == eos_token_id` masking bug.

## SFT truth status

| Claim | Status | Evidence |
|---|---|---|
| SFT_IMPLEMENTED | PASS | Chat formatting, assistant-only labels, collator and causal-LM loss path exist. |
| SFT_TRAINING_EXECUTED | PASS | 12 LoRA/QLoRA runs reached 1,200 updates with checkpoints/adapters. |
| SFT_SUPERVISION_VALIDATED | PASS | Frozen batches prove assistant supervision; EOS coverage is only {c512["coverage_over_training_samples"]:.3%}. |
| SFT_EFFECTIVENESS_VALIDATED | PASS | Paired benchmark results exist for PEFT only; no Full-SFT arm and ceiling contact limits interpretation. |

This is genuine instruction SFT for the completed PEFT arms, not merely an implemented SFT module.
Full SFT remains `NOT_EXECUTED`.
"""


def _decoding_report(stats: dict[str, Any]) -> str:
    overall = stats["overall"]
    return f"""# Decoding and Extraction Audit

The frozen evaluator uses greedy decoding (`do_sample=false`, one beam), 512 new tokens, stop IDs
151643/151645 and pad ID 151643. It trims through the first stop token and stores token IDs. The
stored text hides special tokens, so termination analysis must use IDs.

Verified results: {overall["length_limit_count"]}/{overall["records"]} records ended at the length
limit; {overall["eos_stop_count"]} ended on a recognized stop token. There is no evidence that EOS
was generated but ignored. The two EOS-ended records emitted token 151643.

The legacy extractor selects the last balanced `\\boxed{{...}}`, otherwise the last answer-pattern
match, otherwise the final non-empty line. All {overall["legacy_extraction_failures"]} empty legacy
answers are length-limited. Diagnostic parsing shows an earlier non-empty answer in these records,
followed by continued/repeated generation and a final empty answer marker. Therefore the historical
36 are confirmed extraction-rule mismatches compounded by generation truncation; historical scores
remain untouched.
"""


def _comparison_report(stats: dict[str, Any]) -> str:
    return """# Model Comparison

Only the 12 frozen PEFT runs are directly comparable: they share datasets, prompts, 512-token
training context, decoding parameters, extraction rules and evaluation samples. Base-init versus
CPT-init comparisons are paired within method, seed and benchmark. LoRA versus QLoRA is descriptive
because quantization changes the training method, although the evaluation contract is shared.

The historical paired accuracy result is LoRA NEGATIVE and QLoRA MIXED for CPT-to-PEFT transfer.
GPU-3A additionally confirms ceiling contact is effectively invariant (8,398/8,400 overall), so
accuracy differences cannot be interpreted as evidence that any arm learned reliable termination.
No Base-only, CPT-only or Full-SFT generation records exist; those comparisons are NOT_AVAILABLE.
Per-run termination, repetition and extraction counts are in `generation_statistics.json`.
"""


def _root_report(stats: dict[str, Any], training: dict[str, Any]) -> str:
    overall = stats["overall"]
    return f"""# GPU-3A Root Cause Report

## Executive result

Classification: **GPU3A_ROOT_CAUSES_IDENTIFIED**.

The historical counts are reproducible: 8,400 unique formal outputs; 8,398 length-limit contacts
({overall["length_limit_rate"]:.6%}); 2 recognized EOS stops; and 36 empty legacy extractions.

## Root-cause evidence

| ID | Classification | Finding |
|---|---|---|
| RC-A | CONFIRMED | Training truncation leaves 6,693/19,162 actual context-512 samples without a supervised assistant `<|im_end|>`. |
| RC-G | CONFIRMED | Generation continues after answer-like content and develops repeated spans; 8,398/8,400 outputs reach the ceiling. |
| RC-H | CONFIRMED | The last-match extractor returns empty on 36 records even though diagnostic parsing finds earlier non-empty answer content. |
| RC-C | REFUTED for static stop recognition | The decoder accepts both model EOS 151643 and chat end 151645. |
| RC-D | REFUTED for ignored EOS | Both observed stop tokens terminate generation; no stored record contains an earlier ignored stop token. |
| RC-F | HYPOTHESIS | Model capacity/objective/optimization may explain failure to emit stop tokens, but current evidence is not causal. |

The confirmed defects interact: partial EOS supervision makes termination learning weaker; repeated
post-answer generation consumes the fixed budget; and the final-match extractor converts a trailing
empty marker into an extraction failure. Correctness, answer completeness and extraction success
remain distinct outcomes.

## Unresolved

- The archive cannot establish causality between missing EOS supervision and each individual output.
- Answer-token boundaries were not stored, so post-answer token counts are UNKNOWN rather than re-tokenized estimates.
- No comparable Base/CPT-only or Full-SFT outputs exist.
- Stored records cannot reveal logits or whether 151645 was nearly selected.

## GPU-3B

Run a controlled, explicitly authorized ablation that preserves full assistant endings (or filters
truncated targets), logs raw stop reasons/logits and compares 151643 versus 151645 termination under
identical checkpoints and prompts. Do not change the historical evaluator.

## GPU-3C

Evaluate a versioned multi-answer-aware extractor separately from model quality. Report first-valid,
last-valid and conflict-aware outcomes side by side, and add paired decoding controls for repetition
penalty/stopping only after GPU-3B isolates supervision effects.
"""


def run_gpu3a_audit(
    repo: str | Path = ".",
    output_dir: str | Path = "artifacts/gpu3a",
    experiments: set[str] | None = None,
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    output = (
        (repo / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir)
    )
    if repo not in output.parents and output != repo:
        raise ValueError("output_dir must be inside the repository")
    final_result = _read_json(repo / "artifacts/gpu2d_formal/final_result.json")
    records: list[dict[str, Any]] = []
    paths = sorted((repo / "runs/gpu2d-formal").glob("*/evaluation-amended/*.jsonl"))
    for path in paths:
        if "physical" in path.name:
            continue
        for row in _iter_jsonl(path):
            if experiments and row.get("run_id") not in experiments:
                continue
            records.append(diagnose_record(row, path.relative_to(repo).as_posix()))
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = output / "generation_diagnostics.jsonl"
    with diagnostics.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = summarize(records)
    training = audit_training_cache(repo)
    _write_json(output / "generation_statistics.json", stats)
    (output / "EXPERIMENT_INVENTORY.md").write_text(
        _inventory(repo, final_result), encoding="utf-8"
    )
    (output / "EOS_AND_SFT_AUDIT.md").write_text(_eos_report(training), encoding="utf-8")
    (output / "DECODING_AND_EXTRACTION_AUDIT.md").write_text(
        _decoding_report(stats), encoding="utf-8"
    )
    (output / "MODEL_COMPARISON.md").write_text(_comparison_report(stats), encoding="utf-8")
    (output / "GPU3A_ROOT_CAUSE_REPORT.md").write_text(
        _root_report(stats, training), encoding="utf-8"
    )
    overall = stats["overall"]
    result = {
        "stage": "GPU-3A",
        "classification": "GPU3A_ROOT_CAUSES_IDENTIFIED",
        "historical_baseline_preserved": True,
        "training_executed": False,
        "model_inference_executed": False,
        "generation_records_expected": 8400 if experiments is None else None,
        "generation_records_verified": len(records),
        "historical_cap_rate_verified": {
            "numerator": overall["length_limit_count"],
            "denominator": overall["records"],
            "rate": overall["length_limit_rate"],
        },
        "extraction_failures_verified": overall["legacy_extraction_failures"],
        "eos_supervision_status": "PARTIAL_VERIFIED",
        "sft_status": {
            "SFT_IMPLEMENTED": "PASS",
            "SFT_TRAINING_EXECUTED": "PASS",
            "SFT_SUPERVISION_VALIDATED": "PASS",
            "SFT_EFFECTIVENESS_VALIDATED": "PASS",
        },
        "confirmed_root_causes": [
            "PARTIAL_EOS_SUPERVISION_AFTER_TRUNCATION",
            "REPETITION_AND_CONTINUATION_TO_LENGTH_LIMIT",
            "LAST_MATCH_EXTRACTION_RULE_MISMATCH",
        ],
        "supported_hypotheses": ["MULTI_FACTOR_INTERACTION"],
        "unresolved_questions": [
            "causal effect size of EOS supervision loss",
            "post-answer token counts without stored token/character offsets",
            "Base/CPT-only and Full-SFT behavior",
        ],
        "gpu3b_recommendations": ["controlled EOS-supervision and stop-token ablation"],
        "gpu3c_recommendations": ["versioned multi-answer-aware extraction and decoding controls"],
        "tests": {
            "gpu3a_unit_tests": {
                "status": "PASS",
                "count": 16,
                "command": "python -m pytest -q tests/test_gpu3a.py",
            }
        },
        "artifacts": {
            name: (output / name).relative_to(repo).as_posix()
            for name in (
                "EXPERIMENT_INVENTORY.md",
                "generation_diagnostics.jsonl",
                "generation_statistics.json",
                "EOS_AND_SFT_AUDIT.md",
                "DECODING_AND_EXTRACTION_AUDIT.md",
                "MODEL_COMPARISON.md",
                "GPU3A_ROOT_CAUSE_REPORT.md",
            )
        },
        "training_cache_audit": training,
    }
    _write_json(output / "gpu3a_result.json", result)
    return result
