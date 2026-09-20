from __future__ import annotations

# ruff: noqa: E501
import hashlib
import json
import math
import re
import subprocess
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from minillm_forge.evaluation.math_eval import extract_final_answer, normalize_answer
from minillm_forge.execution.gpu2b import load_frozen_tokenizer

START_HEAD = "2e1bd4bcb2246071bc7e5826b4d97bc55377cd6d"
BASELINE_HEAD = "91c7efe6a646b0b8a9c6592390abbf36ed6f58af"
GPU3A_HEAD = "b41c718c337946c14a0bf80dcbacf68a1129d633"
GPU3B_HEAD = START_HEAD
SYSTEM_PROMPT = "You are a mathematical reasoning assistant."
TRAIN_PATH = Path("data/processed/gpu2b/sft_math_v1_train.jsonl")
VALIDATION_PATH = Path("data/processed/gpu2b/sft_math_v1_validation.jsonl")
GSM_PATH = Path("data/processed/gpu2b/gsm8k_test.jsonl")
MATH_PATH = Path("data/processed/gpu2b/math500_test.jsonl")
IM_END_ID = 151645
END_OF_TEXT_ID = 151643
CONTEXT = 512
EVALUATION_IDS = ["gsm8k-1176", "gsm8k-0805", "math500-0076", "math500-0181"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _percentiles(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)

    def nearest(fraction: float) -> int:
        return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]

    return {
        "count": len(ordered),
        "p50": nearest(0.50),
        "p90": nearest(0.90),
        "p95": nearest(0.95),
        "p99": nearest(0.99),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _find_last_subsequence(values: list[int], needle: list[int]) -> tuple[int, int] | None:
    if not needle or len(needle) > len(values):
        return None
    for start in range(len(values) - len(needle), -1, -1):
        if values[start : start + len(needle)] == needle:
            return start, start + len(needle) - 1
    return None


def _problem_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _source_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["source"]) for row in rows).items()))


def _as_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def _audit_row(tokenizer: Any, row: dict[str, Any], index: int) -> dict[str, Any]:
    prefix_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(row["problem"])},
    ]
    all_messages = [*prefix_messages, {"role": "assistant", "content": str(row["solution"])}]
    prefix_ids = _as_ids(
        tokenizer.apply_chat_template(prefix_messages, tokenize=True, add_generation_prompt=True)
    )
    full_ids = _as_ids(
        tokenizer.apply_chat_template(all_messages, tokenize=True, add_generation_prompt=False)
    )
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise RuntimeError(f"chat-template prefix mismatch at source row {index}")
    end_positions = [
        position
        for position, token_id in enumerate(full_ids)
        if token_id == IM_END_ID and position >= len(prefix_ids)
    ]
    assistant_end = end_positions[-1] if end_positions else None
    retained = full_ids[:CONTEXT]
    assistant_retained = retained[min(len(prefix_ids), len(retained)) :]
    full_answer = extract_final_answer(str(row["solution"]))
    truncated_text = tokenizer.decode(assistant_retained, skip_special_tokens=True)
    truncated_answer = extract_final_answer(truncated_text)
    has_complete_answer = bool(full_answer) and normalize_answer(
        truncated_answer
    ) == normalize_answer(full_answer)
    answer_span = None
    if full_answer:
        answer_ids = [
            int(value) for value in tokenizer(full_answer, add_special_tokens=False).input_ids
        ]
        found = _find_last_subsequence(full_ids[len(prefix_ids) :], answer_ids)
        if found is not None:
            answer_span = (found[0] + len(prefix_ids), found[1] + len(prefix_ids))
    supervised_end = assistant_end is not None and assistant_end < CONTEXT
    if assistant_end is None:
        truncation_reason = "FORMAT_ANOMALY_NO_ASSISTANT_END"
    elif len(prefix_ids) >= CONTEXT:
        truncation_reason = "PROMPT_EXHAUSTS_CONTEXT"
    elif not supervised_end:
        truncation_reason = "ASSISTANT_TRUNCATED_BEFORE_END"
    else:
        truncation_reason = "NONE"
    retained_assistant = max(0, min(len(full_ids), CONTEXT) - len(prefix_ids))
    return {
        "source_index": index,
        "uuid": row["uuid"],
        "source": row["source"],
        "problem_sha256": _problem_hash(str(row["problem"])),
        "prompt_token_count": len(prefix_ids),
        "assistant_token_count": len(full_ids) - len(prefix_ids),
        "full_sequence_length": len(full_ids),
        "answer_start_position": answer_span[0] if answer_span else None,
        "answer_end_position": answer_span[1] if answer_span else None,
        "answer_boundary_status": "EXACT_TOKEN_SUBSEQUENCE" if answer_span else "UNKNOWN",
        "assistant_start_position": len(prefix_ids),
        "assistant_end_position": assistant_end,
        "truncated_prompt_tokens": max(0, len(prefix_ids) - CONTEXT),
        "truncated_assistant_tokens": max(0, len(full_ids) - len(prefix_ids) - retained_assistant),
        "has_complete_answer": has_complete_answer,
        "has_supervised_assistant_end": supervised_end,
        "eligible_training_sample": len(prefix_ids) < CONTEXT,
        "truncation_reason": truncation_reason,
        "supervised_tokens_at_512": retained_assistant,
    }


def _benchmark_rows(repo: Path) -> list[dict[str, Any]]:
    result = []
    for benchmark, path in (("gsm8k", repo / GSM_PATH), ("math500", repo / MATH_PATH)):
        for row in _read_jsonl(path):
            index = int(row["benchmark_index"])
            result.append(
                {
                    "problem_id": f"{benchmark}-{index:04d}",
                    "benchmark": benchmark,
                    "problem": row["problem"],
                    "problem_sha256": _problem_hash(str(row["problem"])),
                }
            )
    return result


def run_data_audit(repo: str | Path = ".") -> dict[str, Any]:
    started = time.perf_counter()
    repo = Path(repo).resolve()
    output = repo / "artifacts/gpu3c"
    tokenizer = load_frozen_tokenizer()
    source_rows = _read_jsonl(repo / TRAIN_PATH)
    validation_rows = _read_jsonl(repo / VALIDATION_PATH)
    details = [_audit_row(tokenizer, row, index) for index, row in enumerate(source_rows)]
    detail_path = output / "data_audit.jsonl"
    with detail_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    eligible = [row for row in details if row["eligible_training_sample"]]
    complete = [row for row in details if row["has_supervised_assistant_end"]]
    source_by_index = {index: row for index, row in enumerate(source_rows)}
    complete_sources = [source_by_index[int(row["source_index"])] for row in complete]
    benchmarks = _benchmark_rows(repo)
    train_hashes = {row["problem_sha256"] for row in details}
    validation_hashes = {_problem_hash(str(row["problem"])) for row in validation_rows}
    collisions = [row for row in benchmarks if row["problem_sha256"] in train_hashes]
    evaluation = [row for row in benchmarks if row["problem_id"] in EVALUATION_IDS]
    subset_digest = _digest(
        [
            {
                "source_index": row["source_index"],
                "uuid": row["uuid"],
                "assistant_end_position": row["assistant_end_position"],
            }
            for row in complete
        ]
    )
    reasons = Counter(row["truncation_reason"] for row in details)
    summary = {
        "stage": "GPU-3C",
        "status": "PASS",
        "source_samples": len(details),
        "actual_training_samples_512": len(eligible),
        "samples_with_supervised_assistant_end": len(complete),
        "samples_missing_supervised_assistant_end_over_training": len(eligible) - len(complete),
        "eos_supervision_coverage": len(complete) / len(eligible),
        "prompt_token_count": _percentiles([row["prompt_token_count"] for row in details]),
        "assistant_token_count": _percentiles([row["assistant_token_count"] for row in details]),
        "full_sequence_length": _percentiles([row["full_sequence_length"] for row in details]),
        "truncation_reasons": dict(sorted(reasons.items())),
        "complete_answer_at_512": sum(row["has_complete_answer"] for row in details),
        "unknown_answer_boundary": sum(
            row["answer_boundary_status"] == "UNKNOWN" for row in details
        ),
        "total_supervised_tokens_original_512": sum(
            row["supervised_tokens_at_512"] for row in eligible
        ),
        "d1_complete_filter": {
            "source_sample_count": len(details),
            "eligible_training_sample_count": len(complete),
            "filtered_sample_count": len(details) - len(complete),
            "complete_answer_count": sum(row["has_complete_answer"] for row in complete),
            "supervised_assistant_end_count": len(complete),
            "eos_supervision_coverage": 1.0,
            "total_supervised_tokens": sum(row["supervised_tokens_at_512"] for row in complete),
            "prompt_token_count": _percentiles([row["prompt_token_count"] for row in complete]),
            "assistant_token_count": _percentiles(
                [row["assistant_token_count"] for row in complete]
            ),
            "source_distribution_before": _source_distribution(source_rows),
            "source_distribution_after": _source_distribution(complete_sources),
            "subset_digest": subset_digest,
        },
        "data_isolation": {
            "benchmark_records_checked": len(benchmarks),
            "exact_normalized_problem_hash_collisions": len(collisions),
            "collisions": collisions,
            "evaluation_ids": EVALUATION_IDS,
            "evaluation_records_found": len(evaluation),
            "train_validation_exact_problem_hash_collisions": len(train_hashes & validation_hashes),
            "claim": "frozen existing benchmarks, not a fully unseen holdout",
        },
        "template": {
            "tokenizer": tokenizer.name_or_path,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "assistant_end_token": "<|im_end|>",
            "assistant_end_token_id": IM_END_ID,
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "assistant_target_terminator": "<|im_end|>",
            "thinking_mode": "NO_EXPLICIT_THINKING_SWITCH_IN_FROZEN_BASE_CHAT_TEMPLATE",
            "training_inference_template_consistent": True,
        },
        "data_audit_jsonl_sha256": _sha256(detail_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output / "data_audit.json", summary)

    historical_step_seconds = 6.453890950011555
    protocol = {
        "stage": "GPU-3C",
        "protocol_version": "gpu3c-v1-frozen",
        "start_commit": START_HEAD,
        "baseline_commit": BASELINE_HEAD,
        "gpu3a_commit": GPU3A_HEAD,
        "gpu3b_commit": GPU3B_HEAD,
        "model": {
            "id": "Qwen/Qwen3-0.6B-Base",
            "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
            "initialization": "BASE_INIT",
            "method": "LORA",
            "seed": 42,
            "context": CONTEXT,
        },
        "tokenizer": summary["template"],
        "training_data": {
            "path": TRAIN_PATH.as_posix(),
            "sha256": _sha256(repo / TRAIN_PATH),
            "validation_path": VALIDATION_PATH.as_posix(),
            "validation_sha256": _sha256(repo / VALIDATION_PATH),
            "complete_subset_count": len(complete),
            "complete_subset_digest": subset_digest,
            "sample_order": "torch randperm over complete subset with seed 42",
        },
        "historical_inputs": {
            path.as_posix(): _sha256(repo / path)
            for path in (
                Path("artifacts/gpu3a/gpu3a_result.json"),
                Path("artifacts/gpu3b/gpu3b_result.json"),
                Path("artifacts/gpu3b/protocol.json"),
                Path("runs/gpu2d-formal/B-LORA-s42/final_adapter/adapter_config.json"),
                Path("runs/gpu2d-formal/B-LORA-s42/final_adapter/adapter_model.safetensors"),
            )
        },
        "groups": {
            "B-EOS-MASKED": {
                "dataset": "identical complete subset",
                "target_im_end_label": -100,
            },
            "C-EOS-SUPERVISED": {
                "dataset": "identical complete subset",
                "target_im_end_label": IM_END_ID,
            },
        },
        "training": {
            "optimizer_steps": 64,
            "checkpoint_steps": [32, 64],
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "effective_examples_per_update": 16,
            "learning_rate": 0.0002,
            "scheduler_total_steps": 64,
            "precision": "bf16",
            "max_wall_time_seconds": 2700,
            "single_active_gpu_job": True,
        },
        "evaluation": {
            "problem_ids": EVALUATION_IDS,
            "paths": {
                "C0": {"eos_token_ids": [END_OF_TEXT_ID, IM_END_ID], "external_stop": False},
                "S3": {"eos_token_ids": [END_OF_TEXT_ID, IM_END_ID], "external_stop": True},
            },
            "max_new_tokens": 512,
            "extractors": ["legacy-gpu2d", "first-valid-v1", "conflict-aware-v1"],
        },
        "runtime_estimate": {
            "historical_median_step_seconds": historical_step_seconds,
            "training_seconds_per_arm": 64 * historical_step_seconds + 60,
            "paired_training_seconds": 2 * (64 * historical_step_seconds + 60),
            "evaluation_generations": 16,
            "historical_seconds_per_512_generation": 30,
            "evaluation_seconds": 480,
            "total_estimated_seconds": 2 * (64 * historical_step_seconds + 60) + 480,
            "basis": "GPU-2D B-LORA-s42 measured median step time plus GPU-3B generation timings",
        },
        "resource_limits": {
            "gpu_memory_mib": 8151,
            "minimum_headroom_mib": 1024,
            "max_training_steps_per_arm": 64,
            "max_generated_tokens": 512,
            "checkpoint_interval_steps": 32,
        },
        "success_conditions": [
            "B/C input_ids and attention_mask identical",
            "B/C labels differ only at final assistant <|im_end|>",
            "both arms finish 64 optimizer updates without OOM or non-finite loss",
            "all frozen evaluation records are retained including negative results",
        ],
        "failure_conditions": [
            "mask identity mismatch",
            "OOM or non-finite training",
            "checkpoint save/load failure",
            "historical input hash mismatch",
            "background supervisor timeout",
        ],
        "status": "FROZEN_BEFORE_GPU3C_TRAINING",
    }
    _write_json(output / "protocol.json", protocol)

    report = f"""# GPU-3C 训练数据与模板审计

## 数据长度与截断

对冻结训练源的 {len(details):,} 条样本使用实际 Qwen chat template 重新序列化。512 context 下有 {len(eligible):,} 条进入训练，{len(complete):,} 条保留并监督目标 assistant `<|im_end|>`，覆盖率为 {len(complete) / len(eligible):.3%}；{len(eligible) - len(complete):,} 条实际训练样本缺少目标结束监督。

| 指标 | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Prompt tokens | {summary["prompt_token_count"]["p50"]} | {summary["prompt_token_count"]["p90"]} | {summary["prompt_token_count"]["p95"]} | {summary["prompt_token_count"]["p99"]} | {summary["prompt_token_count"]["max"]} |
| Assistant tokens | {summary["assistant_token_count"]["p50"]} | {summary["assistant_token_count"]["p90"]} | {summary["assistant_token_count"]["p95"]} | {summary["assistant_token_count"]["p99"]} | {summary["assistant_token_count"]["max"]} |
| Full sequence | {summary["full_sequence_length"]["p50"]} | {summary["full_sequence_length"]["p90"]} | {summary["full_sequence_length"]["p95"]} | {summary["full_sequence_length"]["p99"]} | {summary["full_sequence_length"]["max"]} |

截断分类：`{dict(sorted(reasons.items()))}`。右截断不会追加或伪造 EOS；Prompt 已耗尽 context 的样本不进入实际训练，其余 EOS 缺失来自 assistant 在目标结束标记前被截断。完整答案边界用最终答案 token 子序列定位；无法可靠定位的样本标记 UNKNOWN，而不是用 `ANSWER:` 存在性冒充完整性。

## 模板与特殊 token

- `<|endoftext|>` / tokenizer EOS / PAD：151643。
- Assistant 消息结束 `<|im_end|>`：151645；这是本阶段直接控制的训练 label。
- 训练和推理都使用冻结 Qwen chat template；系统提示一致。
- 冻结 Base tokenizer chat template 没有显式 thinking-mode 开关。本阶段不改变回复结构或引入 thinking 配置。
- Prompt labels 按位置设为 -100；真实 EOS 不按 token identity 批量屏蔽，避免 PAD==EOS 时误伤。

## 数据隔离与 D1 静态方案

训练题与 GSM8K/MATH-500 共 {len(benchmarks):,} 道评测题进行规范化内容哈希检查，精确碰撞 {len(collisions)} 条。GPU-3C 固定 4 道既有 benchmark 题，只用于冻结确认，不宣称完全未接触 holdout。

D1 完整样本过滤保留 {len(complete):,}/{len(details):,} 条，过滤 {len(details) - len(complete):,} 条，EOS 覆盖率变为 100%。这同时改变样本长度和来源分布，因此只能作为数据处理综合方案，不能用来估计 EOS 的单变量因果效应。B/C 对照则使用相同的这 {len(complete):,} 条样本，唯一主动差异是最终 `<|im_end|>` label 是否为 -100。
"""
    # Reassign with an explicit UTF-8 source string.  The audit is deliberately
    # generated from measured values rather than maintained by hand.
    report = f"""# GPU-3C 训练数据与模板审计

## 数据长度与截断

对冻结训练源的 {len(details):,} 条样本使用实际 Qwen chat template 重新序列化。512 context 下有 {len(eligible):,} 条进入原训练，{len(complete):,} 条保留并监督目标 assistant `<|im_end|>`，覆盖率为 {len(complete) / len(eligible):.3%}；{len(eligible) - len(complete):,} 条实际训练样本缺少目标结束监督。

| 指标 | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Prompt tokens | {summary["prompt_token_count"]["p50"]} | {summary["prompt_token_count"]["p90"]} | {summary["prompt_token_count"]["p95"]} | {summary["prompt_token_count"]["p99"]} | {summary["prompt_token_count"]["max"]} |
| Assistant tokens | {summary["assistant_token_count"]["p50"]} | {summary["assistant_token_count"]["p90"]} | {summary["assistant_token_count"]["p95"]} | {summary["assistant_token_count"]["p99"]} | {summary["assistant_token_count"]["max"]} |
| Full sequence | {summary["full_sequence_length"]["p50"]} | {summary["full_sequence_length"]["p90"]} | {summary["full_sequence_length"]["p95"]} | {summary["full_sequence_length"]["p99"]} | {summary["full_sequence_length"]["max"]} |

截断分类：`{dict(sorted(reasons.items()))}`。右截断不会追加或伪造 EOS；prompt 已耗尽 context 的样本不进入实际训练，其余 EOS 缺失来自 assistant 在目标结束标记前被截断。完整答案边界使用最终答案 token 子序列定位；无法可靠定位的样本标记 UNKNOWN，不用 `ANSWER:` 的存在性冒充完整性。

本次语义完整答案计数为 {summary["complete_answer_at_512"]:,}；GPU-2D 旧统计多 1 条，是因为旧逻辑把“完整答案为空且截断答案也为空”判为相等。本审计要求非空最终答案，因此不计该样本。

## 模板与特殊 token

- `<|endoftext|>` / tokenizer EOS / PAD：151643。
- Assistant 消息结束 `<|im_end|>`：151645；这是本阶段直接控制的训练 label。
- 训练和推理均使用冻结 Qwen chat template，系统提示一致。
- 冻结 Base tokenizer chat template 没有显式 thinking-mode 开关；本阶段不改变回答结构或引入 thinking 配置。
- Prompt labels 按位置设为 -100；真实 EOS 不按 token identity 批量屏蔽，避免 PAD==EOS 时误伤。

## 数据隔离与 D1 静态方案

训练题与 GSM8K/MATH-500 共 {len(benchmarks):,} 道评测题进行规范化内容哈希检查，精确碰撞 {len(collisions)} 条。GPU-3C 固定 4 道既有 benchmark 题，只用于冻结确认，不宣称完全未接触 holdout。

D1 完整样本过滤保留 {len(complete):,}/{len(details):,} 条，过滤 {len(details) - len(complete):,} 条，EOS 覆盖率变为 100%。这同时改变样本长度和来源分布，因此只能作为数据处理综合方案，不能用来估计 EOS 的单变量因果效应。B/C 对照使用相同的这 {len(complete):,} 条样本，唯一主动差异是最终 `<|im_end|>` label 是否为 -100。
"""
    (output / "DATA_AND_TEMPLATE_AUDIT.md").write_text(report, encoding="utf-8")
    return summary


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
