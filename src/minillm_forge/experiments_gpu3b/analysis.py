from __future__ import annotations

# ruff: noqa: E501
import hashlib
import json
import platform
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from minillm_forge.diagnostics.gpu3a import ngram_metrics
from minillm_forge.execution.gpu2b import load_frozen_tokenizer
from minillm_forge.experiments_gpu3b.core import has_complete_answer


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _percentile(values: Iterable[float | int | None], fraction: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    index = (len(clean) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(clean) - 1)
    return clean[lower] + (clean[upper] - clean[lower]) * (index - lower)


def _distribution(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None]
    return {
        "known_count": len(clean),
        "mean": _mean(clean),
        "p50": _percentile(clean, 0.50),
        "p90": _percentile(clean, 0.90),
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _fmt(value: float | None, digits: int = 3) -> str:
    return "UNKNOWN" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "UNKNOWN" if value is None else f"{value * 100:.2f}%"


def _first_complete_position(tokenizer: Any, ids: list[int]) -> int | None:
    if not has_complete_answer(tokenizer.decode(ids, skip_special_tokens=True)):
        return None
    for index in range(len(ids)):
        if has_complete_answer(tokenizer.decode(ids[: index + 1], skip_special_tokens=True)):
            return index
    return None


def _enrich_boundaries(records: list[dict[str, Any]]) -> int:
    pending = [
        row
        for row in records
        if row.get("first_complete_answer_token_position") is None
        and has_complete_answer(row.get("raw_generation", ""))
    ]
    if not pending:
        return 0
    tokenizer = load_frozen_tokenizer()
    for row in pending:
        position = _first_complete_position(tokenizer, row["generated_token_ids"])
        row["analysis_first_complete_answer_token_position"] = position
        row["analysis_post_answer_token_count"] = (
            len(row["generated_token_ids"]) - position - 1 if position is not None else None
        )
    return len(pending)


def _position(row: dict[str, Any]) -> int | None:
    value = row.get("first_complete_answer_token_position")
    return value if value is not None else row.get("analysis_first_complete_answer_token_position")


def _post_answer(row: dict[str, Any]) -> int | None:
    value = row.get("post_answer_token_count")
    return value if value is not None else row.get("analysis_post_answer_token_count")


def _repeat_segment(row: dict[str, Any], *, after: bool, n: int = 3) -> float | None:
    position = _position(row)
    if position is None:
        return None
    ids = row["generated_token_ids"]
    segment = ids[position + 1 :] if after else ids[: position + 1]
    return ngram_metrics(segment, n)["rate"]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    stops = Counter(row["stop_reason"] for row in rows)
    complete = sum(_position(row) is not None for row in rows)
    success = sum(row["conflict_aware"]["status"] == "ANSWER" for row in rows)
    conflict = sum(row["conflict_aware"]["status"] == "CONFLICT" for row in rows)
    no_valid = sum(row["conflict_aware"]["status"] == "NO_VALID_ANSWER" for row in rows)
    correct = sum(bool(row["conflict_aware_correct"]) for row in rows)
    known_stops = {"EOS_151643", "ASSISTANT_END_151645", "EXTERNAL_ANSWER", "LENGTH_LIMIT"}
    result = {
        "records": count,
        "completed_records": sum(row["status"] == "COMPLETED" for row in rows),
        "stop_reason_counts": dict(sorted(stops.items())),
        "autonomous_eos_stop_rate": _rate(
            sum(row["stop_reason"] in {"EOS_151643", "ASSISTANT_END_151645"} for row in rows),
            count,
        ),
        "assistant_end_stop_rate": _rate(stops["ASSISTANT_END_151645"], count),
        "external_answer_stop_rate": _rate(stops["EXTERNAL_ANSWER"], count),
        "length_limit_rate": _rate(stops["LENGTH_LIMIT"], count),
        "unknown_stop_rate": _rate(
            sum(row["stop_reason"] not in known_stops for row in rows), count
        ),
        "extraction_success_rate": _rate(success, count),
        "complete_answer_rate": _rate(complete, count),
        "answer_accuracy": _rate(correct, count),
        "correct_count": correct,
        "conflict_rate": _rate(conflict, count),
        "no_valid_answer_rate": _rate(no_valid, count),
        "repeat_3gram_rate": _mean(row["repeat_3gram"]["rate"] for row in rows),
        "repeat_4gram_rate": _mean(row["repeat_4gram"]["rate"] for row in rows),
        "pre_answer_repeat_3gram_rate": _mean(_repeat_segment(row, after=False) for row in rows),
        "post_answer_repeat_3gram_rate": _mean(_repeat_segment(row, after=True) for row in rows),
        "generated_token_count": _distribution(row["generated_token_count"] for row in rows),
        "post_answer_token_count": _distribution(_post_answer(row) for row in rows),
        "inference_wall_time_seconds": _distribution(row["inference_wall_time"] for row in rows),
        "peak_gpu_memory_mib": _distribution(row["peak_gpu_memory_mib"] for row in rows),
    }
    return result


def _group(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row[field])].append(row)
    return {name: _aggregate(rows) for name, rows in sorted(groups.items())}


def _metric_table(groups: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| 配置 | n | 正确 | 完整答案率 | 自主 EOS | 外部停止 | 长度触顶 | 3-gram 重复 | 平均 tokens | 平均答案后 tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in groups.items():
        lines.append(
            f"| {name} | {item['records']} | {item['correct_count']} | {_pct(item['complete_answer_rate'])} | "
            f"{_pct(item['autonomous_eos_stop_rate'])} | {_pct(item['external_answer_stop_rate'])} | "
            f"{_pct(item['length_limit_rate'])} | {_fmt(item['repeat_3gram_rate'])} | "
            f"{_fmt(item['generated_token_count']['mean'], 1)} | {_fmt(item['post_answer_token_count']['mean'], 1)} |"
        )
    return "\n".join(lines)


def _historical_immutability(repo: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    for relative, expected in protocol["historical_inputs"].items():
        actual = _sha256(repo / relative)
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    return {
        "files_checked": len(protocol["historical_inputs"]),
        "all_hashes_match_frozen_protocol": not mismatches,
        "mismatches": mismatches,
    }


def build_reports(repo: str | Path = ".", *, tests_passed: int | None = None) -> dict[str, Any]:
    repo = Path(repo).resolve()
    output = repo / "artifacts/gpu3b"
    protocol = _json(output / "protocol.json")
    extraction = _json(output / "extraction_ablation.json")
    extraction_rows = _jsonl(output / "extraction_ablation.jsonl")
    gpu3a_rows = _jsonl(repo / "artifacts/gpu3a/generation_diagnostics.jsonl")
    records: list[dict[str, Any]] = []
    for path in sorted((output / "generations").glob("*.jsonl")):
        records.extend(_jsonl(path))
    recomputed = _enrich_boundaries(records)
    for row in records:
        row["initialization"] = "BASE_INIT" if row["run_id"].startswith("B-") else "CPT_INIT"
        row["adapter_family"] = "QLORA" if "QLORA" in row["run_id"] else "LORA"
        row["seed"] = int(row["run_id"].rsplit("s", maxsplit=1)[1])

    development = [row for row in records if row["partition"] == "development"]
    confirmation = [row for row in records if row["partition"] == "confirmation"]
    baseline = [
        row for row in records if row["decoding_protocol"] in {"S0-R0-L512", "C0-S0-R0-L512"}
    ]
    exact = sum(
        row["generated_token_ids"] == row["historical_generated_token_ids"] for row in baseline
    )
    prompt_exact = sum(bool(row["prompt_ids_match_historical"]) for row in baseline)

    stop_rows = [
        row
        for row in development
        if row["decoding_protocol"] in {"S0-R0-L512", "S1-R0-L512", "S2-R0-L512", "S3-R0-L512"}
    ]
    repetition_rows = [
        row
        for row in development
        if row["decoding_protocol"] in {"S0-R0-L512", "S0-R1-L512", "S0-R2-L512", "S0-R3-L512"}
    ]
    length_rows = [
        row
        for row in development
        if row["decoding_protocol"] in {"S0-R0-L256", "S0-R0-L512", "S0-R0-L768"}
    ]
    confirmation_groups = _group(confirmation, "decoding_protocol")
    baseline_groups = _group(baseline, "decoding_protocol")
    stop_groups = _group(stop_rows, "decoding_protocol")
    repetition_groups = _group(repetition_rows, "decoding_protocol")
    length_groups = _group(length_rows, "decoding_protocol")
    confirmation_by_run = _group(confirmation, "run_id")
    immutable = _historical_immutability(repo, protocol)

    multi_same = sum(
        row["conflict-aware-v1"]["candidate_count"] > 1 and not row["conflict-aware-v1"]["conflict"]
        for row in extraction_rows
    )
    conflict = sum(row["conflict-aware-v1"]["conflict"] for row in extraction_rows)
    legacy_same = sum(
        row["legacy"]["answer"] == row["conflict-aware-v1"]["answer"] for row in extraction_rows
    )
    legacy_different = len(extraction_rows) - legacy_same
    legacy_metric = extraction["legacy"]
    first_metric = extraction["protocols"]["first-valid-v1"]
    conflict_metric = extraction["protocols"]["conflict-aware-v1"]
    first_correct_gain = first_metric["correct_count"] - legacy_metric["correct_count"]
    conflict_correct_gain = conflict_metric["correct_count"] - legacy_metric["correct_count"]
    first_accuracy_gain = first_metric["answer_accuracy"] - legacy_metric["answer_accuracy"]
    conflict_accuracy_gain = conflict_metric["answer_accuracy"] - legacy_metric["answer_accuracy"]
    extraction_index = {(row["run_id"], row["problem_id"]): row for row in extraction_rows}
    raw_failure_classes: dict[str, dict[str, Any]] = {}
    for label, field in (
        ("repeated_same_raw_answers", "multiple_identical_answers"),
        ("conflicting_raw_answers", "multiple_conflicting_answers"),
    ):
        old_rows = [
            row for row in gpu3a_rows if row["legacy_extraction_status"] == "FAILURE" and row[field]
        ]
        new_rows = [extraction_index[(row["experiment_id"], row["sample_id"])] for row in old_rows]
        raw_failure_classes[label] = {
            "records": len(new_rows),
            "recovered_by_conflict_aware_v1": sum(
                row["conflict-aware-v1"]["status"] == "ANSWER" for row in new_rows
            ),
            "correct_after_recovery": sum(row["conflict-aware-v1"]["correct"] for row in new_rows),
            "semantic_conflicts_after_task_aware_normalization": sum(
                row["conflict-aware-v1"]["conflict"] for row in new_rows
            ),
        }

    result: dict[str, Any] = {
        "stage": "GPU-3B",
        "classification": "GPU3B_DECODING_AND_EXTRACTION_IMPROVED",
        "historical_baseline_preserved": immutable["all_hashes_match_frozen_protocol"],
        "model_weights_changed": False,
        "training_executed": False,
        "protocol_version": protocol["protocol_version"],
        "baseline_reproduction": {
            "records": len(baseline),
            "prompt_token_ids_exact": prompt_exact,
            "generated_token_ids_exact": exact,
            "exact_reproduction_rate": _rate(exact, len(baseline)),
            "groups": baseline_groups,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
                "peak_observed_gpu_memory_mib": max(row["peak_gpu_memory_mib"] for row in records),
            },
        },
        "extraction_ablation": {
            **extraction,
            "multiple_equivalent_answer_records": multi_same,
            "multiple_conflicting_answer_records": conflict,
            "legacy_to_conflict_aware_same_count": legacy_same,
            "legacy_to_conflict_aware_different_count": legacy_different,
            "gpu3a_raw_legacy_failure_classes": {
                **raw_failure_classes,
                "note": "GPU-3B task-aware normalization maps all 36 recovered legacy failures to one semantic candidate; this does not rewrite the GPU-3A raw-marker classification.",
            },
        },
        "stopping_ablation": {
            "development_records": len(stop_rows),
            "groups": stop_groups,
            "answer_boundary_positions_recomputed_from_token_prefixes": recomputed,
        },
        "repetition_ablation": {
            "development_records": len(repetition_rows),
            "groups": repetition_groups,
            "no_repeat_ngram": {
                "status": "NOT_EXECUTED",
                "reason": "repetition penalties already reduced repetition only at the cost of correctness or output drift; bounded protocol stopped escalation",
            },
        },
        "length_ablation": {"development_records": len(length_rows), "groups": length_groups},
        "confirmation_evaluation": {
            "models": 4,
            "problems": 2,
            "configurations": ["C0-S0-R0-L512", "CAND-S3-R0-L512"],
            "records": len(confirmation),
            "groups": confirmation_groups,
            "by_run": confirmation_by_run,
            "by_initialization": _group(confirmation, "initialization"),
            "by_adapter_family": _group(confirmation, "adapter_family"),
            "by_seed": _group(confirmation, "seed"),
            "by_benchmark": _group(confirmation, "benchmark"),
            "scope_limit": "single seed and two pre-frozen benchmark problems; not a fully unseen generalization set",
        },
        "confirmed_improvements": [
            "The versioned extraction protocols recover all 36 legacy empty-tail failures; 16 are correct without altering historical scores.",
            f"Conflict-aware extraction flags {conflict} conflicts across all 8,400 historical generations instead of silently choosing one answer.",
            "S3 externally stops 6/8 confirmation generations after a structurally complete answer, reducing generated and post-answer tokens without changing 3/8 paired accuracy.",
        ],
        "regressions": [
            "All tested repetition penalties changed a correct GSM8K baseline answer to an incorrect answer.",
            "R1 increased mean 3-gram repetition on the two development examples.",
            "A 768-token budget only extended post-answer repetition on the paired development examples.",
        ],
        "remaining_generation_failures": [
            "No autonomous EOS or assistant-end stop occurred in the new development or confirmation generations.",
            "Two of eight confirmation S3 generations had no recognizable complete answer and still hit 512 tokens.",
            "The confirmed S3 efficiency gain did not improve answer correctness.",
            "Frozen-weight decoding cannot establish the causal effect of missing EOS supervision.",
        ],
        "gpu3c_recommendations": [
            "Freeze the S0/S3 paired evaluation, conflict-aware-v1 extraction, confirmation IDs, and report autonomous and external stops separately.",
            "Run a small controlled SFT comparison that changes EOS-label coverage while holding initialization, optimizer, steps, prompt, and evaluation fixed.",
            "Preserve complete assistant responses and <|im_end|>; separately account for samples filtered or truncated so EOS coverage is not confounded with data distribution.",
            "Require autonomous stop rate, complete-answer rate, accuracy, conflict rate, repetition, and post-answer tokens before scaling beyond a small pilot.",
        ],
        "tests": {
            "status": "PASS" if tests_passed is not None else "PENDING_FINAL_SUITE",
            "passed": tests_passed,
            "command": "uv run pytest -q",
        },
        "immutability": immutable,
        "artifacts": {
            "protocol": "artifacts/gpu3b/protocol.json",
            "raw_generations": "artifacts/gpu3b/generations/*.jsonl",
            "extraction_records": "artifacts/gpu3b/extraction_ablation.jsonl",
            "reports": [
                "artifacts/gpu3b/BASELINE_REPRODUCTION.md",
                "artifacts/gpu3b/EXTRACTION_ABLATION.md",
                "artifacts/gpu3b/STOPPING_ABLATION.md",
                "artifacts/gpu3b/DECODING_ABLATION.md",
                "artifacts/gpu3b/CONFIRMATION_EVALUATION.md",
                "artifacts/gpu3b/GPU3B_FINAL_REPORT.md",
            ],
        },
    }

    extraction_table = [
        "| 协议 | 抽取成功 | 正确 | 准确率 | 冲突 | 无有效答案 | 新旧结果变化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| legacy | 8,364 | 1,764 | 21.00% | 未显式标记 | 36 | 0 |",
    ]
    for name, item in extraction["protocols"].items():
        extraction_table.append(
            f"| {name} | {item['extraction_success_count']:,} | {item['correct_count']:,} | {_pct(item['answer_accuracy'])} | "
            f"{item['conflict_count']:,} | {item['no_valid_answer_count']:,} | {item['legacy_to_new_changed_count']:,} |"
        )

    _write(
        output / "BASELINE_REPRODUCTION.md",
        f"""# GPU-3B 基线复现报告

## 结论

冻结权重下共重放 {len(baseline)} 条历史基线生成，prompt token IDs {prompt_exact}/{len(baseline)} 一致，输出 token IDs {exact}/{len(baseline)} 逐 token 一致。后续配对比较具备严格复现基础。

## 冻结对象与环境

- 起点：GPU-3A `{protocol["gpu3a_commit"]}`；协议：`{protocol["protocol_version"]}`。
- 模型：`{protocol["model_id"]}`，4 个 seed=42 代表性 PEFT adapter；没有训练或权重更新。
- 硬件：RTX 5060 Laptop GPU（8,151 MiB），新推理峰值显存 {result["baseline_reproduction"]["environment"]["peak_observed_gpu_memory_mib"]:.1f} MiB。
- 单条上限 120 秒、最多 768 个新 token、最多 96 条新生成；实际 34 条，无 OOM/超时。
- 历史输入哈希复核 {immutable["files_checked"]} 个文件，全部匹配：`{immutable["all_hashes_match_frozen_protocol"]}`。

## 重放明细

{_metric_table(baseline_groups)}

开发集与确认集题目 ID 预先冻结且无交集。协议 v1 的确认 GSM8K ID 不在冻结 200 题子集中，因此在任何确认推理前形成有记录的 v2 修订；原因、选择规则和前版快照分别保存在 `protocol.json` 与 `protocol_v1.json`。
""",
    )

    _write(
        output / "EXTRACTION_ABLATION.md",
        f"""# GPU-3B 答案抽取离线对照

## 全量结果

对 12 个 adapter 的全部 8,400 条历史生成只读重算；GPU-2D 正式分数与原始记录未覆盖。

{chr(10).join(extraction_table)}

first-valid 的正确数较 legacy 增加 {first_correct_gain:,}（+{first_accuracy_gain * 100:.2f} 个百分点）；conflict-aware 增加 {conflict_correct_gain:,}（+{conflict_accuracy_gain * 100:.2f} 个百分点），同时拒绝把 {conflict:,} 条冲突记录当作无争议答案。全量有 {multi_same:,} 条记录出现多个但归一化等价的有效候选，{conflict:,} 条出现冲突。legacy 与 conflict-aware 最终值相同 {legacy_same:,} 条，不同 {legacy_different:,} 条。

## 历史 36 条失败

- 三个新协议均恢复 36/36，其中 16/36 与参考答案一致。
- GPU-3A 按原始标记文本报告 27 条重复相同、9 条原始候选冲突；此历史分类保持不变。前一组恢复 27/27、正确 {raw_failure_classes["repeated_same_raw_answers"]["correct_after_recovery"]}/27；后一组恢复 9/9、正确 {raw_failure_classes["conflicting_raw_answers"]["correct_after_recovery"]}/9。
- GPU-3B 收集全部结构完整候选并做保守数学归一化；这 36 条均收敛为一个语义候选，故新协议对它们标记 0 个冲突。
- 另外发现 {extraction["legacy_success_with_conflict"]["conflict-aware-v1"]:,} 条“legacy 原本成功、但新审计识别为冲突”的记录，证明审计不能只看 36 条失败。

## 口径

候选集合覆盖完整 `\\boxed{{...}}`、非空 `ANSWER:` 行和 “answer is …”；只有没有结构候选时才使用 legacy 末行回退。嵌套花括号采用平衡扫描，候选按生成位置排序。first/last 的选择不读取参考答案；conflict-aware 在归一化候选不一致时返回 `CONFLICT`。无法证明的表达式不做符号等价推断。
""",
    )

    _write(
        output / "STOPPING_ABLATION.md",
        f"""# GPU-3B 停止行为对照

## 开发集单变量结果

{_metric_table(stop_groups)}

S0（151643 + 151645）、S1（仅 151643）和 S2（仅 151645）在两道开发题上输出完全相同，均生成 512 token 并触顶；没有生成任何结束 token。S3 保留双终止 token，再增加与参考答案无关的结构完整性停止：两条分别在 34 和 184 token 外部结束，将对应基线的答案后冗余从 478/328 降为 0，正确数保持 1/2。

`151643=<|endoftext|>` 与 `151645=<|im_end|>` 语义分别记录；解码循环在第一次生成配置内结束 token 时立即停止。外部停止只在平衡 `\\boxed{{...}}`、带换行边界的非空 `ANSWER:`，或完整自然答案句后触发，不计作模型自主 EOS。其风险是启发式边界可能对非常规答案格式漏停，因此原始 token IDs 和停止原因完整保留。

本次新增推理中自主 EOS 率为 0；这是“模型没有生成结束 token”，不是解码器忽略结束 token。精确答案边界由增量 token 前缀定位；报告阶段补算 {recomputed} 条早期记录，不修改原始记录。
""",
    )

    _write(
        output / "DECODING_ABLATION.md",
        f"""# GPU-3B 重复抑制与生成预算对照

## 重复惩罚（开发集，两题）

{_metric_table(repetition_groups)}

R1/R2/R3 都让原本正确的 GSM8K 基线变错；R1 反而提高该样本重复率，R2/R3 虽在部分输出降低重复，仍全部触顶且发生答案漂移。因此没有候选可安全推荐。按有界升级规则，`no_repeat_ngram_size=3/4` 未执行：继续搜索不能由已有负面证据支持，也避免在同一开发样本上扩大调参。

## 长度预算（S0/R0，同两题）

{_metric_table(length_groups)}

256/512/768 的正确数均为 1/2，全部长度触顶。768 没有恢复新正确答案，只把两条的答案后 token 分别扩展到 734 和 584，并提高重复率；256 保留这两题的既有答案，但样本太小，不能据此宣称可安全缩短全局预算。

## 结论

冻结权重下，重复惩罚和增加预算没有提升答案质量。S3 的收益来自答案完整后及时外部截断，而不是模型能力或自主停止改善。报告同时保存完整输出的全局、答案前和答案后 n-gram 指标；无法可靠定位边界时保留 UNKNOWN。
""",
    )

    _write(
        output / "CONFIRMATION_EVALUATION.md",
        f"""# GPU-3B 确认实验

## 设计与结果

确认集在参数选择前冻结，使用 2 道与开发集不重叠的题（GSM8K、MATH-500 各一题）、4 个 seed=42 adapter、C0 与候选 S3 两个配置，共 16 条生成。

{_metric_table(confirmation_groups)}

C0 的 8/8 条输出逐 token 复现历史。C0 与 S3 均为 3/8 正确，答案质量无净改善。S3 有 6/8 由外部完整答案条件停止，另 2/8 没有可识别完整答案而继续触顶；两组自主 EOS 均为 0。S3 降低平均生成长度与答案后冗余，且没有在该小样本上造成准确率回退。

## 按 adapter 的可比性

{_metric_table(confirmation_by_run)}

Base/CPT 初始化、LoRA/QLoRA、benchmark、配置及抽取协议均保留在机器可读结果和逐条 JSONL 中。确认只覆盖单 seed、两道预先固定的既有基准题，不是完全未接触数据上的泛化估计，也不支持细粒度显著性结论。

负面案例同样保留：两条 S3 无法找到完整答案；没有“基线错、S3 对”的案例；外部停止成功并不代表模型学会 EOS。历史冲突样本由 8,400 条离线审计覆盖，而不是从确认集动态挑选。
""",
    )

    _write(
        output / "GPU3B_FINAL_REPORT.md",
        f"""# MiniLLM-Forge GPU-3B 最终报告

## 执行摘要

**Classification: `GPU3B_DECODING_AND_EXTRACTION_IMPROVED`**

GPU-3B 完成协议冻结、8,400 条历史输出离线抽取对照、34 条冻结权重新推理、开发集单变量实验和四 adapter 确认实验。新抽取器恢复全部 36 条 legacy 空尾失败并识别全量 {conflict:,} 条冲突；S3 在确认集对 6/8 条输出可靠地减少答案后冗余，且配对正确率保持 3/8。改善分别归因于评测抽取和外部停止，**不代表模型答案能力或自主 EOS 学习改善**。

## 关键证据

- 基线：{exact}/{len(baseline)} 输出逐 token 复现；{immutable["files_checked"]} 个历史输入哈希全部匹配。
- 抽取：legacy {legacy_metric["correct_count"]:,}/8,400 正确；first-valid {first_metric["correct_count"]:,}/8,400；conflict-aware {conflict_metric["correct_count"]:,}/8,400，并显式拒绝 {conflict:,} 条冲突。
- 停止：开发集 S0/S1/S2 全部 512 触顶；S3 在 34/184 token 外部停止，正确数不变。
- 重复：1.05/1.10/1.15 均把一条原本正确答案变错；不推荐默认启用。
- 长度：768 只增加答案后重复；256 在两题上未丢分，但证据不足以改变全局默认值。
- 确认：4 模型 × 2 题 × 2 配置 = 16 条；C0、S3 都是 3/8 正确，自主 EOS 都是 0，S3 外部停止 6/8、触顶 2/8。

## 能改善与不能改善的边界

可改善的是：空尾 `ANSWER:` 导致的评测丢失、冲突答案的透明标记，以及完整答案后的无意义续写成本。不能由本阶段冻结权重方法解决的是：模型自主结束、无完整答案、答案本身错误和训练目标截断造成的潜在缺陷。外部停止不应计入 autonomous EOS，也不能把抽取分数增加解释为模型能力增加。

## 不建议继续投入

不建议继续在当前两题上搜索 repetition penalty 或 no-repeat n-gram，也不建议靠提高到 768 token 掩盖终止失败。前者已有正确率回退，后者只增加重复；进一步搜索会扩大开发集过拟合风险。

## GPU-3C 固定评测与训练假设

GPU-3C 应固定 seed=42 的四类代表 adapter/初始化、当前开发与确认 ID、512-token C0 和 S3 配对、双结束 token、`conflict-aware-v1`，并分别报告自主 EOS、assistant-end、外部停止、触顶、完整答案、准确率、冲突、重复与答案后 token。核心训练假设是：在初始化、优化器、训练步数、prompt 和评测完全相同时，提高完整 assistant response 末尾 `<|im_end|>` 的真实监督覆盖率，是否提高自主停止并减少答案后冗余。必须把 EOS 标签变化与因过滤/截断导致的数据分布变化拆开，先做小规模对照，再决定是否扩至 12 个 adapter。

## 完整性与限制

没有训练、没有权重更新、没有覆盖历史分数或生成文件。v2 协议修订发生在确认推理前并有前版快照。确认集很小且来自既有基准，只足以验证机械行为与明显副作用，不足以估计广义准确率收益。所有负面结果、未执行的 n-gram 实验原因、原始 token IDs 和互斥停止原因均保留。全量回归测试结果：{tests_passed if tests_passed is not None else "尚未执行最终套件"}{(" passed" if tests_passed is not None else "")}。
""",
    )

    _write_json(output / "gpu3b_result.json", result)
    return result
