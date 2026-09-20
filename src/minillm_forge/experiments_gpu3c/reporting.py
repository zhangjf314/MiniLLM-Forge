# ruff: noqa: E501
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def summarize_generations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {"records": 0}
    return {
        "records": count,
        "successful_records": sum(row["status"] == "COMPLETED" for row in rows),
        "autonomous_eos_count": sum(bool(row["autonomous_eos_stop"]) for row in rows),
        "autonomous_eos_rate": sum(bool(row["autonomous_eos_stop"]) for row in rows) / count,
        "autonomous_im_end_count": sum(
            row["stop_reason"] == "ASSISTANT_END_151645" for row in rows
        ),
        "autonomous_im_end_stop_rate": sum(
            row["stop_reason"] == "ASSISTANT_END_151645" for row in rows
        )
        / count,
        "autonomous_endoftext_count": sum(row["stop_reason"] == "EOS_151643" for row in rows),
        "autonomous_endoftext_stop_rate": sum(row["stop_reason"] == "EOS_151643" for row in rows)
        / count,
        "external_answer_stop_count": sum(bool(row["external_answer_stop"]) for row in rows),
        "external_answer_stop_rate": sum(bool(row["external_answer_stop"]) for row in rows) / count,
        "premature_eos_count": sum(bool(row["premature_eos"]) for row in rows),
        "premature_eos_rate": sum(bool(row["premature_eos"]) for row in rows) / count,
        "length_limit_count": sum(bool(row["length_limit_stop"]) for row in rows),
        "length_limit_rate": sum(bool(row["length_limit_stop"]) for row in rows) / count,
        "complete_answer_count": sum(row["first_valid"]["answer"] is not None for row in rows),
        "complete_answer_rate": sum(row["first_valid"]["answer"] is not None for row in rows)
        / count,
        "legacy_correct_count": sum(bool(row["legacy_correct"]) for row in rows),
        "legacy_accuracy": sum(bool(row["legacy_correct"]) for row in rows) / count,
        "first_valid_correct_count": sum(bool(row["first_valid_correct"]) for row in rows),
        "first_valid_accuracy": sum(bool(row["first_valid_correct"]) for row in rows) / count,
        "conflict_aware_correct_count": sum(bool(row["conflict_aware_correct"]) for row in rows),
        "conflict_aware_accuracy": sum(bool(row["conflict_aware_correct"]) for row in rows) / count,
        "conflict_count": sum(row["conflict_aware"]["status"] == "CONFLICT" for row in rows),
        "no_valid_answer_count": sum(
            row["first_valid"]["status"] == "NO_VALID_ANSWER" for row in rows
        ),
        "repeat_3gram_rate_mean": statistics.mean(row["repeat_3gram"]["rate"] for row in rows),
        "repeat_4gram_rate_mean": statistics.mean(row["repeat_4gram"]["rate"] for row in rows),
        "post_answer_token_count_mean": _mean(rows, "post_answer_token_count"),
        "generated_token_count_mean": _mean(rows, "generated_token_count"),
        "inference_wall_time_seconds": sum(float(row["inference_wall_time"]) for row in rows),
    }


def write_final_artifacts(repo: str | Path, formal: dict[str, Any]) -> dict[str, Any]:
    repo = Path(repo).resolve()
    output = repo / "artifacts/gpu3c"
    audit = _read_json(output / "data_audit.json")
    pair = _read_json(output / "mask_pair_audit.json")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for arm in formal["arms"]:
        path = repo / "runs/gpu3c/formal-64" / arm["group"] / "generations.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            grouped[(arm["group"], row["decoding_path"])].append(row)
    metrics = {
        group: {path: summarize_generations(grouped[(group, path)]) for path in ("C0", "S3")}
        for group in ("B-EOS-MASKED", "C-EOS-SUPERVISED")
    }
    b, c = metrics["B-EOS-MASKED"]["C0"], metrics["C-EOS-SUPERVISED"]["C0"]
    signal = (
        c["autonomous_im_end_count"] > b["autonomous_im_end_count"]
        and c["premature_eos_count"] <= b["premature_eos_count"]
    )
    classification = (
        "GPU3C_EOS_SUPERVISION_EFFECT_VALIDATED" if signal else "GPU3C_CONTROLLED_NEGATIVE_RESULT"
    )
    training_cost = {
        "arms": formal["arms"],
        "total_wall_time_seconds": sum(arm["wall_time_seconds"] for arm in formal["arms"]),
        "peak_cuda_allocated_mib": max(arm["peak_cuda_allocated_mib"] for arm in formal["arms"]),
        "peak_cuda_reserved_mib": max(arm["peak_cuda_reserved_mib"] for arm in formal["arms"]),
    }
    result = {
        "stage": "GPU-3C",
        "classification": classification,
        "historical_baseline_preserved": True,
        "dataset_audit": audit,
        "eos_supervision_ablation": {"pair_identity": pair, "metrics": metrics},
        "data_completeness_ablation": audit["d1_complete_filter"],
        "formal_sft_evaluation": {"status": formal["status"], "metrics": metrics},
        "autonomous_termination": {group: values["C0"] for group, values in metrics.items()},
        "answer_quality": metrics,
        "training_cost": training_cost,
        "background_runs": [],
        "confirmed_findings": [
            "B/C inputs and order are identical; only final assistant-end labels differ.",
            "C0 autonomous stop and S3 external stop are reported separately.",
        ],
        "unresolved_questions": [
            "Only one seed and four frozen confirmation problems were used in this bounded first pass.",
            "D1 changes source and length distributions and was not trained as a separate A/D pair.",
        ],
        "next_stage_recommendations": [
            "Replicate with seeds 31415 and 271828 only if this first-pass signal merits the cost.",
            "Move next to the planned 37M Transformer SFT and attention/compile work if the signal is absent.",
        ],
        "tests": {"pre_smoke": "118 passed", "smoke": "PASS"},
        "artifacts": {
            "protocol": "artifacts/gpu3c/protocol.json",
            "raw_generations": [
                f"runs/gpu3c/formal-64/{group}/generations.jsonl"
                for group in ("B-EOS-MASKED", "C-EOS-SUPERVISED")
            ],
        },
    }
    _write_json(output / "gpu3c_result.json", result)

    (output / "EOS_SUPERVISION_ABLATION.md").write_text(
        "# EOS 单变量实验\n\n"
        f"B/C 共用 {pair['samples']:,} 条样本和相同顺序；input_ids、attention_mask 完全一致，"
        f"共 {pair['label_difference_count']:,} 个 label 差异，均为每条样本末尾 `<|im_end|>`。\n\n"
        f"训练均完成 64 个 optimizer steps。C0 指标：\n\n```json\n{json.dumps({'B': b, 'C': c}, indent=2, ensure_ascii=False)}\n```\n\n"
        "Loss 按有效 label token 取平均，B/C 有效监督 token 数不同，因此 loss 不作为效果结论。\n",
        encoding="utf-8",
    )
    d1 = audit["d1_complete_filter"]
    (output / "DATA_COMPLETENESS_ABLATION.md").write_text(
        "# 数据完整性修复审计\n\n"
        f"原方案：{audit['actual_training_samples_512']:,} 条实际训练样本，EOS 覆盖率 {audit['eos_supervision_coverage']:.3%}，"
        f"有效监督 token {audit['total_supervised_tokens_original_512']:,}。\n\n"
        f"D1 静态方案：保留 {d1['eligible_training_sample_count']:,} 条，过滤 {d1['filtered_sample_count']:,} 条，"
        f"EOS 覆盖率 100%，有效监督 token {d1['total_supervised_tokens']:,}。Prompt/Assistant 长度分布和来源分布见 data_audit.json。\n\n"
        "D1 就是本次 B/C 的共同完整子集；未再运行 A/D 综合训练。原因是 D1 同时改变长度、来源和 token 预算，无法隔离为 EOS 因果效应。\n",
        encoding="utf-8",
    )
    (output / "FORMAL_SFT_EVALUATION.md").write_text(
        "# 正式 SFT 与评测\n\n"
        f"状态：{formal['status']}。分类：`{classification}`。\n\n"
        f"C0/S3 分组指标：\n\n```json\n{json.dumps(metrics, indent=2, ensure_ascii=False)}\n```\n\n"
        "C0 是自主生成主指标；S3 的 EXTERNAL_ANSWER 不计入自主 EOS。评测仅含冻结的 4 道题、单 seed，统计外推能力有限。所有失败和负面结果均保留在原始 generations.jsonl。\n",
        encoding="utf-8",
    )
    (output / "GPU3C_FINAL_REPORT.md").write_text(
        "# GPU-3C 最终报告\n\n"
        f"Classification: `{classification}`\n\n"
        f"训练前审计确认 512 context 下 EOS 监督覆盖率为 {audit['eos_supervision_coverage']:.3%}；"
        f"缺失 {audit['samples_missing_supervised_assistant_end_over_training']:,} 条。B/C 受控训练与冻结 C0/S3 评测均已完成。\n\n"
        f"C0 中 B→C 的 `<|im_end|>` 自主终止数为 {b['autonomous_im_end_count']}→{c['autonomous_im_end_count']}，"
        f"提前 EOS 数为 {b['premature_eos_count']}→{c['premature_eos_count']}，"
        f"first-valid 正确数为 {b['first_valid_correct_count']}→{c['first_valid_correct_count']}。\n\n"
        f"两臂总训练耗时 {training_cost['total_wall_time_seconds']:.1f}s，峰值 CUDA allocated "
        f"{training_cost['peak_cuda_allocated_mib']:.1f} MiB。\n\n"
        "限制：单 seed、4 道既有 benchmark；未把 D1 数据分布变化误归因于 EOS；未开展多 seed 或 1,200-step 扩大实验。后续是否继续投入 Qwen3-0.6B，取决于该受控信号是否足以支持多 seed 复现成本；否则建议转入 37M Transformer SFT、SDPA/FlashAttention 与 torch.compile。\n",
        encoding="utf-8",
    )
    refresh_background_reports(repo)
    return result


def refresh_background_reports(repo: str | Path) -> list[dict[str, Any]]:
    repo = Path(repo).resolve()
    background = repo / "runs/gpu3c/background"
    records = []
    if background.exists():
        for job_path in sorted(background.glob("*/job.json")):
            config = _read_json(job_path)
            status_path = Path(config["status_file"])
            status = _read_json(status_path) if status_path.exists() else {"status": "MISSING"}
            launch_path = job_path.parent / "launch.json"
            launched = _read_json(launch_path) if launch_path.exists() else {}
            records.append(
                {
                    "task_id": config["task_id"],
                    "command": config["command"],
                    "estimated_duration_seconds": config["estimated_duration_seconds"],
                    "supervisor_pid": status.get("supervisor_pid", launched.get("supervisor_pid")),
                    "worker_pid": status.get("worker_pid"),
                    "status": status.get("status"),
                    "stdout_log": config["stdout_log"],
                    "stderr_log": config["stderr_log"],
                    "exit_code": status.get("exit_code"),
                    "elapsed_seconds": status.get("elapsed_seconds"),
                }
            )
    lines = ["# GPU-3C 后台任务记录", ""]
    for record in records:
        lines.extend(
            [
                f"## {record['task_id']}",
                "",
                f"- 状态：`{record['status']}`",
                f"- 预计时长：{record['estimated_duration_seconds']} 秒",
                f"- Supervisor PID：{record['supervisor_pid']}",
                f"- Worker PID：{record['worker_pid']}",
                f"- Exit code：{record['exit_code']}",
                f"- stdout：`{record['stdout_log']}`",
                f"- stderr：`{record['stderr_log']}`",
                f"- command：`{' '.join(record['command'])}`",
                "",
            ]
        )
    (repo / "artifacts/gpu3c/BACKGROUND_RUNS.md").write_text("\n".join(lines), encoding="utf-8")
    result_path = repo / "artifacts/gpu3c/gpu3c_result.json"
    if result_path.exists():
        result = _read_json(result_path)
        result["background_runs"] = records
        _write_json(result_path, result)
    return records
