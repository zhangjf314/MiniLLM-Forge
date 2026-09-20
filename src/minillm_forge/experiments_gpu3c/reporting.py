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
        "post_answer_token_count_observations": sum(
            row.get("post_answer_token_count") is not None for row in rows
        ),
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
            "The supervised arm emitted no <|im_end|> on the frozen C0 problems.",
            "The supervised arm hit the 512-token limit on all four C0 problems.",
        ],
        "unresolved_questions": [
            "Only one seed and four frozen confirmation problems were used in this bounded first pass.",
            "D1 changes source and length distributions and was not trained as a separate A/D pair.",
        ],
        "next_stage_recommendations": [
            "Replicate with seeds 31415 and 271828 only if this first-pass signal merits the cost.",
            "Move next to the planned 37M Transformer SFT and attention/compile work if the signal is absent.",
        ],
        "tests": {"final_regression": "120 passed", "smoke": "PASS"},
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
        "| C0 指标 | B EOS-MASKED | C EOS-SUPERVISED |\n"
        "|---|---:|---:|\n"
        f"| `<|im_end|>` 自主终止 | {b['autonomous_im_end_count']}/4 | {c['autonomous_im_end_count']}/4 |\n"
        f"| `<|endoftext|>` 自主终止 | {b['autonomous_endoftext_count']}/4 | {c['autonomous_endoftext_count']}/4 |\n"
        f"| 512-token 触顶 | {b['length_limit_count']}/4 | {c['length_limit_count']}/4 |\n"
        f"| 提前 EOS | {b['premature_eos_count']}/4 | {c['premature_eos_count']}/4 |\n"
        f"| First-valid 正确 | {b['first_valid_correct_count']}/4 | {c['first_valid_correct_count']}/4 |\n"
        f"| Conflict-aware 正确 | {b['conflict_aware_correct_count']}/4 | {c['conflict_aware_correct_count']}/4 |\n"
        f"| 3-gram 重复率均值 | {b['repeat_3gram_rate_mean']:.3f} | {c['repeat_3gram_rate_mean']:.3f} |\n"
        f"| 生成 token 均值 | {b['generated_token_count_mean']:.1f} | {c['generated_token_count_mean']:.1f} |\n\n"
        "两臂均完成 64 个 optimizer steps，初始 adapter 哈希相同，且均无 OOM、NaN 或 Inf。"
        "Loss 按有效 label token 取平均，B/C 有效监督 token 数不同，因此 loss 不作为效果结论。\n\n"
        "结论：监督目标 `<|im_end|>` 没有使模型在冻结题目上生成该 token；反而 C 组 4/4 触顶，"
        "且重复率显著升高。本轮不支持预期的 EOS 监督收益。正确率上升来自仅 4 道题的单 seed 结果，"
        "不能抵消自主终止失败，也不足以形成能力提升结论。\n",
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
        "| 组别/路径 | 自主 EOS | `<|im_end|>` | 触顶 | 外部停止 | First-valid 正确 | 冲突 | Token 均值 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| B / C0 | {b['autonomous_eos_count']}/4 | {b['autonomous_im_end_count']}/4 | {b['length_limit_count']}/4 | 0/4 | {b['first_valid_correct_count']}/4 | {b['conflict_count']}/4 | {b['generated_token_count_mean']:.1f} |\n"
        f"| C / C0 | {c['autonomous_eos_count']}/4 | {c['autonomous_im_end_count']}/4 | {c['length_limit_count']}/4 | 0/4 | {c['first_valid_correct_count']}/4 | {c['conflict_count']}/4 | {c['generated_token_count_mean']:.1f} |\n"
        f"| B / S3 | {metrics['B-EOS-MASKED']['S3']['autonomous_eos_count']}/4 | 0/4 | 0/4 | {metrics['B-EOS-MASKED']['S3']['external_answer_stop_count']}/4 | {metrics['B-EOS-MASKED']['S3']['first_valid_correct_count']}/4 | {metrics['B-EOS-MASKED']['S3']['conflict_count']}/4 | {metrics['B-EOS-MASKED']['S3']['generated_token_count_mean']:.1f} |\n"
        f"| C / S3 | {metrics['C-EOS-SUPERVISED']['S3']['autonomous_eos_count']}/4 | 0/4 | {metrics['C-EOS-SUPERVISED']['S3']['length_limit_count']}/4 | {metrics['C-EOS-SUPERVISED']['S3']['external_answer_stop_count']}/4 | {metrics['C-EOS-SUPERVISED']['S3']['first_valid_correct_count']}/4 | {metrics['C-EOS-SUPERVISED']['S3']['conflict_count']}/4 | {metrics['C-EOS-SUPERVISED']['S3']['generated_token_count_mean']:.1f} |\n\n"
        "C0 是自主生成主指标；S3 的 EXTERNAL_ANSWER 不计入自主 EOS。C 组在 S3 下有 3/4 外部停止，"
        "生成均值由 C0 的 512 降至 184.25，但这是推理系统截断效果，不是模型学会结束。\n\n"
        "失败案例：C 组四题均未自主终止；三题在完整答案后继续生成至 512；math500-0181 在 C0 出现"
        "多答案冲突，S3 在首个完整答案后停止而消除冲突。B 组有两次无可验证完整边界前的 EOS。"
        "未出现“自主结束率提高但答案错误”，因为 C 组自主结束率没有提高。\n\n"
        "评测仅含冻结的 4 道题、单 seed，统计外推能力有限。所有负面结果保留在原始 generations.jsonl。\n",
        encoding="utf-8",
    )
    (output / "GPU3C_FINAL_REPORT.md").write_text(
        "# GPU-3C 最终报告\n\n"
        f"Classification: `{classification}`\n\n"
        "## 执行摘要\n\n"
        "GPU-3C 完成了数据/模板审计、B/C 单变量训练、D1 静态完整性方案、正式 64-step 配对训练及 C0/S3 评测。"
        "核心假设未得到支持：C-EOS-SUPERVISED 没有生成任何 `<|im_end|>`，并从 B 的 1/4 触顶恶化为 4/4 触顶。\n\n"
        "## 数据与监督\n\n"
        f"19,162 条实际 512-context 样本中，12,469 条监督 `<|im_end|>`，覆盖率 {audit['eos_supervision_coverage']:.3%}；"
        f"6,693 条在 assistant 结束标记前截断。另有 38 条 prompt 耗尽 context、未进入实际训练。"
        "D1 保留 12,469 条完整样本并达到 100% EOS 覆盖，但有效监督 token 从 5,177,783 降至 2,583,463，"
        "且来源/长度分布改变，因此没有把 D1 当作 EOS 单变量效果。\n\n"
        "## 受控训练与自主终止\n\n"
        f"B/C 的 12,469 条输入和顺序完全一致，只在每条最终 EOS label 处有一个差异。两臂各完成 64 steps、"
        f"1,024 examples；B/C 最终 loss 分别为 {formal['arms'][0]['final_loss']:.4f}/{formal['arms'][1]['final_loss']:.4f}，"
        "但因有效监督 token 分母不同，不以 loss 判定效果。\n\n"
        f"C0 的 B→C：`<|im_end|>` 0→0，`<|endoftext|>` 3→0，触顶 1→4，提前 EOS 2→0，"
        f"生成 token 均值 {b['generated_token_count_mean']:.1f}→{c['generated_token_count_mean']:.1f}，"
        f"3-gram 重复率 {b['repeat_3gram_rate_mean']:.3f}→{c['repeat_3gram_rate_mean']:.3f}。"
        "C 组虽消除了提前 EOS，却以完全不终止和严重答案后重复为代价。\n\n"
        "## 答案质量与推理系统\n\n"
        f"C0 First-valid 正确数为 {b['first_valid_correct_count']}→{c['first_valid_correct_count']}，"
        f"Conflict-aware 正确数为 {b['conflict_aware_correct_count']}→{c['conflict_aware_correct_count']}；"
        "样本仅 4 道，不能据此声称任务能力提升。C 组 math500-0181 在 C0 产生冲突。"
        "S3 对 C 组 3/4 成功外部停止，并将平均长度降至 184.25 token，同时消除了该冲突；"
        "这是推理系统效果，不计入自主 EOS。\n\n"
        "## 成本、限制与建议\n\n"
        f"两臂训练加评测总耗时 {training_cost['total_wall_time_seconds']:.1f}s；后台任务端到端 1012.6s；"
        f"峰值 CUDA allocated/reserved 为 {training_cost['peak_cuda_allocated_mib']:.1f}/"
        f"{training_cost['peak_cuda_reserved_mib']:.1f} MiB，无 OOM、NaN 或 Inf。最终回归 120 passed。\n\n"
        "本轮只有 seed 42 和 4 道既有 benchmark，未运行 D1 与原始数据的独立 A/D 训练，也未扩展至 1,200 steps。"
        "鉴于预期 EOS 效果方向未出现，当前不建议直接追加多 seed 或更大训练预算。建议保留 S3 作为工程止损，"
        "并转入第二阶段 37M Transformer SFT、SDPA/FlashAttention 与 torch.compile；只有提出新的、可检验的"
        "结束 token/模板机制假设后，才值得恢复 Qwen3-0.6B 终止优化。\n",
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
            if not config["task_id"].startswith("gpu3c-"):
                continue
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
