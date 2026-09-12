from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from statistics import median
from typing import Any

import torch

from minillm_forge.cli.common import read_jsonl, write_json
from minillm_forge.config import config_hash, load_config
from minillm_forge.data.qwen_cpt import (
    freeze_general_partition,
    freeze_math_partitions,
    tensor_digest,
    write_jsonl,
)
from minillm_forge.training.qwen_cpt import (
    QwenCPTTrainer,
    cuda_memory,
    evaluate_domains,
    file_sha256,
)

DEFAULT_CONFIG = "configs/cpt/qwen3_math.yaml"
MODEL_MANIFEST = Path("artifacts/training/qwen_base_model_manifest.json")
BASE_EVAL = Path("artifacts/eval_manifests/qwen_base_pre_cpt.json")
CALIBRATION = Path("artifacts/training/qwen_cpt_memory_calibration.json")
QUALIFICATION = Path("artifacts/training/qwen_cpt_long_run_qualification.json")
FORMAL_RESULT = Path("artifacts/training/qwen_math_cpt_result.json")
POST_EVAL = Path("artifacts/eval_manifests/qwen_math_cpt_post_eval.json")


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_tokenizer(config: dict[str, Any]):
    from transformers import AutoTokenizer

    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model["name_or_path"], revision=model["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _tokenizer_digest(tokenizer: Any) -> str:
    return _hash_json(
        {
            "backend": tokenizer.backend_tokenizer.to_str(),
            "special_tokens_map": tokenizer.special_tokens_map,
            "vocab_size": len(tokenizer),
        }
    )


def _processed_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["data"]["processed_dir"])
    return {
        "root": root,
        "train_text": root / "train.jsonl",
        "math_text": root / "math_validation.jsonl",
        "general_text": root / "general_validation.jsonl",
        "train_tokens": root / "train_tokens.pt",
        "math_tokens": root / "math_validation_tokens.pt",
        "general_tokens": root / "general_validation_tokens.pt",
    }


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    from datasets import load_dataset

    data = config["data"]
    paths = _processed_paths(config)
    paths["root"].mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer(config)
    tokenizer_hash = _tokenizer_digest(tokenizer)
    general_source = Path(data["general_source"])
    with general_source.open("r", encoding="utf-8") as handle:
        general_validation = freeze_general_partition(
            handle, tokenizer, token_budget=data["general_validation_tokens"]
        )
    benchmark_rows = read_jsonl(data["controlled_math_source"])
    benchmark_texts = [str(row["problem"]) for row in benchmark_rows]
    benchmark_texts.extend(general_validation.documents)
    stream = load_dataset(
        data["dataset"],
        data["subset"],
        split=data["split"],
        revision=data["revision"],
        streaming=True,
    )
    train, math_validation, audit = freeze_math_partitions(
        stream,
        tokenizer,
        text_field=data["text_field"],
        math_validation_tokens=data["math_validation_tokens"],
        training_tokens=data["training_tokens"],
        benchmark_texts=benchmark_texts,
        ngram_size=data["ngram_size"],
        near_threshold=data["near_threshold"],
    )
    write_jsonl(paths["train_text"], train.documents)
    write_jsonl(paths["math_text"], math_validation.documents)
    write_jsonl(paths["general_text"], general_validation.documents)
    torch.save(train.tokens, paths["train_tokens"])
    torch.save(math_validation.tokens, paths["math_tokens"])
    torch.save(general_validation.tokens, paths["general_tokens"])

    common = {
        "tokenizer": config["model"]["name_or_path"],
        "tokenizer_revision": config["model"]["revision"],
        "tokenizer_digest": tokenizer_hash,
        "sequence_length": data["sequence_length"],
        "git_commit": _git_commit(),
    }
    train_manifest = {
        **common,
        "dataset": data["dataset"],
        "dataset_configuration": data["subset"],
        "revision": data["revision"],
        "split": data["split"],
        "selected_shard_or_range": data["sampling_rule"],
        "sampling_rule": data["sampling_rule"],
        "document_count": len(train.documents),
        "raw_text_bytes": train.raw_text_bytes,
        "normalized_count": len(train.documents),
        "duplicate_removals": train.duplicate_removals,
        "token_count": int(train.tokens.numel()),
        "selected_token_budget": data["training_tokens"],
        "data_digest": tensor_digest(train.tokens),
        "text_file": str(paths["train_text"]),
        "text_file_sha256": file_sha256(paths["train_text"]),
        "token_file": str(paths["train_tokens"]),
        "token_file_sha256": file_sha256(paths["train_tokens"]),
        "filter_configuration": {
            "normalization": "normalize_text + normalized SHA-256 deduplication",
            "ngram_size": data["ngram_size"],
            "near_threshold": data["near_threshold"],
            "controlled_benchmark": data["controlled_math_source"],
            "general_validation": data["general_validation_manifest"],
        },
    }
    math_manifest = {
        **common,
        "dataset": data["dataset"],
        "dataset_configuration": data["subset"],
        "revision": data["revision"],
        "partition": "first accepted normalized records, frozen before CPT train range",
        "document_count": len(math_validation.documents),
        "raw_text_bytes": math_validation.raw_text_bytes,
        "normalized_count": len(math_validation.documents),
        "duplicate_removals": math_validation.duplicate_removals,
        "token_count": int(math_validation.tokens.numel()),
        "data_digest": tensor_digest(math_validation.tokens),
        "text_file": str(paths["math_text"]),
        "text_file_sha256": file_sha256(paths["math_text"]),
        "token_file": str(paths["math_tokens"]),
        "token_file_sha256": file_sha256(paths["math_tokens"]),
    }
    general_manifest = {
        **common,
        "dataset": "HuggingFaceFW/fineweb-edu",
        "dataset_configuration": "sample-10BT",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "source_contract": "GPU-1 frozen validation text, independently Qwen-tokenized",
        "source_file": str(general_source),
        "source_file_sha256": file_sha256(general_source),
        "document_count": len(general_validation.documents),
        "raw_text_bytes": general_validation.raw_text_bytes,
        "normalized_count": len(general_validation.documents),
        "duplicate_removals": general_validation.duplicate_removals,
        "token_count": int(general_validation.tokens.numel()),
        "data_digest": tensor_digest(general_validation.tokens),
        "text_file": str(paths["general_text"]),
        "text_file_sha256": file_sha256(paths["general_text"]),
        "token_file": str(paths["general_tokens"]),
        "token_file_sha256": file_sha256(paths["general_tokens"]),
    }
    write_json(data["train_manifest"], train_manifest)
    write_json(data["math_validation_manifest"], math_manifest)
    write_json(data["general_validation_manifest"], general_manifest)
    audit.update(
        {
            "dataset": data["dataset"],
            "revision": data["revision"],
            "train_manifest": data["train_manifest"],
            "math_validation_manifest": data["math_validation_manifest"],
            "general_validation_manifest": data["general_validation_manifest"],
            "controlled_benchmark": data["controlled_math_source"],
        }
    )
    write_json(data["contamination_report"], audit)
    decision = {
        "decision": "10M-token bounded formal full-parameter CPT",
        "formal_training_tokens": config["training"]["formal_tokens"],
        "formal_optimizer_steps": config["training"]["formal_steps"],
        "qualification_tokens": config["training"]["qualification_tokens"],
        "sequence_length": data["sequence_length"],
        "micro_batch": config["training"]["micro_batch_size"],
        "gradient_accumulation": config["training"]["gradient_accumulation_steps"],
        "learning_rate": config["optimizer"]["lr"],
        "learning_rate_reason": (
            "2e-5 is a conservative full-parameter CPT rate for an already pretrained 0.6B LM; "
            "no LR sweep is authorized."
        ),
        "selected_before_formal_run": True,
        "config_hash": config_hash(config),
        "git_commit": _git_commit(),
    }
    write_json("artifacts/training/qwen_cpt_budget_decision.json", decision)
    return {
        "train": train_manifest,
        "math_validation": math_manifest,
        "general_validation": general_manifest,
        "contamination": audit,
        "budget": decision,
    }


def baseline(config: dict[str, Any]) -> dict[str, Any]:
    from minillm_forge.finetuning.full_sft import load_full_sft_model

    data = config["data"]
    paths = _processed_paths(config)
    tokenizer = _load_tokenizer(config)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = load_full_sft_model(
        config["model"]["name_or_path"],
        revision=config["model"]["revision"],
        precision="bf16",
        gradient_checkpointing=False,
        low_cpu_mem_usage=True,
    ).to("cuda")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    manifest = {
        "model_id": config["model"]["name_or_path"],
        "revision": config["model"]["revision"],
        "parameter_count": parameters,
        "config_digest": _hash_json(model.config.to_dict()),
        "tokenizer_revision": config["model"]["revision"],
        "tokenizer_digest": _tokenizer_digest(tokenizer),
        "tokenizer_vocab_size": len(tokenizer),
        "dtype": "bfloat16",
        "git_commit": _git_commit(),
    }
    write_json(MODEL_MANIFEST, manifest)
    started = time.perf_counter()
    evaluation = evaluate_domains(
        model,
        math_tokens_path=paths["math_tokens"],
        general_tokens_path=paths["general_tokens"],
        sequence_length=data["sequence_length"],
    )
    memory = cuda_memory()
    result = {
        "classification": "QWEN_BASELINE_FROZEN",
        "experiment_id": "E00",
        "model": config["model"]["name_or_path"],
        "revision": config["model"]["revision"],
        "parameters": parameters,
        "tokenizer_digest": manifest["tokenizer_digest"],
        "seed": config["training"]["seed"],
        **evaluation,
        "controlled_math_em": None,
        "controlled_math_note": (
            "Not run: generative EM is secondary and not a reliable primary metric "
            "for this Base LM."
        ),
        "peak_allocated_vram_mib": memory["peak_allocated_mib"],
        "peak_reserved_vram_mib": memory["peak_reserved_mib"],
        "elapsed_seconds": time.perf_counter() - started,
        "math_manifest": data["math_validation_manifest"],
        "general_manifest": data["general_validation_manifest"],
        "git_commit": _git_commit(),
    }
    write_json(BASE_EVAL, result)
    write_json("experiments/results/E00.json", result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _trainer(config: dict[str, Any], *, total_steps: int, output_dir: str) -> QwenCPTTrainer:
    data = config["data"]
    training = config["training"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    paths = _processed_paths(config)
    return QwenCPTTrainer(
        model_name=config["model"]["name_or_path"],
        revision=config["model"]["revision"],
        train_tokens_path=paths["train_tokens"],
        math_tokens_path=paths["math_tokens"],
        general_tokens_path=paths["general_tokens"],
        sequence_length=data["sequence_length"],
        gradient_accumulation=training["gradient_accumulation_steps"],
        total_steps=total_steps,
        learning_rate=optimizer["lr"],
        betas=tuple(optimizer["betas"]),
        weight_decay=optimizer["weight_decay"],
        warmup_ratio=scheduler["warmup_ratio"],
        min_lr_ratio=scheduler["min_lr_ratio"],
        grad_clip=training["grad_clip"],
        output_dir=output_dir,
        run_config=config,
        seed=training["seed"],
        empty_cache_after_step=training["empty_cache_after_step"],
    )


def calibrate(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path("runs/E04-qwen3-math-cpt-calibration")
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    trainer = _trainer(
        config,
        total_steps=config["training"]["calibration_steps"],
        output_dir=str(output_dir),
    )
    trainer.train_to(
        config["training"]["calibration_steps"],
        log_every=1,
    )
    trainer_summary = trainer_summary_with_status(trainer)
    qualified = (
        trainer_summary["estimated_min_system_headroom_mib"] >= 1536
        and trainer_summary["nan_events"] == 0
        and trainer_summary["inf_events"] == 0
        and trainer_summary["oom_events"] == 0
    )
    result = {
        "classification": "CALIBRATION_PASS" if qualified else "CALIBRATION_HEADROOM_WARNING",
        "configuration": {
            "sequence_length": config["data"]["sequence_length"],
            "micro_batch": config["training"]["micro_batch_size"],
            "gradient_accumulation": config["training"]["gradient_accumulation_steps"],
            "precision": config["training"]["precision"],
            "gradient_checkpointing": config["training"]["gradient_checkpointing"],
            "optimizer_foreach": config["optimizer"]["foreach"],
        },
        **trainer_summary,
    }
    write_json(CALIBRATION, result)
    report = f"""# Qwen CPT Memory Calibration

## Decision

**{result["classification"]}**

The frozen long-run candidate is sequence length 512, micro-batch 1, gradient
accumulation 8, BF16, gradient checkpointing, and full-parameter AdamW with
`foreach=false`. The latter avoids list-wide optimizer temporaries; it does not change
the full-parameter CPT objective.

## Measured calibration

- Optimizer steps: {result["global_step"]}
- Input tokens: {result["tokens_seen"]}
- Peak allocated: {result["peak_allocated_mib"]:.2f} MiB
- Peak reserved: {result["peak_reserved_mib"]:.2f} MiB
- Estimated minimum system headroom: {result["estimated_min_system_headroom_mib"]:.2f} MiB
- Median throughput: {result["median_tokens_per_second"]:.2f} tokens/s
- NaN / Inf / OOM: {result["nan_events"]} / {result["inf_events"]} / {result["oom_events"]}

The gate target is at least 1536 MiB practical system-level headroom. A short calibration
does not itself establish long-run stability; the 503,808-token qualification is separate.
"""
    report_path = Path("reports/ablations/QWEN_CPT_MEMORY_CALIBRATION.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def trainer_summary_with_status(trainer: QwenCPTTrainer) -> dict[str, Any]:
    summary = trainer.summary()
    state = summary.pop("state")
    return {
        **summary,
        **state,
        "peak_allocated_mib": summary["peak_allocated_vram_mib"],
        "peak_reserved_mib": summary["peak_reserved_vram_mib"],
    }


def _rng_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def qualify(config: dict[str, Any]) -> dict[str, Any]:
    total_steps = config["training"]["qualification_steps"]
    output_dir = Path("runs/E04-qwen3-math-cpt-qualification")
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    first = _trainer(config, total_steps=total_steps, output_dir=str(output_dir))
    first.train_to(total_steps - 1, log_every=config["training"]["log_every"])
    checkpoint = first.output_dir / "resume-source.pt"
    first.save("resume-source.pt")
    first_summary = trainer_summary_with_status(first)
    expected = {
        "global_step": first.state.global_step,
        "tokens_seen": first.state.tokens_seen,
        "learning_rate": first.optimizer.param_groups[0]["lr"],
        "scheduler_last_epoch": first.scheduler.last_epoch,
        "torch_rng": _rng_digest(torch.get_rng_state()),
        "cuda_rng": [_rng_digest(item) for item in torch.cuda.get_rng_state_all()],
    }
    del first
    gc.collect()
    torch.cuda.empty_cache()

    resumed = _trainer(config, total_steps=total_steps, output_dir=str(output_dir))
    payload = resumed.resume(checkpoint)
    restored = {
        "global_step": resumed.state.global_step,
        "tokens_seen": resumed.state.tokens_seen,
        "learning_rate": resumed.optimizer.param_groups[0]["lr"],
        "scheduler_last_epoch": resumed.scheduler.last_epoch,
        "torch_rng": _rng_digest(torch.get_rng_state()),
        "cuda_rng": [_rng_digest(item) for item in torch.cuda.get_rng_state_all()],
    }
    resume_match = expected == restored
    resumed.train_to(
        total_steps,
        eval_steps={total_steps},
        checkpoint_steps={total_steps: "last.pt"},
        log_every=1,
    )
    second_summary = trainer_summary_with_status(resumed)
    evaluations = _read_metrics(metrics_path, "evaluation")
    rates = [record["tokens_per_second"] for record in _read_metrics(metrics_path, "train")]
    peak_allocated = max(first_summary["peak_allocated_mib"], second_summary["peak_allocated_mib"])
    peak_reserved = max(first_summary["peak_reserved_mib"], second_summary["peak_reserved_mib"])
    min_headroom = min(
        first_summary["estimated_min_system_headroom_mib"],
        second_summary["estimated_min_system_headroom_mib"],
    )
    nan_events = first_summary["nan_events"] + second_summary["nan_events"]
    inf_events = first_summary["inf_events"] + second_summary["inf_events"]
    oom_events = first_summary["oom_events"] + second_summary["oom_events"]
    passed = (
        resumed.state.tokens_seen == config["training"]["qualification_tokens"]
        and nan_events == 0
        and inf_events == 0
        and oom_events == 0
        and resume_match
        and bool(evaluations)
        and min_headroom >= 1536
    )
    result = {
        "classification": (
            "FULL_CPT_LONG_RUN_QUALIFIED" if passed else "FULL_CPT_8GB_LONG_RUN_BLOCKED"
        ),
        "global_step": resumed.state.global_step,
        "tokens_seen": resumed.state.tokens_seen,
        "peak_allocated_vram_mib": peak_allocated,
        "peak_reserved_vram_mib": peak_reserved,
        "estimated_min_system_headroom_mib": min_headroom,
        "median_tokens_per_second": median(rates),
        "elapsed_seconds": first_summary["elapsed_seconds"] + second_summary["elapsed_seconds"],
        "nan_events": nan_events,
        "inf_events": inf_events,
        "oom_events": oom_events,
        "evaluation": evaluations[-1] if evaluations else None,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "resume_validation": (
            "CPT_RESUME_SEMANTICS_VALIDATED" if resume_match else "CPT_RESUME_INVALID"
        ),
        "resume_expected": expected,
        "resume_restored": restored,
        "resume_continued_to_step": resumed.state.global_step,
        "git_commit": _git_commit(),
    }
    write_json(QUALIFICATION, result)
    del resumed, payload
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _read_metrics(path: str | Path, event: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if event is None or record.get("event") == event:
                records.append(record)
    return records


def formal(config: dict[str, Any], resume: str | None = None) -> dict[str, Any]:
    qualification = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    if qualification["classification"] != "FULL_CPT_LONG_RUN_QUALIFIED":
        raise RuntimeError("formal CPT is blocked until FULL_CPT_LONG_RUN_QUALIFIED")
    base = json.loads(BASE_EVAL.read_text(encoding="utf-8"))
    training = config["training"]
    output_dir = Path(training["output_dir"])
    metrics_path = output_dir / "metrics.jsonl"
    if resume is None and metrics_path.exists():
        metrics_path.unlink()
    trainer = _trainer(config, total_steps=training["formal_steps"], output_dir=str(output_dir))
    if resume:
        trainer.resume(resume)
    else:
        from minillm_forge.training.qwen_cpt import append_jsonl

        append_jsonl(
            metrics_path,
            {
                "event": "evaluation",
                "global_step": 0,
                "tokens_seen": 0,
                "math_loss": base["math_loss"],
                "math_ppl": base["math_ppl"],
                "math_tokens": base["math_tokens"],
                "general_loss": base["general_loss"],
                "general_ppl": base["general_ppl"],
                "general_tokens": base["general_tokens"],
            },
        )
    all_milestones = config["evaluation"]["milestone_steps"]
    milestones = [step for step in all_milestones if step > trainer.state.global_step]
    names = {
        all_milestones[0]: "early.pt",
        all_milestones[1]: "mid.pt",
        all_milestones[2]: "three-quarter.pt",
        all_milestones[3]: "last.pt",
    }
    degradation_streak = 0
    early_stopped = False
    previous_math_ppl = float(base["math_ppl"])
    for milestone in milestones:
        trainer.train_to(
            milestone,
            eval_steps={milestone},
            checkpoint_steps={milestone: names[milestone]},
            log_every=training["log_every"],
            save_best=True,
        )
        current = _read_metrics(metrics_path, "evaluation")[-1]
        general_relative = current["general_loss"] / base["general_loss"] - 1.0
        math_progress = (previous_math_ppl - current["math_ppl"]) / previous_math_ppl
        if general_relative >= config["evaluation"]["general_loss_relative_stop_threshold"]:
            degradation_streak += 1
        else:
            degradation_streak = 0
        if (
            degradation_streak >= config["evaluation"]["sustained_evaluations"]
            and math_progress <= config["evaluation"]["math_ppl_plateau_relative_threshold"]
        ):
            early_stopped = True
            break
        previous_math_ppl = current["math_ppl"]
    summary = trainer_summary_with_status(trainer)
    evaluations = _read_metrics(metrics_path, "evaluation")
    train_records = _read_metrics(metrics_path, "train")
    final_eval = evaluations[-1]
    if early_stopped:
        trainer.save("last.pt", final_eval)
    tokenizer = _load_tokenizer(config)
    export_path = output_dir / "final_model"
    trainer.model.save_pretrained(export_path, safe_serialization=True)
    tokenizer.save_pretrained(export_path)
    write_json(
        POST_EVAL,
        {
            "classification": "POST_CPT_FROZEN_EVALUATION",
            "model_path": str(export_path),
            "base_revision": config["model"]["revision"],
            "math_manifest": config["data"]["math_validation_manifest"],
            "general_manifest": config["data"]["general_validation_manifest"],
            **final_eval,
            "controlled_math_em": None,
        },
    )
    math_delta = final_eval["math_ppl"] - base["math_ppl"]
    general_delta = final_eval["general_ppl"] - base["general_ppl"]
    math_relative = math_delta / base["math_ppl"]
    general_relative = general_delta / base["general_ppl"]
    if math_delta >= 0:
        classification = "QWEN_MATH_CPT_NO_DOMAIN_GAIN"
    elif general_relative > 0.01:
        classification = "QWEN_MATH_CPT_VALIDATED_WITH_GENERAL_DEGRADATION"
    else:
        classification = "QWEN_MATH_CPT_VALIDATED"
    hypothesis_results = {
        "H-CPT-1": (
            "supported"
            if math_relative <= -0.01
            else "partially-supported"
            if math_delta < 0
            else "rejected"
        ),
        "H-CPT-2": (
            "supported"
            if general_relative > 0.01
            else "partially-supported"
            if general_delta > 0
            else "rejected"
        ),
        "H-CPT-3": "supported",
    }
    result = {
        "classification": classification,
        "experiment_id": "E04",
        "base_model": config["model"]["name_or_path"],
        "base_revision": config["model"]["revision"],
        "parameters": sum(parameter.numel() for parameter in trainer.model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
        ),
        "sequence_length": config["data"]["sequence_length"],
        "training_tokens": trainer.state.tokens_seen,
        "optimizer_steps": trainer.state.global_step,
        "micro_batch": training["micro_batch_size"],
        "gradient_accumulation": training["gradient_accumulation_steps"],
        "precision": training["precision"],
        "optimizer": config["optimizer"]["name"],
        "learning_rate": config["optimizer"]["lr"],
        "warmup_ratio": config["scheduler"]["warmup_ratio"],
        "initial_math_loss": base["math_loss"],
        "final_math_loss": final_eval["math_loss"],
        "initial_math_ppl": base["math_ppl"],
        "final_math_ppl": final_eval["math_ppl"],
        "initial_general_loss": base["general_loss"],
        "final_general_loss": final_eval["general_loss"],
        "initial_general_ppl": base["general_ppl"],
        "final_general_ppl": final_eval["general_ppl"],
        "final_train_loss": train_records[-1]["train_loss"] if train_records else None,
        "peak_allocated_vram_mib": summary["peak_allocated_mib"],
        "peak_reserved_vram_mib": summary["peak_reserved_mib"],
        "estimated_min_system_headroom_mib": summary["estimated_min_system_headroom_mib"],
        "median_tokens_per_second": (
            median(record["tokens_per_second"] for record in train_records) if train_records else 0
        ),
        "elapsed_seconds": summary["elapsed_seconds"],
        "resume_validation": qualification["resume_validation"],
        "nan_events": summary["nan_events"],
        "inf_events": summary["inf_events"],
        "oom_events": summary["oom_events"],
        "early_stopped": early_stopped,
        "termination_reason": (
            "CPT_EARLY_STOP_GENERAL_DEGRADATION" if early_stopped else "TOKEN_BUDGET_REACHED"
        ),
        "math_ppl_delta": math_delta,
        "math_ppl_relative_delta": math_relative,
        "general_ppl_delta": general_delta,
        "general_ppl_relative_delta": general_relative,
        "controlled_math_em": None,
        "hypothesis_results": hypothesis_results,
        "metrics_path": str(metrics_path),
        "model_path": str(export_path),
        "git_commit": _git_commit(),
    }
    write_json(FORMAL_RESULT, result)
    write_json("experiments/results/E04.json", result)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    from minillm_forge.evaluation.qwen_cpt_evidence import finalize, run_regression

    parser = argparse.ArgumentParser(description="Run a frozen Stage GPU-2A phase")
    parser.add_argument(
        "phase",
        choices=[
            "prepare",
            "baseline",
            "calibrate",
            "qualify",
            "formal",
            "regression",
            "finalize",
        ],
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    operation = {
        "prepare": prepare,
        "baseline": baseline,
        "calibrate": calibrate,
        "qualify": qualify,
        "formal": lambda value: formal(value, args.resume),
        "regression": run_regression,
        "finalize": finalize,
    }[args.phase]
    print(json.dumps(operation(config), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
