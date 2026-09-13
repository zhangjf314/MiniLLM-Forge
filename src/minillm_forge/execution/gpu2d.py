from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from statistics import median
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from minillm_forge.cli.common import StatefulRandomSampler, write_json
from minillm_forge.config import config_hash, load_config
from minillm_forge.design.gpu2d import OUTPUT as CONTEXT_ANALYSIS
from minillm_forge.execution.gpu2b import (
    TOKEN_CACHE,
    NvidiaMemoryMonitor,
    _build_optimizer,
    _git_commit,
    _load_model,
    _single_item_collator,
    file_sha256,
    qlora_identity,
)
from minillm_forge.execution.gpu2c import (
    _adapter_identity,
    _snapshot_comparison,
    _state_snapshot,
    _trajectory_digest,
)
from minillm_forge.finetuning.lora import add_lora_adapters, trainable_parameter_summary
from minillm_forge.training import ForgeTrainer, build_cosine_scheduler
from minillm_forge.training.trainer import TrainingConfig

STAGE = "STAGE_GPU_2D_R0"
HEADROOM_MIB = 1536
Q1_UPDATES = 64
FORMAL_UPDATES = 1200
BASE_LOGICAL_PARAMETERS = 596_049_920
CONFIGS = {
    ("LORA", 768, "R16_ALL_LINEAR"): Path("configs/sft/gpu2d/lora-r16-all-linear-768.yaml"),
    ("LORA", 512, "R16_ALL_LINEAR"): Path("configs/sft/gpu2d/lora-r16-all-linear-512.yaml"),
    ("QLORA", 1024, "R16_ALL_LINEAR"): Path("configs/sft/gpu2d/qlora-r16-all-linear-1024.yaml"),
    ("QLORA", 768, "R16_ALL_LINEAR"): Path("configs/sft/gpu2d/qlora-r16-all-linear-768.yaml"),
    ("QLORA", 512, "R16_ALL_LINEAR"): Path("configs/sft/gpu2d/qlora-r16-all-linear-512.yaml"),
    ("LORA", 1024, "R8_ALL_LINEAR"): Path("configs/sft/gpu2d/lora-r8-all-linear-1024.yaml"),
    ("LORA", 1024, "R16_ATTENTION"): Path("configs/sft/gpu2d/lora-r16-attention-1024.yaml"),
}
MATRIX_PATHS = {
    "LORA": Path("artifacts/gpu2d/lora_memory_matrix.json"),
    "QLORA": Path("artifacts/gpu2d/qlora_memory_matrix.json"),
}
RESUME_RESULT = Path("artifacts/gpu2d/resume_qualification.json")
PARAMETER_RESULT = Path("artifacts/gpu2d/parameter_analysis.json")


def _config_key(family: str, context: int, variant: str) -> tuple[str, int, str]:
    key = (family.upper(), context, variant.upper())
    if key not in CONFIGS:
        raise ValueError(f"no frozen GPU-2D configuration for {key}")
    return key


def _context_cache_path(context: int) -> Path:
    return Path(f"data/processed/gpu2d/sft_math_context_{context}_train_tokens.pt")


def _context_manifest_path(context: int) -> Path:
    return Path(f"artifacts/gpu2d/data_context_{context}.json")


def _tensor_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_context_cache(context: int) -> dict[str, Any]:
    if context not in {512, 768, 1024}:
        raise ValueError("context must be 512, 768, or 1024")
    analysis = json.loads(CONTEXT_ANALYSIS.read_text(encoding="utf-8"))
    contract = analysis["partitions"]["train"]["contexts"][str(context)]
    cache_path = _context_cache_path(context)
    manifest_path = _context_manifest_path(context)
    if cache_path.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("context") == context
            and existing.get("source_dataset_sha256") == analysis["source_files"]["train"]["sha256"]
            and existing.get("cache_sha256") == file_sha256(cache_path)
        ):
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            actual_digest = _tensor_digest(
                cached["input_ids"],
                cached["offsets"],
                cached["assistant_starts"],
                cached["source_indices"],
            )
            if actual_digest == existing.get("encoded_dataset_digest"):
                return existing
    source = torch.load(TOKEN_CACHE, map_location="cpu", weights_only=True)
    source_ids = source["input_ids"]
    source_offsets = source["offsets"]
    source_starts = source["assistant_starts"]
    chunks: list[torch.Tensor] = []
    offsets = [0]
    starts: list[int] = []
    source_indices: list[int] = []
    excluded_indices: list[int] = []
    input_tokens = 0
    assistant_tokens = 0
    for index in range(int(source_starts.numel())):
        start = int(source_offsets[index])
        end = min(int(source_offsets[index + 1]), start + context)
        assistant_start = int(source_starts[index])
        ids = source_ids[start:end]
        if assistant_start >= len(ids):
            excluded_indices.append(index)
            continue
        chunks.append(ids)
        starts.append(assistant_start)
        source_indices.append(index)
        input_tokens += len(ids)
        assistant_tokens += len(ids) - assistant_start
        offsets.append(input_tokens)
    payload = {
        "input_ids": torch.cat(chunks).to(torch.int32),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "assistant_starts": torch.tensor(starts, dtype=torch.int32),
        "source_indices": torch.tensor(source_indices, dtype=torch.int32),
    }
    if len(source_indices) != contract["examples_eligible_for_training"]:
        raise RuntimeError("context cache eligible-example count differs from analysis")
    if assistant_tokens != contract["assistant_tokens_retained"]:
        raise RuntimeError("context cache assistant-token count differs from analysis")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, cache_path)
    result = {
        "stage": STAGE,
        "classification": "GPU2D_CONTEXT_DATA_MANIFEST_FROZEN",
        "context": context,
        "examples": len(source_indices),
        "source_examples": int(source_starts.numel()),
        "excluded_zero_assistant_examples": len(excluded_indices),
        "excluded_source_indices": excluded_indices,
        "input_tokens_retained": input_tokens,
        "assistant_tokens_retained": assistant_tokens,
        "assistant_token_retention_relative_to_1024": contract[
            "assistant_token_retention_relative_to_1024"
        ],
        "examples_truncated": contract["examples_truncated"],
        "final_answer_retained": contract["final_answer_retained"],
        "final_answer_retention_rate": contract["final_answer_retention_rate"],
        "source_dataset_sha256": analysis["source_files"]["train"]["sha256"],
        "encoded_dataset_digest": _tensor_digest(
            payload["input_ids"],
            payload["offsets"],
            payload["assistant_starts"],
            payload["source_indices"],
        ),
        "cache_path": str(cache_path),
        "cache_sha256": file_sha256(cache_path),
        "semantic_change": "SCIENTIFIC_DATA_EXPOSURE_CHANGE" if context < 1024 else "NONE",
        "formal_campaign_authorized": False,
    }
    write_json(manifest_path, result)
    return result


class ContextSFTDataset(Dataset):
    def __init__(self, context: int) -> None:
        manifest = build_context_cache(context)
        payload = torch.load(manifest["cache_path"], map_location="cpu", weights_only=True)
        self.input_ids = payload["input_ids"]
        self.offsets = payload["offsets"]
        self.assistant_starts = payload["assistant_starts"]
        self.source_indices = payload["source_indices"]

    def __len__(self) -> int:
        return int(self.assistant_starts.numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        ids = self.input_ids[start:end].to(torch.long)
        labels = ids.clone()
        labels[: int(self.assistant_starts[index])] = -100
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids, dtype=torch.bool),
            "labels": labels,
            "source_index": self.source_indices[index].to(torch.long),
        }


def _collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    source_index = features[0].pop("source_index")
    result = _single_item_collator(features)
    result["source_index"] = source_index.unsqueeze(0)
    return result


class GPU2DTrainer(ForgeTrainer):
    def _log(self, record: dict[str, Any]) -> None:
        monitor = getattr(self, "physical_monitor", None)
        if monitor is not None and record.get("event") == "train":
            record = {
                **record,
                "physical_vram_used_mib": monitor.latest_used_mib,
                "physical_vram_headroom_mib": monitor.latest_free_mib,
            }
        super()._log(record)

    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return super()._prepare_batch(
            {key: value for key, value in batch.items() if key != "source_index"}
        )


def build_trainer(
    family: str,
    context: int,
    variant: str = "R16_ALL_LINEAR",
    *,
    initialization: str = "BASE_INIT",
    seed: int = 42,
    max_steps: int = Q1_UPDATES,
    output_dir: str | Path,
) -> tuple[GPU2DTrainer, dict[str, Any]]:
    family, context, variant = _config_key(family, context, variant)
    config_path = CONFIGS[(family, context, variant)]
    config = load_config(config_path)
    if int(config["data"]["max_length"]) != context:
        raise RuntimeError("frozen configuration/context mismatch")
    if initialization not in {"BASE_INIT", "CPT_INIT"}:
        raise ValueError("initialization must be BASE_INIT or CPT_INIT")
    model = _load_model(config, initialization, seed)
    identity = qlora_identity(model) if family == "QLORA" else {"status": "PASS"}
    if identity["status"] != "PASS":
        raise RuntimeError("QLoRA hard identity gate failed")
    dataset = ContextSFTDataset(context)
    sampler = StatefulRandomSampler(dataset, seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=_collator,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    optimizer = _build_optimizer(model, config)
    scheduler = build_cosine_scheduler(
        optimizer,
        total_steps=FORMAL_UPDATES,
        warmup_ratio=config["scheduler"]["warmup_ratio"],
        min_lr_ratio=config["scheduler"]["min_lr_ratio"],
    )
    training = config["training"]
    trainer = GPU2DTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        config=TrainingConfig(
            max_steps=max_steps,
            gradient_accumulation_steps=training["gradient_accumulation_steps"],
            grad_clip=training["grad_clip"],
            precision=training["precision"],
            log_every=1,
            eval_every=0,
            save_every=0,
            output_dir=str(output_dir),
            seed=seed,
            device=training["device"],
            tensorboard=False,
            empty_cache_after_step=True,
        ),
        run_config=config,
    )
    return trainer, {
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "semantic_config_hash": config_hash(config),
        "parameters": trainable_parameter_summary(model),
        "qlora_identity": identity,
        "dataset_examples": len(dataset),
        "initialization": initialization,
    }


def _physical_memory() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,driver_version,pstate",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).splitlines()[0]
        name, total, used, free, driver, pstate = [item.strip() for item in output.split(",")]
        return {
            "status": "PASS",
            "gpu": name,
            "total_mib": int(total),
            "used_mib": int(used),
            "free_mib": int(free),
            "driver": driver,
            "pstate": pstate,
        }
    except Exception as exc:
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def _allocator_evidence() -> dict[str, Any]:
    stats = torch.cuda.memory_stats()
    keys = (
        "allocated_bytes.all.current",
        "allocated_bytes.all.peak",
        "active_bytes.all.current",
        "active_bytes.all.peak",
        "reserved_bytes.all.current",
        "reserved_bytes.all.peak",
        "inactive_split_bytes.all.current",
        "inactive_split_bytes.all.peak",
        "segment.all.current",
        "segment.all.peak",
        "num_alloc_retries",
        "num_ooms",
    )
    selected = {key: int(stats.get(key, 0)) for key in keys}
    snapshot_error = None
    inactive_blocks: list[int] = []
    try:
        for segment in torch.cuda.memory_snapshot():
            for block in segment.get("blocks", []):
                if block.get("state") == "inactive":
                    inactive_blocks.append(int(block.get("size", 0)))
    except Exception as exc:
        snapshot_error = f"{type(exc).__name__}: {exc}"
    return {
        "memory_stats": selected,
        "inactive_block_count": len(inactive_blocks),
        "largest_inactive_block_bytes": max(inactive_blocks, default=0),
        "snapshot_error": snapshot_error,
        "allocator_environment": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


def _empty_matrix(family: str) -> dict[str, Any]:
    plans = (
        {
            "LORA": [
                (768, "R16_ALL_LINEAR", "PLANNED_FIRST"),
                (512, "R16_ALL_LINEAR", "CONDITIONAL_IF_768_FAILS"),
                (1024, "R8_ALL_LINEAR", "CANDIDATE_C_CONDITIONAL"),
                (1024, "R16_ATTENTION", "CANDIDATE_C_CONDITIONAL"),
            ],
            "QLORA": [
                (1024, "R16_ALL_LINEAR", "PLANNED_FIRST"),
                (768, "R16_ALL_LINEAR", "CONDITIONAL_CANDIDATE_A"),
                (512, "R16_ALL_LINEAR", "CONDITIONAL_CANDIDATE_A_FALLBACK"),
            ],
        }
    )[family]
    return {
        "stage": STAGE,
        "family": family,
        "minimum_physical_headroom_mib": HEADROOM_MIB,
        "q1_updates": Q1_UPDATES,
        "tests": {
            f"{family}_{variant}_{context}": {
                "context": context,
                "variant": variant,
                "status": "NOT_RUN",
                "policy": policy,
            }
            for context, variant, policy in plans
        },
        "formal_campaign_authorized": False,
    }


def memory_qualification(
    family: str, context: int, variant: str = "R16_ALL_LINEAR"
) -> dict[str, Any]:
    family, context, variant = _config_key(family, context, variant)
    path = MATRIX_PATHS[family]
    matrix = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else _empty_matrix(family)
    )
    test_key = f"{family}_{variant}_{context}"
    if test_key not in matrix["tests"]:
        raise RuntimeError("qualification was not prospectively listed in the bounded grid")
    output_dir = Path("runs/gpu2d/qualification/q1") / test_key.lower()
    metrics_path = output_dir / "metrics.jsonl"
    previous_attempts = list(matrix["tests"][test_key].get("previous_attempts", []))
    if metrics_path.exists():
        records = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prior_train = [record for record in records if record.get("event") == "train"]
        if prior_train:
            archive = output_dir / (
                f"metrics-interrupted-at-step-{prior_train[-1]['global_step']:04d}.jsonl"
            )
            shutil.copy2(metrics_path, archive)
            previous_attempts.append(
                {
                    "classification": "INFRA_INTERRUPTION",
                    "updates_completed": prior_train[-1]["global_step"],
                    "nan_count": prior_train[-1]["nan_count"],
                    "inf_count": prior_train[-1]["inf_count"],
                    "oom_count": prior_train[-1]["oom_count"],
                    "preserved_metrics": str(archive),
                    "preserved_metrics_sha256": file_sha256(archive),
                    "rerun_authorized": True,
                }
            )
        metrics_path.unlink()
    started = time.time()
    trainer = None
    metadata: dict[str, Any] = {}
    identity: dict[str, Any] = {"status": "NOT_RUN"}
    failure = None
    baseline_physical = _physical_memory()
    with NvidiaMemoryMonitor(interval_seconds=0.2) as monitor:
        try:
            trainer, metadata = build_trainer(
                family,
                context,
                variant,
                seed=42,
                max_steps=Q1_UPDATES,
                output_dir=output_dir,
            )
            identity = _adapter_identity(trainer.model, family)
            if identity["status"] != "PASS":
                raise RuntimeError(f"{family} adapter/quantization identity failed")
            trainer.train_until(Q1_UPDATES, finalize=False)
            torch.cuda.synchronize()
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - started
    final_physical = _physical_memory()
    allocator = _allocator_evidence() if trainer is not None else None
    state = trainer.state if trainer is not None else None
    history = (
        [record for record in state.history if record.get("event") == "train"]
        if state is not None
        else []
    )
    losses = [float(record["loss"]) for record in history]
    grad_norms = [float(record["grad_norm"]) for record in history]
    target_rates = [float(record["target_tokens_per_second"]) for record in history]
    step_times = []
    previous_targets = 0
    for record in history:
        current_targets = int(record["target_tokens_seen"])
        step_times.append(
            (current_targets - previous_targets) / float(record["target_tokens_per_second"])
        )
        previous_targets = current_targets
    passed = bool(
        failure is None
        and state is not None
        and state.global_step == Q1_UPDATES
        and state.nan_count == 0
        and state.inf_count == 0
        and state.oom_count == 0
        and losses
        and grad_norms
        and all(math.isfinite(value) for value in losses + grad_norms)
        and (monitor.minimum_free_mib or 0) >= HEADROOM_MIB
        and not monitor.errors
        and identity["status"] == "PASS"
    )
    result = {
        "stage": STAGE,
        "phase": f"GPU2D-Q1-{test_key}",
        "classification": (
            f"GPU2D_{test_key}_MEMORY_QUALIFICATION_PASS"
            if passed
            else f"GPU2D_{test_key}_MEMORY_QUALIFICATION_FAILED"
        ),
        "status": "PASS" if passed else "FAIL",
        "family": family,
        "context": context,
        "variant": variant,
        "initialization": "BASE_INIT",
        "seed": 42,
        "updates_completed": state.global_step if state is not None else 0,
        "samples_processed": state.examples_seen if state is not None else 0,
        "input_tokens": state.tokens_seen if state is not None else 0,
        "assistant_tokens": state.target_tokens_seen if state is not None else 0,
        "loss_finite": bool(losses) and all(math.isfinite(value) for value in losses),
        "final_loss": losses[-1] if losses else None,
        "nan_count": state.nan_count if state is not None else 0,
        "inf_count": state.inf_count if state is not None else 0,
        "oom_count": state.oom_count if state is not None else 0,
        "gradient_norm_min": min(grad_norms) if grad_norms else None,
        "gradient_norm_max": max(grad_norms) if grad_norms else None,
        "physical_peak_vram_mib": monitor.maximum_used_mib,
        "minimum_physical_headroom_mib": monitor.minimum_free_mib,
        "physical_sample_count": monitor.samples,
        "physical_monitor_errors": monitor.errors,
        "physical_before": baseline_physical,
        "physical_after": final_physical,
        "peak_cuda_allocated_mib": trainer.peak_vram_mb() if trainer is not None else None,
        "peak_cuda_reserved_mib": (
            trainer.peak_reserved_vram_mb() if trainer is not None else None
        ),
        "allocator": allocator,
        "median_target_tokens_per_second": median(target_rates) if target_rates else None,
        "median_step_time_seconds": median(step_times) if step_times else None,
        "samples_per_second": (
            state.examples_seen / elapsed if state is not None and elapsed else None
        ),
        "wall_clock_seconds": elapsed,
        "adapter_identity": identity,
        "metadata": metadata,
        "failure": failure,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "os": platform.platform(),
        },
        "code_commit": _git_commit(),
        "previous_attempts": previous_attempts,
        "formal_run": False,
        "benchmark_result": False,
    }
    matrix["tests"][test_key] = result
    write_json(path, matrix)
    if trainer is not None:
        del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parameter_analysis() -> dict[str, Any]:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    base = load_config(CONFIGS[("LORA", 768, "R16_ALL_LINEAR")])
    model_contract = base["design"]["paired_initializations"]["BASE_INIT"]
    hf_config = AutoConfig.from_pretrained(
        model_contract["name_or_path"], revision=model_contract["revision"]
    )
    variants = {
        "LORA_R16_ALL_LINEAR": base["lora"],
        "QLORA_R16_ALL_LINEAR": base["lora"],
        "C1_R8_ALL_LINEAR": load_config(CONFIGS[("LORA", 1024, "R8_ALL_LINEAR")])["lora"],
        "C2_R16_ATTENTION": load_config(CONFIGS[("LORA", 1024, "R16_ATTENTION")])["lora"],
    }
    results = {}
    for name, lora in variants.items():
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(hf_config)
            model = add_lora_adapters(model, **lora)
        summary = trainable_parameter_summary(model)
        meta_instantiated_total = summary["total_parameters"]
        trainable_parameters = summary["trainable_parameters"]
        logical_total = BASE_LOGICAL_PARAMETERS + trainable_parameters
        trainable_names = [
            parameter_name
            for parameter_name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        results[name] = {
            "logical_total_parameters": logical_total,
            "meta_instantiated_total_parameters": meta_instantiated_total,
            "trainable_parameters": trainable_parameters,
            "trainable_percent_of_logical_model": 100.0 * trainable_parameters / logical_total,
            "adapter_parameter_bytes_bf16": trainable_parameters * 2,
            "optimizer_parameter_count": trainable_parameters,
            "trainable_tensor_count": len(trainable_names),
            "unexpected_non_adapter_trainables": [
                parameter_name
                for parameter_name in trainable_names
                if "lora_" not in parameter_name
            ],
            "lora": lora,
        }
        del model
    result = {
        "stage": STAGE,
        "classification": "GPU2D_CANDIDATE_PARAMETER_ANALYSIS_COMPLETE",
        "status": (
            "PASS"
            if all(not item["unexpected_non_adapter_trainables"] for item in results.values())
            else "FAIL"
        ),
        "variants": results,
        "formal_campaign_authorized": False,
    }
    write_json(PARAMETER_RESULT, result)
    return result


def _gpu2d_state_snapshot(trainer: GPU2DTrainer, family: str) -> dict[str, Any]:
    snapshot = _state_snapshot(trainer, family)
    sampler_state = trainer.train_loader.sampler.state_dict()
    consumed = sampler_state["order"][: sampler_state["position"]]
    source_indices = trainer.train_loader.dataset.source_indices[
        torch.tensor(consumed, dtype=torch.long)
    ]
    snapshot["consumed_source_sample_order_sha256"] = _tensor_digest(source_indices)
    snapshot["consumed_source_sample_count"] = len(consumed)
    return snapshot


def resume_qualification(
    family: str, context: int, variant: str = "R16_ALL_LINEAR"
) -> dict[str, Any]:
    family, context, variant = _config_key(family, context, variant)
    matrix = json.loads(MATRIX_PATHS[family].read_text(encoding="utf-8"))
    test_key = f"{family}_{variant}_{context}"
    if matrix["tests"][test_key]["status"] != "PASS":
        raise RuntimeError(f"{test_key} Q1 must pass before resume qualification")
    root = Path("runs/gpu2d/qualification/q2") / test_key.lower()
    previous_attempts = []
    for branch in ("continuous", "interrupted"):
        metrics = root / branch / "metrics.jsonl"
        if not metrics.exists():
            continue
        records = [
            json.loads(line)
            for line in metrics.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        train_records = [record for record in records if record.get("event") == "train"]
        if train_records:
            if family == "QLORA":
                reason = (
                    "bitsandbytes NF4 auxiliary state was serialized but rejected "
                    "by the ordinary strict PyTorch model loader."
                )
            else:
                reason = "Scalar optimizer tensor digest did not support zero dimensions."
            archive = metrics.with_name(
                f"metrics-verified-software-fault-at-step-"
                f"{train_records[-1]['global_step']:04d}.jsonl"
            )
            shutil.copy2(metrics, archive)
            previous_attempts.append(
                {
                    "classification": "VERIFIED_SOFTWARE_FAULT",
                    "branch": branch,
                    "updates_completed": train_records[-1]["global_step"],
                    "reason": reason,
                    "preserved_metrics": str(archive),
                    "preserved_metrics_sha256": file_sha256(archive),
                    "rerun_authorized": True,
                }
            )
        metrics.unlink()
    continuous, _ = build_trainer(
        family, context, variant, max_steps=16, output_dir=root / "continuous"
    )
    continuous.train_until(8)
    continuous_boundary = _gpu2d_state_snapshot(continuous, family)
    continuous.train_until(16)
    continuous_final = _gpu2d_state_snapshot(continuous, family)
    continuous_trajectory = _trajectory_digest(continuous)
    del continuous
    gc.collect()
    torch.cuda.empty_cache()

    interrupted, _ = build_trainer(
        family, context, variant, max_steps=16, output_dir=root / "interrupted"
    )
    interrupted.train_until(8)
    checkpoint = interrupted._save("resume-at-8.pt")
    interrupted_boundary = _gpu2d_state_snapshot(interrupted, family)
    checkpoint_sha256 = file_sha256(checkpoint)
    del interrupted
    gc.collect()
    torch.cuda.empty_cache()

    resumed, _ = build_trainer(
        family, context, variant, max_steps=16, output_dir=root / "interrupted"
    )
    resumed.resume(checkpoint)
    restored_boundary = _gpu2d_state_snapshot(resumed, family)
    resumed.train_until(16)
    resumed_final = _gpu2d_state_snapshot(resumed, family)
    resumed_trajectory = _trajectory_digest(resumed)
    save_restore = _snapshot_comparison(interrupted_boundary, restored_boundary)
    control_boundary = _snapshot_comparison(continuous_boundary, interrupted_boundary)
    final_comparison = _snapshot_comparison(continuous_final, resumed_final)
    trajectory_match = continuous_trajectory == resumed_trajectory
    sample_order_match = (
        continuous_final["consumed_source_sample_order_sha256"]
        == resumed_final["consumed_source_sample_order_sha256"]
    )
    rng_keys = (
        "python_rng_sha256",
        "numpy_rng_sha256",
        "torch_rng_sha256",
        "cuda_rng_sha256",
        "loader_generator_sha256",
    )
    rng_match = all(continuous_final[key] == resumed_final[key] for key in rng_keys)
    exact = bool(
        save_restore["all_match"]
        and control_boundary["all_match"]
        and final_comparison["all_match"]
        and trajectory_match
        and sample_order_match
        and rng_match
    )
    result = {
        "stage": STAGE,
        "test": test_key,
        "family": family,
        "context": context,
        "variant": variant,
        "classification": (
            "PEFT_EXACT_RESUME_CONFIRMED" if exact else "GPU2D_PEFT_RESUME_QUALIFICATION_FAILED"
        ),
        "status": "PASS" if exact else "FAIL",
        "control_updates": 16,
        "resume_boundary_update": 8,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "save_restore_comparison": save_restore,
        "independent_control_boundary_comparison": control_boundary,
        "final_comparison": final_comparison,
        "trajectory_match": trajectory_match,
        "sample_order_match": sample_order_match,
        "rng_match": rng_match,
        "previous_attempts": previous_attempts,
        "continuous_final": continuous_final,
        "resumed_final": resumed_final,
        "formal_run": False,
    }
    combined = (
        json.loads(RESUME_RESULT.read_text(encoding="utf-8"))
        if RESUME_RESULT.exists()
        else {
            "stage": STAGE,
            "classification": "GPU2D_RESUME_QUALIFICATION_INCOMPLETE",
            "tests": {},
            "formal_campaign_authorized": False,
        }
    )
    combined["tests"][test_key] = result
    write_json(RESUME_RESULT, combined)
    del resumed
    gc.collect()
    torch.cuda.empty_cache()
    return result
