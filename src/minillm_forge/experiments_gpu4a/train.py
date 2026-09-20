from __future__ import annotations

import gc
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from minillm_forge.cli.common import StatefulRandomSampler
from minillm_forge.experiments_gpu4a.data import TASKS, SFTDataset, collate_rows
from minillm_forge.experiments_gpu4a.runtime import (
    CHECKPOINT,
    file_sha256,
    generate_record,
    load_native_model,
    release,
    teacher_forced_metrics,
    write_json,
)
from minillm_forge.model import MiniLLM
from minillm_forge.training import (
    ForgeTrainer,
    TrainingConfig,
    build_adamw,
    build_cosine_scheduler,
)

SEED = 42


def _state_digest(model: MiniLLM) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class NativeSFTTrainer(ForgeTrainer):
    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return super()._prepare_batch(
            {key: value for key, value in batch.items() if key != "sample_ids"}
        )

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        if self.validation_loader is None:
            return {}
        result = teacher_forced_metrics(self.model, self.validation_loader, str(self.device))
        self.state.best_validation_loss = min(self.state.best_validation_loss, result["loss"])
        return {
            "validation_loss": result["loss"],
            "validation_perplexity": result["perplexity"],
            "validation_target_token_accuracy": result["target_token_accuracy"],
            "validation_targets": result["targets"],
        }


def _loader(
    dataset: SFTDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=StatefulRandomSampler(dataset, seed) if shuffle else None,
        shuffle=False,
        collate_fn=collate_rows,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )


def build_trainer(
    repo: str | Path,
    *,
    run_dir: str | Path,
    steps: int,
    learning_rate: float,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    train_dataset: SFTDataset | None = None,
    validation_dataset: SFTDataset | None = None,
    eval_every: int = 0,
    save_every: int = 0,
) -> tuple[NativeSFTTrainer, Any, dict[str, Any]]:
    repo = Path(repo).resolve()
    model, tokenizer, identity = load_native_model(repo, CHECKPOINT)
    model.gradient_checkpointing = True
    train_dataset = train_dataset or SFTDataset(repo / "data/processed/gpu4a/train.jsonl")
    validation_dataset = validation_dataset or SFTDataset(
        repo / "data/processed/gpu4a/validation.jsonl"
    )
    train_loader = _loader(train_dataset, batch_size=micro_batch_size, shuffle=True, seed=SEED)
    validation_loader = _loader(
        validation_dataset, batch_size=micro_batch_size, shuffle=False, seed=SEED
    )
    optimizer = build_adamw(
        model,
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    scheduler = build_cosine_scheduler(
        optimizer, total_steps=steps, warmup_ratio=0.05, min_lr_ratio=0.1
    )
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = repo / run_dir
    trainer = NativeSFTTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=TrainingConfig(
            max_steps=steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grad_clip=1.0,
            precision="bf16",
            log_every=1,
            eval_every=eval_every,
            save_every=save_every,
            output_dir=str(run_dir),
            seed=SEED,
            device="cuda",
            tensorboard=False,
        ),
        run_config={
            "stage": "GPU-4A",
            "model": identity["model_config"],
            "initial_checkpoint": identity,
            "optimizer": {
                "name": "AdamW",
                "lr": learning_rate,
                "betas": [0.9, 0.95],
                "weight_decay": 0.01,
            },
            "scheduler": {"name": "cosine", "warmup_ratio": 0.05, "min_lr_ratio": 0.1},
        },
    )
    optimizer_parameters = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    trainable_parameters = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if optimizer_parameters != trainable_parameters:
        raise RuntimeError("optimizer does not cover exactly all trainable parameters")
    return trainer, tokenizer, identity


def run_tiny_overfit(repo: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo).resolve()
    source = SFTDataset(repo / "data/processed/gpu4a/train.jsonl")
    chosen = []
    for task in TASKS:
        task_ids = [row["sample_id"] for row in source.rows if row["task"] == task]
        chosen.extend(task_ids[:4])
    tiny = SFTDataset(repo / "data/processed/gpu4a/train.jsonl", sample_ids=set(chosen))
    run_dir = repo / "runs/gpu4a/tiny-overfit"
    trainer, tokenizer, identity = build_trainer(
        repo,
        run_dir=run_dir,
        steps=61,
        learning_rate=1e-3,
        micro_batch_size=8,
        gradient_accumulation_steps=1,
        train_dataset=tiny,
        validation_dataset=tiny,
    )
    initial_digest = _state_digest(trainer.model)
    initial_metrics = teacher_forced_metrics(trainer.model, trainer.validation_loader)
    initial_generation = [generate_record(trainer.model, tokenizer, row) for row in tiny.rows]
    started = time.perf_counter()
    trainer.train_until(60)
    checkpoint = trainer._save("checkpoints/step-0060.pt")
    trained_metrics = teacher_forced_metrics(trainer.model, trainer.validation_loader)
    trained_generation = [generate_record(trainer.model, tokenizer, row) for row in tiny.rows]
    final_digest = _state_digest(trainer.model)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    resumed, resumed_tokenizer, _ = build_trainer(
        repo,
        run_dir=repo / "runs/gpu4a/tiny-overfit-resumed",
        steps=61,
        learning_rate=1e-3,
        micro_batch_size=8,
        gradient_accumulation_steps=1,
        train_dataset=tiny,
        validation_dataset=tiny,
    )
    resumed.resume(checkpoint)
    restored_digest = _state_digest(resumed.model)
    restored_generation = [
        generate_record(resumed.model, resumed_tokenizer, row) for row in tiny.rows
    ]
    restored_step = resumed.state.global_step
    restored_sampler_position = resumed.train_loader.sampler.position
    resumed.train_until(61)
    result = {
        "status": "PASS",
        "initial_checkpoint": identity,
        "sample_ids": chosen,
        "steps_before_resume": 60,
        "steps_after_resume": resumed.state.global_step,
        "initial_metrics": initial_metrics,
        "trained_metrics": trained_metrics,
        "initial_exact_generation": sum(row["task_score"] for row in initial_generation),
        "trained_exact_generation": sum(row["task_score"] for row in trained_generation),
        "restored_generation_matches": [row["generated_token_ids"] for row in restored_generation]
        == [row["generated_token_ids"] for row in trained_generation],
        "initial_model_sha256": initial_digest,
        "trained_model_sha256": final_digest,
        "restored_model_sha256": restored_digest,
        "parameters_updated": initial_digest != final_digest,
        "checkpoint_model_exact": final_digest == restored_digest,
        "restored_step": restored_step,
        "restored_sampler_position": restored_sampler_position,
        "finite_loss": all(
            math.isfinite(value) for value in (initial_metrics["loss"], trained_metrics["loss"])
        ),
        "wall_time_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_mib": resumed.peak_vram_mb(),
        "peak_cuda_reserved_mib": resumed.peak_reserved_vram_mb(),
    }
    if not (
        result["parameters_updated"]
        and result["checkpoint_model_exact"]
        and result["restored_generation_matches"]
        and trained_metrics["loss"] < initial_metrics["loss"]
        and trained_metrics["target_token_accuracy"] > initial_metrics["target_token_accuracy"]
    ):
        result["status"] = "FAIL"
    write_json(repo / "artifacts/gpu4a/tiny_overfit_result.json", result)
    release(resumed.model)
    return result


def run_prewarm(repo: str | Path = ".", steps: int = 8) -> dict[str, Any]:
    repo = Path(repo).resolve()
    run_dir = repo / "runs/gpu4a/prewarm"
    trainer, _, identity = build_trainer(
        repo,
        run_dir=run_dir,
        steps=steps,
        learning_rate=1e-4,
        micro_batch_size=16,
        gradient_accumulation_steps=2,
    )
    initial_validation = trainer.evaluate()
    started = time.perf_counter()
    trainer.train_until(steps)
    training_seconds = time.perf_counter() - started
    checkpoint_started = time.perf_counter()
    checkpoint = trainer._save(f"checkpoints/step-{steps:04d}.pt")
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    final_validation = trainer.evaluate()
    history = [row for row in trainer.state.history if row.get("event") == "train"]
    result = {
        "status": "PASS",
        "steps": trainer.state.global_step,
        "initial_checkpoint": identity,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "training_seconds": training_seconds,
        "median_step_seconds": training_seconds / steps,
        "checkpoint_seconds": checkpoint_seconds,
        "tokens_seen": trainer.state.tokens_seen,
        "target_tokens_seen": trainer.state.target_tokens_seen,
        "median_tokens_per_second": statistics.median(row["tokens_per_second"] for row in history),
        "median_target_tokens_per_second": statistics.median(
            row["target_tokens_per_second"] for row in history
        ),
        "peak_cuda_allocated_mib": trainer.peak_vram_mb(),
        "peak_cuda_reserved_mib": trainer.peak_reserved_vram_mb(),
        "minimum_headroom_mib": trainer.state.minimum_headroom_mb,
        "nan_count": trainer.state.nan_count,
        "inf_count": trainer.state.inf_count,
        "oom_count": trainer.state.oom_count,
        "checkpoint": str(checkpoint),
    }
    write_json(repo / "artifacts/gpu4a/prewarm_result.json", result)
    release(trainer.model)
    return result


def run_formal_sft(repo: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo).resolve()
    steps = 300
    checkpoint_steps = list(range(50, steps + 1, 50))
    prewarm = json.loads((repo / "artifacts/gpu4a/prewarm_result.json").read_text(encoding="utf-8"))
    estimated = (
        steps * prewarm["median_step_seconds"]
        + len(checkpoint_steps) * prewarm["checkpoint_seconds"]
        + 25
    )
    protocol_path = repo / "artifacts/gpu4a/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["formal_training_budget"] = {
        "experiment_id": "gpu4a-native-full-sft-s42-v1",
        "method": "FULL_SFT",
        "seed": SEED,
        "optimizer_steps": steps,
        "micro_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 32,
        "learning_rate": 0.0001,
        "optimizer": "AdamW",
        "betas": [0.9, 0.95],
        "weight_decay": 0.01,
        "precision": "bf16",
        "checkpoint_steps": checkpoint_steps,
        "validation_steps": checkpoint_steps,
        "checkpoint_selection": "lowest validation response-only loss; earliest step breaks ties",
        "estimated_duration_seconds": estimated,
        "estimation_basis": prewarm,
    }
    protocol["status"] = "FROZEN_BEFORE_FORMAL_FULL_SFT"
    write_json(protocol_path, protocol)
    run_dir = repo / "runs/gpu4a/formal-full-sft-s42-v1"
    trainer, _, identity = build_trainer(
        repo,
        run_dir=run_dir,
        steps=steps,
        learning_rate=1e-4,
        micro_batch_size=16,
        gradient_accumulation_steps=2,
        eval_every=50,
    )
    initial_digest = _state_digest(trainer.model)
    initial_validation = trainer.evaluate()
    started = time.perf_counter()
    checkpoints = []
    for step in checkpoint_steps:
        trainer.train_until(step)
        evaluation = next(
            row
            for row in reversed(trainer.state.history)
            if row.get("event") == "evaluation" and row["global_step"] == step
        )
        path = trainer._save(f"checkpoints/step-{step:04d}.pt", evaluation)
        checkpoints.append(
            {
                "step": step,
                "path": str(path),
                "sha256": file_sha256(path),
                "validation_loss": evaluation["validation_loss"],
                "validation_perplexity": evaluation["validation_perplexity"],
                "validation_target_token_accuracy": evaluation["validation_target_token_accuracy"],
            }
        )
    training_wall = time.perf_counter() - started
    final_digest = _state_digest(trainer.model)
    history = [row for row in trainer.state.history if row.get("event") == "train"]
    best = min(checkpoints, key=lambda item: (item["validation_loss"], item["step"]))
    state = trainer.state
    result = {
        "stage": "GPU-4A-3",
        "status": "COMPLETED",
        "experiment_id": "gpu4a-native-full-sft-s42-v1",
        "method": "FULL_SFT",
        "initial_checkpoint": identity,
        "trainable_parameters": trainer.model.num_parameters(trainable_only=True),
        "optimizer_parameter_count": sum(
            parameter.numel()
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        ),
        "steps": state.global_step,
        "examples_seen": state.examples_seen,
        "input_tokens_seen": state.tokens_seen,
        "supervised_tokens_seen": state.target_tokens_seen,
        "initial_validation": initial_validation,
        "checkpoints": checkpoints,
        "selected_checkpoint": best,
        "final_train_loss": history[-1]["loss"],
        "median_train_loss": statistics.median(row["loss"] for row in history),
        "median_tokens_per_second": statistics.median(row["tokens_per_second"] for row in history),
        "median_target_tokens_per_second": statistics.median(
            row["target_tokens_per_second"] for row in history
        ),
        "peak_cuda_allocated_mib": trainer.peak_vram_mb(),
        "peak_cuda_reserved_mib": trainer.peak_reserved_vram_mb(),
        "minimum_headroom_mib": state.minimum_headroom_mb,
        "nan_count": state.nan_count,
        "inf_count": state.inf_count,
        "oom_count": state.oom_count,
        "initial_model_sha256": initial_digest,
        "final_model_sha256": final_digest,
        "parameters_updated": initial_digest != final_digest,
        "wall_time_seconds": training_wall,
    }
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    selected_dataset = SFTDataset(repo / "data/processed/gpu4a/train.jsonl")
    validation_dataset = SFTDataset(repo / "data/processed/gpu4a/validation.jsonl")
    restored, _, _ = build_trainer(
        repo,
        run_dir=repo / "runs/gpu4a/formal-full-sft-s42-v1-reload",
        steps=steps,
        learning_rate=1e-4,
        micro_batch_size=16,
        gradient_accumulation_steps=2,
        train_dataset=selected_dataset,
        validation_dataset=validation_dataset,
    )
    restored.resume(best["path"])
    reload_validation = restored.evaluate()
    result["selected_checkpoint_reload"] = {
        "status": "PASS",
        "restored_step": restored.state.global_step,
        "validation": reload_validation,
        "matches_recorded_validation": abs(
            reload_validation["validation_loss"] - best["validation_loss"]
        )
        < 1e-8,
    }
    if not result["selected_checkpoint_reload"]["matches_recorded_validation"]:
        result["status"] = "FAILED"
    write_json(repo / "artifacts/gpu4a/formal_sft_result.json", result)
    release(restored.model)
    return result
