from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from minillm_forge.evaluation.perplexity import evaluate_perplexity
from minillm_forge.finetuning.full_sft import load_full_sft_model
from minillm_forge.training.checkpoint import load_checkpoint, save_checkpoint
from minillm_forge.training.optimizer import build_adamw
from minillm_forge.training.scheduler import build_cosine_scheduler
from minillm_forge.training.trainer import set_seed


class FixedTokenBlockDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, tokens: torch.Tensor, sequence_length: int) -> None:
        flat = tokens.detach().cpu().to(torch.long).flatten()
        if sequence_length <= 1 or flat.numel() < sequence_length:
            raise ValueError("token tensor is too small for sequence_length")
        usable = flat.numel() - flat.numel() % sequence_length
        self.tokens = flat[:usable].view(-1, sequence_length)

    def __len__(self) -> int:
        return self.tokens.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        values = self.tokens[index]
        return {
            "input_ids": values,
            "attention_mask": torch.ones_like(values),
            "labels": values.clone(),
        }


def load_token_tensor(path: str | Path) -> torch.Tensor:
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} does not contain a tensor")
    return value.to(torch.long)


def evaluation_loader(
    token_path: str | Path, sequence_length: int, *, batch_size: int = 1
) -> DataLoader:
    return DataLoader(
        FixedTokenBlockDataset(load_token_tensor(token_path), sequence_length),
        batch_size=batch_size,
        shuffle=False,
    )


def evaluate_domains(
    model: Any,
    *,
    math_tokens_path: str | Path,
    general_tokens_path: str | Path,
    sequence_length: int,
    device: str = "cuda",
) -> dict[str, float | int]:
    math_result = evaluate_perplexity(
        model,
        evaluation_loader(math_tokens_path, sequence_length),
        device=device,
    )
    general_result = evaluate_perplexity(
        model,
        evaluation_loader(general_tokens_path, sequence_length),
        device=device,
    )
    return {
        "math_loss": math_result["loss"],
        "math_ppl": math_result["perplexity"],
        "math_tokens": math_result["evaluated_tokens"],
        "general_loss": general_result["loss"],
        "general_ppl": general_result["perplexity"],
        "general_tokens": general_result["evaluated_tokens"],
    }


def cuda_memory() -> dict[str, float]:
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mib": torch.cuda.memory_reserved() / 1024**2,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "system_free_mib": free / 1024**2,
        "system_total_mib": total / 1024**2,
    }


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class QwenCPTState:
    global_step: int = 0
    micro_batches_seen: int = 0
    tokens_seen: int = 0
    data_cursor: int = 0
    nan_events: int = 0
    inf_events: int = 0
    oom_events: int = 0
    best_math_loss: float = math.inf
    best_step: int = 0


class QwenCPTTrainer:
    """Deterministic full-parameter Qwen CPT loop with dual-domain evaluation."""

    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        train_tokens_path: str | Path,
        math_tokens_path: str | Path,
        general_tokens_path: str | Path,
        sequence_length: int,
        gradient_accumulation: int,
        total_steps: int,
        learning_rate: float,
        betas: tuple[float, float],
        weight_decay: float,
        warmup_ratio: float,
        min_lr_ratio: float,
        grad_clip: float,
        output_dir: str | Path,
        run_config: dict[str, Any],
        seed: int = 42,
        empty_cache_after_step: bool = True,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen CPT requires CUDA and refuses CPU fallback")
        if gradient_accumulation <= 0 or total_steps <= 0:
            raise ValueError("gradient_accumulation and total_steps must be positive")
        set_seed(seed)
        self.sequence_length = sequence_length
        self.gradient_accumulation = gradient_accumulation
        self.total_steps = total_steps
        self.grad_clip = grad_clip
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.run_config = run_config
        self.empty_cache_after_step = empty_cache_after_step
        tokens = load_token_tensor(train_tokens_path)
        self.dataset = FixedTokenBlockDataset(tokens, sequence_length)
        required_blocks = total_steps * gradient_accumulation
        if len(self.dataset) < required_blocks:
            raise ValueError(
                f"training partition has {len(self.dataset)} blocks, needs {required_blocks}"
            )
        self.math_tokens_path = Path(math_tokens_path)
        self.general_tokens_path = Path(general_tokens_path)
        self.model = load_full_sft_model(
            model_name,
            revision=revision,
            precision="bf16",
            gradient_checkpointing=True,
            low_cpu_mem_usage=True,
        )
        self.model.to("cuda")
        self.optimizer = build_adamw(
            self.model,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            foreach=False,
        )
        self.scheduler = build_cosine_scheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio,
        )
        self.state = QwenCPTState()
        self.started = time.perf_counter()
        self.compute_seconds = 0.0
        self.step_rates: list[float] = []
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = cuda_memory()
        self.external_memory_mib = (
            initial_memory["system_total_mib"]
            - initial_memory["system_free_mib"]
            - initial_memory["reserved_mib"]
        )

    def _batch(self, index: int) -> dict[str, torch.Tensor]:
        raw = self.dataset[index]
        return {key: value.unsqueeze(0).to("cuda") for key, value in raw.items()}

    def evaluate(self) -> dict[str, float | int]:
        return evaluate_domains(
            self.model,
            math_tokens_path=self.math_tokens_path,
            general_tokens_path=self.general_tokens_path,
            sequence_length=self.sequence_length,
        )

    def train_to(
        self,
        target_step: int,
        *,
        eval_steps: set[int] | None = None,
        checkpoint_steps: dict[int, str] | None = None,
        log_every: int = 5,
        save_best: bool = False,
    ) -> None:
        if target_step > self.total_steps or target_step < self.state.global_step:
            raise ValueError("target_step is outside this run")
        eval_steps = eval_steps or set()
        checkpoint_steps = checkpoint_steps or {}
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        while self.state.global_step < target_step:
            step_started = time.perf_counter()
            loss_sum = 0.0
            try:
                for _ in range(self.gradient_accumulation):
                    batch = self._batch(self.state.data_cursor)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = self.model(**batch).loss
                        scaled_loss = loss / self.gradient_accumulation
                    if bool(torch.isnan(loss)):
                        self.state.nan_events += 1
                        raise FloatingPointError("NaN training loss")
                    if bool(torch.isinf(loss)):
                        self.state.inf_events += 1
                        raise FloatingPointError("Inf training loss")
                    scaled_loss.backward()
                    loss_sum += float(loss.detach())
                    self.state.micro_batches_seen += 1
                    self.state.data_cursor += 1
                    self.state.tokens_seen += self.sequence_length
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if not bool(torch.isfinite(grad_norm)):
                    if bool(torch.isnan(grad_norm)):
                        self.state.nan_events += 1
                    else:
                        self.state.inf_events += 1
                    raise FloatingPointError("non-finite gradient norm")
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            except torch.OutOfMemoryError:
                self.state.oom_events += 1
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                raise
            if self.empty_cache_after_step:
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            step_seconds = max(time.perf_counter() - step_started, 1e-9)
            self.compute_seconds += step_seconds
            self.state.global_step += 1
            rate = self.sequence_length * self.gradient_accumulation / step_seconds
            self.step_rates.append(rate)
            memory = cuda_memory()
            estimated_headroom = (
                memory["system_total_mib"] - self.external_memory_mib - memory["peak_reserved_mib"]
            )
            if self.state.global_step % log_every == 0 or self.state.global_step == 1:
                append_jsonl(
                    self.metrics_path,
                    {
                        "event": "train",
                        "global_step": self.state.global_step,
                        "tokens_seen": self.state.tokens_seen,
                        "train_loss": loss_sum / self.gradient_accumulation,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "gradient_norm": float(grad_norm),
                        "step_time": step_seconds,
                        "tokens_per_second": rate,
                        **memory,
                        "estimated_min_system_headroom_mib": estimated_headroom,
                    },
                )
            if self.state.global_step in eval_steps:
                evaluation = self.evaluate()
                evaluation_record = {
                    "event": "evaluation",
                    "global_step": self.state.global_step,
                    "tokens_seen": self.state.tokens_seen,
                    **evaluation,
                }
                append_jsonl(self.metrics_path, evaluation_record)
                math_loss = float(evaluation["math_loss"])
                if math_loss < self.state.best_math_loss:
                    self.state.best_math_loss = math_loss
                    self.state.best_step = self.state.global_step
                    if save_best:
                        self.save("best_math_validation.pt", evaluation_record)
                self.model.train()
            if self.state.global_step in checkpoint_steps:
                self.save(checkpoint_steps[self.state.global_step])

    def save(self, name: str, metrics: dict[str, Any] | None = None) -> Path:
        return save_checkpoint(
            self.output_dir / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=None,
            trainer_state=asdict(self.state),
            config=self.run_config,
            metrics=metrics,
        )

    def resume(self, path: str | Path) -> dict[str, Any]:
        payload = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=None,
            map_location="cpu",
        )
        self.state = QwenCPTState(**payload["trainer_state"])
        return payload

    def summary(self) -> dict[str, Any]:
        memory = cuda_memory()
        return {
            "state": asdict(self.state),
            "peak_allocated_vram_mib": memory["peak_allocated_mib"],
            "peak_reserved_vram_mib": memory["peak_reserved_mib"],
            "estimated_min_system_headroom_mib": (
                memory["system_total_mib"] - self.external_memory_mib - memory["peak_reserved_mib"]
            ),
            "median_tokens_per_second": median(self.step_rates) if self.step_rates else 0.0,
            "elapsed_seconds": time.perf_counter() - self.started,
            "compute_seconds": self.compute_seconds,
        }


def release_trainer(trainer: QwenCPTTrainer) -> None:
    del trainer.model
    del trainer.optimizer
    del trainer.scheduler
    gc.collect()
    torch.cuda.empty_cache()
