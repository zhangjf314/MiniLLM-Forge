from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from minillm_forge.training.amp import AMPController
from minillm_forge.training.checkpoint import load_checkpoint, save_checkpoint
from minillm_forge.training.metrics import (
    JsonlMetricLogger,
    global_parameter_norm,
    gradients_are_finite,
    safe_perplexity,
    tensor_statistics,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    max_steps: int
    gradient_accumulation_steps: int = 8
    grad_clip: float = 1.0
    precision: str = "bf16"
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 100
    output_dir: str = "runs/default"
    seed: int = 42
    device: str = "auto"
    fail_on_non_finite: bool = True
    tensorboard: bool = True

    def __post_init__(self) -> None:
        for name in ("max_steps", "gradient_accumulation_steps", "log_every"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.eval_every < 0 or self.save_every < 0:
            raise ValueError("eval_every and save_every must be non-negative")


@dataclass
class TrainingState:
    global_step: int = 0
    epoch: int = 0
    batches_seen: int = 0
    examples_seen: int = 0
    tokens_seen: int = 0
    target_tokens_seen: int = 0
    best_validation_loss: float = float("inf")
    nan_count: int = 0
    inf_count: int = 0
    oom_count: int = 0
    minimum_headroom_mb: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    sampler_state: dict[str, Any] | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was explicitly requested but is unavailable; refusing CPU fallback"
            )
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ForgeTrainer:
    """Small, explicit PyTorch trainer used by both MiniLLM and HF causal LMs."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        config: TrainingConfig,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        validation_loader: DataLoader | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.config = config
        self.run_config = run_config or {}
        self.state = TrainingState()
        self.device = resolve_device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = JsonlMetricLogger(self.output_dir / "metrics.jsonl")
        if config.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.output_dir / "tensorboard")
        else:
            self.writer = None
        self.amp = AMPController(config.precision, self.device.type)
        set_seed(config.seed)
        if not getattr(model, "hf_device_map", None):
            self.model.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _batch_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.device

    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        device = self._batch_device()
        return {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    @staticmethod
    def _extract_loss(output: Any) -> torch.Tensor:
        loss = output.get("loss") if isinstance(output, dict) else getattr(output, "loss", None)
        if loss is None:
            raise ValueError("model output must provide a loss when labels are supplied")
        return loss

    @staticmethod
    def _extract_logits(output: Any) -> torch.Tensor | None:
        logits = (
            output.get("logits") if isinstance(output, dict) else getattr(output, "logits", None)
        )
        return logits if isinstance(logits, torch.Tensor) else None

    def _log(self, record: dict[str, Any]) -> None:
        self.metrics.log(record)
        if self.writer is None:
            return
        step = int(record.get("global_step", self.state.global_step))
        event = str(record.get("event", "metrics"))
        for key, value in record.items():
            if key in {"event", "global_step"} or not isinstance(value, (int, float)):
                continue
            if math.isfinite(float(value)):
                self.writer.add_scalar(f"{event}/{key}", value, step)

    def peak_vram_mb(self) -> float | None:
        if self.device.type != "cuda":
            return None
        return torch.cuda.max_memory_allocated(self.device) / 1024**2

    def peak_reserved_vram_mb(self) -> float | None:
        if self.device.type != "cuda":
            return None
        return torch.cuda.max_memory_reserved(self.device) / 1024**2

    def _save(self, name: str, extra_metrics: dict[str, Any] | None = None) -> Path:
        sampler = self.train_loader.sampler
        if hasattr(sampler, "state_dict"):
            self.state.sampler_state = sampler.state_dict()
        return save_checkpoint(
            self.output_dir / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.amp.scaler,
            trainer_state=asdict(self.state),
            config={"training": asdict(self.config), **self.run_config},
            metrics=extra_metrics,
        )

    def resume(self, path: str | Path) -> None:
        checkpoint_state = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.amp.scaler,
            map_location=self._batch_device(),
        )
        self.state = TrainingState(**checkpoint_state["trainer_state"])
        sampler = self.train_loader.sampler
        if self.state.sampler_state is not None and hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(self.state.sampler_state)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        if self.validation_loader is None:
            return {}
        was_training = self.model.training
        self.model.eval()
        loss_sum, batches = 0.0, 0
        for raw_batch in self.validation_loader:
            batch = self._prepare_batch(raw_batch)
            with self.amp.autocast():
                loss = self._extract_loss(self.model(**batch))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite validation loss")
            loss_sum += float(loss)
            batches += 1
        self.model.train(was_training)
        mean_loss = loss_sum / max(batches, 1)
        result = {"validation_loss": mean_loss, "validation_perplexity": safe_perplexity(mean_loss)}
        self.state.best_validation_loss = min(self.state.best_validation_loss, mean_loss)
        return result

    def train_until(self, target_step: int, *, finalize: bool = False) -> TrainingState:
        """Train to an explicit optimizer step, preserving resumable iterator state.

        ``target_step`` is bounded by the configured schedule length.  Qualification
        controls use this method to stop exactly on an accumulation boundary, seal a
        checkpoint, and continue without changing the scheduler's total-step contract.
        """
        if not self.state.global_step <= target_step <= self.config.max_steps:
            raise ValueError("target_step must be between global_step and configured max_steps")
        if len(self.train_loader) == 0:
            raise ValueError("train_loader must contain at least one batch")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        window_loss = 0.0
        window_tokens = 0
        window_target_tokens = 0
        window_batches = 0
        window_started = time.perf_counter()
        activation_stats: dict[str, float] = {}

        while self.state.global_step < target_step:
            for raw_batch in self.train_loader:
                if self.state.global_step >= target_step:
                    break
                batch = self._prepare_batch(raw_batch)
                try:
                    with self.amp.autocast():
                        output = self.model(**batch)
                        loss = self._extract_loss(output)
                        scaled_loss = loss / self.config.gradient_accumulation_steps
                    will_step = (
                        self.state.batches_seen + 1
                    ) % self.config.gradient_accumulation_steps == 0
                    will_log = (
                        will_step and (self.state.global_step + 1) % self.config.log_every == 0
                    )
                    logits = self._extract_logits(output)
                    if will_log and logits is not None:
                        activation_stats = tensor_statistics(logits)
                    if not bool(torch.isfinite(loss)):
                        if bool(torch.isnan(loss)):
                            self.state.nan_count += 1
                        else:
                            self.state.inf_count += 1
                        raise FloatingPointError(
                            f"non-finite training loss at step {self.state.global_step}"
                        )
                    self.amp.backward(scaled_loss)
                except torch.OutOfMemoryError:
                    self.state.oom_count += 1
                    self._log(
                        {
                            "event": "oom",
                            "global_step": self.state.global_step,
                            "oom_count": self.state.oom_count,
                        }
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise

                self.state.batches_seen += 1
                self.state.examples_seen += int(batch["input_ids"].shape[0])
                tokens = int(batch.get("attention_mask", batch["input_ids"].ne(-1)).sum())
                target_tokens = int(batch.get("labels", batch["input_ids"]).ne(-100).sum())
                self.state.tokens_seen += tokens
                self.state.target_tokens_seen += target_tokens
                if self.device.type == "cuda":
                    free_bytes, _ = torch.cuda.mem_get_info(self.device)
                    free_mb = free_bytes / 1024**2
                    self.state.minimum_headroom_mb = (
                        free_mb
                        if self.state.minimum_headroom_mb is None
                        else min(self.state.minimum_headroom_mb, free_mb)
                    )
                window_tokens += tokens
                window_target_tokens += target_tokens
                window_loss += float(loss.detach())
                window_batches += 1
                accumulation_boundary = (
                    self.state.batches_seen % self.config.gradient_accumulation_steps == 0
                )
                if not accumulation_boundary:
                    continue

                self.amp.unscale_(self.optimizer)
                if not gradients_are_finite(self.model):
                    self.state.nan_count += 1
                    if self.config.fail_on_non_finite:
                        raise FloatingPointError("model contains a non-finite gradient")
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                self.amp.step(self.optimizer)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.state.global_step += 1

                if self.state.global_step % self.config.log_every == 0:
                    elapsed = max(time.perf_counter() - window_started, 1e-9)
                    record = {
                        "event": "train",
                        "global_step": self.state.global_step,
                        "epoch": self.state.epoch,
                        "loss": window_loss / max(window_batches, 1),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm),
                        "parameter_norm": global_parameter_norm(self.model),
                        "tokens_per_second": window_tokens / elapsed,
                        "target_tokens_per_second": window_target_tokens / elapsed,
                        "tokens_seen": self.state.tokens_seen,
                        "target_tokens_seen": self.state.target_tokens_seen,
                        "nan_count": self.state.nan_count,
                        "inf_count": self.state.inf_count,
                        "oom_count": self.state.oom_count,
                        "peak_vram_mb": self.peak_vram_mb(),
                        **activation_stats,
                    }
                    self.state.history.append(record)
                    self._log(record)
                    window_loss, window_tokens, window_target_tokens, window_batches = 0.0, 0, 0, 0
                    activation_stats = {}
                    window_started = time.perf_counter()

                if self.config.eval_every and self.state.global_step % self.config.eval_every == 0:
                    evaluation = {
                        "event": "evaluation",
                        "global_step": self.state.global_step,
                        **self.evaluate(),
                    }
                    self.state.history.append(evaluation)
                    self._log(evaluation)
                    self.model.train()
                if self.config.save_every and self.state.global_step % self.config.save_every == 0:
                    self._save(f"step-{self.state.global_step:08d}.pt")
                if self.state.global_step >= target_step:
                    break
            self.state.epoch += 1

        if finalize:
            final_evaluation = self.evaluate()
            self._save("last.pt", final_evaluation)
            if self.writer is not None:
                self.writer.close()
        return self.state

    def train(self) -> TrainingState:
        return self.train_until(self.config.max_steps, finalize=True)
