"""Evidence-oriented MiniLLM training on frozen, full-length causal-LM blocks.

Reuses the project trainer's AMP, optimizer, logging and checkpoint mechanics. Unlike
the general SFT path, batches here have equal supervised lengths and no padding.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from minillm_forge.training.checkpoint import load_checkpoint
from minillm_forge.training.metrics import global_parameter_norm
from minillm_forge.training.trainer import ForgeTrainer, TrainingState


@dataclass
class FormalState(TrainingState):
    inf_count: int = 0
    supervised_tokens_seen: int = 0
    elapsed_seconds: float = 0.0
    compute_seconds: float = 0.0
    best_step: int = 0
    peak_allocated_mib: float = 0.0
    peak_reserved_mib: float = 0.0


class FormalTrainer(ForgeTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.train_loader.num_workers != 0:
            raise ValueError("exact formal cursor recovery requires num_workers=0")
        if self.train_loader.drop_last is not True:
            raise ValueError("fixed full micro-batches are required for formal accumulation")
        self.state = FormalState()
        self._clock = time.perf_counter()

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _tick(self):
        now = time.perf_counter()
        self.state.elapsed_seconds += now - self._clock
        self._clock = now

    def _iterator(self):
        # DataLoader iterator creation must not perturb model dropout RNG after resume.
        rng = torch.get_rng_state()
        iterator = iter(self.train_loader)
        torch.set_rng_state(rng)
        return iterator

    def resume(self, path):
        state = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.amp.scaler,
            map_location="cpu",
        )
        old = state["config"]
        for key in ("model", "data_identity", "optimizer", "scheduler"):
            if old.get(key) != self.run_config.get(key):
                raise ValueError(f"resume config mismatch: {key}")
        for key in (
            "max_steps",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "seed",
            "precision",
        ):
            if old["training"].get(key) != self.run_config["training"].get(key):
                raise ValueError(f"resume training config mismatch: {key}")
        self.state = FormalState(**state["trainer_state"])
        self.train_loader.sampler.load_state_dict(self.state.sampler_state)
        self._clock = time.perf_counter()
        self._log(
            {
                "event": "resume",
                "global_step": self.state.global_step,
                "tokens_seen": self.state.tokens_seen,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "scheduler_last_epoch": self.scheduler.last_epoch,
                "cuda_rng_restored": "cuda" in state["rng_state"],
            }
        )

    @torch.no_grad()
    def evaluate_loader(self, loader):
        was_training = self.model.training
        self.model.eval()
        total, count = 0.0, 0
        rng = torch.get_rng_state()
        try:
            for raw in loader:
                batch = self._prepare_batch(raw)
                n = int(batch["labels"][:, 1:].ne(-100).sum())
                with self.amp.autocast():
                    loss = self._extract_loss(self.model(**batch))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite held-out loss")
                total += float(loss) * n
                count += n
        finally:
            self.model.train(was_training)
            torch.set_rng_state(rng)
        if not count:
            raise ValueError("evaluation requires supervised tokens")
        loss = total / count
        return {
            "validation_loss": loss,
            "validation_perplexity": math.exp(loss),
            "evaluated_targets": count,
        }

    def evaluate(self):
        return self.evaluate_loader(self.validation_loader)

    def evaluate_and_save(self):
        result = self.evaluate()
        self._tick()
        record = {
            "event": "evaluation",
            "global_step": self.state.global_step,
            "tokens_seen": self.state.tokens_seen,
            "elapsed_seconds": self.state.elapsed_seconds,
            **result,
        }
        self.state.history.append(record)
        self._log(record)
        if result["validation_loss"] < self.state.best_validation_loss:
            self.state.best_validation_loss = result["validation_loss"]
            self.state.best_step = self.state.global_step
            self._save("best.pt", result)
        print(
            f"eval step={self.state.global_step} tokens={self.state.tokens_seen} "
            f"loss={result['validation_loss']:.5f} ppl={result['validation_perplexity']:.3f}",
            flush=True,
        )
        return result

    def train(self, stop_after_step=None):
        stop = min(stop_after_step or self.config.max_steps, self.config.max_steps)
        if stop <= self.state.global_step:
            raise ValueError("stop step must exceed restored step")
        iterator = self._iterator()
        interval_loss, interval_tokens, interval_time, interval_steps = 0.0, 0, 0.0, 0
        save_steps = self.run_config.get("checkpoint_steps", [])
        self.model.train()
        try:
            while self.state.global_step < stop:
                self._sync()
                started = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                update_loss, update_tokens, update_targets = 0.0, 0, 0
                used_lr = self.optimizer.param_groups[0]["lr"]
                for _ in range(self.config.gradient_accumulation_steps):
                    try:
                        raw = next(iterator)
                    except StopIteration:
                        self.state.epoch += 1
                        iterator = self._iterator()
                        raw = next(iterator)
                    batch = self._prepare_batch(raw)
                    if not bool(batch["attention_mask"].all()):
                        raise ValueError("formal blocks may not contain padding")
                    with self.amp.autocast():
                        output = self.model(**batch)
                        loss = self._extract_loss(output)
                    if not bool(torch.isfinite(loss)):
                        self.state.nan_count += int(bool(torch.isnan(loss)))
                        self.state.inf_count += int(bool(torch.isinf(loss)))
                        raise FloatingPointError("non-finite training loss")
                    update_loss += float(loss.detach()) / self.config.gradient_accumulation_steps
                    self.amp.backward(loss / self.config.gradient_accumulation_steps)
                    update_tokens += batch["input_ids"].numel()
                    update_targets += batch["input_ids"].shape[0] * (
                        batch["input_ids"].shape[1] - 1
                    )
                    self.state.batches_seen += 1
                    del output, loss, batch
                self.amp.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip, error_if_nonfinite=True
                )
                self.amp.step(self.optimizer)
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._sync()
                duration = time.perf_counter() - started
                self.state.global_step += 1
                self.state.tokens_seen += update_tokens
                self.state.supervised_tokens_seen += update_targets
                self.state.compute_seconds += duration
                self.state.peak_allocated_mib = max(
                    self.state.peak_allocated_mib, self.peak_vram_mb() or 0
                )
                self.state.peak_reserved_mib = max(
                    self.state.peak_reserved_mib, self.peak_reserved_vram_mb() or 0
                )
                interval_loss += update_loss
                interval_tokens += update_tokens
                interval_time += duration
                interval_steps += 1
                self._tick()
                if (
                    self.state.global_step % self.config.log_every == 0
                    or self.state.global_step == stop
                ):
                    record = {
                        "event": "train",
                        "global_step": self.state.global_step,
                        "tokens_seen": self.state.tokens_seen,
                        "epoch": self.state.epoch,
                        "train_loss": interval_loss / interval_steps,
                        "loss": interval_loss / interval_steps,
                        "learning_rate": used_lr,
                        "next_learning_rate": self.optimizer.param_groups[0]["lr"],
                        "gradient_norm": float(norm),
                        "grad_norm": float(norm),
                        "parameter_norm": global_parameter_norm(self.model),
                        "step_time": interval_time / interval_steps,
                        "tokens_per_second": interval_tokens / interval_time,
                        "allocated_cuda_memory": torch.cuda.memory_allocated() / 1024**2
                        if self.device.type == "cuda"
                        else 0,
                        "reserved_cuda_memory": torch.cuda.memory_reserved() / 1024**2
                        if self.device.type == "cuda"
                        else 0,
                        "peak_allocated_mib": self.state.peak_allocated_mib,
                        "peak_reserved_mib": self.state.peak_reserved_mib,
                        "elapsed_seconds": self.state.elapsed_seconds,
                        "nan_count": self.state.nan_count,
                        "inf_count": self.state.inf_count,
                        "oom_count": self.state.oom_count,
                    }
                    self.state.history.append(record)
                    self._log(record)
                    print(
                        f"step={record['global_step']} tokens={record['tokens_seen']} "
                        f"loss={record['loss']:.4f} tok/s={record['tokens_per_second']:.0f}",
                        flush=True,
                    )
                    interval_loss, interval_tokens, interval_time, interval_steps = 0.0, 0, 0.0, 0
                if self.config.eval_every and self.state.global_step % self.config.eval_every == 0:
                    self.evaluate_and_save()
                if self.state.global_step in save_steps:
                    self._save(f"step-{self.state.global_step:08d}.pt")
            result = self.evaluate_and_save() if self.validation_loader is not None else {}
            self._tick()
            self._save("last.pt", result)
        except Exception as error:
            self.state.oom_count += int(isinstance(error, torch.OutOfMemoryError))
            self._log(
                {
                    "event": "failure",
                    "global_step": self.state.global_step,
                    "tokens_seen": self.state.tokens_seen,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            raise
        finally:
            if self.writer is not None:
                self.writer.close()
        return self.state
