# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from minillm_forge.experiments_gpu4a.runtime import write_json


@dataclass(frozen=True)
class BenchmarkConfig:
    backend: str
    batch_size: int
    sequence_length: int
    dtype: str
    warmup_count: int
    measurement_count: int
    operation: str

    @property
    def sha256(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def summarize_measurements(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("at least one measurement is required")
    ordered = sorted(values)
    mean = statistics.mean(values)
    return {
        "median": statistics.median(values),
        "mean": mean,
        "p10": ordered[max(0, math_index(len(ordered), 0.1))],
        "p90": ordered[min(len(ordered) - 1, math_index(len(ordered), 0.9))],
        "std": statistics.pstdev(values),
        "cv": statistics.pstdev(values) / mean if mean else 0.0,
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def math_index(length: int, quantile: float) -> int:
    return round((length - 1) * quantile)


def prepare_cuda_measurement(
    synchronize: Callable[[], None] = torch.cuda.synchronize,
    reset_peak: Callable[[], None] = torch.cuda.reset_peak_memory_stats,
) -> None:
    synchronize()
    reset_peak()


def run_warmups(
    operation: Callable[[], Any],
    count: int,
    synchronize: Callable[[], None] = torch.cuda.synchronize,
) -> None:
    for _ in range(count):
        operation()
    synchronize()


def run_cuda_measurements(operation: Callable[[], Any], config: BenchmarkConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"status": "CUDA_UNAVAILABLE", "backend": config.backend}
    try:
        run_warmups(operation, config.warmup_count)
        prepare_cuda_measurement()
        values = []
        raw = []
        for index in range(config.measurement_count):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_started = time.perf_counter()
            start.record()
            operation()
            end.record()
            torch.cuda.synchronize()
            milliseconds = float(start.elapsed_time(end))
            values.append(milliseconds)
            raw.append(
                {
                    "measurement": index,
                    "cuda_milliseconds": milliseconds,
                    "wall_milliseconds": (time.perf_counter() - wall_started) * 1000,
                }
            )
        return {
            "status": "COMPLETED",
            "backend": config.backend,
            "config": asdict(config),
            "config_sha256": config.sha256,
            "raw_measurements": raw,
            "cuda_milliseconds": summarize_measurements(values),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
    except torch.OutOfMemoryError as error:
        return {"status": "OOM", "backend": config.backend, "error": str(error)}
    except TimeoutError as error:
        return {"status": "TIMEOUT", "backend": config.backend, "error": str(error)}


def write_gate_blocked_benchmarks(repo: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo).resolve()
    output = repo / "artifacts/gpu4b"
    correctness = json.loads((output / "attention_correctness.json").read_text(encoding="utf-8"))
    sequence_lengths = [256, 512, 768, 1024]
    records = []
    for sequence_length in sequence_lengths:
        for backend in ("manual", "sdpa_math", "sdpa_auto", "sdpa_flash"):
            if backend == "manual":
                reason = "no correctness-qualified optimized backend for a paired benchmark"
            elif backend == "sdpa_flash":
                reason = "FLASH_BACKEND_NOT_AVAILABLE"
            else:
                reason = "ATTENTION_CORRECTNESS_GATE_FAILED"
            config = BenchmarkConfig(
                backend=backend,
                batch_size=1,
                sequence_length=sequence_length,
                dtype="bfloat16",
                warmup_count=5,
                measurement_count=20,
                operation="attention_forward_backward",
            )
            records.append(
                {
                    "status": "NOT_RUN",
                    "reason": reason,
                    "backend": backend,
                    "sequence_length": sequence_length,
                    "config_sha256": config.sha256,
                }
            )
    result = {
        "stage": "GPU-4B-3",
        "status": "NOT_RUN_CORRECTNESS_GATE_FAILED",
        "attention_only": records,
        "full_model": [
            {
                "status": "NOT_RUN",
                "reason": "ATTENTION_CORRECTNESS_GATE_FAILED",
                "sequence_length": length,
            }
            for length in sequence_lengths
        ],
        "real_sft": {
            "status": "NOT_RUN",
            "reason": "NO_OPTIMIZED_ATTENTION_BACKEND_PASSED_B2",
            "gpu4a_manual_baseline_retained": True,
        },
        "backend_gates": correctness["backend_gates"],
    }
    write_json(output / "attention_benchmarks.json", result)
    (output / "ATTENTION_BENCHMARK.md").write_text(
        """# Attention performance benchmark

Performance execution was stopped by the pre-declared correctness gate. SDPA Math and Auto did not satisfy the frozen BF16 full-logit, selected-gradient, and one-step update tolerances; built-in Flash is unavailable. Therefore no optimized backend was eligible for the 256/512/768/1024 attention-only, full-model, or 300-step real-SFT benchmarks.

`attention_benchmarks.json` retains an explicit backend/sequence-length matrix with `NOT_RUN` reasons and configuration hashes. No microbenchmark, model throughput, memory, or speedup number is fabricated from an ineligible implementation. The unchanged GPU-4A Manual Eager measurement remains the only formal real-SFT baseline.
""",
        encoding="utf-8",
    )
    return result
