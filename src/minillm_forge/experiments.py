from __future__ import annotations

import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from minillm_forge.config import config_hash

REGISTRY_FIELDS = [
    "experiment_id",
    "created_at",
    "git_commit",
    "config_hash",
    "dataset_hash",
    "model",
    "dataset",
    "seed",
    "learning_rate",
    "batch_size",
    "effective_batch_size",
    "rank",
    "target_modules",
    "precision",
    "trainable_params",
    "peak_vram_mb",
    "tokens_per_second",
    "final_loss",
    "eval_score",
    "status",
    "environment",
    "device",
    "gpu_name",
    "torch_version",
    "torch_cuda_version",
    "peak_cuda_reserved_mb",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def environment_metadata() -> dict[str, Any]:
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "not-installed"
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "none"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "transformers": transformers_version,
        "cuda": torch.version.cuda,
        "gpu": gpu,
    }


def register_experiment(path: str | Path, record: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    exists = destination.exists() and destination.stat().st_size > 0
    normalized = {field: record.get(field, "") for field in REGISTRY_FIELDS}
    for field, value in normalized.items():
        if isinstance(value, (dict, list, tuple)):
            normalized[field] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    with destination.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)


def base_experiment_record(experiment_id: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config_hash": config_hash(config),
        "seed": config.get("training", {}).get("seed", 42),
        "precision": config.get("training", {}).get("precision", ""),
        "status": "running",
        "environment": environment_metadata(),
        "device": config.get("training", {}).get("device", "auto"),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "",
    }
