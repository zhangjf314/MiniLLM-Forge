from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

from minillm_forge.config import load_config
from minillm_forge.experiments import base_experiment_record, register_experiment

MODULES = {
    "pretrain": "minillm_forge.cli.pretrain",
    "cpt": "minillm_forge.cli.cpt",
    "sft": "minillm_forge.cli.sft",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and register one controlled experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--id", dest="experiment_id")
    parser.add_argument("--registry", default="experiments/registry.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    phase = config.get("experiment", {}).get("phase")
    if phase not in MODULES:
        raise ValueError(f"experiment.phase must be one of {sorted(MODULES)}")
    experiment_id = args.experiment_id or config.get("experiment", {}).get("id")
    experiment_id = experiment_id or f"run-{uuid.uuid4().hex[:8]}"
    record = base_experiment_record(experiment_id, config)
    record.update(
        {
            "model": config.get("model", {}).get("name_or_path", "MiniLLM"),
            "dataset": config.get("data", {}).get(
                "dataset", config.get("data", {}).get("files", "")
            ),
            "dataset_hash": config.get("data", {}).get("sha256", "unversioned"),
            "learning_rate": config.get("optimizer", {}).get("lr", ""),
            "batch_size": config.get("training", {}).get("micro_batch_size", ""),
            "effective_batch_size": (
                config.get("training", {}).get("micro_batch_size", 1)
                * config.get("training", {}).get("gradient_accumulation_steps", 1)
                * config.get("training", {}).get("world_size", 1)
            ),
            "rank": config.get("lora", {}).get("rank", ""),
            "target_modules": config.get("lora", {}).get("target_modules", ""),
        }
    )
    command = [sys.executable, "-m", MODULES[phase], "--config", args.config]
    completed = subprocess.run(command, check=False)
    record["status"] = "completed" if completed.returncode == 0 else "failed"
    output_dir = Path(config["training"]["output_dir"])
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        record["dataset_hash"] = summary.get("dataset_hash", record["dataset_hash"])
        record["trainable_params"] = summary.get("trainable_parameters", "")
        record["peak_vram_mb"] = summary.get("peak_vram_mb", "")
        record["peak_cuda_reserved_mb"] = summary.get("peak_reserved_vram_mb", "")
        record["device"] = summary.get("device", record["device"])
        record["gpu_name"] = summary.get("gpu_name", record["gpu_name"])
        record["torch_version"] = summary.get("torch_version", record["torch_version"])
        record["torch_cuda_version"] = summary.get(
            "torch_cuda_version", record["torch_cuda_version"]
        )
        record["final_loss"] = summary.get("final", {}).get(
            "validation_loss", summary.get("final_evaluation", {}).get("validation_loss", "")
        )
        metrics_path = output_dir / "metrics.jsonl"
        if metrics_path.exists():
            metric_records = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            train_records = [item for item in metric_records if item.get("event") == "train"]
            if train_records:
                record["tokens_per_second"] = train_records[-1].get("tokens_per_second", "")
    register_experiment(args.registry, record)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
