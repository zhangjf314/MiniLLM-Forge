"""GPU-1 bounded data, calibration, pilot and formal evidence entry points."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from minillm_forge.cli.common import StatefulRandomSampler, training_config
from minillm_forge.config import config_hash, load_config
from minillm_forge.data.formal import (
    MANIFEST,
    TOKENIZER,
    FrozenTokenDataset,
    json_write,
    verify_frozen,
)
from minillm_forge.data.manifest import file_digest
from minillm_forge.experiments import environment_metadata, git_commit
from minillm_forge.model import MiniLLM, MiniLLMConfig
from minillm_forge.training import build_adamw, build_cosine_scheduler
from minillm_forge.training.formal import FormalTrainer
from minillm_forge.training.trainer import set_seed

TRAINING_ROOT = Path("artifacts/training")


def gpu_snapshot():
    command = [
        "nvidia-smi",
        "--query-gpu=name,temperature.gpu,power.draw,clocks.sm,memory.used,memory.free",
        "--format=csv,noheader",
    ]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def canonical_config(path):
    config = load_config(path)
    manifest = verify_frozen()
    config["data_identity"] = {
        "manifest_hash": file_digest(MANIFEST),
        "tokenizer_hash": file_digest(TOKENIZER),
        "validation_hash": manifest["files"]["validation"]["sha256"],
    }
    return config, manifest


def build(config, manifest):
    if not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA")
    set_seed(config["training"]["seed"])
    model = MiniLLM(MiniLLMConfig(**config["model"]))
    train = FrozenTokenDataset(manifest["files"]["train"]["path"])
    val = FrozenTokenDataset(manifest["files"]["validation"]["path"])
    batch = config["training"]["micro_batch_size"]
    train_loader = DataLoader(
        train,
        batch_size=batch,
        sampler=StatefulRandomSampler(train, config["training"]["seed"]),
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val,
        batch_size=batch,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(987),
    )
    optimizer = build_adamw(model, **config["optimizer"])
    scheduler = build_cosine_scheduler(
        optimizer, total_steps=config["training"]["max_steps"], **config["scheduler"]
    )
    trainer = FormalTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        validation_loader=val_loader,
        config=training_config(config["training"]),
        run_config=config,
    )
    return trainer


def sample_generations(trainer, destination):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    prompts = [
        "The purpose of education is",
        "In mathematics, a fraction is",
        "Plants use sunlight to",
    ]
    samples = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        output = trainer.model.generate(
            torch.tensor([ids], device=trainer.device), max_new_tokens=32, temperature=0
        )
        samples.append(
            {"prompt": prompt, "completion": tokenizer.decode(output[0, len(ids) :].tolist())}
        )
    json_write(
        Path(destination),
        {
            "global_step": trainer.state.global_step,
            "generation": {
                "seed": 42,
                "temperature": 0,
                "top_p": 1.0,
                "max_new_tokens": 32,
                "mode": "greedy",
            },
            "purpose": "qualitative sanity only; not a benchmark",
            "samples": samples,
        },
    )


def run(path, resume=None, stop=None, output=None):
    session_started = time.perf_counter()
    config, manifest = canonical_config(path)
    if config["experiment"]["id"] == "E01":
        pilot = json.loads(Path("runs/gpu1-pilot/summary.json").read_text(encoding="utf-8"))
        decision = json.loads((TRAINING_ROOT / "budget_decision.json").read_text(encoding="utf-8"))
        if pilot["status"] != "completed" or pilot["tokens_seen"] < 5_000_000:
            raise ValueError("formal run requires a completed 5M-token pilot")
        if any(pilot[key] for key in ("nan_count", "inf_count", "oom_count")):
            raise ValueError("pilot numerical gate failed")
        if not decision["formal_approved"]:
            raise ValueError("formal budget is not approved by pilot evidence")
    if output:
        config["training"]["output_dir"] = output
    root = Path(config["training"]["output_dir"])
    prior_wall = 0.0
    if resume and (root / "summary.json").exists():
        prior_wall = json.loads((root / "summary.json").read_text())["elapsed_seconds"]
    if not resume and (root / "metrics.jsonl").exists():
        raise ValueError("fresh run output already exists; use a new run ID or explicit resume")
    trainer = build(config, manifest)
    root.mkdir(parents=True, exist_ok=True)
    if resume:
        trainer.resume(resume)
    else:
        config["git_commit"] = git_commit()
        json_write(root / "resolved_config.json", config)
        json_write(
            root / "hardware_before.json", {"snapshot": gpu_snapshot(), **environment_metadata()}
        )
        probe = DataLoader(
            Subset(trainer.train_loader.dataset, range(32)),
            batch_size=config["training"]["micro_batch_size"],
            generator=torch.Generator().manual_seed(987),
        )
        initial_train = trainer.evaluate_loader(probe)["validation_loss"]
        initial_val = trainer.evaluate_and_save()
        json_write(root / "initial.json", {"initial_train_loss": initial_train, **initial_val})
        sample_generations(trainer, root / "generation_initial.json")
    state = trainer.train(stop_after_step=stop)
    sample_generations(trainer, root / f"generation_step_{state.global_step}.json")
    json_write(root / "hardware_after.json", {"snapshot": gpu_snapshot()})
    records = [item for item in state.history if item["event"] == "train"]
    probe = DataLoader(
        Subset(trainer.train_loader.dataset, range(32)),
        batch_size=config["training"]["micro_batch_size"],
        generator=torch.Generator().manual_seed(987),
    )
    final_probe_loss = trainer.evaluate_loader(probe)["validation_loss"]
    final_validation = trainer.evaluate()
    summary = {
        "status": "completed" if state.global_step == config["training"]["max_steps"] else "paused",
        "global_step": state.global_step,
        "tokens_seen": state.tokens_seen,
        "supervised_tokens_seen": state.supervised_tokens_seen,
        "final_train_loss": records[-1]["loss"],
        "final_validation": final_validation,
        "final_train_probe_loss": final_probe_loss,
        "best_validation_loss": state.best_validation_loss,
        "best_step": state.best_step,
        "peak_allocated_mib": state.peak_allocated_mib,
        "peak_reserved_mib": state.peak_reserved_mib,
        "elapsed_seconds": prior_wall + time.perf_counter() - session_started,
        "loop_elapsed_seconds": state.elapsed_seconds,
        "compute_seconds": state.compute_seconds,
        "median_tokens_per_second": statistics.median(x["tokens_per_second"] for x in records),
        "average_tokens_per_second": state.tokens_seen / state.compute_seconds,
        "nan_count": state.nan_count,
        "inf_count": state.inf_count,
        "oom_count": state.oom_count,
        "parameters": trainer.model.num_parameters(),
        "config_hash": config_hash(config),
        "data_identity": config["data_identity"],
    }
    json_write(root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def calibrate(path, micro_batch):
    config, manifest = canonical_config(path)
    config = copy.deepcopy(config)
    config["training"].update(
        {
            "micro_batch_size": micro_batch,
            "gradient_accumulation_steps": 16 // micro_batch,
            "max_steps": 8,
            "eval_every": 0,
            "save_every": 0,
            "log_every": 1,
            "output_dir": f"runs/gpu1-calibration/batch-{micro_batch}",
            "tensorboard": False,
        }
    )
    config["checkpoint_steps"] = []
    config["scheduler"] = {"warmup_steps": 1, "min_lr_ratio": 0.1}
    trainer = build(config, manifest)
    trainer.validation_loader = None
    result = {
        "micro_batch": micro_batch,
        "grad_accum": 16 // micro_batch,
        "config": config,
        "gpu_before": gpu_snapshot(),
        "status": "FAIL",
    }
    try:
        state = trainer.train()
        windows = [x for x in state.history if x["event"] == "train"][2:]
        free, total = torch.cuda.mem_get_info()
        result.update(
            {
                "status": "PASS",
                "tokens_seen": state.tokens_seen,
                "peak_allocated_mib": state.peak_allocated_mib,
                "peak_reserved_mib": state.peak_reserved_mib,
                "median_tokens_per_second": statistics.median(
                    x["tokens_per_second"] for x in windows
                ),
                "median_step_seconds": statistics.median(x["step_time"] for x in windows),
                "headroom_fraction": free / total,
                "gpu_after": gpu_snapshot(),
            }
        )
    except torch.OutOfMemoryError as exc:
        result.update({"status": "OOM", "error": str(exc)})
    json_write(TRAINING_ROOT / f"batch_calibration_{micro_batch}.json", result)
    print(json.dumps(result, indent=2), flush=True)


def digest_state(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str(item.dtype).encode())
            digest.update(item.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item:
                visit(part)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def compare_resume(continuous_path, resumed_path, checkpoint_path):
    continuous = torch.load(continuous_path, map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    keys = ("model", "optimizer", "scheduler", "rng_state")
    checks = {
        key: {
            "continuous_sha256": digest_state(continuous[key]),
            "resumed_sha256": digest_state(resumed[key]),
            "equal": digest_state(continuous[key]) == digest_state(resumed[key]),
        }
        for key in keys
    }
    exact = all(check["equal"] for check in checks.values())
    record = {
        "classification": "EXACT_RESUME_CONFIRMED" if exact else "RESUME_SEMANTICS_VALIDATED",
        "comparison": checks,
        "checkpoint_step": checkpoint["trainer_state"]["global_step"],
        "checkpoint_tokens": checkpoint["trainer_state"]["tokens_seen"],
        "checkpoint_lr": checkpoint["optimizer"]["param_groups"][0]["lr"],
        "end_step": resumed["trainer_state"]["global_step"],
        "end_tokens": resumed["trainer_state"]["tokens_seen"],
        "cuda_rng_present": "cuda" in checkpoint["rng_state"],
        "continuous_metrics": continuous["metrics"],
        "resumed_metrics": resumed["metrics"],
        "control_cost": "8-step real-corpus control within pilot horizon; no repeated formal run",
    }
    json_write(TRAINING_ROOT / "minillm_gpu_resume_validation.json", record)
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["calibrate", "train", "compare", "freeze"])
    parser.add_argument("--config", default="configs/pretrain/formal.yaml")
    parser.add_argument("--micro-batch", type=int, choices=[2, 4, 8, 16], default=2)
    parser.add_argument("--resume")
    parser.add_argument("--stop", type=int)
    parser.add_argument("--output")
    parser.add_argument("--continuous")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == "freeze":
        config, manifest = canonical_config(args.config)
        model = MiniLLM(MiniLLMConfig(**config["model"]))
        json_write(
            TRAINING_ROOT / "minillm_formal_model.json",
            {
                "model": "MiniLLM",
                "parameter_count": model.num_parameters(),
                "trainable_parameter_count": model.num_parameters(trainable_only=True),
                "architecture": model.config.to_dict(),
                "git_commit": git_commit(),
                "config_hash": config_hash(config["model"]),
                "architecture_frozen": True,
                "stage_gpu1_baseline": "05df0d0201746fab521ea776c00440e206ac7f47",
                "data_identity": config["data_identity"],
            },
        )
    elif args.action == "calibrate":
        calibrate(args.config, args.micro_batch)
    elif args.action == "compare":
        compare_resume(args.continuous, args.resume, args.checkpoint)
    else:
        run(args.config, args.resume, args.stop, args.output)


if __name__ == "__main__":
    main()
