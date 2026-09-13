from __future__ import annotations

import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

from minillm_forge.cli.common import write_json
from minillm_forge.config import config_hash, load_config
from minillm_forge.execution.gpu2b import (
    CONTAMINATION_REPORT,
    EVALUATION_MANIFEST,
    FAMILY_CONFIGS,
    SEED_MANIFEST,
    SEEDS,
    TOKEN_CACHE,
    NvidiaMemoryMonitor,
    _git_commit,
    _json_sha256,
    _validate_frozen_records,
    build_token_cache,
    build_trainer,
    file_sha256,
    permutation_digest,
    qlora_identity,
    tokenizer_digest,
)
from minillm_forge.training.checkpoint import capture_rng_state

STAGE = "STAGE_GPU_2C_Q"
PROTOCOL = Path("artifacts/gpu2c/gpu2c_protocol.json")
Q0_RESULT = Path("artifacts/gpu2c/q0_preflight.json")
Q1_RESULTS = {
    "LORA": Path("artifacts/gpu2c/q1_lora_memory.json"),
    "QLORA": Path("artifacts/gpu2c/q1_qlora_memory.json"),
}
Q2_RESULT = Path("artifacts/gpu2c/q2_peft_resume.json")
QUALIFICATION_RESULT = Path("artifacts/gpu2c/qualification-result.json")
BASE_MANIFEST = Path("artifacts/training/qwen_base_model_manifest.json")
CPT_RESULT = Path("artifacts/training/qwen_math_cpt_result.json")
CPT_BUDGET = Path("artifacts/training/qwen_cpt_budget_decision.json")
CPT_MODEL = Path("runs/E04-qwen3-math-cpt/final_model/model.safetensors")
PROMPT = Path("reports/gpu2b/prompt-format.md")
DECISION_RULES = Path("reports/gpu2b/decision-rules.md")
SCORER = Path("src/minillm_forge/evaluation/math_eval.py")
HEADROOM_MIB = 1536


def _run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "command": command,
        "output": (completed.stdout + completed.stderr)[-12_000:],
    }


def _regression() -> dict[str, Any]:
    checks = {
        "pytest": _run_command([sys.executable, "-m", "pytest", "-q"]),
        "ruff": _run_command([sys.executable, "-m", "ruff", "check", "."]),
        "format": _run_command([sys.executable, "-m", "ruff", "format", "--check", "."]),
        "lock": _run_command(["uv", "lock", "--check"]),
        "build": _run_command(["uv", "build"]),
    }
    import yaml

    tracked = subprocess.check_output(
        ["git", "ls-files", "*.yaml", "*.yml"], text=True
    ).splitlines()
    yaml_errors = []
    for path_text in tracked:
        try:
            yaml.safe_load(Path(path_text).read_text(encoding="utf-8"))
        except Exception as exc:
            yaml_errors.append(f"{path_text}: {type(exc).__name__}: {exc}")
    checks["yaml"] = {
        "status": "PASS" if not yaml_errors else "FAIL",
        "files_parsed": len(tracked),
        "errors": yaml_errors,
    }
    return {
        "status": (
            "PASS" if all(check["status"] == "PASS" for check in checks.values()) else "FAIL"
        ),
        "checks": checks,
    }


def _assistant_mask_digest() -> str:
    payload = torch.load(TOKEN_CACHE, map_location="cpu", weights_only=True)
    starts = payload["assistant_starts"].contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(starts).hexdigest()


def _base_identity(protocol: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoConfig

    contract = protocol["initialization_contracts"]["B"]
    manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(contract["model"], revision=contract["revision"])
    actual_config = _json_sha256(config.to_dict())
    passed = (
        manifest["model_id"] == contract["model"]
        and manifest["revision"] == contract["revision"]
        and manifest["parameter_count"] == contract["parameter_count"]
        and actual_config == contract["config_sha256"]
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "model": contract["model"],
        "revision": contract["revision"],
        "parameter_count": manifest["parameter_count"],
        "expected_config_sha256": contract["config_sha256"],
        "actual_config_sha256": actual_config,
    }


def _cpt_identity(protocol: dict[str, Any]) -> dict[str, Any]:
    contract = protocol["initialization_contracts"]["C"]
    result = json.loads(CPT_RESULT.read_text(encoding="utf-8"))
    budget = json.loads(CPT_BUDGET.read_text(encoding="utf-8"))
    actual_model_hash = file_sha256(CPT_MODEL)
    passed = (
        actual_model_hash == contract["model_sha256"]
        and result["base_revision"] == contract["base_revision"]
        and result["training_tokens"] == contract["cpt_tokens"]
        and result["optimizer_steps"] == contract["cpt_updates"]
        and budget["config_hash"] == contract["cpt_config_hash"]
        and result["classification"] == "QWEN_MATH_CPT_VALIDATED_WITH_GENERAL_DEGRADATION"
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "path": contract["path"],
        "expected_model_sha256": contract["model_sha256"],
        "actual_model_sha256": actual_model_hash,
        "base_revision": result["base_revision"],
        "cpt_tokens": result["training_tokens"],
        "cpt_updates": result["optimizer_steps"],
        "cpt_config_hash": budget["config_hash"],
        "validation_classification": result["classification"],
    }


def preflight() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    records, data_check = _validate_frozen_records()
    cache = build_token_cache()
    data_contract = protocol["dataset"]
    data_pass = (
        data_check["status"] == "PASS"
        and len(records) == protocol["train_records"]
        and data_check["partitions"]["validation"]["examples"] == protocol["validation_records"]
        and data_check["train_input_tokens"] == protocol["input_tokens"]
        and data_check["train_assistant_target_tokens"] == protocol["assistant_tokens"]
        and data_check["partitions"]["train"]["actual_sha256"] == data_contract["train_sha256"]
        and data_check["partitions"]["validation"]["actual_sha256"]
        == data_contract["validation_sha256"]
        and cache["cache_sha256"] == data_contract["token_cache_sha256"]
        and _assistant_mask_digest() == data_contract["assistant_starts_int32_sha256"]
    )
    seed_manifest = json.loads(SEED_MANIFEST.read_text(encoding="utf-8"))
    seeds: dict[str, Any] = {}
    seed_pass = True
    for seed in SEEDS:
        order = torch.randperm(len(records), generator=torch.Generator().manual_seed(seed))
        actual = permutation_digest(len(records), seed)
        expected = seed_manifest["permutations"][str(seed)]["uint32_le_sha256"]
        seeds[str(seed)] = {
            "status": "PASS" if actual == expected else "FAIL",
            "sample_count": len(records),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "first_indices": order[:8].tolist(),
            "last_indices": order[-8:].tolist(),
        }
        seed_pass = seed_pass and actual == expected
    from transformers import AutoTokenizer

    tokenizer_contract = protocol["initialization_contracts"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_contract["B"]["model"], revision=tokenizer_contract["tokenizer_revision"]
    )
    tokenizer_actual = tokenizer_digest(tokenizer)
    base = _base_identity(protocol)
    cpt = _cpt_identity(protocol)
    configs = {}
    for family in ("LORA", "QLORA"):
        family_key = family.lower() + "_config"
        path = Path(protocol[family_key]["source"])
        loaded = load_config(path)
        actual_file_hash = file_sha256(path)
        configs[family] = {
            "status": ("PASS" if actual_file_hash == protocol[family_key]["sha256"] else "FAIL"),
            "path": str(path),
            "expected_file_sha256": protocol[family_key]["sha256"],
            "actual_file_sha256": actual_file_hash,
            "semantic_config_hash": config_hash(loaded),
        }
    evaluation = json.loads(EVALUATION_MANIFEST.read_text(encoding="utf-8"))
    benchmarks = {}
    benchmark_pass = True
    for name, contract in evaluation["primary_benchmarks"].items():
        actual = file_sha256(contract["snapshot_path"])
        item_pass = actual == contract["snapshot_sha256"]
        benchmark_pass = benchmark_pass and item_pass
        benchmarks[name] = {
            "status": "PASS" if item_pass else "FAIL",
            "examples": contract["examples"],
            "revision": contract["revision"],
            "expected_sha256": contract["snapshot_sha256"],
            "actual_sha256": actual,
        }
    static_hashes = {
        "prompt": file_sha256(PROMPT),
        "evaluation_manifest": file_sha256(EVALUATION_MANIFEST),
        "scorer": file_sha256(SCORER),
        "decision_rules": file_sha256(DECISION_RULES),
        "contamination_report": file_sha256(CONTAMINATION_REPORT),
    }
    expected_hashes = {
        "prompt": "77c3e381b5975746b71f359405cd465c8921c0ad79ec22f9961e596f0aad7b7c",
        "evaluation_manifest": "eb13f3e67d78b14a4c177b08f2727380daad8a3fb2aa0b3c7b4f3f9174b0718e",
        "scorer": evaluation["scoring"]["evaluator_sha256"],
        "decision_rules": protocol["transfer_rule"]["sha256"],
        "contamination_report": "8cd2796c8a989b18b468b14db43c7e16868ac1b3d26155b7e36ba71d47fc4524",
    }
    static_pass = static_hashes == expected_hashes
    regression = _regression()
    assistant_mask_actual = _assistant_mask_digest()
    checks = {
        "dataset_identity": {**data_check, "status": "PASS" if data_pass else "FAIL"},
        "token_cache": {
            **cache,
            "status": (
                "PASS" if cache["cache_sha256"] == data_contract["token_cache_sha256"] else "FAIL"
            ),
        },
        "assistant_mask": {
            "status": (
                "PASS"
                if assistant_mask_actual == data_contract["assistant_starts_int32_sha256"]
                else "FAIL"
            ),
            "expected_sha256": data_contract["assistant_starts_int32_sha256"],
            "actual_sha256": assistant_mask_actual,
        },
        "seed_permutations": {"status": "PASS" if seed_pass else "FAIL", "seeds": seeds},
        "base_checkpoint": base,
        "cpt_checkpoint": cpt,
        "tokenizer": {
            "status": (
                "PASS" if tokenizer_actual == tokenizer_contract["tokenizer_sha256"] else "FAIL"
            ),
            "revision": tokenizer_contract["tokenizer_revision"],
            "expected_sha256": tokenizer_contract["tokenizer_sha256"],
            "actual_sha256": tokenizer_actual,
        },
        "configs": {
            "status": (
                "PASS" if all(item["status"] == "PASS" for item in configs.values()) else "FAIL"
            ),
            "families": configs,
        },
        "benchmarks": {"status": "PASS" if benchmark_pass else "FAIL", **benchmarks},
        "static_artifacts": {
            "status": "PASS" if static_pass else "FAIL",
            "expected": expected_hashes,
            "actual": static_hashes,
        },
        "regression": regression,
    }
    passed = all(check["status"] == "PASS" for check in checks.values())
    if not data_pass:
        classification = "GPU2C_FROZEN_DATA_IDENTITY_MISMATCH"
    elif passed:
        classification = "GPU2C_Q0_P_PASS"
    else:
        classification = "GPU2C_SCIENTIFIC_IDENTITY_FAILURE"
    result = {
        "stage": STAGE,
        "phase": "GPU2C-Q0-P",
        "classification": classification,
        "status": "PASS" if passed else "FAIL",
        "formal_campaign_authorized": False,
        "formal_training_launches": 0,
        "benchmark_launches": 0,
        "code_commit": _git_commit(),
        "protocol_sha256": file_sha256(PROTOCOL),
        "checks": checks,
    }
    write_json(Q0_RESULT, result)
    return result


def _adapter_identity(model: Any, family: str) -> dict[str, Any]:
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    unexpected = [name for name in trainable_names if "lora_" not in name]
    summary = {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    summary["trainable_percent"] = (
        100.0 * summary["trainable_parameters"] / summary["total_parameters"]
    )
    result: dict[str, Any] = {
        "status": "PASS" if trainable_names and not unexpected else "FAIL",
        **summary,
        "trainable_parameter_names": trainable_names,
        "unexpected_base_trainables": unexpected,
    }
    if family == "QLORA":
        linear4bit = []
        missing_state = []
        for name, module in model.named_modules():
            if type(module).__name__ != "Linear4bit":
                continue
            quant_state_present = (
                getattr(getattr(module, "weight", None), "quant_state", None) is not None
            )
            linear4bit.append(
                {
                    "name": name,
                    "module_type": type(module).__name__,
                    "weight_type": type(module.weight).__name__,
                    "quant_state_present": quant_state_present,
                    "weight_requires_grad": module.weight.requires_grad,
                }
            )
            if not quant_state_present:
                missing_state.append(name)
        hard_identity = qlora_identity(model)
        result.update(
            {
                "qlora_backbone_4bit": bool(linear4bit) and not missing_state,
                "linear4bit_modules": linear4bit,
                "linear4bit_module_count": len(linear4bit),
                "missing_quantization_state": missing_state,
                "frozen_backbone": not hard_identity["unintended_trainable"],
                "quantized_backbone_parameters": hard_identity["quantized_backbone_parameters"],
            }
        )
        if not result["qlora_backbone_4bit"] or hard_identity["status"] != "PASS":
            result["status"] = "FAIL"
    return result


def memory_qualification(family: str) -> dict[str, Any]:
    family = family.upper()
    if family not in Q1_RESULTS:
        raise ValueError("family must be LORA or QLORA")
    q0 = json.loads(Q0_RESULT.read_text(encoding="utf-8"))
    if q0["status"] != "PASS":
        raise RuntimeError("GPU2C-Q0-P must pass before Q1")
    output_dir = Path("runs/gpu2c/qualification/q1") / family.lower()
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    started = time.time()
    with NvidiaMemoryMonitor(interval_seconds=0.2) as monitor:
        trainer, metadata = build_trainer(
            family, initialization="BASE_INIT", seed=42, max_steps=64, output_dir=output_dir
        )
        identity = _adapter_identity(trainer.model, family)
        if identity["status"] != "PASS":
            raise RuntimeError(f"{family} adapter identity failed")
        trainer.train_until(64, finalize=True)
    elapsed = time.time() - started
    history = [record for record in trainer.state.history if record.get("event") == "train"]
    grad_norms = [float(record["grad_norm"]) for record in history]
    target_rates = [float(record["target_tokens_per_second"]) for record in history]
    losses = [float(record["loss"]) for record in history]
    step_times = []
    prior_target_tokens = 0
    for record in history:
        step_target_tokens = int(record["target_tokens_seen"]) - prior_target_tokens
        prior_target_tokens = int(record["target_tokens_seen"])
        step_times.append(step_target_tokens / float(record["target_tokens_per_second"]))
    state = trainer.state
    passed = (
        state.global_step == 64
        and state.examples_seen == 1024
        and state.nan_count == 0
        and state.inf_count == 0
        and state.oom_count == 0
        and all(math.isfinite(value) for value in losses + grad_norms)
        and (monitor.minimum_free_mib or 0) >= HEADROOM_MIB
        and not monitor.errors
        and identity["status"] == "PASS"
    )
    result = {
        "stage": STAGE,
        "phase": f"GPU2C-Q1-{family}",
        "classification": (
            f"GPU2C_{family}_MEMORY_QUALIFICATION_PASS"
            if passed
            else f"GPU2C_{family}_MEMORY_QUALIFICATION_FAILED"
        ),
        "status": "PASS" if passed else "FAIL",
        "initialization": "B",
        "seed": 42,
        "updates_completed": state.global_step,
        "samples_processed": state.examples_seen,
        "input_tokens": state.tokens_seen,
        "assistant_tokens": state.target_tokens_seen,
        "loss_finite": all(math.isfinite(value) for value in losses),
        "final_loss": losses[-1] if losses else None,
        "nan_count": state.nan_count,
        "inf_count": state.inf_count,
        "oom_count": state.oom_count,
        "physical_peak_vram_mib": monitor.maximum_used_mib,
        "minimum_physical_headroom_mib": monitor.minimum_free_mib,
        "physical_sample_count": monitor.samples,
        "physical_monitor_errors": monitor.errors,
        "peak_cuda_allocated_mib": trainer.peak_vram_mb(),
        "peak_cuda_reserved_mib": trainer.peak_reserved_vram_mb(),
        "median_target_tokens_per_second": median(target_rates),
        "samples_per_second": state.examples_seen / elapsed,
        "median_step_time_seconds": median(step_times),
        "wall_clock_seconds": elapsed,
        "gradient_norm_min": min(grad_norms),
        "gradient_norm_max": max(grad_norms),
        "adapter_identity": identity,
        "config_file_sha256": file_sha256(FAMILY_CONFIGS[family]),
        "semantic_config_hash": metadata["config_hash"],
        "code_commit": _git_commit(),
        "formal_run": False,
        "benchmark_result": False,
    }
    write_json(Q1_RESULTS[family], result)
    return result


def _digest_value(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b"ndarray")
            digest.update(str(item.dtype).encode())
            digest.update(str(item.shape).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda entry: str(entry)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for entry in item:
                update(entry)
        else:
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def _state_snapshot(trainer: Any, family: str) -> dict[str, Any]:
    model_state = trainer.model.state_dict()
    adapter_state = {name: value for name, value in model_state.items() if "lora_" in name}
    sampler_state = trainer.train_loader.sampler.state_dict()
    loader_generator = trainer.train_loader.generator.get_state()
    rng_state = capture_rng_state()
    state = trainer.state
    order = sampler_state["order"][: sampler_state["position"]]
    return {
        "global_step": state.global_step,
        "examples_seen": state.examples_seen,
        "input_tokens_seen": state.tokens_seen,
        "assistant_tokens_seen": state.target_tokens_seen,
        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
        "model_sha256": _digest_value(model_state),
        "adapter_sha256": _digest_value(adapter_state),
        "optimizer_sha256": _digest_value(trainer.optimizer.state_dict()),
        "scheduler_sha256": _digest_value(trainer.scheduler.state_dict()),
        "sampler_sha256": _digest_value(sampler_state),
        "consumed_sample_order_sha256": _digest_value(order),
        "sampler_position": sampler_state["position"],
        "loader_generator_sha256": _digest_value(loader_generator),
        "python_rng_sha256": _digest_value(rng_state["python"]),
        "numpy_rng_sha256": _digest_value(rng_state["numpy"]),
        "torch_rng_sha256": _digest_value(rng_state["torch"]),
        "cuda_rng_sha256": _digest_value(rng_state.get("cuda", [])),
        "qlora_identity": (
            {
                key: value
                for key, value in _adapter_identity(trainer.model, family).items()
                if key
                in {
                    "status",
                    "qlora_backbone_4bit",
                    "linear4bit_module_count",
                    "frozen_backbone",
                    "quantized_backbone_parameters",
                }
            }
            if family == "QLORA"
            else None
        ),
    }


def _trajectory_digest(trainer: Any) -> str:
    records = []
    for record in trainer.state.history:
        if record.get("event") != "train":
            continue
        records.append(
            {
                key: record[key]
                for key in (
                    "global_step",
                    "loss",
                    "learning_rate",
                    "grad_norm",
                    "tokens_seen",
                    "target_tokens_seen",
                )
            }
        )
    return _digest_value(records)


def _snapshot_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(left) & set(right))
    matches = {key: left[key] == right[key] for key in keys}
    return {"all_match": all(matches.values()), "field_matches": matches}


def resume_qualification(family: str) -> dict[str, Any]:
    family = family.upper()
    if family not in Q1_RESULTS:
        raise ValueError("family must be LORA or QLORA")
    for required_family, path in Q1_RESULTS.items():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "PASS":
            raise RuntimeError(f"GPU2C-Q1-{required_family} must pass before Q2")
    root = Path("runs/gpu2c/qualification/q2") / family.lower()
    continuous_dir = root / "continuous"
    interrupted_dir = root / "interrupted"
    for metrics in (continuous_dir / "metrics.jsonl", interrupted_dir / "metrics.jsonl"):
        if metrics.exists():
            metrics.unlink()
    continuous, _ = build_trainer(
        family,
        initialization="BASE_INIT",
        seed=42,
        max_steps=16,
        output_dir=continuous_dir,
    )
    continuous.train_until(8)
    continuous_boundary = _state_snapshot(continuous, family)
    continuous.train_until(16)
    continuous_final = _state_snapshot(continuous, family)
    continuous_trajectory = _trajectory_digest(continuous)
    del continuous
    gc.collect()
    torch.cuda.empty_cache()

    interrupted, _ = build_trainer(
        family,
        initialization="BASE_INIT",
        seed=42,
        max_steps=16,
        output_dir=interrupted_dir,
    )
    interrupted.train_until(8)
    checkpoint = interrupted._save("resume-at-8.pt")
    interrupted_boundary = _state_snapshot(interrupted, family)
    checkpoint_sha256 = file_sha256(checkpoint)
    del interrupted
    gc.collect()
    torch.cuda.empty_cache()

    resumed, _ = build_trainer(
        family,
        initialization="BASE_INIT",
        seed=42,
        max_steps=16,
        output_dir=interrupted_dir,
    )
    resumed.resume(checkpoint)
    restored_boundary = _state_snapshot(resumed, family)
    resumed.train_until(16)
    resumed_final = _state_snapshot(resumed, family)
    resumed_trajectory = _trajectory_digest(resumed)

    save_restore = _snapshot_comparison(interrupted_boundary, restored_boundary)
    control_boundary = _snapshot_comparison(continuous_boundary, interrupted_boundary)
    final_comparison = _snapshot_comparison(continuous_final, resumed_final)
    trajectory_match = continuous_trajectory == resumed_trajectory
    sample_order_match = (
        continuous_final["consumed_sample_order_sha256"]
        == resumed_final["consumed_sample_order_sha256"]
    )
    rng_keys = (
        "python_rng_sha256",
        "numpy_rng_sha256",
        "torch_rng_sha256",
        "cuda_rng_sha256",
        "loader_generator_sha256",
    )
    rng_match = all(continuous_final[key] == resumed_final[key] for key in rng_keys)
    exact = (
        save_restore["all_match"]
        and control_boundary["all_match"]
        and final_comparison["all_match"]
        and trajectory_match
        and sample_order_match
        and rng_match
    )
    classification = "PEFT_EXACT_RESUME_CONFIRMED" if exact else "GPU2C_RESUME_QUALIFICATION_FAILED"
    family_result = {
        "family": family,
        "status": "PASS" if exact else "FAIL",
        "classification": classification,
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
        "continuous_boundary": continuous_boundary,
        "restored_boundary": restored_boundary,
        "continuous_final": continuous_final,
        "resumed_final": resumed_final,
        "formal_run": False,
    }
    combined = (
        json.loads(Q2_RESULT.read_text(encoding="utf-8"))
        if Q2_RESULT.exists()
        else {
            "stage": STAGE,
            "phase": "GPU2C-Q2-PEFT-RESUME",
            "formal_campaign_authorized": False,
            "methods": {},
        }
    )
    combined["methods"][family] = family_result
    statuses = [combined["methods"].get(name, {}).get("status") for name in ("LORA", "QLORA")]
    combined["status"] = "PASS" if statuses == ["PASS", "PASS"] else "INCOMPLETE_OR_FAIL"
    combined["classification"] = (
        "GPU2C_PEFT_RESUME_QUALIFICATION_PASS"
        if combined["status"] == "PASS"
        else "GPU2C_RESUME_QUALIFICATION_PENDING_OR_FAILED"
    )
    combined["code_commit"] = _git_commit()
    write_json(Q2_RESULT, combined)
    return family_result


def finalize() -> dict[str, Any]:
    q0 = json.loads(Q0_RESULT.read_text(encoding="utf-8"))
    q1 = {
        family: json.loads(path.read_text(encoding="utf-8")) for family, path in Q1_RESULTS.items()
    }
    q2 = json.loads(Q2_RESULT.read_text(encoding="utf-8"))
    regression = _regression()
    gates = {
        "Q0": q0["status"],
        "Q1_LORA": q1["LORA"]["status"],
        "Q1_QLORA": q1["QLORA"]["status"],
        "Q2_LORA": q2["methods"]["LORA"]["status"],
        "Q2_QLORA": q2["methods"]["QLORA"]["status"],
        "REGRESSION": regression["status"],
    }
    passed = all(value == "PASS" for value in gates.values())
    result = {
        "stage": STAGE,
        "classification": (
            "GPU2C_FORMAL_CAMPAIGN_QUALIFIED" if passed else "GPU2C_RESUME_QUALIFICATION_FAILED"
        ),
        "status": "PASS" if passed else "FAIL",
        "gates": gates,
        "formal_campaign_authorized": passed,
        "formal_training_launches": 0,
        "benchmark_launches": 0,
        "protocol_sha256": file_sha256(PROTOCOL),
        "q0_sha256": file_sha256(Q0_RESULT),
        "q1_lora_sha256": file_sha256(Q1_RESULTS["LORA"]),
        "q1_qlora_sha256": file_sha256(Q1_RESULTS["QLORA"]),
        "q2_sha256": file_sha256(Q2_RESULT),
        "regression": regression,
        "protocol_commit": q0["code_commit"],
        "qualification_commit": _git_commit(),
        "recommended_next_stage": "STAGE_GPU_2C_F" if passed else None,
    }
    write_json(QUALIFICATION_RESULT, result)
    return result
