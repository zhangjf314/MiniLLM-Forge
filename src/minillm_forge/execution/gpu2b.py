from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from minillm_forge.cli.common import StatefulRandomSampler, read_jsonl, write_json
from minillm_forge.config import config_hash, load_config
from minillm_forge.data.sft import encode_sft_example
from minillm_forge.finetuning import add_lora_adapters, load_full_sft_model, load_qlora_model
from minillm_forge.finetuning.lora import trainable_parameter_summary
from minillm_forge.training import ForgeTrainer, build_cosine_scheduler
from minillm_forge.training.optimizer import build_adamw, build_adamw_8bit
from minillm_forge.training.trainer import TrainingConfig, set_seed

STAGE = "STAGE_GPU_2B"
DESIGN_STAGE = "STAGE_GPU_2B_D_COMPLETE"
SEEDS = (42, 31415, 271828)
ARMS = ("B-FULL", "C-FULL", "B-LORA", "C-LORA", "B-QLORA", "C-QLORA")
FAMILY_CONFIGS = {
    "FULL": Path("configs/sft/gpu2b/full-sft-config.yaml"),
    "LORA": Path("configs/sft/gpu2b/lora-config.yaml"),
    "QLORA": Path("configs/sft/gpu2b/qlora-config.yaml"),
}
DESIGN_CHECKSUMS = Path("artifacts/design/gpu2b/checksums.txt")
SFT_MANIFEST = Path("artifacts/data_manifests/sft-dataset-manifest.json")
CONTAMINATION_REPORT = Path("artifacts/data_manifests/sft-contamination-report.json")
EVALUATION_MANIFEST = Path("artifacts/eval_manifests/evaluation-manifest.json")
SEED_MANIFEST = Path("artifacts/training/seed-manifest.json")
DESIGN_RESULT = Path("artifacts/training/gpu2b-design-result.json")
PREFLIGHT_RESULT = Path("artifacts/training/gpu2b-preflight.json")
FORMAL_MANIFEST = Path("artifacts/training/gpu2b-formal-run-manifest.json")
TOKEN_CACHE = Path("data/processed/gpu2b/sft_math_v1_train_tokens.pt")
TOKEN_CACHE_MANIFEST = Path("data/processed/gpu2b/sft_math_v1_train_tokens.json")
EXPECTED_TRAIN_EXAMPLES = 19_200
EXPECTED_VALIDATION_EXAMPLES = 800
EXPECTED_TARGET_TOKENS = 7_024_493
EXPECTED_UPDATES = 1_200


class NvidiaMemoryMonitor:
    """Sample physical GPU memory independently of the CUDA allocator on WDDM."""

    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval_seconds = interval_seconds
        self.minimum_free_mib: int | None = None
        self.maximum_used_mib: int | None = None
        self.samples = 0
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            )
            used_text, free_text = output.splitlines()[0].split(",")
            used, free = int(used_text.strip()), int(free_text.strip())
            self.maximum_used_mib = (
                used if self.maximum_used_mib is None else max(self.maximum_used_mib, used)
            )
            self.minimum_free_mib = (
                free if self.minimum_free_mib is None else min(self.minimum_free_mib, free)
            )
            self.samples += 1
        except Exception as exc:
            if len(self.errors) < 10:
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> NvidiaMemoryMonitor:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._sample()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_clean() -> bool:
    return not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()


def tokenizer_digest(tokenizer: Any) -> str:
    return _json_sha256(
        {
            "backend": tokenizer.backend_tokenizer.to_str(),
            "special_tokens_map": tokenizer.special_tokens_map,
            "vocab_size": len(tokenizer),
        }
    )


def load_frozen_tokenizer():
    from transformers import AutoTokenizer

    manifest = json.loads(SFT_MANIFEST.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["tokenizer"], revision=manifest["tokenizer_revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    actual = tokenizer_digest(tokenizer)
    if actual != manifest["tokenizer_digest"]:
        raise RuntimeError(
            f"frozen tokenizer digest mismatch: {actual} != {manifest['tokenizer_digest']}"
        )
    return tokenizer


def permutation_digest(size: int, seed: int) -> str:
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    values = order.to(torch.int64).tolist()
    payload = b"".join(int(value).to_bytes(4, "little", signed=False) for value in values)
    return hashlib.sha256(payload).hexdigest()


def _validate_frozen_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(SFT_MANIFEST.read_text(encoding="utf-8"))
    errors: list[str] = []
    partitions: dict[str, Any] = {}
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name, expected_count in (
        ("all", 20_000),
        ("train", EXPECTED_TRAIN_EXAMPLES),
        ("validation", EXPECTED_VALIDATION_EXAMPLES),
    ):
        contract = manifest[name]
        path = Path(contract["path"])
        actual_hash = file_sha256(path) if path.exists() else None
        if actual_hash != contract["file_sha256"]:
            errors.append(f"{name} dataset file hash mismatch")
            records: list[dict[str, Any]] = []
        else:
            records = read_jsonl(path)
            if len(records) != expected_count:
                errors.append(f"{name} dataset count mismatch")
        loaded[name] = records
        partitions[name] = {
            "path": str(path),
            "expected_sha256": contract["file_sha256"],
            "actual_sha256": actual_hash,
            "examples": len(records),
        }
    train_target_tokens = sum(int(row["target_tokens"]) for row in loaded["train"])
    train_input_tokens = sum(int(row["input_tokens"]) for row in loaded["train"])
    if train_target_tokens != EXPECTED_TARGET_TOKENS:
        errors.append("stored assistant-target token sum mismatch")
    if train_input_tokens != manifest["train"]["input_tokens_after_truncation"]:
        errors.append("stored input-token sum mismatch")
    return loaded["train"], {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "partitions": partitions,
        "train_input_tokens": train_input_tokens,
        "train_assistant_target_tokens": train_target_tokens,
    }


def build_token_cache(*, force: bool = False) -> dict[str, Any]:
    records, record_check = _validate_frozen_records()
    if record_check["status"] != "PASS":
        raise RuntimeError(f"frozen record validation failed: {record_check['errors']}")
    source_hash = record_check["partitions"]["train"]["actual_sha256"]
    tokenizer = load_frozen_tokenizer()
    token_hash = tokenizer_digest(tokenizer)
    if not force and TOKEN_CACHE.exists() and TOKEN_CACHE_MANIFEST.exists():
        cached = json.loads(TOKEN_CACHE_MANIFEST.read_text(encoding="utf-8"))
        if (
            cached.get("source_sha256") == source_hash
            and cached.get("tokenizer_digest") == token_hash
            and cached.get("cache_sha256") == file_sha256(TOKEN_CACHE)
        ):
            return cached

    input_chunks: list[torch.Tensor] = []
    offsets = [0]
    assistant_starts: list[int] = []
    actual_target_tokens = 0
    actual_input_tokens = 0
    for index, row in enumerate(records):
        encoded = encode_sft_example(
            tokenizer,
            user=str(row["problem"]),
            assistant=str(row["solution"]),
            system="You are a mathematical reasoning assistant.",
            max_length=1024,
        )
        input_ids = encoded["input_ids"]
        target_tokens = sum(label != -100 for label in encoded["labels"])
        if len(input_ids) != int(row["input_tokens"]):
            raise RuntimeError(f"input-token mismatch at frozen train row {index}")
        if target_tokens != int(row["target_tokens"]):
            raise RuntimeError(f"assistant-token mismatch at frozen train row {index}")
        input_chunks.append(torch.tensor(input_ids, dtype=torch.int32))
        actual_input_tokens += len(input_ids)
        actual_target_tokens += target_tokens
        offsets.append(actual_input_tokens)
        assistant_starts.append(len(input_ids) - target_tokens)
    if actual_target_tokens != EXPECTED_TARGET_TOKENS:
        raise RuntimeError("encoded assistant-target token budget mismatch")
    payload = {
        "input_ids": torch.cat(input_chunks),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "assistant_starts": torch.tensor(assistant_starts, dtype=torch.int32),
    }
    TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = TOKEN_CACHE.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, TOKEN_CACHE)
    result = {
        "classification": "GPU2B_TOKEN_CACHE_VALIDATED",
        "source_sha256": source_hash,
        "tokenizer_digest": token_hash,
        "examples": len(records),
        "input_tokens": actual_input_tokens,
        "assistant_target_tokens": actual_target_tokens,
        "cache_path": str(TOKEN_CACHE),
        "cache_sha256": file_sha256(TOKEN_CACHE),
    }
    write_json(TOKEN_CACHE_MANIFEST, result)
    return result


class FrozenSFTDataset(Dataset):
    def __init__(self, cache_path: str | Path = TOKEN_CACHE) -> None:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        self.input_ids = payload["input_ids"]
        self.offsets = payload["offsets"]
        self.assistant_starts = payload["assistant_starts"]

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
        }


def _single_item_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(features) != 1:
        raise RuntimeError("GPU-2B frozen micro-batch must equal one")
    return {key: value.unsqueeze(0) for key, value in features[0].items()}


def _run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "command": command,
        "output": (completed.stdout + completed.stderr)[-12_000:],
    }


def preflight() -> dict[str, Any]:
    started = time.time()
    checks = {
        "pytest": _run_command([sys.executable, "-m", "pytest", "-q"]),
        "ruff": _run_command([sys.executable, "-m", "ruff", "check", "."]),
        "format": _run_command([sys.executable, "-m", "ruff", "format", "--check", "."]),
        "lock": _run_command(["uv", "lock", "--check"]),
        "build": _run_command(["uv", "build"]),
    }
    import yaml

    tracked_yaml = [
        Path(line)
        for line in subprocess.check_output(
            ["git", "ls-files", "*.yaml", "*.yml"], text=True
        ).splitlines()
        if line.strip()
    ]
    yaml_errors = []
    for path in tracked_yaml:
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            yaml_errors.append(f"{path}: {type(exc).__name__}: {exc}")
    checks["yaml"] = {
        "status": "PASS" if not yaml_errors else "FAIL",
        "files_parsed": len(tracked_yaml),
        "files": [path.as_posix() for path in tracked_yaml],
        "errors": yaml_errors,
    }

    checksum_items = []
    checksum_errors = []
    for line in DESIGN_CHECKSUMS.read_text(encoding="utf-8").splitlines():
        expected, path_text = line.split("  ", 1)
        actual = file_sha256(path_text)
        status = "PASS" if actual == expected else "FAIL"
        checksum_items.append(
            {
                "path": path_text,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": status,
            }
        )
        if status == "FAIL":
            checksum_errors.append(path_text)
    checks["frozen_design_artifacts"] = {
        "status": "PASS" if not checksum_errors else "FAIL",
        "items": checksum_items,
        "errors": checksum_errors,
    }
    records, data_identity = _validate_frozen_records()
    checks["data_identity"] = data_identity
    contamination = json.loads(CONTAMINATION_REPORT.read_text(encoding="utf-8"))
    checks["contamination"] = {
        "status": "PASS" if contamination.get("remaining_known_collisions") == 0 else "FAIL",
        "report_sha256": file_sha256(CONTAMINATION_REPORT),
        "remaining_known_collisions": contamination.get("remaining_known_collisions"),
    }
    seed_contract = json.loads(SEED_MANIFEST.read_text(encoding="utf-8"))
    permutation_items = {}
    permutation_errors = []
    for seed in SEEDS:
        expected = seed_contract["permutations"][str(seed)]["uint32_le_sha256"]
        actual = permutation_digest(len(records), seed)
        permutation_items[str(seed)] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "PASS" if actual == expected else "FAIL",
        }
        if actual != expected:
            permutation_errors.append(seed)
    checks["seed_permutations"] = {
        "status": "PASS" if not permutation_errors else "FAIL",
        "items": permutation_items,
        "paired_across_all_initializations_and_families": True,
        "errors": permutation_errors,
    }
    try:
        cache = build_token_cache()
        cache_status = "PASS"
        cache_error = None
    except Exception as exc:
        cache, cache_status = {}, "FAIL"
        cache_error = f"{type(exc).__name__}: {exc}"
    checks["token_accounting"] = {
        "status": cache_status,
        "contract_assistant_target_tokens": EXPECTED_TARGET_TOKENS,
        "contract_optimizer_updates": EXPECTED_UPDATES,
        "cache": cache,
        "error": cache_error,
    }
    cpt_path = Path("runs/E04-qwen3-math-cpt/final_model")
    checks["cpt_checkpoint"] = {
        "status": "PASS" if (cpt_path / "model.safetensors").exists() else "FAIL",
        "path": str(cpt_path),
        "tree_sha256": tree_sha256(cpt_path),
        "model_sha256": file_sha256(cpt_path / "model.safetensors"),
    }
    design = json.loads(DESIGN_RESULT.read_text(encoding="utf-8"))
    checks["predecessor"] = {
        "status": "PASS" if design.get("classification") == DESIGN_STAGE else "FAIL",
        "classification": design.get("classification"),
    }
    all_pass = all(value["status"] == "PASS" for value in checks.values())
    result = {
        "stage": STAGE,
        "phase": "GPU2B-Q0",
        "classification": "GPU2B_Q0_PASS" if all_pass else "STAGE_GPU_2B_BLOCKED_BY_PREFLIGHT",
        "status": "PASS" if all_pass else "FAIL",
        "formal_training_runs_launched": 0,
        "code_commit": _git_commit(),
        "working_tree_clean": _git_clean(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "checks": checks,
        "started_at_unix": started,
        "ended_at_unix": time.time(),
    }
    write_json(PREFLIGHT_RESULT, result)
    if all_pass:
        freeze_formal_manifest(result)
    return result


def freeze_formal_manifest(preflight_result: dict[str, Any] | None = None) -> dict[str, Any]:
    seed_contract = json.loads(SEED_MANIFEST.read_text(encoding="utf-8"))
    runs = []
    for run_id in seed_contract["formal_runs"]:
        arm, seed_text = run_id.rsplit("-s", 1)
        initialization = "BASE_INIT" if arm.startswith("B-") else "CPT_INIT"
        family = arm.split("-", 1)[1]
        config_path = FAMILY_CONFIGS[family]
        config = load_config(config_path)
        runs.append(
            {
                "stage": STAGE,
                "run_id": run_id,
                "arm": arm,
                "initialization": initialization,
                "adaptation_family": "FULL_SFT" if family == "FULL" else family,
                "seed": int(seed_text),
                "config_path": str(config_path),
                "config_hash": config_hash(config),
                "dataset_digest": json.loads(SFT_MANIFEST.read_text(encoding="utf-8"))["train"][
                    "record_digest"
                ],
                "tokenizer_digest": json.loads(SFT_MANIFEST.read_text(encoding="utf-8"))[
                    "tokenizer_digest"
                ],
                "status": "PLANNED_NOT_RUN",
            }
        )
    result = {
        "stage": STAGE,
        "phase": "GPU2B-T",
        "execution_order_authority": "artifacts/training/seed-manifest.json:formal_runs",
        "code_commit": (preflight_result or {}).get("code_commit", _git_commit()),
        "formal_training_authorized": False,
        "runs": runs,
    }
    write_json(FORMAL_MANIFEST, result)
    return result


def _load_model(config: dict[str, Any], initialization: str, seed: int):
    set_seed(seed)
    model_contract = config["design"]["paired_initializations"][initialization]
    model_path = model_contract["name_or_path"]
    revision = model_contract.get("revision")
    method = config["model"]["method"]
    if method == "qlora":
        model = load_qlora_model(model_path, revision=revision, **config["lora"])
    else:
        model = load_full_sft_model(
            model_path,
            revision=revision,
            precision=config["training"]["precision"],
            gradient_checkpointing=config["training"]["gradient_checkpointing"],
            low_cpu_mem_usage=True,
        )
        if method == "lora":
            model = add_lora_adapters(model, **config["lora"])
    return model


def _build_optimizer(model: Any, config: dict[str, Any]):
    values = config["optimizer"]
    common = {
        "lr": values["lr"],
        "betas": tuple(values["betas"]),
        "weight_decay": values["weight_decay"],
        "eps": values["eps"],
    }
    name = values["name"].lower()
    if name == "adamw":
        return build_adamw(model, **common, foreach=False, fused=False)
    if name == "adamw8bit":
        return build_adamw_8bit(model, **common)
    if name == "pagedadamw8bit":
        import bitsandbytes as bnb

        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return bnb.optim.PagedAdamW8bit(parameters, **common)
    raise ValueError(f"unsupported frozen GPU-2B optimizer: {values['name']}")


def qlora_identity(model: Any) -> dict[str, Any]:
    inventory = []
    unintended_trainable = []
    quantized_backbone = 0
    nonquantized_backbone_weights = []
    for name, parameter in model.named_parameters():
        class_name = type(parameter).__name__
        is_adapter = "lora_" in name
        is_4bit = class_name == "Params4bit"
        if is_4bit:
            quantized_backbone += parameter.numel()
        if parameter.requires_grad and not is_adapter:
            unintended_trainable.append(name)
        if (
            not is_adapter
            and name.endswith(".weight")
            and parameter.ndim >= 2
            and "embed_tokens" not in name
            and not is_4bit
        ):
            nonquantized_backbone_weights.append(name)
        inventory.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "dtype": str(parameter.dtype),
                "class": class_name,
                "requires_grad": parameter.requires_grad,
                "adapter": is_adapter,
            }
        )
    passed = (
        bool(quantized_backbone) and not unintended_trainable and not nonquantized_backbone_weights
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "quantized_backbone_parameters": quantized_backbone,
        "unintended_trainable": unintended_trainable,
        "nonquantized_backbone_matrix_weights": nonquantized_backbone_weights,
        "inventory": inventory,
    }


def build_trainer(
    family: str,
    *,
    initialization: str = "BASE_INIT",
    seed: int = 42,
    max_steps: int = EXPECTED_UPDATES,
    output_dir: str | Path,
) -> tuple[ForgeTrainer, dict[str, Any]]:
    family = family.upper()
    config = load_config(FAMILY_CONFIGS[family])
    model = _load_model(config, initialization, seed)
    identity = qlora_identity(model) if family == "QLORA" else {"status": "PASS"}
    if identity["status"] != "PASS":
        raise RuntimeError("QLoRA hard identity gate failed")
    dataset = FrozenSFTDataset()
    sampler = StatefulRandomSampler(dataset, seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=_single_item_collator,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    optimizer = _build_optimizer(model, config)
    scheduler = build_cosine_scheduler(
        optimizer,
        total_steps=EXPECTED_UPDATES,
        warmup_ratio=config["scheduler"]["warmup_ratio"],
        min_lr_ratio=config["scheduler"]["min_lr_ratio"],
    )
    training = config["training"]
    trainer = ForgeTrainer(
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
    metadata = {
        "config": config,
        "config_hash": config_hash(config),
        "parameters": trainable_parameter_summary(model),
        "qlora_identity": identity,
    }
    return trainer, metadata


def memory_qualification(family: str, *, steps: int = 64) -> dict[str, Any]:
    family = family.upper()
    if steps != 64:
        raise ValueError("formal GPU2B-Q1 qualification requires exactly 64 updates")
    preflight_result = json.loads(PREFLIGHT_RESULT.read_text(encoding="utf-8"))
    if preflight_result["status"] != "PASS":
        raise RuntimeError("GPU2B-Q0 must pass before memory qualification")
    output_dir = Path("runs/gpu2b/qualification/memory") / family.lower()
    destination = Path(f"artifacts/training/gpu2b-memory-qualification-{family.lower()}.json")
    previous_attempts = []
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        previous_attempts.extend(previous.get("previous_attempts", []))
        previous_attempts.append(
            {
                "status": previous.get("status"),
                "classification": previous.get("classification"),
                "code_commit": previous.get("code_commit"),
                "actual_optimizer_updates": previous.get("actual_optimizer_updates"),
                "peak_allocated_vram_mib": previous.get("peak_allocated_vram_mib"),
                "peak_reserved_vram_mib": previous.get("peak_reserved_vram_mib"),
                "minimum_observed_headroom_mib": previous.get("minimum_observed_headroom_mib"),
                "nan_count": previous.get("nan_count"),
                "inf_count": previous.get("inf_count"),
                "oom_count": previous.get("oom_count"),
            }
        )
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    started = time.time()
    with NvidiaMemoryMonitor() as memory_monitor:
        trainer, metadata = build_trainer(family, max_steps=steps, output_dir=output_dir)
        trainer.train_until(steps, finalize=True)
    history = [record for record in trainer.state.history if record.get("event") == "train"]
    grad_norms = [float(record["grad_norm"]) for record in history]
    rates = [float(record["target_tokens_per_second"]) for record in history]
    state = trainer.state
    passed = (
        state.global_step == 64
        and state.nan_count == 0
        and state.inf_count == 0
        and state.oom_count == 0
        and bool(grad_norms)
        and all(torch.isfinite(torch.tensor(grad_norms)))
        and (memory_monitor.minimum_free_mib or 0) >= 1536
        and metadata["qlora_identity"]["status"] == "PASS"
    )
    result = {
        "stage": STAGE,
        "phase": "GPU2B-Q1",
        "family": family,
        "classification": (
            f"{family}_MEMORY_QUALIFICATION_PASS"
            if passed
            else "STAGE_GPU_2B_BLOCKED_BY_MEMORY_QUALIFICATION"
        ),
        "status": "PASS" if passed else "FAIL",
        "previous_attempts": previous_attempts,
        "actual_optimizer_updates": state.global_step,
        "actual_examples": state.examples_seen,
        "actual_input_tokens": state.tokens_seen,
        "actual_target_tokens": state.target_tokens_seen,
        "peak_allocated_vram_mib": trainer.peak_vram_mb(),
        "peak_reserved_vram_mib": trainer.peak_reserved_vram_mb(),
        "minimum_observed_headroom_mib": memory_monitor.minimum_free_mib,
        "maximum_observed_system_vram_used_mib": memory_monitor.maximum_used_mib,
        "system_vram_monitor_samples": memory_monitor.samples,
        "system_vram_monitor_errors": memory_monitor.errors,
        "cuda_mem_get_info_minimum_free_mib": state.minimum_headroom_mb,
        "memory_measurement_note": (
            "The nvidia-smi physical-memory monitor is authoritative for the frozen "
            "system-headroom gate. CUDA allocator counters are retained separately because "
            "WDDM may report virtual reservations above physical capacity."
        ),
        "median_target_tokens_per_second": median(rates) if rates else None,
        "nan_count": state.nan_count,
        "inf_count": state.inf_count,
        "oom_count": state.oom_count,
        "gradient_norm_min": min(grad_norms) if grad_norms else None,
        "gradient_norm_max": max(grad_norms) if grad_norms else None,
        "termination_reason": "QUALIFICATION_UPDATE_BUDGET_REACHED",
        "configuration_identity": metadata["config_hash"],
        "parameter_summary": metadata["parameters"],
        "qlora_identity": metadata["qlora_identity"],
        "code_commit": _git_commit(),
        "started_at_unix": started,
        "ended_at_unix": time.time(),
        "trainer_state": {
            key: value
            for key, value in asdict(state).items()
            if key not in {"history", "sampler_state"}
        },
    }
    write_json(destination, result)
    return result
