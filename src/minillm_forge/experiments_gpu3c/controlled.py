# ruff: noqa: E501
from __future__ import annotations

import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from minillm_forge.cli.common import StatefulRandomSampler
from minillm_forge.config import load_config
from minillm_forge.evaluation.math_eval import normalize_answer
from minillm_forge.execution.gpu2b import _build_optimizer, _load_model, file_sha256
from minillm_forge.execution.gpu2c import _digest_value
from minillm_forge.execution.gpu2d import CONFIGS
from minillm_forge.experiments_gpu3b.inference import config as generation_config
from minillm_forge.experiments_gpu3b.inference import generate_one
from minillm_forge.training import ForgeTrainer, build_cosine_scheduler
from minillm_forge.training.checkpoint import load_checkpoint
from minillm_forge.training.trainer import TrainingConfig

IM_END_ID = 151645
SEED = 42
CONTEXT_CACHE = Path("data/processed/gpu2d/sft_math_context_512_train_tokens.pt")
AUDIT_DETAILS = Path("artifacts/gpu3c/data_audit.jsonl")
PROTOCOL_PATH = Path("artifacts/gpu3c/protocol.json")


def apply_eos_policy(
    ids: torch.Tensor, assistant_start: int, assistant_end: int, group: str
) -> torch.Tensor:
    """Build position-based labels without treating PAD/EOS token IDs as padding."""
    if group not in {"B-EOS-MASKED", "C-EOS-SUPERVISED"}:
        raise ValueError("unknown GPU-3C group")
    if not 0 <= assistant_start <= assistant_end < len(ids):
        raise ValueError("invalid assistant span")
    if int(ids[assistant_end]) != IM_END_ID:
        raise ValueError("assistant_end does not point to <|im_end|>")
    labels = ids.clone().to(torch.long)
    labels[:assistant_start] = -100
    if group == "B-EOS-MASKED":
        labels[assistant_end] = -100
    return labels


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class ControlledEOSDataset(Dataset):
    """The frozen complete subset with exactly one controlled target-label difference."""

    def __init__(self, repo: str | Path, group: str) -> None:
        self.repo = Path(repo).resolve()
        if group not in {"B-EOS-MASKED", "C-EOS-SUPERVISED"}:
            raise ValueError("unknown GPU-3C group")
        self.group = group
        payload = torch.load(self.repo / CONTEXT_CACHE, map_location="cpu", weights_only=True)
        details = _read_jsonl(self.repo / AUDIT_DETAILS)
        complete = {
            int(row["source_index"]): int(row["assistant_end_position"])
            for row in details
            if row["has_supervised_assistant_end"]
        }
        source_to_cache = {
            int(source_index): cache_index
            for cache_index, source_index in enumerate(payload["source_indices"].tolist())
        }
        self.entries = [
            (source_to_cache[source_index], source_index, complete[source_index])
            for source_index in sorted(complete)
        ]
        self.input_ids = payload["input_ids"]
        self.offsets = payload["offsets"]
        self.assistant_starts = payload["assistant_starts"]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        cache_index, source_index, assistant_end = self.entries[index]
        start, end = int(self.offsets[cache_index]), int(self.offsets[cache_index + 1])
        ids = self.input_ids[start:end].to(torch.long)
        assistant_start = int(self.assistant_starts[cache_index])
        if assistant_end >= len(ids) or int(ids[assistant_end]) != IM_END_ID:
            raise RuntimeError(f"invalid frozen assistant end for source row {source_index}")
        labels = apply_eos_policy(ids, assistant_start, assistant_end, self.group)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids, dtype=torch.bool),
            "labels": labels,
            "source_index": torch.tensor(source_index, dtype=torch.long),
            "assistant_end_position": torch.tensor(assistant_end, dtype=torch.long),
        }


def _collate(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(features) != 1:
        raise ValueError("GPU-3C frozen micro-batch size is one")
    return {key: value.unsqueeze(0) for key, value in features[0].items()}


class GPU3CTrainer(ForgeTrainer):
    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return super()._prepare_batch(
            {
                key: value
                for key, value in batch.items()
                if key not in {"source_index", "assistant_end_position"}
            }
        )


def audit_pair_identity(repo: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo).resolve()
    masked = ControlledEOSDataset(repo, "B-EOS-MASKED")
    supervised = ControlledEOSDataset(repo, "C-EOS-SUPERVISED")
    if len(masked) != len(supervised):
        raise RuntimeError("B/C sample counts differ")
    differences = 0
    masked_targets = 0
    supervised_targets = 0
    order_payload = []
    for index in range(len(masked)):
        left, right = masked[index], supervised[index]
        if not torch.equal(left["input_ids"], right["input_ids"]):
            raise RuntimeError(f"B/C input IDs differ at {index}")
        if not torch.equal(left["attention_mask"], right["attention_mask"]):
            raise RuntimeError(f"B/C attention masks differ at {index}")
        changed = torch.nonzero(left["labels"] != right["labels"]).flatten().tolist()
        end = int(right["assistant_end_position"])
        if (
            changed != [end]
            or int(left["labels"][end]) != -100
            or int(right["labels"][end]) != IM_END_ID
        ):
            raise RuntimeError(f"B/C labels differ outside target EOS at {index}")
        differences += len(changed)
        masked_targets += int(left["labels"].ne(-100).sum())
        supervised_targets += int(right["labels"].ne(-100).sum())
        order_payload.append(int(right["source_index"]))
    result = {
        "status": "PASS",
        "samples": len(masked),
        "input_ids_identical": True,
        "attention_mask_identical": True,
        "label_difference_count": differences,
        "label_difference_per_sample": 1,
        "only_difference": "final assistant <|im_end|> label",
        "masked_effective_supervised_tokens": masked_targets,
        "supervised_effective_supervised_tokens": supervised_targets,
        "target_token_difference": supervised_targets - masked_targets,
        "sample_order_sha256": _digest_value(order_payload),
        "packing": "DISABLED",
        "causal_shift": "Hugging Face causal LM shifts labels internally; label at position t is predicted from logits t-1",
        "pad_masking": "position based; token ID 151643 is not globally masked",
    }
    _write_json(repo / "artifacts/gpu3c/mask_pair_audit.json", result)
    return result


def build_trainer(
    repo: str | Path,
    group: str,
    *,
    steps: int,
    output_dir: Path,
) -> tuple[GPU3CTrainer, dict[str, Any]]:
    repo = Path(repo).resolve()
    protocol = _read_json(repo / PROTOCOL_PATH)
    if steps > int(protocol["resource_limits"]["max_training_steps_per_arm"]):
        raise ValueError("steps exceed frozen GPU-3C limit")
    config = load_config(repo / CONFIGS[("LORA", 512, "R16_ALL_LINEAR")])
    model = _load_model(config, "BASE_INIT", SEED)
    dataset = ControlledEOSDataset(repo, group)
    sampler = StatefulRandomSampler(dataset, SEED)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=_collate,
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    optimizer = _build_optimizer(model, config)
    scheduler = build_cosine_scheduler(
        optimizer,
        total_steps=steps,
        warmup_ratio=config["scheduler"]["warmup_ratio"],
        min_lr_ratio=config["scheduler"]["min_lr_ratio"],
    )
    training = config["training"]
    trainer = GPU3CTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        config=TrainingConfig(
            max_steps=steps,
            gradient_accumulation_steps=training["gradient_accumulation_steps"],
            grad_clip=training["grad_clip"],
            precision=training["precision"],
            log_every=1,
            eval_every=0,
            save_every=0,
            output_dir=str(output_dir),
            seed=SEED,
            device="cuda",
            tensorboard=False,
            empty_cache_after_step=True,
        ),
        run_config={"stage": "GPU-3C", "group": group, "protocol": protocol},
    )
    adapter_state = {name: value for name, value in model.state_dict().items() if "lora_" in name}
    return trainer, {
        "initial_adapter_sha256": _digest_value(adapter_state),
        "dataset_samples": len(dataset),
        "cache_sha256": file_sha256(repo / CONTEXT_CACHE),
    }


def _historical_eval_rows(repo: Path) -> dict[str, dict[str, Any]]:
    result = {}
    root = repo / "runs/gpu2d-formal/B-LORA-s42/evaluation-amended"
    for path in (root / "gsm8k-fixed-200.jsonl", root / "math500.jsonl"):
        for row in _read_jsonl(path):
            result[row["problem_id"]] = row
    return result


def _evaluate_model(
    repo: Path, model: Any, tokenizer: Any, group: str, problem_ids: list[str]
) -> list[dict[str, Any]]:
    source = _historical_eval_rows(repo)
    records = []
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    for problem_id in problem_ids:
        historical = source[problem_id]
        for identifier, stop in (("C0", "S0"), ("S3", "S3")):
            cfg = generation_config(
                f"GPU3C-{identifier}", stop=stop, repetition_penalty=1.0, max_new_tokens=512
            )
            generated = generate_one(
                model,
                tokenizer,
                [int(value) for value in historical["prompt_token_ids"]],
                eos_token_ids=cfg["eos_token_ids"],
                max_new_tokens=512,
                external_answer_stop=cfg["external_answer_stop"],
                wall_time_seconds=120,
            )
            reference = str(historical["reference_answer"])
            generated.update(
                {
                    "generation_id": f"gpu3c:{group}:{problem_id}:{identifier}",
                    "group": group,
                    "sample_id": problem_id,
                    "benchmark": historical["benchmark"],
                    "decoding_path": identifier,
                    "reference_answer": reference,
                    "legacy_correct": normalize_answer(generated["legacy_extracted_answer"])
                    == normalize_answer(reference),
                    "first_valid_correct": normalize_answer(
                        generated["first_valid"]["answer"] or ""
                    )
                    == normalize_answer(reference),
                    "conflict_aware_correct": normalize_answer(
                        generated["conflict_aware"]["answer"] or ""
                    )
                    == normalize_answer(reference),
                    "premature_eos": generated["autonomous_eos_stop"]
                    and generated["first_complete_answer_token_position"] is None,
                }
            )
            records.append(generated)
    return records


def run_arm(
    repo: str | Path,
    group: str,
    *,
    steps: int,
    evaluate: bool,
    run_name: str,
) -> dict[str, Any]:
    from minillm_forge.execution.gpu2b import load_frozen_tokenizer

    repo = Path(repo).resolve()
    protocol = _read_json(repo / PROTOCOL_PATH)
    run_dir = repo / "runs/gpu3c" / run_name / group
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trainer, metadata = build_trainer(repo, group, steps=steps, output_dir=run_dir)
    initial_adapter = metadata["initial_adapter_sha256"]
    checkpoints = sorted(
        set(step for step in protocol["training"]["checkpoint_steps"] if step <= steps) | {steps}
    )
    for checkpoint_step in checkpoints:
        trainer.train_until(checkpoint_step)
        trainer._save(f"checkpoints/step-{checkpoint_step:04d}.pt")
    final_adapter_state = {
        name: value for name, value in trainer.model.state_dict().items() if "lora_" in name
    }
    final_adapter = _digest_value(final_adapter_state)
    adapter_dir = run_dir / "final_adapter"
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_files = {
        path.name: file_sha256(path) for path in sorted(adapter_dir.iterdir()) if path.is_file()
    }
    records = []
    if evaluate:
        tokenizer = load_frozen_tokenizer()
        records = _evaluate_model(
            repo, trainer.model, tokenizer, group, protocol["evaluation"]["problem_ids"]
        )
        generation_path = run_dir / "generations.jsonl"
        with generation_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    state = trainer.state
    history = [row for row in state.history if row.get("event") == "train"]
    result = {
        "stage": "GPU-3C",
        "group": group,
        "status": "COMPLETED",
        "steps": state.global_step,
        "examples_seen": state.examples_seen,
        "input_tokens_seen": state.tokens_seen,
        "supervised_tokens_seen": state.target_tokens_seen,
        "final_loss": history[-1]["loss"] if history else None,
        "nan_count": state.nan_count,
        "inf_count": state.inf_count,
        "oom_count": state.oom_count,
        "initial_adapter_sha256": initial_adapter,
        "final_adapter_sha256": final_adapter,
        "adapter_changed": initial_adapter != final_adapter,
        "adapter_files": adapter_files,
        "peak_cuda_allocated_mib": trainer.peak_vram_mb(),
        "peak_cuda_reserved_mib": trainer.peak_reserved_vram_mb(),
        "wall_time_seconds": time.perf_counter() - started,
        "generation_records": len(records),
        "median_target_tokens_per_second": statistics.median(
            row["target_tokens_per_second"] for row in history
        )
        if history
        else None,
        "metadata": metadata,
    }
    _write_json(run_dir / "result.json", result)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_smoke(repo: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo).resolve()
    pair = audit_pair_identity(repo)
    results = []
    for group in ("B-EOS-MASKED", "C-EOS-SUPERVISED"):
        result = run_arm(repo, group, steps=1, evaluate=False, run_name="smoke")
        checkpoint = repo / "runs/gpu3c/smoke" / group / "checkpoints/step-0001.pt"
        trainer, _ = build_trainer(
            repo,
            group,
            steps=1,
            output_dir=repo / "runs/gpu3c/smoke-reload" / group,
        )
        load_checkpoint(
            checkpoint,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.amp.scaler,
            map_location=trainer._batch_device(),
        )
        from minillm_forge.execution.gpu2b import load_frozen_tokenizer

        tokenizer = load_frozen_tokenizer()
        historical = next(iter(_historical_eval_rows(repo).values()))
        trainer.model.gradient_checkpointing_disable()
        trainer.model.config.use_cache = True
        trainer.model.eval()
        generated = generate_one(
            trainer.model,
            tokenizer,
            [int(value) for value in historical["prompt_token_ids"]],
            eos_token_ids=[151643, IM_END_ID],
            max_new_tokens=2,
            wall_time_seconds=30,
        )
        result["checkpoint_reload"] = "PASS"
        result["generation_smoke"] = {
            "status": generated["status"],
            "generated_token_count": generated["generated_token_count"],
        }
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        results.append(result)
    summary = {"status": "PASS", "pair_audit": pair, "arms": results}
    _write_json(repo / "artifacts/gpu3c/smoke_result.json", summary)
    return summary


def run_formal_pair(repo: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo).resolve()
    protocol = _read_json(repo / PROTOCOL_PATH)
    pair = audit_pair_identity(repo)
    results = []
    for group in ("B-EOS-MASKED", "C-EOS-SUPERVISED"):
        results.append(
            run_arm(
                repo,
                group,
                steps=int(protocol["training"]["optimizer_steps"]),
                evaluate=True,
                run_name="formal-64",
            )
        )
    summary = {"status": "COMPLETED", "pair_audit": pair, "arms": results}
    _write_json(repo / "artifacts/gpu3c/formal_pair_result.json", summary)
    from minillm_forge.experiments_gpu3c.reporting import write_final_artifacts

    write_final_artifacts(repo, summary)
    return summary
