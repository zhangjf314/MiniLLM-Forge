from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from minillm_forge.experiments_gpu4a.data import EOS_ID, TASKS, format_prompt
from minillm_forge.model import MiniLLM, MiniLLMConfig

CHECKPOINT = Path("runs/E01-minillm-formal/best.pt")
MODEL_MANIFEST = Path("artifacts/training/minillm_formal_model.json")
TOKENIZER_PATH = Path("artifacts/tokenizers/minillm-tokenizer.json")
TOKENIZER_MANIFEST = Path("artifacts/tokenizers/minillm-tokenizer-manifest.json")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_native_model(
    repo: str | Path, checkpoint: str | Path = CHECKPOINT, *, device: str = "cuda"
) -> tuple[MiniLLM, Any, dict[str, Any]]:
    from tokenizers import Tokenizer

    repo = Path(repo).resolve()
    manifest = read_json(repo / MODEL_MANIFEST)
    config = MiniLLMConfig(**manifest["architecture"])
    model = MiniLLM(config)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo / checkpoint_path
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.gradient_checkpointing = False
    model.to(device)
    model.eval()
    tokenizer = Tokenizer.from_file(str(repo / TOKENIZER_PATH))
    identity = {
        "path": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "global_step": payload["trainer_state"]["global_step"],
        "tokens_seen": payload["trainer_state"]["tokens_seen"],
        "best_validation_loss": payload["trainer_state"]["best_validation_loss"],
        "parameter_count": model.num_parameters(),
        "model_config": config.to_dict(),
        "tokenizer_sha256": file_sha256(repo / TOKENIZER_PATH),
    }
    return model, tokenizer, identity


@torch.no_grad()
def generate_record(
    model: MiniLLM,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_new_tokens: int = 12,
) -> dict[str, Any]:
    prompt = format_prompt(row["task"], row["input"])
    prompt_ids = [
        model.config.bos_token_id,
        *tokenizer.encode(prompt, add_special_tokens=False).ids,
    ]
    tensor = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    started = time.perf_counter()
    output = model.generate(tensor, max_new_tokens=max_new_tokens, temperature=0.0)
    elapsed = time.perf_counter() - started
    generated = output[0, len(prompt_ids) :].tolist()
    eos_position = generated.index(EOS_ID) if EOS_ID in generated else None
    content_ids = generated[:eos_position] if eos_position is not None else generated
    text = tokenizer.decode(content_ids, skip_special_tokens=True)
    extracted = text.strip().splitlines()[0].strip() if text.strip() else ""
    if row["task"] == TASKS[0]:
        format_valid = extracted in {"BILLING", "TECHNICAL", "ACCOUNT"}
    else:
        format_valid = re.fullmatch(r"-?\d+", extracted) is not None
    repeated = (
        len(content_ids) >= 2
        and len(set(zip(content_ids, content_ids[1:], strict=False))) < len(content_ids) - 1
    )
    return {
        "sample_id": row["sample_id"],
        "task": row["task"],
        "split": row["split"],
        "prompt": prompt,
        "generated_text": text,
        "generated_token_ids": generated,
        "stop_reason": "EOS" if eos_position is not None else "LENGTH_LIMIT",
        "generated_token_count": len(generated),
        "extracted_answer": extracted,
        "reference_answer": row["target"],
        "task_score": float(extracted == row["target"]),
        "format_validity": format_valid,
        "instruction_compliance": format_valid,
        "complete_answer": format_valid,
        "repetition": repeated,
        "inference_wall_time": elapsed,
    }


def evaluation_ids(repo: str | Path, *, per_task_split: int = 64) -> list[str]:
    repo = Path(repo).resolve()
    selected = []
    for split in ("train", "validation", "test"):
        rows = [
            json.loads(line)
            for line in (repo / f"data/processed/gpu4a/{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        for task in TASKS:
            selected.extend(
                row["sample_id"]
                for row in sorted(
                    (row for row in rows if row["task"] == task),
                    key=lambda item: item["sample_id"],
                )[:per_task_split]
            )
    return selected


def evaluate_generations(
    repo: str | Path,
    model: MiniLLM,
    tokenizer: Any,
    *,
    model_id: str,
    output_path: str | Path,
    per_task_split: int = 64,
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    wanted = set(evaluation_ids(repo, per_task_split=per_task_split))
    rows = []
    for split in ("train", "validation", "test"):
        rows.extend(
            row
            for row in (
                json.loads(line)
                for line in (repo / f"data/processed/gpu4a/{split}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            if row["sample_id"] in wanted
        )
    records = []
    for row in rows:
        record = generate_record(model, tokenizer, row)
        record["model_id"] = model_id
        records.append(record)
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = repo / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    grouped = {}
    for split in ("train", "validation", "test"):
        for task in TASKS:
            subset = [row for row in records if row["split"] == split and row["task"] == task]
            grouped[f"{split}:{task}"] = {
                "count": len(subset),
                "correct": sum(row["task_score"] for row in subset),
                "accuracy": statistics.mean(row["task_score"] for row in subset),
                "format_validity_rate": statistics.mean(row["format_validity"] for row in subset),
                "instruction_compliance_rate": statistics.mean(
                    row["instruction_compliance"] for row in subset
                ),
                "eos_stop_rate": statistics.mean(row["stop_reason"] == "EOS" for row in subset),
                "length_limit_rate": statistics.mean(
                    row["stop_reason"] == "LENGTH_LIMIT" for row in subset
                ),
                "complete_answer_rate": statistics.mean(row["complete_answer"] for row in subset),
                "invalid_output_rate": statistics.mean(
                    not row["format_validity"] for row in subset
                ),
                "repetition_rate": statistics.mean(row["repetition"] for row in subset),
                "generated_token_count_mean": statistics.mean(
                    row["generated_token_count"] for row in subset
                ),
            }
    return {
        "model_id": model_id,
        "records": len(records),
        "per_task_split": per_task_split,
        "metrics": grouped,
        "generation_path": str(destination),
        "wall_time_seconds": sum(row["inference_wall_time"] for row in records),
    }


@torch.no_grad()
def teacher_forced_metrics(model: MiniLLM, loader: DataLoader, device: str = "cuda") -> dict:
    was_training = model.training
    model.eval()
    loss_sum, targets, correct = 0.0, 0, 0
    for raw in loader:
        batch = {
            key: value.to(device) for key, value in raw.items() if isinstance(value, torch.Tensor)
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**batch)
        shifted_labels = batch["labels"][:, 1:]
        mask = shifted_labels.ne(-100)
        count = int(mask.sum())
        predictions = output.logits[:, :-1].argmax(dim=-1)
        correct += int((predictions[mask] == shifted_labels[mask]).sum())
        loss_sum += float(output.loss) * count
        targets += count
    model.train(was_training)
    return {
        "loss": loss_sum / targets,
        "perplexity": math.exp(loss_sum / targets),
        "target_token_accuracy": correct / targets,
        "targets": targets,
    }


def release(model: MiniLLM) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()
