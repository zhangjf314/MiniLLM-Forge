from __future__ import annotations

# ruff: noqa: E501, I001

import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from minillm_forge.diagnostics.gpu3a import max_consecutive_repeat_span, ngram_metrics
from minillm_forge.evaluation.math_eval import extract_final_answer, normalize_answer
from minillm_forge.execution.gpu2b import load_frozen_tokenizer
from minillm_forge.execution.gpu2d_formal import _load_formal_model
from minillm_forge.experiments_gpu3b.core import (
    CONFIRMATION_IDS,
    DEVELOPMENT_IDS,
    SELECTED_ADAPTERS,
    extract_protocol,
    freeze_protocol,
    has_complete_answer,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _historical_rows(repo: Path, run_id: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    root = repo / "runs/gpu2d-formal" / run_id / "evaluation-amended"
    for path in (root / "math500.jsonl", root / "gsm8k-fixed-200.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    result[row["problem_id"]] = row
    return result


def _physical_memory() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).splitlines()[0]
        name, total, used, free, driver = [item.strip() for item in output.split(",")]
        return {
            "status": "PASS",
            "name": name,
            "total_mib": int(total),
            "used_mib": int(used),
            "free_mib": int(free),
            "driver": driver,
        }
    except Exception as exc:
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def _first_position(ids: list[int], token_id: int) -> int | None:
    try:
        return ids.index(token_id)
    except ValueError:
        return None


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    *,
    eos_token_ids: list[int],
    max_new_tokens: int,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    external_answer_stop: bool = False,
    wall_time_seconds: float = 120.0,
) -> dict[str, Any]:
    from transformers import NoRepeatNGramLogitsProcessor, RepetitionPenaltyLogitsProcessor

    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    generated: list[int] = []
    first_complete_answer_position: int | None = None
    past_key_values = None
    current_ids = input_ids
    stop_reason = "LENGTH_LIMIT"
    status = "COMPLETED"
    error = None
    processors = []
    if repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if no_repeat_ngram_size:
        processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for _step in range(max_new_tokens):
                if time.perf_counter() - started > wall_time_seconds:
                    stop_reason = "TIMEOUT"
                    status = "FAILED"
                    break
                output = model(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = output.past_key_values
                scores = output.logits[:, -1, :]
                full_ids = torch.tensor([prompt_ids + generated], dtype=torch.long, device=device)
                for processor in processors:
                    scores = processor(full_ids, scores)
                current_ids = scores.argmax(dim=-1, keepdim=True)
                token_id = int(current_ids.item())
                generated.append(token_id)
                if token_id in eos_token_ids:
                    stop_reason = "EOS_151643" if token_id == 151643 else "ASSISTANT_END_151645"
                    break
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=device,
                        ),
                    ),
                    dim=1,
                )
                if external_answer_stop:
                    decoded = tokenizer.decode(generated, skip_special_tokens=True)
                    if has_complete_answer(decoded):
                        first_complete_answer_position = len(generated) - 1
                        stop_reason = "EXTERNAL_ANSWER"
                        break
    except torch.OutOfMemoryError as exc:
        stop_reason, status, error = "OOM", "FAILED", f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        stop_reason, status, error = "ERROR", "FAILED", f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    if first_complete_answer_position is None and has_complete_answer(raw):
        # Incremental replay is exact because it uses the same token prefix; find the first token
        # whose decoded prefix satisfies the frozen structural completion rule.
        for index in range(len(generated)):
            if has_complete_answer(
                tokenizer.decode(generated[: index + 1], skip_special_tokens=True)
            ):
                first_complete_answer_position = index
                break
    first_valid = extract_protocol(raw, "first-valid-v1")
    last_valid = extract_protocol(raw, "last-valid-v1")
    conflict = extract_protocol(raw, "conflict-aware-v1")
    post_answer = (
        len(generated) - first_complete_answer_position - 1
        if first_complete_answer_position is not None
        else None
    )
    return {
        "status": status,
        "error": error,
        "generated_token_ids": generated,
        "raw_generation": raw,
        "first_151643_position": _first_position(generated, 151643),
        "first_151645_position": _first_position(generated, 151645),
        "stop_reason": stop_reason,
        "autonomous_eos_stop": stop_reason in {"EOS_151643", "ASSISTANT_END_151645"},
        "assistant_end_stop": stop_reason == "ASSISTANT_END_151645",
        "external_answer_stop": stop_reason == "EXTERNAL_ANSWER",
        "length_limit_stop": stop_reason == "LENGTH_LIMIT",
        "generated_token_count": len(generated),
        "first_complete_answer_token_position": first_complete_answer_position,
        "post_answer_token_count": post_answer,
        "post_answer_token_ratio": post_answer / len(generated)
        if post_answer is not None and generated
        else None,
        "repeat_3gram": ngram_metrics(generated, 3),
        "repeat_4gram": ngram_metrics(generated, 4),
        "max_consecutive_repeat_span": max_consecutive_repeat_span(generated),
        "legacy_extracted_answer": extract_final_answer(raw),
        "first_valid": first_valid,
        "last_valid": last_valid,
        "conflict_aware": conflict,
        "inference_wall_time": elapsed,
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        ),
    }


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_jobs(
    repo: str | Path,
    *,
    run_id: str,
    sample_ids: list[str],
    configs: list[dict[str, Any]],
    partition: str,
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    protocol = freeze_protocol(repo)
    if run_id not in SELECTED_ADAPTERS:
        raise ValueError(f"run_id is not frozen in GPU-3B protocol: {run_id}")
    allowed = set(DEVELOPMENT_IDS if partition == "development" else CONFIRMATION_IDS)
    if not set(sample_ids) <= allowed:
        raise ValueError(f"samples are outside frozen {partition} partition")
    output_path = repo / f"artifacts/gpu3b/generations/{partition}-{run_id}.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    existing[item["generation_id"]] = item
    source = _historical_rows(repo, run_id)
    summary = _read_json(repo / f"runs/gpu2d-formal/{run_id}/final_summary.json")
    pending = []
    for sample_id in sample_ids:
        if sample_id not in source:
            raise ValueError(f"sample {sample_id} is absent from historical run {run_id}")
        for config in configs:
            generation_id = f"gpu3b:{partition}:{run_id}:{sample_id}:{config['id']}"
            if generation_id not in existing:
                pending.append((generation_id, sample_id, config))
    if not pending:
        return {
            "status": "COMPLETE",
            "executed": 0,
            "resumed": len(existing),
            "path": str(output_path),
        }
    before = _physical_memory()
    if before.get("free_mib", 0) < protocol["resource_limits"]["minimum_headroom_mib"]:
        raise RuntimeError("insufficient GPU headroom before model load")
    model = _load_formal_model(summary)
    tokenizer = load_frozen_tokenizer()
    executed = 0
    for generation_id, sample_id, config in pending:
        historical = source[sample_id]
        result = generate_one(
            model,
            tokenizer,
            [int(value) for value in historical["prompt_token_ids"]],
            eos_token_ids=list(config["eos_token_ids"]),
            max_new_tokens=int(config["max_new_tokens"]),
            repetition_penalty=float(config.get("repetition_penalty", 1.0)),
            no_repeat_ngram_size=int(config.get("no_repeat_ngram_size", 0)),
            external_answer_stop=bool(config.get("external_answer_stop", False)),
            wall_time_seconds=float(
                protocol["resource_limits"]["per_generation_wall_time_seconds"]
            ),
        )
        result.update(
            {
                "generation_id": generation_id,
                "protocol_version": protocol["protocol_version"],
                "partition": partition,
                "decoding_protocol": config["id"],
                "generation_config": config,
                "model_id": protocol["model_id"],
                "run_id": run_id,
                "adapter_id": summary["adapter"]["state_sha256"],
                "checkpoint_id": summary["checkpoint"]["final_sha256"],
                "sample_id": sample_id,
                "benchmark": historical["benchmark"],
                "prompt_token_ids": historical["prompt_token_ids"],
                "reference_answer": historical["reference_answer"],
                "first_valid_correct": normalize_answer(result["first_valid"]["answer"] or "")
                == normalize_answer(historical["reference_answer"]),
                "last_valid_correct": normalize_answer(result["last_valid"]["answer"] or "")
                == normalize_answer(historical["reference_answer"]),
                "conflict_aware_correct": normalize_answer(result["conflict_aware"]["answer"] or "")
                == normalize_answer(historical["reference_answer"]),
                "historical_generated_token_ids": historical["generated_token_ids"],
                "historical_legacy_answer": historical["extracted_answer"],
                "historical_correct": historical["correct"],
                "prompt_ids_match_historical": True,
            }
        )
        _append(output_path, result)
        existing[generation_id] = result
        executed += 1
        if result["status"] == "FAILED":
            break
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "status": "COMPLETE" if executed == len(pending) else "PARTIAL",
        "executed": executed,
        "pending_initially": len(pending),
        "records_total": len(existing),
        "path": output_path.relative_to(repo).as_posix(),
        "memory_before": before,
        "memory_after": _physical_memory(),
    }


def config(
    identifier: str,
    *,
    stop: str = "S0",
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    stops = {
        "S0": ([151643, 151645], False),
        "S1": ([151643], False),
        "S2": ([151645], False),
        "S3": ([151643, 151645], True),
    }
    eos_ids, external = stops[stop]
    return {
        "id": identifier,
        "stop_protocol": stop,
        "eos_token_ids": eos_ids,
        "external_answer_stop": external,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
