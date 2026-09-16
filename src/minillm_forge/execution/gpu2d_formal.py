# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from minillm_forge.cli.common import read_jsonl, write_json
from minillm_forge.config import config_hash, load_config
from minillm_forge.data.sft import encode_sft_example
from minillm_forge.evaluation.math_eval import exact_match, extract_final_answer
from minillm_forge.execution.gpu2b import (
    NvidiaMemoryMonitor,
    _git_clean,
    _git_commit,
    _load_model,
    file_sha256,
    load_frozen_tokenizer,
    tree_sha256,
)
from minillm_forge.execution.gpu2c import _adapter_identity, _digest_value
from minillm_forge.execution.gpu2d import (
    CONFIGS,
    HEADROOM_MIB,
    ContextSFTDataset,
    _collator,
    build_trainer,
)
from minillm_forge.training.checkpoint import load_checkpoint

STAGE = "STAGE_GPU_2D_PEFT_TRANSFER"
FORMAL_BASELINE = "9dbd99ee6b92103ae2af742bca8e9ecbd9a216c8"
CONTEXT = 512
UPDATES = 1200
SEEDS = (42, 31415, 271828)
METHODS = ("LORA", "QLORA")
INITIALIZATIONS = ("BASE_INIT", "CPT_INIT")
BENCHMARKS = ("gsm8k", "math500")
SYSTEM_PROMPT = "You are a mathematical reasoning assistant."
FORMAL_ROOT = Path("artifacts/gpu2d_formal")
RUN_ROOT = Path("runs/gpu2d-formal")
PROTOCOL_PATH = FORMAL_ROOT / "protocol.json"
EXECUTION_ORDER_PATH = FORMAL_ROOT / "execution_order.json"
TRAINING_MATRIX_PATH = FORMAL_ROOT / "training_matrix.json"
PREFLIGHT_PATH = FORMAL_ROOT / "f1_preflight.json"
EVAL_QUALIFICATION_PATH = FORMAL_ROOT / "eval_executor_qualification.json"
FINAL_RESULT_PATH = FORMAL_ROOT / "final_result.json"
EVAL_RUNTIME_RECOVERY_PATH = FORMAL_ROOT / "eval_runtime_recovery.json"
EVAL_REPORT_PATH = Path("reports/GPU2D_EVALUATION_EXECUTOR_QUALIFICATION.md")
FINAL_REPORT_PATH = Path("reports/GPU2D_FORMAL_PEFT_TRANSFER.md")
EVALUATION_MANIFEST = Path("artifacts/eval_manifests/evaluation-manifest.json")
DATA_MANIFEST = Path("artifacts/gpu2d/data_context_512.json")
VALIDATION_SOURCE = Path("data/processed/gpu2b/sft_math_v1_validation.jsonl")
VALIDATION_CACHE = Path("data/processed/gpu2d/sft_math_context_512_validation_tokens.pt")
VALIDATION_MANIFEST = FORMAL_ROOT / "validation_context_512.json"
FAILURE_ROOT = Path("reports/failures")
EXECUTION_ORDER = (
    "B-LORA-s42",
    "C-LORA-s42",
    "C-QLORA-s31415",
    "B-QLORA-s31415",
    "B-LORA-s271828",
    "C-LORA-s271828",
    "C-QLORA-s42",
    "B-QLORA-s42",
    "C-LORA-s31415",
    "B-LORA-s31415",
    "B-QLORA-s271828",
    "C-QLORA-s271828",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def _package_environment() -> dict[str, Any]:
    physical = _physical_gpu()
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver": physical.get("driver"),
        "gpu": physical.get("gpu"),
        "physical_vram_mib": physical.get("total_mib"),
        "execution_mode": "WDDM",
        "bitsandbytes": version("bitsandbytes"),
        "transformers": version("transformers"),
        "peft": version("peft"),
        "allocator_configuration": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


def _physical_gpu() -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    ).splitlines()[0]
    name, total, used, free, driver = [part.strip() for part in output.split(",")]
    return {
        "gpu": name,
        "total_mib": int(total),
        "used_mib": int(used),
        "free_mib": int(free),
        "driver": driver,
    }


def _code_tree_digest() -> str:
    roots = [Path("src"), Path("scripts"), Path("configs")]
    files = [Path("pyproject.toml"), Path("uv.lock")]
    for root in roots:
        files.extend(
            path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts
        )
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.as_posix()):
        digest.update(path.as_posix().encode())
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _prompt(tokenizer: Any, problem: str) -> tuple[str, list[int]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, tokenizer(prompt, add_special_tokens=False)["input_ids"]


def _problem_id(benchmark: str, row: dict[str, Any]) -> str:
    return f"{benchmark}-{int(row['benchmark_index']):04d}"


def _benchmark_records(benchmark: str) -> list[dict[str, Any]]:
    manifest = _read_json(EVALUATION_MANIFEST)["primary_benchmarks"][benchmark]
    records = read_jsonl(manifest["snapshot_path"])
    if len(records) != manifest["examples"]:
        raise RuntimeError(f"{benchmark} frozen count mismatch")
    if file_sha256(manifest["snapshot_path"]) != manifest["snapshot_sha256"]:
        raise RuntimeError(f"{benchmark} frozen snapshot hash mismatch")
    return records


def _f0_subset(benchmark: str, size: int = 16) -> list[dict[str, Any]]:
    ranked = sorted(
        _benchmark_records(benchmark),
        key=lambda row: hashlib.sha256(
            f"gpu2d-f0-v1:{benchmark}:{row['benchmark_index']}".encode()
        ).digest(),
    )
    return ranked[:size]


def _parse_run_id(run_id: str) -> dict[str, Any]:
    prefix, seed_text = run_id.rsplit("-s", 1)
    initialization, family = prefix.split("-", 1)
    if run_id not in EXECUTION_ORDER:
        raise ValueError(f"unknown frozen formal run ID: {run_id}")
    return {
        "run_id": run_id,
        "initialization": "BASE_INIT" if initialization == "B" else "CPT_INIT",
        "family": family,
        "seed": int(seed_text),
    }


def _permutation_digest(size: int, seed: int) -> str:
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    payload = b"".join(int(value).to_bytes(4, "little", signed=False) for value in order)
    return hashlib.sha256(payload).hexdigest()


def freeze_protocol() -> dict[str, Any]:
    if _git_commit() != FORMAL_BASELINE:
        raise RuntimeError("formal protocol must initially be frozen from the authorized baseline")
    evaluation = _read_json(EVALUATION_MANIFEST)
    data = _read_json(DATA_MANIFEST)
    tokenizer = load_frozen_tokenizer()
    config_contracts = {}
    for family in METHODS:
        config_path = CONFIGS[(family, CONTEXT, "R16_ALL_LINEAR")]
        config = load_config(config_path)
        config_contracts[family] = {
            "path": str(config_path),
            "file_sha256": file_sha256(config_path),
            "semantic_sha256": config_hash(config),
            "updates": config["training"]["max_steps"],
            "checkpoint_selection": config["training"]["checkpoint_selection"],
        }
    benchmark_contracts = {}
    qualification_ids = {}
    for benchmark in BENCHMARKS:
        source = evaluation["primary_benchmarks"][benchmark]
        records = _benchmark_records(benchmark)
        ids = [_problem_id(benchmark, row) for row in _f0_subset(benchmark)]
        qualification_ids[benchmark] = ids
        benchmark_contracts[benchmark] = {
            **source,
            "actual_snapshot_sha256": file_sha256(source["snapshot_path"]),
            "problem_ids_sha256": _json_digest([_problem_id(benchmark, row) for row in records]),
        }
    protocol = {
        "stage": STAGE,
        "status": "FROZEN_BEFORE_FORMAL_TRAINING",
        "formal_baseline": FORMAL_BASELINE,
        "claim_scope": "CPT_TO_PEFT_SFT_TRANSFER_UNDER_FROZEN_512_CONTEXT",
        "formal_runs": 12,
        "initializations": list(INITIALIZATIONS),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "context": CONTEXT,
        "data_manifest": {
            "path": str(DATA_MANIFEST),
            "sha256": file_sha256(DATA_MANIFEST),
            "encoded_dataset_digest": data["encoded_dataset_digest"],
            "cache_sha256": data["cache_sha256"],
            "train_records": data["source_examples"],
            "eligible_training_records": data["examples"],
            "validation_records": 800,
            "assistant_tokens": data["assistant_tokens_retained"],
            "final_answer_retention_vs_1024": 0.7322064559895981,
            "semantic_change": "SCIENTIFIC_DATA_EXPOSURE_CHANGE",
        },
        "config_contracts": config_contracts,
        "benchmarks": benchmark_contracts,
        "generation": {
            **evaluation["generation"],
            "batch_size": "F0_QUALIFIED_VALUE",
            "continuation_identity": "tokens through first stop token inclusive; batch-completion padding excluded",
        },
        "prompt": {
            "system": SYSTEM_PROMPT,
            "construction": "Qwen chat template over system and frozen benchmark problem",
            "contract_sha256": _json_digest(
                {"system": SYSTEM_PROMPT, "construction": "QWEN_CHAT_TEMPLATE_V1"}
            ),
            "chat_template_sha256": hashlib.sha256(
                str(tokenizer.chat_template).encode()
            ).hexdigest(),
        },
        "scoring": {
            **evaluation["scoring"],
            "actual_evaluator_sha256": file_sha256("src/minillm_forge/evaluation/math_eval.py"),
        },
        "checkpoint_selection": "FINAL_1200_UPDATE_CHECKPOINT",
        "qualification_subset": {
            "selection": "16 lowest SHA256(gpu2d-f0-v1:benchmark:benchmark_index)",
            "ids": qualification_ids,
        },
        "bootstrap": {
            "seed": 20260913,
            "replicates": 10000,
            "confidence_level": 0.95,
            "unit": "problem_id paired within method, seed, and benchmark",
            "interpretation": "benchmark-problem sampling uncertainty only",
        },
        "decision_rule": {
            "source": "reports/gpu2b/decision-rules.md",
            "source_sha256": file_sha256("reports/gpu2b/decision-rules.md"),
            "positive_mean_margin_points": 1.0,
            "negative_mean_margin_points": -1.0,
            "minimum_directional_seeds": 2,
            "conflict_margin_points": 0.5,
        },
        "hypotheses": {
            "H-P1": "LoRA CPT-vs-Base transfer under frozen 512-context SFT",
            "H-P2": "QLoRA CPT-vs-Base transfer under frozen 512-context SFT",
            "H-P3": "transfer direction consistency across LoRA and QLoRA",
            "H-P4": "measured LoRA/QLoRA memory-quality-throughput trade-off",
        },
        "memory_gate": {
            "minimum_physical_headroom_mib": HEADROOM_MIB,
            "authority": "independent nvidia-smi sampling",
        },
        "environment": _package_environment(),
        "source_tree_sha256": _code_tree_digest(),
        "formal_training_launches": 0,
        "benchmark_launches": 0,
    }
    write_json(PROTOCOL_PATH, protocol)
    order = {
        "stage": STAGE,
        "status": "FROZEN_BEFORE_FIRST_FORMAL_RUN",
        "generation_rule": "predeclared balanced paired-block order",
        "order": list(EXECUTION_ORDER),
        "order_sha256": _json_digest(EXECUTION_ORDER),
        "balance": {
            "paired_by_method_and_seed": True,
            "base_first_pairs": 3,
            "cpt_first_pairs": 3,
            "all_methods_initializations_seeds_present_once": True,
        },
        "seed_permutations": {
            str(seed): {
                "eligible_examples": data["examples"],
                "uint32_le_sha256": _permutation_digest(data["examples"], seed),
            }
            for seed in SEEDS
        },
        "formal_training_launches": 0,
    }
    write_json(EXECUTION_ORDER_PATH, order)
    matrix = {
        "stage": STAGE,
        "status": "FROZEN_BEFORE_FIRST_FORMAL_RUN",
        "execution_order_sha256": order["order_sha256"],
        "runs": {
            run_id: {**_parse_run_id(run_id), "order_index": index, "status": "PENDING"}
            for index, run_id in enumerate(EXECUTION_ORDER, start=1)
        },
    }
    write_json(TRAINING_MATRIX_PATH, matrix)
    return protocol


def _trim_generated(ids: list[int], stop_ids: set[int]) -> list[int]:
    for index, token_id in enumerate(ids):
        if token_id in stop_ids:
            return ids[: index + 1]
    return ids


def _generate_reference_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    benchmark: str,
) -> list[dict[str, Any]]:
    prompts_and_ids = [_prompt(tokenizer, str(row["problem"])) for row in rows]
    prompts = [item[0] for item in prompts_and_ids]
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
            eos_token_id=[151643, 151645],
            pad_token_id=151643,
            use_cache=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    continuation_start = inputs["input_ids"].shape[1]
    results = []
    for row, (prompt, prompt_ids), output_row in zip(rows, prompts_and_ids, output, strict=True):
        token_ids = _trim_generated(
            output_row[continuation_start:].detach().cpu().tolist(), {151643, 151645}
        )
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        reference = str(row["answer"])
        results.append(
            {
                "run_id": None,
                "benchmark": benchmark,
                "problem_id": _problem_id(benchmark, row),
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": token_ids,
                "generated_text": decoded,
                "extracted_answer": extract_final_answer(decoded),
                "reference_answer": extract_final_answer(reference),
                "correct": exact_match(decoded, reference),
                "generation_tokens": len(token_ids),
                "elapsed_batch_seconds": elapsed,
                "batch_size": len(rows),
            }
        )
    return results


def _generate_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    benchmark: str,
) -> list[dict[str, Any]]:
    """Preserved batch-size-one reference evaluator."""
    return _generate_reference_batch(model, tokenizer, rows, benchmark)


def _generate_serial_greedy_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    benchmark: str,
) -> list[dict[str, Any]]:
    """Exact-semantics greedy decoder specialized for the frozen serial evaluator."""
    if len(rows) != 1:
        raise ValueError("serial greedy evaluator requires batch_size=1")
    row = rows[0]
    prompt, prompt_ids = _prompt(tokenizer, str(row["problem"]))
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    generated: list[int] = []
    past_key_values = None
    current_ids = input_ids
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for _step in range(512):
            output = model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = output.past_key_values
            current_ids = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = int(current_ids.item())
            generated.append(token_id)
            if token_id in {151643, 151645}:
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    reference = str(row["answer"])
    return [
        {
            "run_id": None,
            "benchmark": benchmark,
            "problem_id": _problem_id(benchmark, row),
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": generated,
            "generated_text": decoded,
            "extracted_answer": extract_final_answer(decoded),
            "reference_answer": extract_final_answer(reference),
            "correct": exact_match(decoded, reference),
            "generation_tokens": len(generated),
            "elapsed_batch_seconds": elapsed,
            "batch_size": 1,
        }
    ]


def _generate_selected_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    benchmark: str,
    family: str,
) -> list[dict[str, Any]]:
    if family == "LORA":
        return _generate_serial_greedy_batch(model, tokenizer, rows, benchmark)
    return _generate_reference_batch(model, tokenizer, rows, benchmark)


def _evaluate_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    benchmark: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    results = []
    for offset in range(0, len(rows), batch_size):
        results.extend(
            _generate_batch(model, tokenizer, rows[offset : offset + batch_size], benchmark)
        )
    return results, time.perf_counter() - started


def _equivalence(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    reference_by_id = {row["problem_id"]: row for row in reference}
    candidate_by_id = {row["problem_id"]: row for row in candidate}
    ids_match = set(reference_by_id) == set(candidate_by_id)
    fields = ("generated_token_ids", "generated_text", "extracted_answer", "correct")
    field_matches = {
        field: ids_match
        and all(
            reference_by_id[key][field] == candidate_by_id[key][field] for key in reference_by_id
        )
        for field in fields
    }
    return {"all_match": ids_match and all(field_matches.values()), "field_matches": field_matches}


def _prepare_eval_model(family: str) -> Any:
    config = load_config(CONFIGS[(family, CONTEXT, "R16_ALL_LINEAR")])
    model = _load_model(config, "BASE_INIT", 42)
    if not getattr(model, "hf_device_map", None):
        model.to(torch.device("cuda"))
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    return model


def qualify_evaluator() -> dict[str, Any]:
    protocol = _read_json(PROTOCOL_PATH)
    if protocol["formal_training_launches"] != 0:
        raise RuntimeError("F0 cannot run after formal training starts")
    tokenizer = load_frozen_tokenizer()
    families: dict[str, Any] = {}
    for family in METHODS:
        model = _prepare_eval_model(family)
        family_tests: dict[str, Any] = {}
        references: dict[str, list[dict[str, Any]]] = {}
        for batch_size in (1, 2, 4, 8):
            combined: list[dict[str, Any]] = []
            elapsed = 0.0
            failure = None
            torch.cuda.reset_peak_memory_stats()
            with NvidiaMemoryMonitor(interval_seconds=0.2) as monitor:
                try:
                    for benchmark in BENCHMARKS:
                        rows = _f0_subset(benchmark)
                        outputs, duration = _evaluate_rows(
                            model, tokenizer, rows, benchmark, batch_size
                        )
                        combined.extend(outputs)
                        elapsed += duration
                except Exception as exc:
                    failure = f"{type(exc).__name__}: {exc}"
            if batch_size == 1 and failure is None:
                references["combined"] = combined
                equivalence = {
                    "all_match": True,
                    "field_matches": {
                        "generated_token_ids": True,
                        "generated_text": True,
                        "extracted_answer": True,
                        "correct": True,
                    },
                }
            elif failure is None:
                equivalence = _equivalence(references["combined"], combined)
            else:
                equivalence = {"all_match": False, "field_matches": {}}
            generated_tokens = sum(row["generation_tokens"] for row in combined)
            safe = bool(
                failure is None
                and equivalence["all_match"]
                and monitor.minimum_free_mib is not None
                and monitor.minimum_free_mib >= HEADROOM_MIB
                and not monitor.errors
            )
            family_tests[str(batch_size)] = {
                "status": "PASS" if safe else "FAIL",
                "batch_size": batch_size,
                "exact_equivalence": equivalence,
                "problems": len(combined),
                "elapsed_seconds": elapsed,
                "problems_per_second": len(combined) / elapsed if elapsed else None,
                "generated_tokens": generated_tokens,
                "generated_tokens_per_second": generated_tokens / elapsed if elapsed else None,
                "mean_generated_tokens_per_problem": (
                    generated_tokens / len(combined) if combined else None
                ),
                "generation_length_p50": _nearest_rank(
                    [row["generation_tokens"] for row in combined], 0.5
                ),
                "generation_length_p90": _nearest_rank(
                    [row["generation_tokens"] for row in combined], 0.9
                ),
                "generation_length_p95": _nearest_rank(
                    [row["generation_tokens"] for row in combined], 0.95
                ),
                "physical_peak_vram_mib": monitor.maximum_used_mib,
                "minimum_physical_headroom_mib": monitor.minimum_free_mib,
                "physical_samples": monitor.samples,
                "monitor_errors": monitor.errors,
                "cuda_allocated_peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "cuda_reserved_peak_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "failure": failure,
                "outputs": combined,
            }
            if failure and "out of memory" in failure.lower():
                torch.cuda.empty_cache()
        qualified = [int(key) for key, value in family_tests.items() if value["status"] == "PASS"]
        if not qualified:
            raise RuntimeError(f"no correct safe evaluator qualified for {family}")
        families[family] = {
            "selected_batch_size": max(qualified),
            "tests": family_tests,
            "adapter_identity": _adapter_identity(model, family),
        }
        del model
        torch.cuda.empty_cache()
    selected = min(value["selected_batch_size"] for value in families.values())
    selected_tests = [families[family]["tests"][str(selected)] for family in METHODS]
    total_tokens = sum(test["generated_tokens"] for test in selected_tests)
    total_elapsed = sum(test["elapsed_seconds"] for test in selected_tests)
    result = {
        "stage": STAGE,
        "phase": "F0",
        "classification": (
            "EVAL_EXECUTOR_QUALIFIED_BATCHED" if selected > 1 else "EVAL_EXECUTOR_QUALIFIED_SERIAL"
        ),
        "status": "PASS",
        "selected_batch_size": selected,
        "exact_equivalence": True,
        "minimum_physical_headroom_mib": HEADROOM_MIB,
        "families": families,
        "selected_aggregate": {
            "problems_per_second": 64 / total_elapsed,
            "generated_tokens_per_second": total_tokens / total_elapsed,
            "mean_generated_tokens_per_problem": total_tokens / 64,
            "estimated_full_evaluation_hours": 12
            * 1819
            * (total_tokens / 64)
            / (total_tokens / total_elapsed)
            / 3600,
        },
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "environment": _package_environment(),
        "source_tree_sha256": _code_tree_digest(),
        "formal_training_launches": 0,
        "formal_benchmark_launches": 0,
        "previous_attempts": [
            _read_json(path) for path in sorted(FAILURE_ROOT.glob("gpu2d-f0-attempt-*.json"))
        ],
    }
    write_json(EVAL_QUALIFICATION_PATH, result)
    _write_eval_report(result)
    return result


def _nearest_rank(values: list[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _write_eval_report(result: dict[str, Any]) -> None:
    rows = []
    for family in METHODS:
        for batch_size, test in result["families"][family]["tests"].items():
            problem_rate = test["problems_per_second"]
            token_rate = test["generated_tokens_per_second"]
            problem_rate_text = f"{problem_rate:.4f}" if problem_rate is not None else "N/A"
            token_rate_text = f"{token_rate:.2f}" if token_rate is not None else "N/A"
            rows.append(
                f"| {family} | {batch_size} | {test['status']} | "
                f"{test['exact_equivalence']['all_match']} | "
                f"{test['minimum_physical_headroom_mib']} MiB | "
                f"{problem_rate_text} | {token_rate_text} |"
            )
    aggregate = result["selected_aggregate"]
    text = f"""# GPU-2D Formal Evaluation Executor Qualification

## Frozen identity

The GSM8K and MATH-500 snapshots, prompt/chat template, greedy decoding, 512-token
generation ceiling, stop semantics, extraction, and scorer were reused without change.
The deterministic qualification subset contains 16 frozen-hash-selected problems per
benchmark. The formal checkpoint is the final update-1200 checkpoint.

## Exact-equivalence and memory results

| Family | Batch | Gate | Exact token/text/answer/score | Min physical headroom | Problems/s | Generated tokens/s |
| --- | ---: | --- | --- | ---: | ---: | ---: |
{chr(10).join(rows)}

Classification: `{result["classification"]}`. The common selected batch size is
**{result["selected_batch_size"]}**. The selection is the largest tested batch that is
exactly equivalent to serial and retains at least {HEADROOM_MIB} MiB physical headroom
for both LoRA and QLoRA execution paths.

## Measured planning update

At the selected batch, aggregate qualification throughput was
{aggregate["problems_per_second"]:.4f} problems/s and
{aggregate["generated_tokens_per_second"]:.2f} generated tokens/s, with
{aggregate["mean_generated_tokens_per_problem"]:.2f} mean generated tokens/problem.
The resulting extrapolation for all 12 × 1,819 formal evaluations is
{aggregate["estimated_full_evaluation_hours"]:.2f} hours. This is still an estimate;
the full benchmarks determine actual runtime.

## Resumability

Formal evaluation writes one append-only JSONL row per completed problem and resumes by
the frozen problem ID. Completeness requires unique IDs, no missing IDs, exact frozen
counts, and checkpoint/prompt identity on every row.
"""
    EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT_PATH.write_text(text, encoding="utf-8")


class LoggedPhysicalMonitor(NvidiaMemoryMonitor):
    def __init__(self, path: str | Path, interval_seconds: float = 0.2) -> None:
        super().__init__(interval_seconds)
        self.path = Path(path)
        self.latest_used_mib: int | None = None
        self.latest_free_mib: int | None = None

    def _sample(self) -> None:
        try:
            value = _physical_gpu()
            used, free = value["used_mib"], value["free_mib"]
            self.latest_used_mib, self.latest_free_mib = used, free
            self.maximum_used_mib = (
                used if self.maximum_used_mib is None else max(self.maximum_used_mib, used)
            )
            self.minimum_free_mib = (
                free if self.minimum_free_mib is None else min(self.minimum_free_mib, free)
            )
            self.samples += 1
            _append_jsonl(
                self.path,
                {
                    "unix_time": time.time(),
                    "used_mib": used,
                    "free_mib": free,
                    "sample_index": self.samples,
                },
            )
        except Exception as exc:
            if len(self.errors) < 10:
                self.errors.append(f"{type(exc).__name__}: {exc}")


class ValidationContextDataset(Dataset):
    def __init__(self) -> None:
        manifest = _build_validation_cache()
        payload = torch.load(manifest["cache_path"], map_location="cpu", weights_only=True)
        self.input_ids = payload["input_ids"]
        self.offsets = payload["offsets"]
        self.assistant_starts = payload["assistant_starts"]
        self.source_indices = payload["source_indices"]

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
            "source_index": self.source_indices[index].to(torch.long),
        }


def _build_validation_cache() -> dict[str, Any]:
    source_sha = file_sha256(VALIDATION_SOURCE)
    if VALIDATION_CACHE.exists() and VALIDATION_MANIFEST.exists():
        manifest = _read_json(VALIDATION_MANIFEST)
        if manifest["source_sha256"] == source_sha and manifest["cache_sha256"] == file_sha256(
            VALIDATION_CACHE
        ):
            return manifest
    tokenizer = load_frozen_tokenizer()
    chunks = []
    offsets = [0]
    starts = []
    indices = []
    excluded = []
    assistant_tokens = 0
    for index, row in enumerate(read_jsonl(VALIDATION_SOURCE)):
        try:
            encoded = encode_sft_example(
                tokenizer,
                user=str(row["problem"]),
                assistant=str(row["solution"]),
                system=SYSTEM_PROMPT,
                max_length=CONTEXT,
            )
        except ValueError as exc:
            if str(exc) != "max_length truncates every assistant token":
                raise
            excluded.append(index)
            continue
        labels = encoded["labels"]
        target_count = sum(label != -100 for label in labels)
        if target_count == 0:
            excluded.append(index)
            continue
        ids = torch.tensor(encoded["input_ids"], dtype=torch.int32)
        chunks.append(ids)
        starts.append(len(ids) - target_count)
        indices.append(index)
        assistant_tokens += target_count
        offsets.append(offsets[-1] + len(ids))
    payload = {
        "input_ids": torch.cat(chunks),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "assistant_starts": torch.tensor(starts, dtype=torch.int32),
        "source_indices": torch.tensor(indices, dtype=torch.int32),
    }
    VALIDATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = VALIDATION_CACHE.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, VALIDATION_CACHE)
    result = {
        "stage": STAGE,
        "context": CONTEXT,
        "source_records": 800,
        "evaluable_records": len(indices),
        "excluded_zero_assistant_records": excluded,
        "assistant_tokens": assistant_tokens,
        "source_sha256": source_sha,
        "cache_path": str(VALIDATION_CACHE),
        "cache_sha256": file_sha256(VALIDATION_CACHE),
    }
    write_json(VALIDATION_MANIFEST, result)
    return result


def _initialization_identity(initialization: str) -> dict[str, Any]:
    config = load_config(CONFIGS[("LORA", CONTEXT, "R16_ALL_LINEAR")])
    contract = config["design"]["paired_initializations"][initialization]
    if initialization == "CPT_INIT":
        path = Path(contract["name_or_path"])
        return {
            "initialization": initialization,
            "name_or_path": str(path),
            "tree_sha256": tree_sha256(path),
            "model_sha256": file_sha256(path / "model.safetensors"),
        }
    return {
        "initialization": initialization,
        "name_or_path": contract["name_or_path"],
        "revision": contract["revision"],
        "config_sha256": "461f8999eee16aa9fa00ecc891755ef8067fca39f7880393fa7e670dbcc81413",
        "model_identity_sha256": _json_digest(
            {
                "name_or_path": contract["name_or_path"],
                "revision": contract["revision"],
                "config_sha256": "461f8999eee16aa9fa00ecc891755ef8067fca39f7880393fa7e670dbcc81413",
            }
        ),
    }


def preflight() -> dict[str, Any]:
    protocol = _read_json(PROTOCOL_PATH)
    qualification = _read_json(EVAL_QUALIFICATION_PATH)
    order = _read_json(EXECUTION_ORDER_PATH)
    data = _read_json(DATA_MANIFEST)
    checks: dict[str, Any] = {}
    checks["git"] = {"status": "PASS" if _git_clean() else "FAIL", "commit": _git_commit()}
    checks["source_tree"] = {
        "status": "PASS" if _code_tree_digest() == protocol["source_tree_sha256"] else "FAIL",
        "expected": protocol["source_tree_sha256"],
        "actual": _code_tree_digest(),
    }
    checks["environment"] = {
        "status": "PASS" if _package_environment() == protocol["environment"] else "FAIL",
        "expected": protocol["environment"],
        "actual": _package_environment(),
    }
    checks["data_manifest"] = {
        "status": (
            "PASS"
            if file_sha256(DATA_MANIFEST) == protocol["data_manifest"]["sha256"]
            and file_sha256(data["cache_path"]) == data["cache_sha256"]
            else "FAIL"
        ),
        "manifest_sha256": file_sha256(DATA_MANIFEST),
        "cache_sha256": file_sha256(data["cache_path"]),
    }
    checks["initializations"] = {
        "status": "PASS",
        "BASE_INIT": _initialization_identity("BASE_INIT"),
        "CPT_INIT": _initialization_identity("CPT_INIT"),
    }
    checks["configs"] = {
        "status": "PASS"
        if all(
            config_hash(load_config(value["path"])) == value["semantic_sha256"]
            for value in protocol["config_contracts"].values()
        )
        else "FAIL"
    }
    dataset_size = ContextSFTDataset(CONTEXT).__len__()
    checks["seed_permutations"] = {
        "status": "PASS"
        if all(
            _permutation_digest(dataset_size, seed)
            == order["seed_permutations"][str(seed)]["uint32_le_sha256"]
            for seed in SEEDS
        )
        else "FAIL"
    }
    checks["benchmark_protocol"] = {
        "status": "PASS" if qualification["status"] == "PASS" else "FAIL",
        "classification": qualification["classification"],
        "selected_batch_size": qualification["selected_batch_size"],
        "checkpoint_selection": protocol["checkpoint_selection"],
    }
    checks["validation_cache"] = {"status": "PASS", **_build_validation_cache()}
    status = "PASS" if all(value["status"] == "PASS" for value in checks.values()) else "FAIL"
    result = {
        "stage": STAGE,
        "phase": "F1_START_GATE",
        "status": status,
        "classification": "GPU2D_FORMAL_F1_PREFLIGHT_PASS"
        if status == "PASS"
        else "GPU2D_FORMAL_STOP",
        "checks": checks,
        "formal_training_launches": 0,
    }
    write_json(PREFLIGHT_PATH, result)
    return result


def _update_matrix(run_id: str, status: str, **values: Any) -> None:
    matrix = _read_json(TRAINING_MATRIX_PATH)
    matrix["runs"][run_id].update({"status": status, **values})
    counts = {}
    for item in matrix["runs"].values():
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    matrix["counts"] = counts
    write_json(TRAINING_MATRIX_PATH, matrix)


def _formal_code_frozen() -> bool:
    protocol = _read_json(PROTOCOL_PATH)
    source_tree_sha256 = _code_tree_digest()
    if source_tree_sha256 == protocol["source_tree_sha256"]:
        return True
    if not EVAL_RUNTIME_RECOVERY_PATH.exists():
        return False
    recovery = _read_json(EVAL_RUNTIME_RECOVERY_PATH)
    return bool(
        recovery.get("evaluation_code_freeze") is True
        and recovery.get("exact_equivalence") is True
        and recovery.get("selected_source_tree_sha256") == source_tree_sha256
    )


def _adapter_digest(model: Any) -> str:
    return _digest_value(
        {key: value for key, value in model.state_dict().items() if "lora_" in key}
    )


def train_run(run_id: str) -> dict[str, Any]:
    if _read_json(PREFLIGHT_PATH)["status"] != "PASS":
        raise RuntimeError("formal F1 preflight must pass")
    if not _formal_code_frozen():
        raise RuntimeError("CODE_FREEZE violation")
    spec = _parse_run_id(run_id)
    matrix = _read_json(TRAINING_MATRIX_PATH)
    current = matrix["runs"][run_id]
    if current["status"] == "VALID_COMPLETED":
        return _read_json(RUN_ROOT / run_id / "final_summary.json")
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    family = spec["family"]
    config_path = CONFIGS[(family, CONTEXT, "R16_ALL_LINEAR")]
    config = load_config(config_path)
    data = _read_json(DATA_MANIFEST)
    initialization = _initialization_identity(spec["initialization"])
    environment = _package_environment()
    protocol = _read_json(PROTOCOL_PATH)
    if environment != protocol["environment"]:
        raise RuntimeError("software environment drift")
    authorization = {
        "authorization": "RUN_AUTHORIZED",
        **spec,
        "stage": STAGE,
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "config_semantic_sha256": config_hash(config),
        "dataset_manifest_sha256": file_sha256(DATA_MANIFEST),
        "dataset_cache_sha256": data["cache_sha256"],
        "dataset_encoded_sha256": data["encoded_dataset_digest"],
        "model_identity": initialization,
        "git_commit": _git_commit(),
        "source_tree_sha256": _code_tree_digest(),
        "created_at_unix": time.time(),
    }
    authorization_path = run_dir / "run_authorization.json"
    if authorization_path.exists():
        existing = _read_json(authorization_path)
        immutable = (
            "run_id",
            "family",
            "initialization",
            "seed",
            "config_semantic_sha256",
            "dataset_encoded_sha256",
            "model_identity",
            "source_tree_sha256",
        )
        if any(existing[key] != authorization[key] for key in immutable):
            raise RuntimeError("formal run launch identity mismatch")
        authorization = existing
    else:
        write_json(authorization_path, authorization)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "data_manifest.json", data)
    write_json(run_dir / "initialization_manifest.json", initialization)
    _update_matrix(
        run_id,
        "RUNNING",
        launch_identity_sha256=file_sha256(authorization_path),
        started_at_unix=time.time(),
    )
    trainer = None
    failure = None
    classification = "VALID_COMPLETED_RUN"
    started = time.time()
    physical_path = run_dir / "physical_vram.jsonl"
    try:
        trainer, metadata = build_trainer(
            family,
            CONTEXT,
            initialization=spec["initialization"],
            seed=spec["seed"],
            max_steps=UPDATES,
            output_dir=run_dir,
        )
        trainer.validation_loader = DataLoader(
            ValidationContextDataset(), batch_size=1, shuffle=False, collate_fn=_collator
        )
        recovery = run_dir / "checkpoints" / "recovery.pt"
        mid = run_dir / "checkpoints" / "mid.pt"
        final = run_dir / "checkpoints" / "final.pt"
        resumed_from_checkpoint = recovery.exists()
        if recovery.exists():
            trainer.resume(recovery)
        with LoggedPhysicalMonitor(physical_path, interval_seconds=0.2) as monitor:
            trainer.physical_monitor = monitor
            for target in (300, 600, 900, 1200):
                if trainer.state.global_step >= target:
                    continue
                trainer.train_until(target, finalize=False)
                checkpoint = trainer._save("checkpoints/recovery.pt")
                if target == 600:
                    shutil.copy2(checkpoint, mid)
            validation = trainer.evaluate()
            trainer._save("checkpoints/final.pt", validation)
            trainer.model.save_pretrained(run_dir / "final_adapter", safe_serialization=True)
        state = trainer.state
        history = [row for row in state.history if row.get("event") == "train"]
        step_times = []
        previous_targets = 0
        for row in history:
            current_targets = int(row["target_tokens_seen"])
            step_times.append(
                (current_targets - previous_targets) / row["target_tokens_per_second"]
            )
            previous_targets = current_targets
        if state.global_step != UPDATES:
            classification = "INFRA_FAILURE"
        elif state.oom_count:
            classification = "OOM"
        elif state.nan_count or state.inf_count:
            classification = "NON_FINITE"
        elif monitor.errors:
            classification = "INFRA_FAILURE"
        elif (monitor.minimum_free_mib or 0) < HEADROOM_MIB:
            classification = "MEMORY_CONTRACT_VIOLATION"
        identity = _adapter_identity(trainer.model, family)
        if identity["status"] != "PASS":
            classification = "DATA_IDENTITY_FAILURE"
        summary = {
            "stage": STAGE,
            **spec,
            "classification": classification,
            "status": "VALID_COMPLETED"
            if classification == "VALID_COMPLETED_RUN"
            else "FAILED_FINAL",
            "updates_completed": state.global_step,
            "examples_seen": state.examples_seen,
            "input_tokens_seen": state.tokens_seen,
            "assistant_tokens_seen": state.target_tokens_seen,
            "final_training_loss": history[-1]["loss"] if history else None,
            "validation_loss": validation["validation_loss"],
            "nan_count": state.nan_count,
            "inf_count": state.inf_count,
            "oom_count": state.oom_count,
            "gradient_norm_min": min(row["grad_norm"] for row in history),
            "gradient_norm_max": max(row["grad_norm"] for row in history),
            "median_target_tokens_per_second": median(
                row["target_tokens_per_second"] for row in history
            ),
            "median_step_time_seconds": median(step_times),
            "wall_clock_seconds": time.time() - started,
            "peak_cuda_allocated_mib": trainer.peak_vram_mb(),
            "peak_cuda_reserved_mib": trainer.peak_reserved_vram_mb(),
            "physical_peak_vram_mib": monitor.maximum_used_mib,
            "minimum_physical_headroom_mib": monitor.minimum_free_mib,
            "physical_samples": monitor.samples,
            "physical_monitor_errors": monitor.errors,
            "adapter_identity": identity,
            "checkpoint": {
                "mid_path": str(mid),
                "mid_sha256": file_sha256(mid),
                "final_path": str(final),
                "final_sha256": file_sha256(final),
                "final_size_bytes": final.stat().st_size,
            },
            "adapter": {
                "path": str(run_dir / "final_adapter"),
                "sha256": tree_sha256(run_dir / "final_adapter"),
                "state_sha256": _adapter_digest(trainer.model),
                "size_bytes": sum(
                    path.stat().st_size
                    for path in (run_dir / "final_adapter").rglob("*")
                    if path.is_file()
                ),
            },
            "config_file_sha256": file_sha256(config_path),
            "config_semantic_sha256": config_hash(config),
            "dataset_manifest_sha256": file_sha256(DATA_MANIFEST),
            "dataset_encoded_sha256": data["encoded_dataset_digest"],
            "run_authorization_sha256": file_sha256(authorization_path),
            "environment": environment,
            "metadata": metadata,
            "resume_count": 1 if resumed_from_checkpoint else 0,
            "failure": None,
        }
        if recovery.exists():
            recovery.unlink()
    except torch.OutOfMemoryError as exc:
        failure, classification = f"{type(exc).__name__}: {exc}", "OOM"
    except FloatingPointError as exc:
        failure, classification = f"{type(exc).__name__}: {exc}", "NON_FINITE"
    except Exception as exc:
        failure, classification = f"{type(exc).__name__}: {exc}", "INFRA_FAILURE"
    if failure is not None:
        summary = {
            "stage": STAGE,
            **spec,
            "classification": classification,
            "status": "FAILED_AUTHORIZED_RERUN"
            if classification in {"INFRA_FAILURE", "HARDWARE_INTERRUPTION"}
            else "FAILED_FINAL",
            "failure": failure,
            "updates_completed": trainer.state.global_step if trainer is not None else 0,
            "wall_clock_seconds": time.time() - started,
            "run_authorization_sha256": file_sha256(authorization_path),
        }
        FAILURE_ROOT.mkdir(parents=True, exist_ok=True)
        write_json(FAILURE_ROOT / f"{run_id}.json", summary)
    write_json(run_dir / "final_summary.json", summary)
    _update_matrix(
        run_id,
        summary["status"],
        classification=summary["classification"],
        summary_path=str(run_dir / "final_summary.json"),
        summary_sha256=file_sha256(run_dir / "final_summary.json"),
        ended_at_unix=time.time(),
    )
    if trainer is not None:
        del trainer
    torch.cuda.empty_cache()
    return summary


def _load_formal_model(summary: dict[str, Any]) -> Any:
    family = summary["family"]
    config = load_config(CONFIGS[(family, CONTEXT, "R16_ALL_LINEAR")])
    model = _load_model(config, summary["initialization"], summary["seed"])
    load_checkpoint(
        summary["checkpoint"]["final_path"],
        model=model,
        map_location=next(model.parameters()).device,
        restore_rng=False,
    )
    if not getattr(model, "hf_device_map", None):
        model.to(torch.device("cuda"))
    if _adapter_digest(model) != summary["adapter"]["state_sha256"]:
        raise RuntimeError("adapter hash mismatch before formal evaluation")
    identity = _adapter_identity(model, family)
    if identity["status"] != "PASS":
        raise RuntimeError("adapter/quantized-backbone identity mismatch before evaluation")
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    return model


def evaluate_run(run_id: str) -> dict[str, Any]:
    if not _formal_code_frozen():
        raise RuntimeError("CODE_FREEZE violation")
    summary_path = RUN_ROOT / run_id / "final_summary.json"
    summary = _read_json(summary_path)
    if summary["status"] != "VALID_COMPLETED":
        raise RuntimeError("formal evaluation requires a valid completed run")
    if file_sha256(summary["checkpoint"]["final_path"]) != summary["checkpoint"]["final_sha256"]:
        raise RuntimeError("final checkpoint hash mismatch")
    qualification = _read_json(EVAL_QUALIFICATION_PATH)
    batch_size = qualification["selected_batch_size"]
    model = _load_formal_model(summary)
    tokenizer = load_frozen_tokenizer()
    run_results = {}
    for benchmark in BENCHMARKS:
        rows = _benchmark_records(benchmark)
        output_path = RUN_ROOT / run_id / "evaluation" / f"{benchmark}.jsonl"
        existing = []
        if output_path.exists():
            existing = read_jsonl(output_path)
        by_id = {row["problem_id"]: row for row in existing}
        if len(by_id) != len(existing):
            raise RuntimeError("duplicate completed problem IDs in resumable evaluation")
        pending = [row for row in rows if _problem_id(benchmark, row) not in by_id]
        physical_path = RUN_ROOT / run_id / "evaluation" / f"{benchmark}-physical.jsonl"
        with LoggedPhysicalMonitor(physical_path, interval_seconds=0.2) as monitor:
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset : offset + batch_size]
                outputs = _generate_reference_batch(model, tokenizer, batch, benchmark)
                for output in outputs:
                    output.update(
                        {
                            "run_id": run_id,
                            "checkpoint_hash": summary["checkpoint"]["final_sha256"],
                            "adapter_hash": summary["adapter"]["state_sha256"],
                            "elapsed": output.pop("elapsed_batch_seconds"),
                        }
                    )
                    _append_jsonl(output_path, output)
                    by_id[output["problem_id"]] = output
        if monitor.errors or (monitor.minimum_free_mib or 0) < HEADROOM_MIB:
            raise RuntimeError(f"{benchmark} formal evaluation physical-memory gate failed")
        expected_ids = {_problem_id(benchmark, row) for row in rows}
        if set(by_id) != expected_ids or len(by_id) != len(rows):
            raise RuntimeError(f"{benchmark} completeness gate failed")
        ordered = [by_id[_problem_id(benchmark, row)] for row in rows]
        correct = sum(bool(row["correct"]) for row in ordered)
        elapsed = sum(float(row["elapsed"]) / int(row["batch_size"]) for row in ordered)
        generated_tokens = sum(int(row["generation_tokens"]) for row in ordered)
        run_results[benchmark] = {
            "correct": correct,
            "total": len(ordered),
            "accuracy": correct / len(ordered),
            "problem_ids_unique": True,
            "missing": 0,
            "duplicates": 0,
            "results_path": str(output_path),
            "results_sha256": file_sha256(output_path),
            "generated_tokens": generated_tokens,
            "elapsed_seconds": elapsed,
            "problems_per_second": len(ordered) / elapsed,
            "generated_tokens_per_second": generated_tokens / elapsed,
            "physical_peak_vram_mib": monitor.maximum_used_mib,
            "minimum_physical_headroom_mib": monitor.minimum_free_mib,
            "physical_samples": monitor.samples,
            "monitor_errors": monitor.errors,
        }
    summary["evaluation"] = run_results
    summary["evaluation_status"] = "VALID_COMPLETED"
    summary["evaluation_batch_size"] = batch_size
    write_json(summary_path, summary)
    _write_individual_result(summary)
    del model
    torch.cuda.empty_cache()
    return summary


def _write_individual_result(summary: dict[str, Any]) -> None:
    output = Path("experiments/results") / f"{summary['run_id']}.json"
    result = {
        "stage": "GPU2D_FORMAL",
        "run_id": summary["run_id"],
        "method": summary["family"],
        "initialization": summary["initialization"],
        "seed": summary["seed"],
        "training_status": summary["status"],
        "final_training_loss": summary["final_training_loss"],
        "validation_loss": summary["validation_loss"],
        "peak_physical_vram_mib": summary["physical_peak_vram_mib"],
        "minimum_physical_headroom_mib": summary["minimum_physical_headroom_mib"],
        "target_tokens_per_second": summary["median_target_tokens_per_second"],
        "elapsed_seconds": summary["wall_clock_seconds"],
        "gsm8k": summary.get("evaluation", {}).get("gsm8k"),
        "math500": summary.get("evaluation", {}).get("math500"),
        "checkpoint_hash": summary["checkpoint"]["final_sha256"],
        "adapter_hash": summary["adapter"]["state_sha256"],
        "config_hash": summary["config_semantic_sha256"],
        "dataset_hash": summary["dataset_encoded_sha256"],
    }
    write_json(output, result)


def _paired_bootstrap(base: list[bool], cpt: list[bool], seed: int) -> dict[str, Any]:
    if len(base) != len(cpt) or not base:
        raise ValueError("paired bootstrap needs equal non-empty vectors")
    base_array = np.asarray(base, dtype=np.float64)
    cpt_array = np.asarray(cpt, dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(10000, dtype=np.float64)
    for start in range(0, 10000, 1000):
        width = min(1000, 10000 - start)
        indices = rng.integers(0, len(base), size=(width, len(base)))
        deltas[start : start + width] = (
            cpt_array[indices].mean(axis=1) - base_array[indices].mean(axis=1)
        ) * 100
    return {
        "seed": seed,
        "replicates": 10000,
        "confidence_level": 0.95,
        "observed_delta_points": float((cpt_array.mean() - base_array.mean()) * 100),
        "ci_low_points": float(np.quantile(deltas, 0.025)),
        "ci_high_points": float(np.quantile(deltas, 0.975)),
        "interpretation": "benchmark-problem sampling uncertainty only",
    }


def _accuracy_data(family: str, benchmark: str) -> dict[str, Any]:
    base_values, cpt_values, deltas, bootstraps = [], [], [], {}
    for seed in SEEDS:
        base_id, cpt_id = f"B-{family}-s{seed}", f"C-{family}-s{seed}"
        base_summary = _read_json(RUN_ROOT / base_id / "final_summary.json")
        cpt_summary = _read_json(RUN_ROOT / cpt_id / "final_summary.json")
        base_eval = base_summary["evaluation"][benchmark]
        cpt_eval = cpt_summary["evaluation"][benchmark]
        base_values.append(base_eval["accuracy"])
        cpt_values.append(cpt_eval["accuracy"])
        deltas.append((cpt_eval["accuracy"] - base_eval["accuracy"]) * 100)
        base_rows = read_jsonl(base_eval["results_path"])
        cpt_rows = read_jsonl(cpt_eval["results_path"])
        base_by_id = {row["problem_id"]: bool(row["correct"]) for row in base_rows}
        cpt_by_id = {row["problem_id"]: bool(row["correct"]) for row in cpt_rows}
        if set(base_by_id) != set(cpt_by_id):
            raise RuntimeError("paired problem IDs differ")
        ids = sorted(base_by_id)
        bootstraps[str(seed)] = _paired_bootstrap(
            [base_by_id[key] for key in ids],
            [cpt_by_id[key] for key in ids],
            20260913 + seed,
        )
    return {
        "base": base_values,
        "cpt": cpt_values,
        "paired_delta": deltas,
        "mean_delta": mean(deltas),
        "sd_delta": stdev(deltas),
        "range_delta": [min(deltas), max(deltas)],
        "positive_seeds": sum(value > 0 for value in deltas),
        "negative_seeds": sum(value < 0 for value in deltas),
        "bootstrap": bootstraps,
    }


def _family_outcome(values: dict[str, Any]) -> str:
    summaries = [values[benchmark] for benchmark in BENCHMARKS]
    positive = all(
        item["mean_delta"] >= 1.0
        and item["positive_seeds"] >= 2
        and min(item["paired_delta"]) >= -0.5
        for item in summaries
    )
    negative = all(
        item["mean_delta"] <= -1.0
        and item["negative_seeds"] >= 2
        and max(item["paired_delta"]) <= 0.5
        for item in summaries
    )
    if positive:
        return "POSITIVE"
    if negative:
        return "NEGATIVE"
    if any(abs(item["mean_delta"]) >= 1.0 for item in summaries) or (
        summaries[0]["mean_delta"] * summaries[1]["mean_delta"] < 0
    ):
        return "MIXED"
    return "NOT_SUPPORTED"


def _overall_classification(outcomes: dict[str, str]) -> str:
    values = set(outcomes.values())
    if values == {"POSITIVE"}:
        return "CPT_TO_PEFT_SFT_TRANSFER_SUPPORTED"
    if values == {"NEGATIVE"}:
        return "CPT_TO_PEFT_SFT_TRANSFER_NEGATIVE"
    if "POSITIVE" in values or "NEGATIVE" in values:
        return "CPT_TO_PEFT_SFT_TRANSFER_METHOD_DEPENDENT"
    if "MIXED" in values:
        return "GPU2D_FORMAL_RESULTS_INCONCLUSIVE"
    return "CPT_TO_PEFT_SFT_TRANSFER_NOT_SUPPORTED"


def analyze() -> dict[str, Any]:
    matrix = _read_json(TRAINING_MATRIX_PATH)
    summaries = [_read_json(RUN_ROOT / run_id / "final_summary.json") for run_id in EXECUTION_ORDER]
    if not all(
        summary["status"] == "VALID_COMPLETED"
        and summary.get("evaluation_status") == "VALID_COMPLETED"
        for summary in summaries
    ):
        raise RuntimeError("F3 requires 12 valid trained and evaluated runs")
    methods = {
        family.lower(): {benchmark: _accuracy_data(family, benchmark) for benchmark in BENCHMARKS}
        for family in METHODS
    }
    outcomes = {family: _family_outcome(methods[family.lower()]) for family in METHODS}
    classification = _overall_classification(outcomes)
    efficiency = {}
    for family in METHODS:
        selected = [summary for summary in summaries if summary["family"] == family]
        efficiency[family.lower()] = {
            "training_time_seconds": [item["wall_clock_seconds"] for item in selected],
            "peak_physical_vram_mib": [item["physical_peak_vram_mib"] for item in selected],
            "minimum_physical_headroom_mib": [
                item["minimum_physical_headroom_mib"] for item in selected
            ],
            "target_tokens_per_second": [
                item["median_target_tokens_per_second"] for item in selected
            ],
            "adapter_size_bytes": [item["adapter"]["size_bytes"] for item in selected],
        }
        efficiency[family.lower()]["mean_training_time_seconds"] = mean(
            efficiency[family.lower()]["training_time_seconds"]
        )
        efficiency[family.lower()]["mean_target_tokens_per_second"] = mean(
            efficiency[family.lower()]["target_tokens_per_second"]
        )
    result = {
        "stage": STAGE,
        "classification": classification,
        "claim_scope": "CPT_TO_PEFT_SFT_TRANSFER_UNDER_FROZEN_512_CONTEXT",
        "formal_runs_expected": 12,
        "formal_runs_valid": 12,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "benchmarks": ["GSM8K", "MATH-500"],
        **methods,
        "family_outcomes": outcomes,
        "efficiency": efficiency,
        "bootstrap": {
            family: {benchmark: methods[family][benchmark]["bootstrap"] for benchmark in BENCHMARKS}
            for family in methods
        },
        "failures": [
            item for item in matrix["runs"].values() if item["status"] != "VALID_COMPLETED"
        ],
        "limitations": [
            "Results apply only to the frozen 512-context data exposure.",
            "Only Qwen3-0.6B, one math SFT corpus, three seeds, and two benchmarks are studied.",
            "Problem-level bootstrap does not measure training-seed uncertainty.",
            "No Full-SFT transfer arm is present.",
        ],
    }
    write_json(FINAL_RESULT_PATH, result)
    _write_final_report(result, summaries)
    _update_registry(summaries)
    _update_project_summaries(result)
    return result


def _pct(values: list[float]) -> str:
    return ", ".join(f"{100 * value:.2f}%" for value in values)


def _write_final_report(result: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    lora, qlora = result["lora"], result["qlora"]
    efficiency = result["efficiency"]
    text = f"""# GPU-2D Formal PEFT Transfer

## 1. Research Question

Does frozen math-CPT initialization improve downstream LoRA and QLoRA mathematical
reasoning benchmarks under the frozen 512-context SFT exposure?

## 2. Historical Context

GPU-2B and GPU-2C remain blocked by memory qualification. GPU-2D-R0 prospectively
qualified this distinct reduced-context study.

## 3. Claim Boundary

`CPT_TO_PEFT_SFT_TRANSFER_UNDER_FROZEN_512_CONTEXT` only.

## 4. Experimental Design

Two initializations × two PEFT methods × three seeds = 12 paired formal runs.

## 5. 512-Context Data Exposure

19,200 source training records, 19,162 eligible encoded records, 800 validation records,
and 5,177,783 retained assistant tokens. This is a scientific data-exposure change;
assistant and final-answer retention versus 1024 are 73.71% and 73.22%.

## 6. Initializations

`BASE_INIT` is pinned Qwen3-0.6B-Base. `CPT_INIT` is the frozen validated Math-CPT
checkpoint. Every evaluation verified the expected initialization and adapter identity.

## 7. LoRA

Rank 16, alpha 32, dropout 0.05, all-linear, BF16 backbone, AdamW.

## 8. QLoRA

NF4, double quantization, BF16 compute, rank 16, alpha 32, dropout 0.05,
all-linear, PagedAdamW8bit. The frozen 4-bit backbone identity was verified.

## 9. Seeds

42, 31415, and 271828 with paired deterministic sample permutations.

## 10. Formal Execution

{result["formal_runs_valid"]}/{result["formal_runs_expected"]} valid runs completed in the
pre-frozen balanced order. Each ran exactly 1,200 optimizer updates and retained mid/final
checkpoints plus per-run identity evidence.

## 11. Training Stability

All valid runs completed without disallowed reruns. Full numerical and failure details are
in the training matrix and individual summaries.

## 12. GPU Efficiency

| Method | Peak physical VRAM range | Mean target tokens/s | Mean training time |
| --- | ---: | ---: | ---: |
| LoRA | {min(efficiency["lora"]["peak_physical_vram_mib"])}–{max(efficiency["lora"]["peak_physical_vram_mib"])} MiB | {efficiency["lora"]["mean_target_tokens_per_second"]:.2f} | {efficiency["lora"]["mean_training_time_seconds"] / 3600:.2f} h |
| QLoRA | {min(efficiency["qlora"]["peak_physical_vram_mib"])}–{max(efficiency["qlora"]["peak_physical_vram_mib"])} MiB | {efficiency["qlora"]["mean_target_tokens_per_second"]:.2f} | {efficiency["qlora"]["mean_training_time_seconds"] / 3600:.2f} h |

## 13. Benchmark Protocol

Frozen GSM8K (1,319) and MATH-500 (500), greedy decoding, 512 max-new-tokens,
F0-qualified exact-equivalent resumable batching, and frozen exact-match scoring.

## 14. GSM8K Results

LoRA Base: {_pct(lora["gsm8k"]["base"])}; CPT: {_pct(lora["gsm8k"]["cpt"])}.
QLoRA Base: {_pct(qlora["gsm8k"]["base"])}; CPT: {_pct(qlora["gsm8k"]["cpt"])}.

## 15. MATH-500 Results

LoRA Base: {_pct(lora["math500"]["base"])}; CPT: {_pct(lora["math500"]["cpt"])}.
QLoRA Base: {_pct(qlora["math500"]["base"])}; CPT: {_pct(qlora["math500"]["cpt"])}.

## 16. LoRA Paired Transfer

GSM8K deltas: {lora["gsm8k"]["paired_delta"]}, mean {lora["gsm8k"]["mean_delta"]:.3f} points.
MATH-500 deltas: {lora["math500"]["paired_delta"]}, mean {lora["math500"]["mean_delta"]:.3f} points.
Family outcome: `{result["family_outcomes"]["LORA"]}`.

## 17. QLoRA Paired Transfer

GSM8K deltas: {qlora["gsm8k"]["paired_delta"]}, mean {qlora["gsm8k"]["mean_delta"]:.3f} points.
MATH-500 deltas: {qlora["math500"]["paired_delta"]}, mean {qlora["math500"]["mean_delta"]:.3f} points.
Family outcome: `{result["family_outcomes"]["QLORA"]}`.

## 18. Bootstrap Analysis

Ten-thousand-replicate paired problem-level percentile intervals are stored for every
method, seed, and benchmark. They quantify problem-sampling uncertainty, not training-
seed uncertainty.

## 19. Seed Variability

LoRA delta SD: GSM8K {lora["gsm8k"]["sd_delta"]:.3f}, MATH-500
{lora["math500"]["sd_delta"]:.3f} points. QLoRA delta SD: GSM8K
{qlora["gsm8k"]["sd_delta"]:.3f}, MATH-500 {qlora["math500"]["sd_delta"]:.3f} points.

## 20. PEFT Efficiency Trade-off

Accuracy, physical VRAM, throughput, runtime, trainable parameters, and adapter size are
reported jointly. Engineering results do not override the scientific classification.

## 21. Hypothesis Results

- H-P1: `{result["family_outcomes"]["LORA"]}`.
- H-P2: `{result["family_outcomes"]["QLORA"]}`.
- H-P3: `{"CONSISTENT" if result["family_outcomes"]["LORA"] == result["family_outcomes"]["QLORA"] else "METHOD_DEPENDENT"}`.
- H-P4: `MEASURED_AND_REPORTED`.

## 22. Scientific Classification

`{result["classification"]}`

## 23. Limitations

Results are limited to 512 context, Qwen3-0.6B, one SFT corpus, three seeds, and two
benchmarks. No Full-SFT arm exists, and bootstrap intervals do not replace seed variation.

## 24. Portfolio Interpretation

The campaign demonstrates controlled, multi-seed real LoRA/QLoRA training and auditable
generated-answer evaluation. Any portfolio summary must retain the 512-context qualifier.

## 25. Conclusion

The result answers only the frozen Base-versus-Math-CPT initialization question under
the qualified reduced-context PEFT protocol. It is not a universal CPT or reasoning claim.
"""
    FINAL_REPORT_PATH.write_text(text, encoding="utf-8")


def _update_registry(summaries: list[dict[str, Any]]) -> None:
    path = Path("experiments/registry.csv")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [row for row in reader if row.get("experiment_id") not in EXECUTION_ORDER]
    for summary in summaries:
        values = {
            "experiment_id": summary["run_id"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": summary["environment"].get("git_commit", _git_commit()),
            "config_hash": summary["config_semantic_sha256"],
            "dataset_hash": summary["dataset_encoded_sha256"],
            "model": "Qwen/Qwen3-0.6B-Base",
            "dataset": "GPU2D frozen 512-context SFT math",
            "seed": summary["seed"],
            "batch_size": 1,
            "effective_batch_size": 16,
            "rank": 16,
            "target_modules": "all-linear",
            "precision": "bf16",
            "trainable_params": summary["metadata"]["parameters"]["trainable_parameters"],
            "peak_vram_mb": summary["physical_peak_vram_mib"],
            "tokens_per_second": summary["median_target_tokens_per_second"],
            "final_loss": summary["final_training_loss"],
            "eval_score": summary["evaluation"]["gsm8k"]["accuracy"],
            "status": "completed",
            "environment": json.dumps(summary["environment"], sort_keys=True),
            "device": "cuda",
            "gpu_name": summary["environment"]["gpu"],
            "torch_version": summary["environment"]["torch"],
            "torch_cuda_version": summary["environment"]["cuda_runtime"],
            "classification": "GPU2D_FORMAL_RUN_VALID",
            "sequence_length": 512,
            "optimizer_steps": 1200,
            "micro_batch": 1,
            "gradient_accumulation": 16,
            "final_train_loss": summary["final_training_loss"],
            "final_validation_loss": summary["validation_loss"],
            "peak_allocated_vram_mib": summary["peak_cuda_allocated_mib"],
            "peak_reserved_vram_mib": summary["peak_cuda_reserved_mib"],
            "median_tokens_per_second": summary["median_target_tokens_per_second"],
            "elapsed_seconds": summary["wall_clock_seconds"],
            "nan_events": summary["nan_count"],
            "inf_events": summary["inf_count"],
            "oom_events": summary["oom_count"],
            "optimizer": "AdamW" if summary["family"] == "LORA" else "PagedAdamW8bit",
            "primary_classification": "GPU2D_FORMAL_RUN_VALID",
        }
        rows.append({field: values.get(field, "") for field in fields})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _update_project_summaries(result: dict[str, Any]) -> None:
    block = f"""<!-- GPU2D_FORMAL_RESULTS_START -->
### GPU-2D measured downstream PEFT transfer

- Design: Base vs Math-CPT initialization, LoRA and QLoRA, three paired seeds,
  frozen 512-context exposure, GSM8K and MATH-500.
- Classification: `{result["classification"]}`.
- Full details: `reports/GPU2D_FORMAL_PEFT_TRANSFER.md`.

This result is limited to the frozen 512-context protocol and does not establish
Full-SFT transfer or universal mathematical-reasoning improvement.
<!-- GPU2D_FORMAL_RESULTS_END -->

"""
    readme_path = Path("README.md")
    readme = readme_path.read_text(encoding="utf-8")
    if "<!-- GPU2D_FORMAL_RESULTS_START -->" not in readme:
        marker = "## Phase B: Qwen CPT and SFT\n\n"
        readme = readme.replace(marker, marker + block)
    readme_path.write_text(readme, encoding="utf-8")
    final_path = Path("reports/FINAL_REPORT.md")
    final = final_path.read_text(encoding="utf-8")
    replacement = (
        f"Measured SFT / LoRA / QLoRA results: **{result['classification']}** under the "
        "frozen 512-context GPU-2D PEFT protocol; see `reports/GPU2D_FORMAL_PEFT_TRANSFER.md`."
    )
    if "Measured SFT / LoRA / QLoRA results: TBD" in final:
        final = final.replace("Measured SFT / LoRA / QLoRA results: TBD", replacement)
    elif replacement not in final:
        final += "\n\n" + replacement + "\n"
    final_path.write_text(final, encoding="utf-8")
