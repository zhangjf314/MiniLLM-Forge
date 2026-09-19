from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minillm_forge.evaluation.math_eval import exact_match, extract_final_answer  # noqa: E402
from minillm_forge.execution.gpu2b import (  # noqa: E402
    file_sha256,
    load_frozen_tokenizer,
    tree_sha256,
)
from minillm_forge.execution.gpu2d_formal import (  # noqa: E402
    _benchmark_records,
    _problem_id,
    _prompt,
)

BASELINE_HEAD = "1ca175ef2802fb1eba4beaef33b22dcc751a7db3"
SOURCE_TREE_SHA256 = "955f9263692fd5b2584509d4bb8b53185e43efaa56aadfeae5c17686b9e16a9f"
EVALUATION_ORDER_HASH = "ff50b516501cd66fad6aa23756b2c8eae527884969d1d2befc7d6a70432aba1d"
GSM8K_SUBSET_HASH = "fb9635d80b6210da35a9603da67f16148b5f7e4ca2ba140638865878c1c4cfc8"
BOOTSTRAP_BASE_SEED = 20260913
BOOTSTRAP_REPLICATES = 10_000
SEEDS = (42, 31415, 271828)
FAMILIES = ("LORA", "QLORA")
RUN_IDS = tuple(
    f"{prefix}-{family}-s{seed}" for family in FAMILIES for seed in SEEDS for prefix in ("B", "C")
)
FORMAL = ROOT / "artifacts/gpu2d_formal"
RUN_ROOT = ROOT / "runs/gpu2d-formal"
SUPERVISOR = ROOT.parent / "MiniLLM-Forge-Eval-Supervisor"
FINAL_RESULT = FORMAL / "final_result.json"
RESULT_ROOT = ROOT / "experiments/results"
REGISTRY = ROOT / "experiments/registry.csv"

FROZEN_FILES = {
    FORMAL / "evaluation_cost_amendment.json": (
        "4e89970dc0e1704851eb75a0b2ca33b6945922f79131ea10ae47f87879fc10ac"
    ),
    FORMAL / "evaluation_order_amended.json": (
        "5c44889c45a657129d60ce6ef6ad8f831498d3f0f9187fe27c8ba214acaa3a32"
    ),
    FORMAL / "gsm8k_200_subset.json": (
        "f2856213af4eda93c54b8998bc1d8dec2df57ba6b9b184ca225596cd47a223b1"
    ),
    FORMAL / "eval_runtime_recovery.json": (
        "5a30b13f95c898a44b4c2515f25f1b0d969eacdcb618cb5755e31409816c51ea"
    ),
}


class IntegrityError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise IntegrityError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"invalid JSONL {path}:{number}: {exc}") from exc
            if not isinstance(value, dict):
                raise IntegrityError(f"non-object JSONL record: {path}:{number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def sha256_json_list(values: list[str]) -> str:
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def parse_run_id(run_id: str) -> tuple[str, str, int]:
    prefix, seed_text = run_id.rsplit("-s", 1)
    initialization, family = prefix.split("-", 1)
    return ("BASE_INIT" if initialization == "B" else "CPT_INIT", family, int(seed_text))


def paired_bootstrap(base: list[bool], cpt: list[bool], seed: int) -> dict[str, Any]:
    require(len(base) == len(cpt) and bool(base), "paired bootstrap needs equal vectors")
    base_array = np.asarray(base, dtype=np.float64)
    cpt_array = np.asarray(cpt, dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 1000):
        width = min(1000, BOOTSTRAP_REPLICATES - start)
        indices = rng.integers(0, len(base), size=(width, len(base)))
        deltas[start : start + width] = (
            cpt_array[indices].mean(axis=1) - base_array[indices].mean(axis=1)
        ) * 100
    return {
        "seed": seed,
        "replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": 0.95,
        "observed_delta_points": float((cpt_array.mean() - base_array.mean()) * 100),
        "ci_low_points": float(np.quantile(deltas, 0.025)),
        "ci_high_points": float(np.quantile(deltas, 0.975)),
        "interpretation": "benchmark-problem sampling uncertainty only",
    }


def family_outcome(values: dict[str, Any]) -> str:
    summaries = [values[benchmark] for benchmark in ("math500", "gsm8k_fixed_200")]
    tolerance = 1e-12
    positive = all(
        item["mean_delta_points"] >= 1.0 - tolerance
        and item["positive_seeds"] >= 2
        and min(item["paired_deltas_points"]) >= -0.5 - tolerance
        for item in summaries
    )
    negative = all(
        item["mean_delta_points"] <= -1.0 + tolerance
        and item["negative_seeds"] >= 2
        and max(item["paired_deltas_points"]) <= 0.5 + tolerance
        for item in summaries
    )
    if positive:
        return "POSITIVE"
    if negative:
        return "NEGATIVE"
    if any(abs(item["mean_delta_points"]) >= 1.0 for item in summaries) or (
        summaries[0]["mean_delta_points"] * summaries[1]["mean_delta_points"] < 0
    ):
        return "MIXED"
    return "NOT_SUPPORTED"


def overall_classification(outcomes: dict[str, str]) -> str:
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


def audit() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    baseline_is_ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASELINE_HEAD, head],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )
    require(baseline_is_ancestor, f"frozen evaluation HEAD is not an ancestor: {head}")
    changed_since_baseline = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            BASELINE_HEAD,
            "--",
            "src",
            "scripts",
            "configs",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    changed_since_baseline = [
        path
        for path in changed_since_baseline
        if path.replace("\\", "/") != "scripts/finalize_gpu2d.py"
    ]
    require(
        not changed_since_baseline,
        f"frozen scientific files changed since evaluation: {changed_since_baseline}",
    )
    drift_lines = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src",
            "scripts",
            "configs",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    drift_lines = [
        line for line in drift_lines if line[3:].replace("\\", "/") != "scripts/finalize_gpu2d.py"
    ]
    require(not drift_lines, f"frozen scientific source/config drift:\n{drift_lines}")

    frozen_hashes: dict[str, str] = {}
    for path, expected in FROZEN_FILES.items():
        actual = file_sha256(path)
        require(actual == expected, f"frozen artifact mismatch: {path}: {actual}")
        frozen_hashes[rel(path)] = actual

    order = read_json(FORMAL / "evaluation_order_amended.json")
    subset = read_json(FORMAL / "gsm8k_200_subset.json")
    jobs = order["jobs"]
    job_ids = [str(job["job_id"]) for job in jobs]
    require(len(jobs) == 24, "evaluation order is not 24 jobs")
    require(sha256_json_list(job_ids) == EVALUATION_ORDER_HASH, "job order hash mismatch")
    selected_ids = [str(value) for value in subset["selected_problem_ids"]]
    require(len(selected_ids) == len(set(selected_ids)) == 200, "GSM8K subset cardinality failure")
    require(sha256_json_list(selected_ids) == GSM8K_SUBSET_HASH, "GSM8K subset hash mismatch")

    supervisor_state = read_json(SUPERVISOR / "state.json")
    events = read_jsonl(SUPERVISOR / "events.jsonl")
    require(supervisor_state.get("status") == "COMPLETED", "supervisor is not completed")
    require(
        len(supervisor_state.get("completed_jobs", [])) == 24, "supervisor completed count mismatch"
    )
    require(not supervisor_state.get("pending_jobs"), "supervisor still has pending jobs")
    require(supervisor_state.get("failed_job") is None, "supervisor records a failed job")
    event_counts = Counter(str(event.get("event")) for event in events)
    require(event_counts["SUPERVISOR_STARTED"] == 1, "supervisor restart detected")
    require(event_counts["JOB_STARTED"] == 24, "job-start event count mismatch")
    require(event_counts["JOB_VALID_COMPLETED"] == 24, "job-completion event count mismatch")
    require(event_counts["CAMPAIGN_VALID_COMPLETED"] == 1, "campaign completion event missing")
    started_events = {
        event["job_id"]: event for event in events if event.get("event") == "JOB_STARTED"
    }
    completed_events = {
        event["job_id"]: event for event in events if event.get("event") == "JOB_VALID_COMPLETED"
    }

    matrix = read_json(FORMAL / "training_matrix.json")
    require(matrix.get("counts", {}).get("VALID_COMPLETED") == 12, "training matrix count mismatch")
    require(set(matrix.get("runs", {})) == set(RUN_IDS), "training matrix run IDs mismatch")

    tokenizer = load_frozen_tokenizer()
    source_by_benchmark: dict[str, dict[str, dict[str, Any]]] = {}
    expected_ids: dict[str, list[str]] = {}
    for benchmark in ("math500", "gsm8k"):
        records = _benchmark_records(benchmark)
        by_id = {_problem_id(benchmark, row): row for row in records}
        source_by_benchmark[benchmark] = by_id
        if benchmark == "math500":
            expected_ids[benchmark] = [_problem_id(benchmark, row) for row in records]
        else:
            expected_ids[benchmark] = selected_ids
            require(not (set(selected_ids) - set(by_id)), "GSM8K subset ID absent from snapshot")

    job_audits: list[dict[str, Any]] = []
    run_results: dict[str, dict[str, Any]] = {}
    row_vectors: dict[tuple[str, str], dict[str, bool]] = {}
    all_rows = 0
    total_duplicates = 0
    total_missing = 0
    examples: dict[str, list[dict[str, Any]]] = {
        "answer_extraction_failure": [],
        "incorrect_extracted_answer": [],
        "token_ceiling": [],
        "base_correct_cpt_wrong": [],
        "base_wrong_cpt_correct": [],
    }

    summary_hashes: dict[str, str] = {}
    for run_id in RUN_IDS:
        expected_initialization, expected_family, expected_seed = parse_run_id(run_id)
        summary_path = RUN_ROOT / run_id / "final_summary.json"
        summary = read_json(summary_path)
        summary_hash = file_sha256(summary_path)
        summary_hashes[run_id] = summary_hash
        matrix_entry = matrix["runs"][run_id]
        checks = {
            "run_id": summary.get("run_id") == run_id,
            "status": summary.get("status") == "VALID_COMPLETED",
            "classification": summary.get("classification") == "VALID_COMPLETED_RUN",
            "initialization": summary.get("initialization") == expected_initialization,
            "family": summary.get("family") == expected_family,
            "seed": summary.get("seed") == expected_seed,
            "updates": summary.get("updates_completed") == 1200,
            "nan": summary.get("nan_count") == 0,
            "inf": summary.get("inf_count") == 0,
            "oom": summary.get("oom_count") == 0,
            "summary_hash": matrix_entry.get("summary_sha256") == summary_hash,
            "matrix_status": matrix_entry.get("status") == "VALID_COMPLETED",
            "matrix_classification": matrix_entry.get("classification") == "VALID_COMPLETED_RUN",
        }
        failed = [name for name, passed in checks.items() if not passed]
        require(not failed, f"training identity failure {run_id}: {failed}")
        checkpoint_path = ROOT / Path(summary["checkpoint"]["final_path"])
        checkpoint_hash = file_sha256(checkpoint_path)
        require(
            checkpoint_hash == summary["checkpoint"]["final_sha256"],
            f"checkpoint hash mismatch: {run_id}",
        )
        adapter_path = ROOT / Path(summary["adapter"]["path"])
        adapter_tree_hash = tree_sha256(adapter_path)
        require(
            adapter_tree_hash == summary["adapter"]["sha256"], f"adapter hash mismatch: {run_id}"
        )
        run_results[run_id] = {
            "run_id": run_id,
            "initialization": expected_initialization,
            "method": expected_family,
            "seed": expected_seed,
            "training": {
                "classification": summary["classification"],
                "updates": summary["updates_completed"],
                "final_training_loss": summary["final_training_loss"],
                "validation_loss": summary["validation_loss"],
                "wall_clock_seconds": summary["wall_clock_seconds"],
                "median_target_tokens_per_second": summary["median_target_tokens_per_second"],
                "peak_cuda_allocated_mib": summary["peak_cuda_allocated_mib"],
                "peak_cuda_reserved_mib": summary["peak_cuda_reserved_mib"],
                "physical_peak_vram_mib": summary["physical_peak_vram_mib"],
                "minimum_physical_headroom_mib": summary["minimum_physical_headroom_mib"],
                "trainable_parameters": summary["metadata"]["parameters"]["trainable_parameters"],
                "adapter_size_bytes": summary["adapter"]["size_bytes"],
                "resume_count": summary["resume_count"],
                "non_finite_events": summary["nan_count"] + summary["inf_count"],
                "oom_events": summary["oom_count"],
            },
            "artifact_identity": {
                "summary_path": rel(summary_path),
                "summary_sha256": summary_hash,
                "checkpoint_path": rel(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "adapter_path": rel(adapter_path),
                "adapter_tree_sha256": adapter_tree_hash,
                "adapter_state_sha256": summary["adapter"]["state_sha256"],
                "config_file_sha256": summary["config_file_sha256"],
                "config_semantic_sha256": summary["config_semantic_sha256"],
                "dataset_manifest_sha256": summary["dataset_manifest_sha256"],
                "dataset_encoded_sha256": summary["dataset_encoded_sha256"],
            },
            "benchmarks": {},
        }

    for job in jobs:
        run_id = str(job["run_id"])
        label = str(job["benchmark"])
        benchmark = "math500" if label == "MATH500_FULL" else "gsm8k"
        result_key = "math500" if benchmark == "math500" else "gsm8k_fixed_200"
        output_name = "math500.jsonl" if benchmark == "math500" else "gsm8k-fixed-200.jsonl"
        completion_name = (
            "completion-math500.json"
            if benchmark == "math500"
            else "completion-gsm8k-fixed-200.json"
        )
        physical_name = f"{label.lower()}-physical.jsonl"
        output_path = RUN_ROOT / run_id / "evaluation-amended" / output_name
        completion_path = RUN_ROOT / run_id / "evaluation-amended" / completion_name
        physical_path = RUN_ROOT / run_id / "evaluation-amended" / physical_name
        rows = read_jsonl(output_path)
        completion = read_json(completion_path)
        physical = read_jsonl(physical_path)
        ids = [str(row.get("problem_id")) for row in rows]
        duplicates = len(ids) - len(set(ids))
        expected = expected_ids[benchmark]
        missing = len(set(expected) - set(ids))
        unexpected = sorted(set(ids) - set(expected))
        require(not duplicates, f"duplicate problem IDs: {job['job_id']}")
        require(not missing, f"missing problem IDs: {job['job_id']}")
        require(not unexpected, f"unexpected problem IDs: {job['job_id']}: {unexpected[:3]}")
        require(len(rows) == int(job["problems"]), f"problem count mismatch: {job['job_id']}")
        summary = read_json(RUN_ROOT / run_id / "final_summary.json")
        source_rows = source_by_benchmark[benchmark]
        correct_vector: dict[str, bool] = {}
        elapsed = 0.0
        generated_tokens = 0
        ceiling = 0
        extraction_failures = 0
        incorrect_extracted = 0
        for row in rows:
            problem_id = str(row["problem_id"])
            source = source_rows[problem_id]
            prompt, prompt_ids = _prompt(tokenizer, str(source["problem"]))
            generated_ids = row.get("generated_token_ids")
            require(
                isinstance(generated_ids, list)
                and all(isinstance(value, int) for value in generated_ids),
                f"invalid generated tokens: {run_id}/{problem_id}",
            )
            stop_positions = [
                index for index, value in enumerate(generated_ids) if value in {151643, 151645}
            ]
            require(
                not stop_positions or stop_positions[0] == len(generated_ids) - 1,
                f"untrimmed stop token: {run_id}/{problem_id}",
            )
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
            reference = str(source["answer"])
            recomputed_correct = exact_match(decoded, reference)
            checks = {
                "run_id": row.get("run_id") == run_id,
                "benchmark": row.get("benchmark") == benchmark,
                "scope": row.get("benchmark_scope") == label,
                "problem_id": problem_id == _problem_id(benchmark, source),
                "prompt_hash": row.get("prompt_hash")
                == hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_ids": row.get("prompt_token_ids") == prompt_ids,
                "checkpoint": row.get("checkpoint_hash") == summary["checkpoint"]["final_sha256"],
                "adapter": row.get("adapter_hash") == summary["adapter"]["state_sha256"],
                "config": row.get("config_semantic_sha256") == summary["config_semantic_sha256"],
                "snapshot": row.get("benchmark_snapshot_sha256")
                == (
                    "f850b2b7554370eb31b6ea5dba6f893ce6f8462912c9d0edf658802feda5a9ec"
                    if benchmark == "math500"
                    else "ee22a458cb44be48e782d5150027ba530911efbfcaa71fe3fb01fa67a48952cf"
                ),
                "subset": row.get("gsm8k_subset_hash")
                == (None if benchmark == "math500" else GSM8K_SUBSET_HASH),
                "order": row.get("evaluation_order_hash") == EVALUATION_ORDER_HASH,
                "head": row.get("repository_head") == BASELINE_HEAD,
                "source_tree": row.get("evaluator_source_tree_sha256") == SOURCE_TREE_SHA256,
                "batch": row.get("batch_size") == 1,
                "generation_length": len(generated_ids) == row.get("generation_tokens")
                and len(generated_ids) <= 512,
                "decoded_text": row.get("generated_text") == decoded,
                "extracted": row.get("extracted_answer") == extract_final_answer(decoded),
                "reference": row.get("reference_answer") == extract_final_answer(reference),
                "correct": row.get("correct") == recomputed_correct,
            }
            failed = [name for name, passed in checks.items() if not passed]
            require(not failed, f"row identity failure {run_id}/{problem_id}: {failed}")
            correct_vector[problem_id] = recomputed_correct
            elapsed += float(row["elapsed"])
            generated_tokens += int(row["generation_tokens"])
            if int(row["generation_tokens"]) == 512:
                ceiling += 1
                if len(examples["token_ceiling"]) < 6:
                    examples["token_ceiling"].append(
                        {
                            "run_id": run_id,
                            "benchmark": label,
                            "problem_id": problem_id,
                            "generation_artifact": rel(output_path),
                        }
                    )
            extracted = row.get("extracted_answer")
            if extracted is None or str(extracted).strip() == "":
                extraction_failures += 1
                if len(examples["answer_extraction_failure"]) < 6:
                    examples["answer_extraction_failure"].append(
                        {
                            "run_id": run_id,
                            "benchmark": label,
                            "problem_id": problem_id,
                            "generation_artifact": rel(output_path),
                        }
                    )
            elif not recomputed_correct:
                incorrect_extracted += 1
                if len(examples["incorrect_extracted_answer"]) < 6:
                    examples["incorrect_extracted_answer"].append(
                        {
                            "run_id": run_id,
                            "benchmark": label,
                            "problem_id": problem_id,
                            "extracted_answer": extracted,
                            "reference_answer": row.get("reference_answer"),
                            "generation_artifact": rel(output_path),
                        }
                    )
        output_hash = file_sha256(output_path)
        minimum_free = min(int(row["free_mib"]) for row in physical)
        maximum_used = max(int(row["used_mib"]) for row in physical)
        completion_checks = {
            "status": completion.get("status") == "JOB_VALID_COMPLETED",
            "job": completion.get("job_id") == job["job_id"],
            "run": completion.get("run_id") == run_id,
            "benchmark": completion.get("benchmark") == label,
            "problems": completion.get("problems") == len(expected),
            "unique": completion.get("unique_problem_ids") == len(expected),
            "missing": completion.get("missing_problem_ids") == 0,
            "expected_ids": completion.get("expected_problem_ids_sha256")
            == sha256_json_list(expected),
            "output_hash": completion.get("output_sha256") == output_hash,
            "checkpoint": completion.get("checkpoint_sha256")
            == summary["checkpoint"]["final_sha256"],
            "adapter": completion.get("adapter_state_sha256") == summary["adapter"]["state_sha256"],
            "config": completion.get("config_semantic_sha256") == summary["config_semantic_sha256"],
            "order": completion.get("evaluation_order_hash") == EVALUATION_ORDER_HASH,
            "subset": completion.get("gsm8k_subset_hash")
            == (None if benchmark == "math500" else GSM8K_SUBSET_HASH),
            "head": completion.get("repository_head") == BASELINE_HEAD,
            "source_tree": completion.get("evaluator_source_tree_sha256") == SOURCE_TREE_SHA256,
            "reference_evaluator": completion.get("reference_evaluator") is True,
            "batch_size": completion.get("batch_size") == 1,
            "max_new_tokens": completion.get("max_new_tokens") == 512,
            "physical_samples": completion.get("physical_memory", {}).get("samples")
            == len(physical),
            "physical_minimum": completion.get("physical_memory", {}).get(
                "minimum_physical_headroom_mib"
            )
            == minimum_free,
            "physical_peak": completion.get("physical_memory", {}).get("physical_peak_vram_mib")
            == maximum_used,
        }
        failed = [name for name, passed in completion_checks.items() if not passed]
        require(not failed, f"completion identity failure {job['job_id']}: {failed}")
        started = datetime.fromisoformat(
            started_events[job["job_id"]]["timestamp"].replace("Z", "+00:00")
        )
        ended = datetime.fromisoformat(
            completed_events[job["job_id"]]["timestamp"].replace("Z", "+00:00")
        )
        correct = sum(correct_vector.values())
        benchmark_result = {
            "scope": label,
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows),
            "problem_ids_unique": True,
            "missing": missing,
            "duplicates": duplicates,
            "generation_tokens": generated_tokens,
            "token_ceiling_count": ceiling,
            "token_ceiling_rate": ceiling / len(rows),
            "answer_extraction_failures": extraction_failures,
            "incorrect_extracted_answers": incorrect_extracted,
            "generation_elapsed_seconds": elapsed,
            "generated_tokens_per_second": generated_tokens / elapsed,
            "job_wall_seconds": (ended - started).total_seconds(),
            "physical_peak_vram_mib": maximum_used,
            "minimum_physical_headroom_mib": minimum_free,
            "output_path": rel(output_path),
            "output_sha256": output_hash,
            "completion_path": rel(completion_path),
            "completion_sha256": file_sha256(completion_path),
            "physical_evidence_path": rel(physical_path),
            "physical_evidence_sha256": file_sha256(physical_path),
        }
        run_results[run_id]["benchmarks"][result_key] = benchmark_result
        row_vectors[(run_id, result_key)] = correct_vector
        job_audits.append(
            {
                "order": job["order"],
                "job_id": job["job_id"],
                "run_id": run_id,
                "method": run_results[run_id]["method"],
                "initialization": run_results[run_id]["initialization"],
                "seed": run_results[run_id]["seed"],
                "benchmark": label,
                "checkpoint_identity": "PASS",
                "protocol_identity": "PASS",
                "completion_classification": "JOB_VALID_COMPLETED",
                "problem_count": len(rows),
                "missing": missing,
                "duplicates": duplicates,
                "output_sha256": output_hash,
                "completion_sha256": file_sha256(completion_path),
            }
        )
        all_rows += len(rows)
        total_duplicates += duplicates
        total_missing += missing

    require(all_rows == 8400, f"formal row count is {all_rows}, not 8400")
    require(total_duplicates == total_missing == 0, "global completeness failure")

    comparisons: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        comparisons[family] = {}
        for benchmark_key in ("math500", "gsm8k_fixed_200"):
            pairs: dict[str, Any] = {}
            deltas: list[float] = []
            for seed in SEEDS:
                base_id = f"B-{family}-s{seed}"
                cpt_id = f"C-{family}-s{seed}"
                base = run_results[base_id]["benchmarks"][benchmark_key]
                cpt = run_results[cpt_id]["benchmarks"][benchmark_key]
                base_vector = row_vectors[(base_id, benchmark_key)]
                cpt_vector = row_vectors[(cpt_id, benchmark_key)]
                require(set(base_vector) == set(cpt_vector), "paired problem set mismatch")
                ids = sorted(base_vector)
                base_correct = [base_vector[value] for value in ids]
                cpt_correct = [cpt_vector[value] for value in ids]
                delta = (cpt["accuracy"] - base["accuracy"]) * 100
                deltas.append(delta)
                base_only = [value for value in ids if base_vector[value] and not cpt_vector[value]]
                cpt_only = [value for value in ids if not base_vector[value] and cpt_vector[value]]
                output_name = (
                    "math500.jsonl" if benchmark_key == "math500" else "gsm8k-fixed-200.jsonl"
                )
                for category, values, example_run in (
                    ("base_correct_cpt_wrong", base_only, cpt_id),
                    ("base_wrong_cpt_correct", cpt_only, cpt_id),
                ):
                    for problem_id in values:
                        if len(examples[category]) >= 6:
                            break
                        examples[category].append(
                            {
                                "run_id": example_run,
                                "paired_base_run_id": base_id,
                                "benchmark": benchmark_key,
                                "problem_id": problem_id,
                                "generation_artifact": rel(
                                    RUN_ROOT / example_run / "evaluation-amended" / output_name
                                ),
                            }
                        )
                pairs[str(seed)] = {
                    "base_run_id": base_id,
                    "cpt_run_id": cpt_id,
                    "base_correct": base["correct"],
                    "cpt_correct": cpt["correct"],
                    "base_accuracy": base["accuracy"],
                    "cpt_accuracy": cpt["accuracy"],
                    "correct_count_delta": cpt["correct"] - base["correct"],
                    "delta_points": delta,
                    "base_correct_cpt_wrong": len(base_only),
                    "base_wrong_cpt_correct": len(cpt_only),
                    "bootstrap": paired_bootstrap(
                        base_correct, cpt_correct, BOOTSTRAP_BASE_SEED + seed
                    ),
                }
            comparisons[family][benchmark_key] = {
                "pairs": pairs,
                "paired_deltas_points": deltas,
                "mean_delta_points": mean(deltas),
                "sample_sd_delta_points": stdev(deltas),
                "min_delta_points": min(deltas),
                "max_delta_points": max(deltas),
                "positive_seeds": sum(value > 0 for value in deltas),
                "negative_seeds": sum(value < 0 for value in deltas),
                "zero_seeds": sum(value == 0 for value in deltas),
                "direction_consistency": (
                    "ALL_POSITIVE"
                    if all(value > 0 for value in deltas)
                    else "ALL_NEGATIVE"
                    if all(value < 0 for value in deltas)
                    else "MIXED"
                ),
            }

    outcomes = {family: family_outcome(comparisons[family]) for family in FAMILIES}
    classification = overall_classification(outcomes)
    efficiency: dict[str, Any] = {}
    for family in FAMILIES:
        selected = [item for item in run_results.values() if item["method"] == family]
        training = [item["training"] for item in selected]
        evaluations = [benchmark for item in selected for benchmark in item["benchmarks"].values()]
        efficiency[family] = {
            "training": {
                "run_count": len(training),
                "mean_final_training_loss": mean(item["final_training_loss"] for item in training),
                "mean_validation_loss": mean(item["validation_loss"] for item in training),
                "cumulative_wall_seconds": sum(item["wall_clock_seconds"] for item in training),
                "mean_wall_seconds": mean(item["wall_clock_seconds"] for item in training),
                "mean_target_tokens_per_second": mean(
                    item["median_target_tokens_per_second"] for item in training
                ),
                "physical_peak_vram_mib_range": [
                    min(item["physical_peak_vram_mib"] for item in training),
                    max(item["physical_peak_vram_mib"] for item in training),
                ],
                "minimum_physical_headroom_mib": min(
                    item["minimum_physical_headroom_mib"] for item in training
                ),
                "trainable_parameters": sorted({item["trainable_parameters"] for item in training}),
                "adapter_size_bytes_range": [
                    min(item["adapter_size_bytes"] for item in training),
                    max(item["adapter_size_bytes"] for item in training),
                ],
            },
            "evaluation": {
                "job_count": len(evaluations),
                "cumulative_job_wall_seconds": sum(
                    item["job_wall_seconds"] for item in evaluations
                ),
                "cumulative_generation_elapsed_seconds": sum(
                    item["generation_elapsed_seconds"] for item in evaluations
                ),
                "generated_tokens": sum(item["generation_tokens"] for item in evaluations),
                "generated_tokens_per_second": sum(
                    item["generation_tokens"] for item in evaluations
                )
                / sum(item["generation_elapsed_seconds"] for item in evaluations),
                "physical_peak_vram_mib_range": [
                    min(item["physical_peak_vram_mib"] for item in evaluations),
                    max(item["physical_peak_vram_mib"] for item in evaluations),
                ],
                "minimum_physical_headroom_mib": min(
                    item["minimum_physical_headroom_mib"] for item in evaluations
                ),
            },
        }

    error_totals = {
        "answer_extraction_failures": sum(
            benchmark["answer_extraction_failures"]
            for run in run_results.values()
            for benchmark in run["benchmarks"].values()
        ),
        "incorrect_extracted_answers": sum(
            benchmark["incorrect_extracted_answers"]
            for run in run_results.values()
            for benchmark in run["benchmarks"].values()
        ),
        "token_ceiling_count": sum(
            benchmark["token_ceiling_count"]
            for run in run_results.values()
            for benchmark in run["benchmarks"].values()
        ),
    }
    error_totals["token_ceiling_rate"] = error_totals["token_ceiling_count"] / all_rows

    script_hash = file_sha256(Path(__file__))
    final = {
        "stage": "STAGE_GPU_2D_FINAL",
        "primary_classification": "MINILLM_FORGE_GPU2D_PORTFOLIO_FINALIZED",
        "scientific_classification": classification,
        "family_outcomes": outcomes,
        "claim_boundary": "CPT_TO_PEFT_SFT_TRANSFER_UNDER_FROZEN_512_CONTEXT",
        "training_campaign_status": {
            "classification": "GPU2D_FORMAL_TRAINING_VALIDATED",
            "valid_runs": 12,
            "expected_runs": 12,
        },
        "evaluation_campaign_status": {
            "classification": "GPU2D_FORMAL_EVALUATION_VALIDATED",
            "valid_jobs": 24,
            "expected_jobs": 24,
            "valid_problem_outputs": all_rows,
            "expected_problem_outputs": 8400,
            "missing": total_missing,
            "duplicates": total_duplicates,
            "checkpoint_identity": "PASS",
            "protocol_identity": "PASS",
            "recovery_integrity": "PASS",
            "supervisor_status": supervisor_state["status"],
            "supervisor_event_counts": dict(sorted(event_counts.items())),
            "interruptions": 0,
            "supervisor_restarts": 0,
            "repeated_job_attempts": 0,
            "recovered_jobs": 0,
        },
        "completion_audit": job_audits,
        "problem_integrity_audit": {
            "valid": all_rows,
            "expected": 8400,
            "math500": 6000,
            "gsm8k_fixed_200": 2400,
            "missing": total_missing,
            "duplicates": total_duplicates,
            "gsm8k_subset_sha256": GSM8K_SUBSET_HASH,
            "evaluation_order_sha256": EVALUATION_ORDER_HASH,
        },
        "per_run_benchmark_results": run_results,
        "paired_transfer": comparisons,
        "efficiency": efficiency,
        "error_analysis": {"totals": error_totals, "traceable_examples": examples},
        "bootstrap_protocol": {
            "base_seed": BOOTSTRAP_BASE_SEED,
            "pair_seed_rule": "base_seed + training_seed",
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": 0.95,
            "unit": "problem_id paired within method, seed, and benchmark",
            "interpretation": "problem-sampling uncertainty; not training-seed uncertainty",
        },
        "limitations": [
            "Only the frozen 512-context PEFT-SFT exposure is studied.",
            "Only Qwen3-0.6B, one math SFT corpus, LoRA/QLoRA, and three seeds are studied.",
            "GSM8K is a frozen 200-problem subset, not the full test set.",
            "No Full-SFT formal transfer arm was completed.",
            "The original 1024-context protocol remained blocked by memory qualification.",
            "Problem-level bootstrap intervals do not estimate training-pipeline randomness.",
            "Widespread 512-token ceiling contact limits interpretation of generated answers.",
        ],
        "artifact_identities": {
            "audit_baseline_head": BASELINE_HEAD,
            "frozen_evaluator_source_tree_sha256": SOURCE_TREE_SHA256,
            "analysis_script": rel(Path(__file__)),
            "analysis_script_sha256": script_hash,
            "frozen_files": frozen_hashes,
            "training_summary_hashes": summary_hashes,
            "supervisor_state_sha256": file_sha256(SUPERVISOR / "state.json"),
            "supervisor_events_sha256": file_sha256(SUPERVISOR / "events.jsonl"),
            "supervisor_log_sha256": file_sha256(SUPERVISOR / "supervisor.log"),
            "campaign_manifest_sha256": file_sha256(SUPERVISOR / "campaign_manifest.json"),
        },
    }
    return final


def write_per_run_results(final: dict[str, Any]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    for run_id, result in final["per_run_benchmark_results"].items():
        write_json(
            RESULT_ROOT / f"{run_id}.json",
            {
                "stage": "GPU2D_FORMAL",
                **result,
                "evaluation_classification": "GPU2D_FORMAL_EVALUATION_VALIDATED",
                "claim_boundary": final["claim_boundary"],
            },
        )


def update_registry(final: dict[str, Any]) -> None:
    with REGISTRY.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [row for row in reader if row.get("experiment_id") not in RUN_IDS]
    for run_id in RUN_IDS:
        result = final["per_run_benchmark_results"][run_id]
        training = result["training"]
        math = result["benchmarks"]["math500"]
        gsm = result["benchmarks"]["gsm8k_fixed_200"]
        values = {
            "experiment_id": run_id,
            "created_at": "2026-09-19T13:36:56Z",
            "git_commit": BASELINE_HEAD,
            "config_hash": result["artifact_identity"]["config_semantic_sha256"],
            "dataset_hash": result["artifact_identity"]["dataset_encoded_sha256"],
            "model": "Qwen/Qwen3-0.6B-Base",
            "dataset": "GPU2D frozen 512-context math SFT",
            "seed": result["seed"],
            "batch_size": 1,
            "effective_batch_size": 16,
            "rank": 16,
            "target_modules": "all-linear",
            "precision": "bf16",
            "trainable_params": training["trainable_parameters"],
            "peak_vram_mb": training["physical_peak_vram_mib"],
            "tokens_per_second": training["median_target_tokens_per_second"],
            "final_loss": training["final_training_loss"],
            "eval_score": math["accuracy"],
            "status": "completed",
            "device": "cuda",
            "gpu_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
            "torch_version": "2.11.0+cu128",
            "torch_cuda_version": "12.8",
            "classification": "GPU2D_FORMAL_RUN_VALID",
            "sequence_length": 512,
            "optimizer_steps": 1200,
            "micro_batch": 1,
            "gradient_accumulation": 16,
            "final_train_loss": training["final_training_loss"],
            "final_validation_loss": training["validation_loss"],
            "peak_allocated_vram_mib": training["peak_cuda_allocated_mib"],
            "peak_reserved_vram_mib": training["peak_cuda_reserved_mib"],
            "median_tokens_per_second": training["median_target_tokens_per_second"],
            "elapsed_seconds": training["wall_clock_seconds"],
            "nan_events": 0,
            "inf_events": 0,
            "oom_events": training["oom_events"],
            "optimizer": "AdamW" if result["method"] == "LORA" else "PagedAdamW8bit",
            "primary_classification": "GPU2D_FORMAL_RUN_VALID",
            "regression_validation": "PASS",
            "regression_evidence": json.dumps(
                {
                    "result": f"experiments/results/{run_id}.json",
                    "MATH500": {"correct": math["correct"], "total": math["total"]},
                    "GSM8K_FIXED_200": {"correct": gsm["correct"], "total": gsm["total"]},
                },
                sort_keys=True,
            ),
        }
        rows.append({field: values.get(field, "") for field in fields})
    with REGISTRY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    try:
        final = audit()
    except IntegrityError as exc:
        print(f"GPU2D_FORMAL_EVALUATION_INTEGRITY_FAILURE: {exc}", file=sys.stderr)
        return 2
    write_per_run_results(final)
    update_registry(final)
    document_paths = [
        ROOT / "README.md",
        ROOT / "reports/FINAL_REPORT.md",
        ROOT / "reports/GPU2D_FORMAL_PEFT_TRANSFER.md",
        ROOT / "reports/PORTFOLIO_RESUME_EVIDENCE.md",
    ]
    final["artifact_identities"]["portfolio_documents"] = {
        rel(path): file_sha256(path) for path in document_paths
    }
    final["artifact_identities"]["per_run_result_hashes"] = {
        rel(RESULT_ROOT / f"{run_id}.json"): file_sha256(RESULT_ROOT / f"{run_id}.json")
        for run_id in RUN_IDS
    }
    final["artifact_identities"]["experiment_registry_sha256"] = file_sha256(REGISTRY)
    write_json(FINAL_RESULT, final)
    (FORMAL / "final_result.sha256").write_text(
        f"{file_sha256(FINAL_RESULT)}  final_result.json\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "status": final["primary_classification"],
                "scientific_classification": final["scientific_classification"],
                "valid_jobs": final["evaluation_campaign_status"]["valid_jobs"],
                "valid_problem_outputs": final["evaluation_campaign_status"][
                    "valid_problem_outputs"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
