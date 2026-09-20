from __future__ import annotations

# ruff: noqa: E501
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minillm_forge.evaluation.math_eval import (
    _balanced_boxed_answers,
    extract_final_answer,
    normalize_answer,
)

GPU3A_HEAD = "b41c718c337946c14a0bf80dcbacf68a1129d633"
BASELINE_HEAD = "91c7efe6a646b0b8a9c6592390abbf36ed6f58af"
SELECTED_ADAPTERS = ["B-LORA-s42", "C-LORA-s42", "B-QLORA-s42", "C-QLORA-s42"]
DEVELOPMENT_IDS = [
    "gsm8k-0240",
    "gsm8k-0483",
    "gsm8k-0579",
    "gsm8k-0845",
    "math500-0153",
    "math500-0254",
    "math500-0269",
    "math500-0374",
]
CONFIRMATION_IDS = [
    "gsm8k-1176",
    "gsm8k-0805",
    "gsm8k-0198",
    "gsm8k-0222",
    "math500-0076",
    "math500-0181",
    "math500-0324",
    "math500-0478",
]

ANSWER_COLON = re.compile(r"(?:final\s+answer|answer)\s*:\s*([^\r\n]*)", re.I)
ANSWER_IS = re.compile(r"(?:final\s+answer|answer)\s+is\s+([^\r\n]*)", re.I)
ANSWER_IS_COMPLETE = re.compile(r"(?:final\s+answer|answer)\s+is\s+.+?\.\s", re.I)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@dataclass(frozen=True)
class AnswerCandidate:
    position: int
    end: int
    value: str
    source: str
    raw_value: str

    @property
    def normalized(self) -> str:
        return normalize_answer(self.value)


def _balanced_boxed_with_positions(text: str) -> list[AnswerCandidate]:
    candidates: list[AnswerCandidate] = []
    marker = r"\boxed{"
    start = 0
    while (marker_index := text.find(marker, start)) >= 0:
        content_start = marker_index + len(marker)
        depth = 1
        index = content_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            value = text[content_start : index - 1].strip()
            if value:
                candidates.append(AnswerCandidate(marker_index, index, value, "boxed", value))
        start = content_start
    return candidates


def _clean_answer_value(value: str) -> str:
    raw = value.strip()
    boxed = _balanced_boxed_with_positions(raw)
    if boxed:
        return boxed[-1].value
    numeric = re.match(r"^\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?)\s*\$?", raw)
    if numeric:
        return numeric.group(1)
    # Preserve decimals but drop a following prose sentence.
    sentence = re.split(r"\.\s+(?=[A-Za-z#])", raw, maxsplit=1)[0]
    return sentence.rstrip(". ")


def answer_candidates(text: str, *, allow_fallback: bool = True) -> list[AnswerCandidate]:
    # Collect every explicit, structurally valid answer across formats. This is intentionally
    # broader than the legacy format-priority extractor so conflict-aware can audit cross-format
    # disagreements rather than silently selecting one representation.
    candidates = _balanced_boxed_with_positions(text)
    for match in ANSWER_COLON.finditer(text):
        raw = match.group(1).strip()
        value = _clean_answer_value(raw)
        if value:
            candidates.append(
                AnswerCandidate(match.start(), match.end(), value, "answer_marker", raw)
            )
    for match in ANSWER_IS.finditer(text):
        raw = match.group(1).strip()
        value = _clean_answer_value(raw)
        if value:
            candidates.append(AnswerCandidate(match.start(), match.end(), value, "answer_is", raw))
    if candidates or not allow_fallback:
        return sorted(candidates, key=lambda candidate: (candidate.position, candidate.end))
    lines = [
        (match.start(), match.end(), match.group().strip())
        for match in re.finditer(r"[^\r\n]+", text)
    ]
    if not lines:
        return []
    position, end, raw = lines[-1]
    value = extract_final_answer(text)
    return [AnswerCandidate(position, end, value, "legacy_fallback", raw)] if value else []


def has_complete_answer(text: str) -> bool:
    if _balanced_boxed_answers(text):
        return True
    # A line-oriented answer is complete only after a newline. Merely seeing ANSWER: is insufficient.
    for match in ANSWER_COLON.finditer(text):
        if match.group(1).strip() and match.end() < len(text) and text[match.end()] in "\r\n":
            return True
    return bool(ANSWER_IS_COMPLETE.search(text))


def extract_protocol(text: str, protocol: str) -> dict[str, Any]:
    candidates = answer_candidates(text)
    normalized = [candidate.normalized for candidate in candidates if candidate.normalized]
    unique = list(dict.fromkeys(normalized))
    conflict = len(unique) > 1
    common = {
        "candidate_count": len(candidates),
        "candidate_values": [candidate.value for candidate in candidates],
        "candidate_raw_values": [candidate.raw_value for candidate in candidates],
        "candidate_sources": [candidate.source for candidate in candidates],
        "normalized_candidate_count": len(unique),
        "conflict": conflict,
    }
    if not candidates:
        return {**common, "status": "NO_VALID_ANSWER", "answer": None}
    if protocol == "first-valid-v1":
        return {**common, "status": "ANSWER", "answer": candidates[0].value}
    if protocol == "last-valid-v1":
        return {**common, "status": "ANSWER", "answer": candidates[-1].value}
    if protocol == "conflict-aware-v1":
        if conflict:
            return {**common, "status": "CONFLICT", "answer": None}
        return {**common, "status": "ANSWER", "answer": candidates[0].value}
    raise ValueError(f"unknown extraction protocol: {protocol}")


def _correct(answer: str | None, reference: str) -> bool:
    return answer is not None and normalize_answer(answer) == normalize_answer(reference)


def _historical_paths(repo: Path) -> list[Path]:
    return sorted(
        path
        for path in (repo / "runs/gpu2d-formal").glob("*/evaluation-amended/*.jsonl")
        if "physical" not in path.name
    )


def freeze_protocol(repo: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo).resolve()
    destination = repo / "artifacts/gpu3b/protocol.json"
    if destination.exists():
        existing = _json(destination)
        if existing.get("protocol_version") == "gpu3b-v2-frozen":
            return existing
        v1_path = destination.with_name("protocol_v1.json")
        if not v1_path.exists():
            _write_json(v1_path, existing)
    gpu2d_protocol = _json(repo / "artifacts/gpu2d_formal/protocol.json")
    summaries = {}
    for run_id in SELECTED_ADAPTERS:
        summary_path = repo / f"runs/gpu2d-formal/{run_id}/final_summary.json"
        summary = _json(summary_path)
        summaries[run_id] = {
            "initialization": summary["initialization"],
            "family": summary["family"],
            "seed": summary["seed"],
            "checkpoint_id": summary["checkpoint"]["final_sha256"],
            "checkpoint_path": summary["checkpoint"]["final_path"],
            "adapter_id": summary["adapter"]["state_sha256"],
            "adapter_path": summary["adapter"]["path"],
        }
    protocol = {
        "stage": "GPU-3B",
        "protocol_version": "gpu3b-v2-frozen",
        "baseline_commit": BASELINE_HEAD,
        "gpu3a_commit": GPU3A_HEAD,
        "branch_start_commit": GPU3A_HEAD,
        "model_id": "Qwen/Qwen3-0.6B-Base",
        "model_revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "selected_adapters": summaries,
        "tokenizer_revision": gpu2d_protocol["prompt"]["chat_template_sha256"],
        "dataset_revision": {
            name: value["revision"] for name, value in gpu2d_protocol["benchmarks"].items()
        },
        "dataset_hash": {
            name: value["snapshot_sha256"] for name, value in gpu2d_protocol["benchmarks"].items()
        },
        "prompt_template": gpu2d_protocol["prompt"],
        "historical_generation_config": gpu2d_protocol["generation"],
        "stop_protocols": {
            "S0": {"eos_token_ids": [151643, 151645], "external_answer_stop": False},
            "S1": {"eos_token_ids": [151643], "external_answer_stop": False},
            "S2": {"eos_token_ids": [151645], "external_answer_stop": False},
            "S3": {"eos_token_ids": [151643, 151645], "external_answer_stop": True},
        },
        "repetition_protocols": {
            "R0": {"repetition_penalty": 1.0, "no_repeat_ngram_size": 0},
            "R1": {"repetition_penalty": 1.05, "no_repeat_ngram_size": 0},
            "R2": {"repetition_penalty": 1.10, "no_repeat_ngram_size": 0},
            "R3": {"repetition_penalty": 1.15, "no_repeat_ngram_size": 0},
            "N3": {"repetition_penalty": 1.0, "no_repeat_ngram_size": 3},
            "N4": {"repetition_penalty": 1.0, "no_repeat_ngram_size": 4},
        },
        "length_protocols": {"L256": 256, "L512": 512, "L768": 768},
        "decoding_strategy": "deterministic greedy, single sample",
        "extraction_protocol_versions": [
            "legacy-gpu2d",
            "first-valid-v1",
            "last-valid-v1",
            "conflict-aware-v1",
        ],
        "development_sample_ids": DEVELOPMENT_IDS,
        "confirmation_sample_ids": CONFIRMATION_IDS,
        "partition_rule": "fixed problem IDs; no problem appears in both partitions; same partition across adapters",
        "random_seed": 20260920,
        "resource_limits": {
            "gpu_memory_mib": 8151,
            "minimum_headroom_mib": 1024,
            "per_generation_wall_time_seconds": 120,
            "maximum_new_tokens": 768,
            "maximum_new_generations": 96,
            "automatic_full_8400_parameter_search": False,
        },
        "historical_inputs": {
            path.relative_to(repo).as_posix(): _sha256(path) for path in _historical_paths(repo)
        },
        "status": "FROZEN_BEFORE_GPU3B_INFERENCE",
        "amendment": {
            "supersedes": "gpu3b-v1-frozen",
            "reason": "v1 confirmation GSM8K IDs were not members of the frozen 200-problem subset",
            "selection_rule": "lowest SHA256(gpu3b-confirm-v2:problem_id) among available frozen GSM8K IDs after excluding development IDs",
            "inference_executed_under_v1_confirmation_partition": 0,
        },
    }
    if set(DEVELOPMENT_IDS) & set(CONFIRMATION_IDS):
        raise RuntimeError("development and confirmation partitions overlap")
    _write_json(destination, protocol)
    return protocol


def _metric(rows: list[dict[str, Any]], protocol: str) -> dict[str, Any]:
    total = len(rows)
    extracted = sum(row[protocol]["status"] == "ANSWER" for row in rows)
    conflicts = sum(row[protocol]["conflict"] for row in rows)
    no_answer = sum(row[protocol]["status"] == "NO_VALID_ANSWER" for row in rows)
    correct = sum(row[protocol]["correct"] for row in rows)
    changed = sum(row[protocol]["answer"] != row["legacy"]["answer"] for row in rows)
    return {
        "records": total,
        "extraction_success_count": extracted,
        "extraction_success_rate": extracted / total,
        "correct_count": correct,
        "answer_accuracy": correct / total,
        "conflict_count": conflicts,
        "conflict_rate": conflicts / total,
        "no_valid_answer_count": no_answer,
        "no_valid_answer_rate": no_answer / total,
        "legacy_to_new_changed_count": changed,
    }


def run_extraction_ablation(repo: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo).resolve()
    freeze_protocol(repo)
    output = repo / "artifacts/gpu3b"
    protocols = ("first-valid-v1", "last-valid-v1", "conflict-aware-v1")
    details: list[dict[str, Any]] = []
    for path in _historical_paths(repo):
        for row in _iter_jsonl(path):
            legacy_answer = row.get("extracted_answer")
            item: dict[str, Any] = {
                "run_id": row["run_id"],
                "benchmark": row["benchmark"],
                "problem_id": row["problem_id"],
                "reference_answer": row["reference_answer"],
                "source_path": path.relative_to(repo).as_posix(),
                "legacy": {
                    "status": "ANSWER" if str(legacy_answer or "").strip() else "NO_VALID_ANSWER",
                    "answer": legacy_answer or None,
                    "correct": bool(row["correct"]),
                },
            }
            for protocol in protocols:
                result = extract_protocol(str(row.get("generated_text") or ""), protocol)
                result["correct"] = _correct(result["answer"], str(row["reference_answer"]))
                item[protocol] = result
            details.append(item)
    detail_path = output / "extraction_ablation.jsonl"
    with detail_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        groups[(row["run_id"], row["benchmark"])].append(row)
    legacy_failures = [row for row in details if row["legacy"]["status"] == "NO_VALID_ANSWER"]
    metrics: dict[str, Any] = {
        "records": len(details),
        "legacy": {
            "records": len(details),
            "extraction_success_count": sum(row["legacy"]["status"] == "ANSWER" for row in details),
            "extraction_success_rate": sum(row["legacy"]["status"] == "ANSWER" for row in details)
            / len(details),
            "correct_count": sum(row["legacy"]["correct"] for row in details),
            "answer_accuracy": sum(row["legacy"]["correct"] for row in details) / len(details),
        },
        "protocols": {protocol: _metric(details, protocol) for protocol in protocols},
        "groups": {
            f"{run_id}:{benchmark}": {protocol: _metric(rows, protocol) for protocol in protocols}
            for (run_id, benchmark), rows in sorted(groups.items())
        },
        "legacy_failures": {
            "count": len(legacy_failures),
            "recovered": {
                protocol: sum(row[protocol]["status"] == "ANSWER" for row in legacy_failures)
                for protocol in protocols
            },
            "correct_after_recovery": {
                protocol: sum(row[protocol]["correct"] for row in legacy_failures)
                for protocol in protocols
            },
            "conflicts": {
                protocol: sum(row[protocol]["conflict"] for row in legacy_failures)
                for protocol in protocols
            },
        },
        "legacy_success_with_conflict": {
            protocol: sum(
                row["legacy"]["status"] == "ANSWER" and row[protocol]["conflict"] for row in details
            )
            for protocol in protocols
        },
    }
    _write_json(output / "extraction_ablation.json", metrics)
    return metrics


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
