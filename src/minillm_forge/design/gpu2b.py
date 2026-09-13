from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from minillm_forge.data.dedup import text_hash
from minillm_forge.data.manifest import dataset_digest, file_digest
from minillm_forge.data.sft import DEFAULT_SYSTEM_PROMPT, _as_ids, encode_sft_example

STAGE = "STAGE_GPU_2B_D"
BASELINE_COMMIT = "5479eaea03432532af6d4eb7ded4cda97176b3d7"
MODEL = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
TOKENIZER_DIGEST = "3fe32a8d9deb69b4b17efbc1af1bbde2dc345949efdc975ff315a3d066821cfe"
CPT_MODEL = "runs/E04-qwen3-math-cpt/final_model"
CPT_RESULT = "artifacts/training/qwen_math_cpt_result.json"
SFT_DATASET = "open-r1/OpenR1-Math-220k"
SFT_REVISION = "e4e141ec9dea9f8326f4d347be56105859b2bd68"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
SEEDS = (42, 31415, 271828)
ARMS = ("B-FULL", "C-FULL", "B-LORA", "C-LORA", "B-QLORA", "C-QLORA")
MAX_LENGTH = 1024
TOTAL_EXAMPLES = 20_000
VALIDATION_EXAMPLES = 800
TRAIN_EXAMPLES = TOTAL_EXAMPLES - VALIDATION_EXAMPLES
NEAR_NGRAM = 5
NEAR_JACCARD = 0.80
OVERLAP_CONTAINMENT = 0.60
OVERLAP_MIN_SHARED = 8

ROOT = Path("data/processed/gpu2b")
SFT_ALL = ROOT / "sft_math_v1.jsonl"
SFT_TRAIN = ROOT / "sft_math_v1_train.jsonl"
SFT_VALIDATION = ROOT / "sft_math_v1_validation.jsonl"
GSM8K_FILE = ROOT / "gsm8k_test.jsonl"
MATH500_FILE = ROOT / "math500_test.jsonl"
SFT_MANIFEST = Path("artifacts/data_manifests/sft-dataset-manifest.json")
CONTAMINATION_REPORT = Path("artifacts/data_manifests/sft-contamination-report.json")
EVALUATION_MANIFEST = Path("artifacts/eval_manifests/evaluation-manifest.json")
SEED_MANIFEST = Path("artifacts/training/seed-manifest.json")
DESIGN_RESULT = Path("artifacts/training/gpu2b-design-result.json")
REGRESSION_RESULT = Path("artifacts/training/gpu2b-design-regression.json")
CHECKSUMS = Path("artifacts/design/gpu2b/checksums.txt")
FORMAL_ARTIFACTS = (
    Path("reports/GPU2B_DESIGN.md"),
    SFT_MANIFEST,
    CONTAMINATION_REPORT,
    EVALUATION_MANIFEST,
    Path("reports/gpu2b/prompt-format.md"),
    Path("configs/sft/gpu2b/full-sft-config.yaml"),
    Path("configs/sft/gpu2b/lora-config.yaml"),
    Path("configs/sft/gpu2b/qlora-config.yaml"),
    SEED_MANIFEST,
    Path("reports/gpu2b/memory-qualification-plan.md"),
    Path("reports/gpu2b/resume-qualification-plan.md"),
    Path("reports/gpu2b/decision-rules.md"),
    Path("reports/gpu2b/claims-and-nonclaims.md"),
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _word_ngrams(text: str, n: int = NEAR_NGRAM) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = re.findall(r"\w+", normalized, flags=re.UNICODE)
    if not words:
        return set()
    if len(words) <= n:
        return {" ".join(words)}
    return {" ".join(words[index : index + n]) for index in range(len(words) - n + 1)}


@dataclass(frozen=True)
class Probe:
    source: str
    index: int
    problem: str
    exact_digest: str
    ngrams: frozenset[str]


class BenchmarkCollisionIndex:
    def __init__(self, benchmark_rows: list[dict[str, str]]) -> None:
        self.probes: list[Probe] = []
        self.exact: dict[str, list[int]] = defaultdict(list)
        self.inverted: dict[str, set[int]] = defaultdict(set)
        for row in benchmark_rows:
            problem = row["problem"]
            probe = Probe(
                source=row["benchmark"],
                index=int(row["benchmark_index"]),
                problem=problem,
                exact_digest=text_hash(problem),
                ngrams=frozenset(_word_ngrams(problem)),
            )
            probe_id = len(self.probes)
            self.probes.append(probe)
            self.exact[probe.exact_digest].append(probe_id)
            for ngram in probe.ngrams:
                self.inverted[ngram].add(probe_id)

    def collisions(self, problem: str) -> list[dict[str, Any]]:
        digest = text_hash(problem)
        collisions: list[dict[str, Any]] = []
        exact_ids = set(self.exact.get(digest, ()))
        for probe_id in sorted(exact_ids):
            probe = self.probes[probe_id]
            collisions.append(
                {
                    "benchmark": probe.source,
                    "benchmark_index": probe.index,
                    "match_type": "normalized_exact",
                    "score": 1.0,
                }
            )
        grams = _word_ngrams(problem)
        candidates: set[int] = set()
        for gram in grams:
            candidates.update(self.inverted.get(gram, ()))
        for probe_id in sorted(candidates - exact_ids):
            probe = self.probes[probe_id]
            shared = len(grams & probe.ngrams)
            if shared == 0:
                continue
            union = len(grams | probe.ngrams)
            jaccard = shared / max(union, 1)
            containment = shared / max(min(len(grams), len(probe.ngrams)), 1)
            if jaccard >= NEAR_JACCARD:
                match_type, score = "near_duplicate_word_5gram_jaccard", jaccard
            elif shared >= OVERLAP_MIN_SHARED and containment >= OVERLAP_CONTAINMENT:
                match_type, score = "problem_statement_overlap_word_5gram_containment", containment
            else:
                continue
            collisions.append(
                {
                    "benchmark": probe.source,
                    "benchmark_index": probe.index,
                    "match_type": match_type,
                    "score": round(score, 6),
                    "shared_ngrams": shared,
                }
            )
        return collisions


def _truthy_collection(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return any(bool(item) for item in value)
    return bool(value)


def _candidate(row: dict[str, Any], source_index: int) -> dict[str, Any] | None:
    problem = str(row.get("problem") or "").strip()
    solution = row.get("solution")
    if isinstance(solution, list):
        solution = solution[0] if solution else None
    solution = str(solution or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not problem or not solution or not answer or not re.search(r"[\w\\]", answer):
        return None
    if len(solution) > 12_000 or "\x00" in problem or "\x00" in solution:
        return None
    if not _truthy_collection(row.get("is_reasoning_complete")):
        return None
    if not _truthy_collection(row.get("correctness_math_verify")):
        return None
    return {
        "source_index": source_index,
        "uuid": str(row.get("uuid") or text_hash(problem)),
        "source": str(row.get("source") or "unknown"),
        "problem": problem,
        "solution": solution,
        "reference_answer": answer,
    }


def _serialized_lengths(tokenizer: Any, record: dict[str, Any]) -> tuple[int, int, int]:
    encoded = encode_sft_example(
        tokenizer,
        user=record["problem"],
        assistant=record["solution"],
        system=DEFAULT_SYSTEM_PROMPT,
        max_length=MAX_LENGTH,
    )
    target_tokens = sum(label != -100 for label in encoded["labels"])
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": record["problem"]},
        {"role": "assistant", "content": record["solution"]},
    ]
    untruncated = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    return len(encoded["input_ids"]), target_tokens, len(_as_ids(untruncated))


def _partition_summary(records: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    return {
        "examples": len(records),
        "input_tokens_after_truncation": sum(item["input_tokens"] for item in records),
        "assistant_target_tokens": sum(item["target_tokens"] for item in records),
        "untruncated_tokens": sum(item["untruncated_tokens"] for item in records),
        "truncated_examples": sum(bool(item["truncated"]) for item in records),
        "record_digest": dataset_digest(records),
        "path": str(path),
        "file_sha256": file_digest(path),
    }


def _load_benchmarks() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    specs = [
        ("gsm8k", "openai/gsm8k", "main", "test", GSM8K_REVISION, "question", "answer"),
        (
            "math500",
            "HuggingFaceH4/MATH-500",
            "default",
            "test",
            MATH500_REVISION,
            "problem",
            "answer",
        ),
    ]
    all_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {}
    output_paths = {"gsm8k": GSM8K_FILE, "math500": MATH500_FILE}
    for name, dataset_name, subset, split, revision, problem_field, answer_field in specs:
        dataset = load_dataset(dataset_name, subset, split=split, revision=revision)
        rows = [
            {
                "benchmark": name,
                "benchmark_index": index,
                "problem": str(row[problem_field]).strip(),
                "answer": str(row[answer_field]).strip(),
            }
            for index, row in enumerate(dataset)
        ]
        _write_jsonl(output_paths[name], rows)
        manifest[name] = {
            "dataset": dataset_name,
            "subset": subset,
            "split": split,
            "revision": revision,
            "license": (
                "MIT"
                if name == "gsm8k"
                else "not declared by HF snapshot; upstream openai/prm800k is MIT"
            ),
            "examples": len(rows),
            "record_digest": dataset_digest(rows),
            "snapshot_path": str(output_paths[name]),
            "snapshot_sha256": file_digest(output_paths[name]),
            "problem_field": problem_field,
            "answer_field": answer_field,
        }
        all_rows.extend(rows)
    controlled = []
    for index, line in enumerate(
        Path("data/eval/controlled_math.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        if line.strip():
            row = json.loads(line)
            controlled.append(
                {
                    "benchmark": "controlled_math",
                    "benchmark_index": index,
                    "problem": str(row["problem"]).strip(),
                    "answer": str(row["answer"]).strip(),
                }
            )
    manifest["controlled_math"] = {
        "dataset": "local controlled-math-v1",
        "split": "test",
        "examples": len(controlled),
        "record_digest": dataset_digest(controlled),
        "snapshot_path": "data/eval/controlled_math.jsonl",
        "snapshot_sha256": file_digest("data/eval/controlled_math.jsonl"),
        "role": "secondary diagnostic only",
    }
    all_rows.extend(controlled)
    for name, path in (
        ("frozen_math_ppl", Path("data/processed/qwen-gpu2a/math_validation.jsonl")),
        ("frozen_general_ppl", Path("data/processed/qwen-gpu2a/general_validation.jsonl")),
    ):
        rows = []
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            source_row = json.loads(line)
            rows.append(
                {
                    "benchmark": name,
                    "benchmark_index": index,
                    "problem": str(source_row["text"]).strip(),
                    "answer": "",
                }
            )
        manifest[name] = {
            "dataset": f"frozen GPU-2A {name}",
            "split": "validation",
            "examples": len(rows),
            "record_digest": dataset_digest(rows),
            "snapshot_path": str(path),
            "snapshot_sha256": file_digest(path),
            "role": "PPL diagnostic and contamination-protection probe",
        }
        all_rows.extend(rows)
    return all_rows, manifest


def _permutation_digest(length: int, seed: int) -> str:
    order = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    digest = hashlib.sha256()
    for index in order:
        digest.update(struct.pack("<I", index))
    return digest.hexdigest()


def prepare() -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    benchmark_rows, benchmark_manifest = _load_benchmarks()
    collision_index = BenchmarkCollisionIndex(benchmark_rows)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    stream = load_dataset(
        SFT_DATASET,
        "default",
        split="train",
        revision=SFT_REVISION,
        streaming=True,
    ).select_columns(
        [
            "problem",
            "solution",
            "answer",
            "source",
            "uuid",
            "is_reasoning_complete",
            "correctness_math_verify",
        ]
    )
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    collisions: list[dict[str, Any]] = []
    scanned = 0
    for source_index, row in enumerate(stream):
        scanned += 1
        if scanned % 5_000 == 0:
            print(
                f"scanned={scanned} accepted={len(accepted)} rejections={dict(rejection_counts)}",
                flush=True,
            )
        candidate = _candidate(dict(row), source_index)
        if candidate is None:
            rejection_counts["quality"] += 1
            continue
        digest = text_hash(candidate["problem"])
        if digest in seen:
            rejection_counts["duplicate"] += 1
            continue
        seen.add(digest)
        matches = collision_index.collisions(candidate["problem"])
        if matches:
            rejection_counts["benchmark_collision"] += 1
            collisions.append(
                {
                    "source_index": source_index,
                    "uuid": candidate["uuid"],
                    "problem_sha256": digest,
                    "matches": matches,
                }
            )
            continue
        try:
            input_tokens, target_tokens, untruncated_tokens = _serialized_lengths(
                tokenizer, candidate
            )
        except ValueError:
            rejection_counts["serialization"] += 1
            continue
        if target_tokens < 16:
            rejection_counts["too_few_target_tokens"] += 1
            continue
        candidate.update(
            {
                "input_tokens": input_tokens,
                "target_tokens": target_tokens,
                "untruncated_tokens": untruncated_tokens,
                "truncated": untruncated_tokens > MAX_LENGTH,
            }
        )
        accepted.append(candidate)
        if len(accepted) == TOTAL_EXAMPLES:
            break
    if len(accepted) != TOTAL_EXAMPLES:
        raise RuntimeError(f"only selected {len(accepted)} clean SFT examples")
    validation_ids = {
        record["uuid"]
        for record in sorted(
            accepted,
            key=lambda record: hashlib.sha256(
                f"gpu2b-validation-v1:{record['uuid']}".encode()
            ).hexdigest(),
        )[:VALIDATION_EXAMPLES]
    }
    train = [record for record in accepted if record["uuid"] not in validation_ids]
    validation = [record for record in accepted if record["uuid"] in validation_ids]
    if len(train) != TRAIN_EXAMPLES or len(validation) != VALIDATION_EXAMPLES:
        raise RuntimeError("hash split did not produce the frozen 19,200/800 partition")
    _write_jsonl(SFT_ALL, accepted)
    _write_jsonl(SFT_TRAIN, train)
    _write_jsonl(SFT_VALIDATION, validation)

    manifest = {
        "stage": STAGE,
        "dataset_id": "SFT_MATH_V1",
        "dataset": SFT_DATASET,
        "subset": "default",
        "split": "train",
        "revision": SFT_REVISION,
        "license": "Apache-2.0",
        "selection": (
            "first 20,000 accepted in pinned streaming order after frozen quality, "
            "deduplication, serialization, and benchmark-collision filters"
        ),
        "scanned_source_rows": scanned,
        "accepted_examples": len(accepted),
        "rejections": dict(sorted(rejection_counts.items())),
        "fields": {
            "prompt": "problem",
            "response": "solution",
            "reference_answer": "reference_answer",
        },
        "serialization": "Qwen chat template; system+user masked; assistant-only loss",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "max_length": MAX_LENGTH,
        "tokenizer": MODEL,
        "tokenizer_revision": MODEL_REVISION,
        "tokenizer_digest": TOKENIZER_DIGEST,
        "all": _partition_summary(accepted, SFT_ALL),
        "train": _partition_summary(train, SFT_TRAIN),
        "validation": _partition_summary(validation, SFT_VALIDATION),
        "split_rule": (
            "800 smallest sha256('gpu2b-validation-v1:' + uuid) values form validation; "
            "source order preserved within each partition"
        ),
        "formal_budget": {
            "epochs": 1,
            "training_examples": TRAIN_EXAMPLES,
            "assistant_target_tokens": sum(item["target_tokens"] for item in train),
            "optimizer_steps": 1200,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 16,
        },
        "baseline_commit": BASELINE_COMMIT,
    }
    _write_json(SFT_MANIFEST, manifest)
    contamination = {
        "stage": STAGE,
        "procedure_frozen_before_training": True,
        "training_candidates_scanned": scanned,
        "accepted_examples": len(accepted),
        "removed_colliding_examples": rejection_counts["benchmark_collision"],
        "normalized_exact": {
            "normalization": "NFKC + casefold + remove non-word",
            "threshold": 1.0,
        },
        "near_duplicate": {"unit": "word 5-gram", "metric": "Jaccard", "threshold": NEAR_JACCARD},
        "problem_statement_overlap": {
            "unit": "word 5-gram",
            "metric": "intersection / smaller-set size",
            "threshold": OVERLAP_CONTAINMENT,
            "minimum_shared_ngrams": OVERLAP_MIN_SHARED,
        },
        "benchmarks": {
            name: {"examples": values["examples"], "record_digest": values["record_digest"]}
            for name, values in benchmark_manifest.items()
        },
        "collisions": collisions,
        "policy": "every matched SFT item is removed before the 20,000-example set is finalized",
        "remaining_known_collisions": 0,
        "limitation": (
            "lexical matching cannot rule out semantic paraphrases or upstream "
            "model-pretraining contamination"
        ),
    }
    _write_json(CONTAMINATION_REPORT, contamination)
    evaluation = {
        "stage": STAGE,
        "primary_metric_class": "GENERATED_MATH_TASK_PERFORMANCE",
        "primary_benchmarks": {name: benchmark_manifest[name] for name in ("gsm8k", "math500")},
        "diagnostic_probes": {
            "controlled_math": benchmark_manifest["controlled_math"],
            "math_ppl": {
                **benchmark_manifest["frozen_math_ppl"],
                "manifest": "artifacts/data_manifests/qwen_math_validation.json",
            },
            "general_ppl": {
                **benchmark_manifest["frozen_general_ppl"],
                "manifest": "artifacts/data_manifests/qwen_general_validation.json",
            },
        },
        "generation": {
            "prompt": "Qwen chat template over frozen system prompt and benchmark problem",
            "do_sample": False,
            "temperature": None,
            "num_beams": 1,
            "max_new_tokens": 512,
            "stop_token_ids": [151643, 151645],
            "pad_token_id": 151643,
            "batch_size": 1,
        },
        "scoring": {
            "primary": "deterministic exact match after frozen extraction/normalization",
            "gsm8k_reference_extraction": "substring after final #### marker",
            "math500_reference_field": "answer",
            "prediction_extraction": (
                "last balanced boxed expression; else final-answer marker; "
                "else final non-empty line"
            ),
            "manual_correction": False,
            "evaluator": "src/minillm_forge/evaluation/math_eval.py",
            "evaluator_sha256": file_digest("src/minillm_forge/evaluation/math_eval.py"),
        },
        "tokenizer_contract": {
            "name": MODEL,
            "revision": MODEL_REVISION,
            "digest": TOKENIZER_DIGEST,
            "vocabulary_resize": False,
            "eos_token_id": 151643,
            "im_end_token_id": 151645,
            "pad_token_id": 151643,
        },
    }
    _write_json(EVALUATION_MANIFEST, evaluation)
    seed_manifest = {
        "stage": STAGE,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "formal_runs": [f"{arm}-s{seed}" for arm in ARMS for seed in SEEDS],
        "paired_across_all_six_arms": True,
        "sampler": "torch.randperm over the frozen 19,200-row train partition",
        "permutations": {
            str(seed): {
                "examples": TRAIN_EXAMPLES,
                "uint32_le_sha256": _permutation_digest(TRAIN_EXAMPLES, seed),
            }
            for seed in SEEDS
        },
        "validation_partition_is_seed_invariant": True,
        "failed_seed_replacement": "forbidden",
    }
    _write_json(SEED_MANIFEST, seed_manifest)
    return {
        "classification": "GPU2B_DATA_FREEZE_COMPLETE",
        "manifest": str(SFT_MANIFEST),
        "contamination_report": str(CONTAMINATION_REPORT),
        "evaluation_manifest": str(EVALUATION_MANIFEST),
        "seed_manifest": str(SEED_MANIFEST),
        "training_examples": len(train),
        "validation_examples": len(validation),
        "training_target_tokens": manifest["train"]["assistant_target_tokens"],
        "removed_collisions": rejection_counts["benchmark_collision"],
    }


def recount() -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    partitions: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    for name, path in (
        ("all", SFT_ALL),
        ("train", SFT_TRAIN),
        ("validation", SFT_VALIDATION),
    ):
        records = _read_jsonl(path)
        for record in records:
            input_tokens, target_tokens, untruncated_tokens = _serialized_lengths(tokenizer, record)
            record.update(
                {
                    "input_tokens": input_tokens,
                    "target_tokens": target_tokens,
                    "untruncated_tokens": untruncated_tokens,
                    "truncated": untruncated_tokens > MAX_LENGTH,
                }
            )
        _write_jsonl(path, records)
        partitions[name] = (path, records)
    manifest = json.loads(SFT_MANIFEST.read_text(encoding="utf-8"))
    for name, (path, records) in partitions.items():
        manifest[name] = _partition_summary(records, path)
    manifest["formal_budget"]["assistant_target_tokens"] = manifest["train"][
        "assistant_target_tokens"
    ]
    _write_json(SFT_MANIFEST, manifest)
    return {
        "classification": "GPU2B_LOCAL_TOKEN_RECOUNT_COMPLETE",
        "training_target_tokens": manifest["train"]["assistant_target_tokens"],
        "training_untruncated_tokens": manifest["train"]["untruncated_tokens"],
        "training_truncated_examples": manifest["train"]["truncated_examples"],
        "training_digest": manifest["train"]["record_digest"],
    }


def validate() -> dict[str, Any]:
    import yaml

    manifest = json.loads(SFT_MANIFEST.read_text(encoding="utf-8"))
    contamination = json.loads(CONTAMINATION_REPORT.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION_MANIFEST.read_text(encoding="utf-8"))
    seeds = json.loads(SEED_MANIFEST.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest["train"]["examples"] != TRAIN_EXAMPLES:
        errors.append("train example count is not 19,200")
    if manifest["validation"]["examples"] != VALIDATION_EXAMPLES:
        errors.append("validation example count is not 800")
    for key in ("train", "validation", "all"):
        path = Path(manifest[key]["path"])
        if not path.exists() or file_digest(path) != manifest[key]["file_sha256"]:
            errors.append(f"{key} data snapshot hash mismatch")
    if contamination["remaining_known_collisions"] != 0:
        errors.append("known benchmark collisions remain")
    if set(evaluation["primary_benchmarks"]) != {"gsm8k", "math500"}:
        errors.append("evaluation benchmark set is incomplete")
    expected_runs = [f"{arm}-s{seed}" for arm in ARMS for seed in SEEDS]
    if (
        seeds["seeds"] != list(SEEDS)
        or seeds.get("arms") != list(ARMS)
        or seeds.get("formal_runs") != expected_runs
        or len(seeds["permutations"]) != 3
    ):
        errors.append("paired seed design is not frozen")
    if evaluation["scoring"]["evaluator_sha256"] != file_digest(evaluation["scoring"]["evaluator"]):
        errors.append("frozen evaluator digest mismatch")
    methods = {
        "full-sft-config.yaml": "full",
        "lora-config.yaml": "lora",
        "qlora-config.yaml": "qlora",
    }
    initialization_contracts = []
    for filename, method in methods.items():
        path = Path("configs/sft/gpu2b") / filename
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        initialization_contracts.append(config["design"]["paired_initializations"])
        if config["model"]["method"] != method:
            errors.append(f"{filename} method mismatch")
        if config["data"]["target_token_budget"] != manifest["train"]["assistant_target_tokens"]:
            errors.append(f"{filename} target-token budget mismatch")
        if config["data"]["train_examples"] != TRAIN_EXAMPLES:
            errors.append(f"{filename} train-example budget mismatch")
        if config["training"]["seeds"] != list(SEEDS):
            errors.append(f"{filename} seed list mismatch")
        if config["training"]["max_steps"] != 1200:
            errors.append(f"{filename} optimizer-step budget mismatch")
        if config["training"]["checkpoint_selection"] != "FINAL_TOKEN_BUDGET_CHECKPOINT":
            errors.append(f"{filename} checkpoint policy mismatch")
    if any(contract != initialization_contracts[0] for contract in initialization_contracts[1:]):
        errors.append("family initialization contracts differ")
    for path in FORMAL_ARTIFACTS:
        if not path.exists():
            errors.append(f"missing formal artifact: {path}")
    if CHECKSUMS.exists():
        for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
            expected, relative_path = line.split("  ", 1)
            if file_digest(relative_path) != expected:
                errors.append(f"checksum mismatch: {relative_path}")
    return {
        "classification": "PASS" if not errors else "FAIL",
        "stage": STAGE,
        "errors": errors,
        "execution_class": "ZERO_TRAINING_WORK",
        "training_executed": False,
    }


def run_regression() -> dict[str, Any]:
    commands = {
        "pytest": [sys.executable, "-m", "pytest", "-q"],
        "ruff": [sys.executable, "-m", "ruff", "check", "."],
        "format": [sys.executable, "-m", "ruff", "format", "--check", "."],
        "lock": ["uv", "lock", "--check"],
        "build": ["uv", "build"],
    }
    checks: dict[str, Any] = {}
    for name, command in commands.items():
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        checks[name] = {
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "returncode": completed.returncode,
            "command": command,
            "output": (completed.stdout + completed.stderr)[-8000:],
        }
    import yaml

    tracked = subprocess.check_output(["git", "ls-files", "*.yaml", "*.yml"], text=True)
    yaml_files = {Path(line) for line in tracked.splitlines() if line.strip()}
    yaml_files.update(Path("configs/sft/gpu2b").glob("*.yaml"))
    yaml_error = None
    try:
        for path in sorted(yaml_files):
            yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        yaml_error = f"{type(exc).__name__}: {exc}"
    checks["yaml"] = {
        "status": "PASS" if yaml_error is None else "FAIL",
        "scope": "version-controlled YAML plus new GPU-2B-D family configs",
        "files_parsed": len(yaml_files),
        "predecessor_files": 26,
        "expected_delta": 3,
        "error": yaml_error,
    }
    result = {
        "classification": (
            "PASS" if all(value["status"] == "PASS" for value in checks.values()) else "FAIL"
        ),
        "stage": STAGE,
        "design_commit": _git_commit(),
        "checks": checks,
        "training_executed": False,
    }
    _write_json(REGRESSION_RESULT, result)
    return result


def finalize() -> dict[str, Any]:
    regression = json.loads(REGRESSION_RESULT.read_text(encoding="utf-8"))
    if regression["classification"] != "PASS":
        raise RuntimeError("design regression must pass before finalization")
    validation = validate()
    if validation["classification"] != "PASS":
        raise RuntimeError(f"design validation failed: {validation['errors']}")
    CHECKSUMS.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUMS.write_text(
        "".join(f"{file_digest(path)}  {path.as_posix()}\n" for path in FORMAL_ARTIFACTS),
        encoding="utf-8",
    )
    validation = validate()
    if validation["classification"] != "PASS":
        raise RuntimeError(f"checksum validation failed: {validation['errors']}")
    manifest = json.loads(SFT_MANIFEST.read_text(encoding="utf-8"))
    result = {
        "classification": "STAGE_GPU_2B_D_COMPLETE",
        "execution_class": "ZERO_TRAINING_WORK",
        "design_baseline": BASELINE_COMMIT,
        "design_commit": _git_commit(),
        "factorial_design": "2_INIT x 3_ADAPTATION",
        "formal_arms": 6,
        "formal_seeds_per_arm": 3,
        "maximum_formal_training_runs": 18,
        "primary_science": "CPT_TO_DOWNSTREAM_SFT_TRANSFER",
        "primary_metric_class": "GENERATED_MATH_TASK_PERFORMANCE",
        "secondary_metrics": ["MATH_PPL", "GENERAL_PPL"],
        "sft_execution": "NOT_RUN",
        "training_executed": False,
        "sft_dataset": SFT_DATASET,
        "sft_revision": SFT_REVISION,
        "training_examples": TRAIN_EXAMPLES,
        "validation_examples": VALIDATION_EXAMPLES,
        "assistant_target_token_budget": manifest["train"]["assistant_target_tokens"],
        "primary_benchmarks": ["GSM8K", "MATH-500"],
        "seeds": list(SEEDS),
        "regression": regression["classification"],
        "checksums": str(CHECKSUMS),
        "formal_artifact_count": len(FORMAL_ARTIFACTS) + 1,
    }
    _write_json(DESIGN_RESULT, result)
    return result
