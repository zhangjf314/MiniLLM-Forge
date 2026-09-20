# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from minillm_forge.portfolio.evidence import (
    build_release_artifacts,
    read_json,
    sha256,
    write_json,
)

REQUIRED_CLAIM_FIELDS = {
    "claim_id", "claim_text", "category", "status", "evidence_artifact",
    "evidence_metric", "limitations", "resume_allowed", "interview_allowed",
}
REQUIRED_EXPERIMENT_FIELDS = {
    "experiment_id", "stage", "model", "initialization", "training_method", "dataset",
    "seed", "steps", "status", "classification", "primary_metrics", "artifact_path",
    "checkpoint_path", "historical_or_current", "resume_safe", "portfolio_status",
}
QUICK_FORBIDDEN_CALLS = (
    "subprocess.run(",
    ".backward(",
    "minillm_forge.training",
    "requests.",
    "urlopen(",
)


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def quick_verify(repo: str | Path = ".", *, write_output: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    repo = Path(repo).resolve()
    out = repo / "artifacts/gpu4c"
    protocol = read_json(out / "protocol.json")
    checks: list[dict[str, Any]] = []
    for relative, expected in protocol["frozen_evidence_hashes"].items():
        path = repo / relative
        actual = sha256(path) if path.is_file() else None
        checks.append(_check(f"hash:{relative}", actual == expected, f"expected={expected}; actual={actual}"))
    manifest = read_json(out / "final_evidence_manifest.json")
    claims = read_json(out / "claim_registry.json")
    experiments = read_json(out / "experiment_registry.json")
    checks.append(_check("manifest_schema", all(key in manifest for key in ("pretrained_checkpoint", "native_sft_checkpoint", "tokenizer", "stage_classifications", "major_dataset_hashes")), "required final evidence fields"))
    allowed = set(claims["allowed_statuses"])
    checks.append(_check("claim_schema", all(REQUIRED_CLAIM_FIELDS <= set(item) and item["status"] in allowed for item in claims["claims"]), f"claims={len(claims['claims'])}"))
    checks.append(_check("validated_claim_evidence", all((repo / item["evidence_artifact"]).exists() for item in claims["claims"] if item["status"] == "VALIDATED"), "every VALIDATED claim has a repository artifact"))
    checks.append(_check("experiment_schema", all(REQUIRED_EXPERIMENT_FIELDS <= set(item) for item in experiments["experiments"]), f"experiments={len(experiments['experiments'])}"))
    tokenizer = Tokenizer.from_file(str(repo / manifest["tokenizer"]))
    checks.append(_check("tokenizer_load", tokenizer.get_vocab_size() == 24000, f"vocab_size={tokenizer.get_vocab_size()}"))
    for label, relative, expected_step in (
        ("pretrained_checkpoint", manifest["pretrained_checkpoint"], 3052),
        ("native_sft_checkpoint", manifest["native_sft_checkpoint"], 300),
    ):
        payload = torch.load(repo / relative, map_location="cpu", weights_only=False, mmap=True)
        trainer = payload.get("trainer_state", {})
        actual_step = trainer.get("global_step", payload.get("global_step"))
        checks.append(_check(f"{label}_metadata", actual_step == expected_step and "model" in payload, f"global_step={actual_step}; model_state={'model' in payload}"))
        del payload
    g4 = read_json(repo / "artifacts/gpu4a/sft_evaluation.json")
    t1 = g4["metrics"]["test:T1_SUPPORT_CLASSIFICATION"]
    checks.append(_check("t1_subset_denominator", t1["count"] == 64 and t1["correct"] == 64, "fixed independent subset=64/64; not full test"))
    g3b = read_json(repo / "artifacts/gpu3b/gpu3b_result.json")
    checks.append(_check("gpu3b_evaluator_boundary", g3b["model_weights_changed"] is False and g3b["training_executed"] is False, "extraction/stopping improvement does not change model"))
    g3c = read_json(repo / "artifacts/gpu3c/gpu3c_result.json")
    checks.append(_check("gpu3c_negative_preserved", g3c["classification"] == "GPU3C_CONTROLLED_NEGATIVE_RESULT", g3c["classification"]))
    g4b = read_json(repo / "artifacts/gpu4b/gpu4b_result.json")
    checks.append(_check("gpu4b_partial_preserved", g4b["classification"] == "GPU4B_PARTIAL_VALIDATION", g4b["classification"]))
    g4br1 = read_json(repo / "artifacts/gpu4b_r1/gpu4b_r1_result.json")
    checks.append(_check("gpu4b_original_gate_preserved", g4br1["original_gate_reproduction"]["bf16_pass"] is False and not g4br1["production_attention_changed"], "BF16=false; production unchanged"))
    source = (repo / "src/minillm_forge/portfolio/verify.py").read_text(encoding="utf-8")
    quick_source = source.split("def quick_verify", 1)[1].split("def full_verify", 1)[0]
    checks.append(_check("quick_has_no_training", all(term not in quick_source for term in QUICK_FORBIDDEN_CALLS), "quick performs read-only verification except its own report"))
    passed = all(item["passed"] for item in checks)
    result = {"mode": "quick", "classification": "PASS" if passed else "FAIL", "checks": checks, "passed": sum(item["passed"] for item in checks), "total": len(checks), "wall_time_seconds": time.perf_counter() - started, "long_training_executed": False, "network_accessed": False}
    if write_output:
        write_json(out / "portfolio_verification.json", result)
        lines = ["# Portfolio Verification", "", f"Quick verification: **{result['classification']}** ({result['passed']}/{result['total']} checks, {result['wall_time_seconds']:.2f} s).", "", "No training, generation campaign, network download, performance benchmark, or API call was executed.", "", "| Check | Status | Detail |", "|---|---|---|"]
        lines.extend(f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | {item['detail']} |" for item in checks)
        (out / "PORTFOLIO_VERIFICATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def full_verify(repo: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo).resolve()
    quick = quick_verify(repo)
    completed = subprocess.run([str(repo / ".venv/Scripts/python.exe"), "-m", "pytest", "-q"], cwd=repo, capture_output=True, text=True)
    return {"mode": "full", "quick": quick, "pytest_returncode": completed.returncode, "pytest_output": completed.stdout[-4000:], "passed": quick["classification"] == "PASS" and completed.returncode == 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MiniLLM-Forge portfolio evidence without retraining")
    parser.add_argument("mode", choices=("build", "quick", "full"), nargs="?", default="quick")
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    if args.mode == "build":
        result = build_release_artifacts(args.repo)
    elif args.mode == "quick":
        result = quick_verify(args.repo)
    else:
        result = full_verify(args.repo)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
