from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

STAGE = "STAGE_GPU_2B_H1_PA"
CLASSIFICATION = "STAGE_GPU_2B_H1_PA_COMPLETE"
PLATFORM_AUTHORITY = "UNAVAILABLE"
NEXT_WORK = "STAGE_GPU_2B_RD2_DESIGN_ONLY"
REPORT_DIRECTORY = Path("reports/gpu2b_h1_pa")
RESULT = Path("artifacts/design/gpu2b_h1_pa/stage-result.json")
RD_RESULT = Path("artifacts/design/gpu2b_rd/stage-result.json")
GPU2B_RESULT = Path("artifacts/training/gpu2b-stage-result.json")
INVENTORY = REPORT_DIRECTORY / "available-platforms.tsv"
CHECKSUMS = REPORT_DIRECTORY / "checksums.txt"
REQUIRED_REPORTS = (
    "results.md",
    "historical-boundary.md",
    "available-platforms.tsv",
    "platform-authority.md",
    "decision.md",
    "next-boundary.md",
    "claims-and-nonclaims.md",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate() -> dict[str, Any]:
    errors: list[str] = []
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    rd = json.loads(RD_RESULT.read_text(encoding="utf-8"))
    gpu2b = json.loads(GPU2B_RESULT.read_text(encoding="utf-8"))
    expected = {
        "stage": STAGE,
        "classification": CLASSIFICATION,
        "execution_class": "ZERO_TRAINING_WORK",
        "historical_rd": "STAGE_GPU_2B_RD_COMPLETE",
        "historical_rd_recovery_mode": "HARDWARE_MIGRATION",
        "historical_gpu2b": "STAGE_GPU_2B_BLOCKED_BY_MEMORY_QUALIFICATION",
        "h1_platform_authority": PLATFORM_AUTHORITY,
        "h1_execution": "NOT_AUTHORIZED",
        "next_authorized_work": NEXT_WORK,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            errors.append(f"stage result mismatch: {key}")
    if rd.get("classification") != "STAGE_GPU_2B_RD_COMPLETE":
        errors.append("RD prerequisite is not complete")
    if rd.get("recovery_mode") != "HARDWARE_MIGRATION":
        errors.append("historical RD recovery mode changed")
    if gpu2b.get("classification") != "STAGE_GPU_2B_BLOCKED_BY_MEMORY_QUALIFICATION":
        errors.append("historical GPU-2B classification changed")
    if file_sha256(RD_RESULT) != result["historical_evidence"]["rd_stage_result_sha256"]:
        errors.append("RD result digest mismatch")
    if file_sha256(GPU2B_RESULT) != result["historical_evidence"]["gpu2b_stage_result_sha256"]:
        errors.append("GPU-2B result digest mismatch")
    if any(result["numerical_launches"].values()):
        errors.append("platform audit numerical launch count must remain zero")
    with INVENTORY.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    eligible = [row for row in rows if row["h1_eligible"] == "YES"]
    if len(rows) != result["candidate_environment_count"]:
        errors.append("candidate environment count mismatch")
    if len(eligible) != result["eligible_environment_count"]:
        errors.append("eligible environment count mismatch")
    if PLATFORM_AUTHORITY == "UNAVAILABLE" and eligible:
        errors.append("UNAVAILABLE authority contradicts eligible inventory")
    if result.get("selected_environment") is not None:
        errors.append("UNAVAILABLE audit must not select an environment")
    for filename in REQUIRED_REPORTS:
        if not (REPORT_DIRECTORY / filename).is_file():
            errors.append(f"missing required report: {filename}")
    if not CHECKSUMS.is_file():
        errors.append("missing checksums.txt")
    else:
        for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
            expected_hash, path_text = line.split("  ", 1)
            path = Path(path_text)
            if not path.is_file() or file_sha256(path) != expected_hash:
                errors.append(f"checksum mismatch: {path_text}")
    return {
        "stage": STAGE,
        "classification": "PASS" if not errors else "FAIL",
        "h1_platform_authority": PLATFORM_AUTHORITY,
        "next_authorized_work": NEXT_WORK,
        "training_executed": False,
        "errors": errors,
    }
