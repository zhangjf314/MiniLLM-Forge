from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

STAGE = "STAGE_GPU_2B_RD"
CLASSIFICATION = "STAGE_GPU_2B_RD_COMPLETE"
PREDECESSOR = "STAGE_GPU_2B_BLOCKED_BY_MEMORY_QUALIFICATION"
RECOVERY_MODE = "HARDWARE_MIGRATION"
FUTURE_STAGE = "STAGE_GPU_2B_H1"
REPORT_DIRECTORY = Path("reports/gpu2b_rd")
RESULT = Path("artifacts/design/gpu2b_rd/stage-result.json")
PREDECESSOR_RESULT = Path("artifacts/training/gpu2b-stage-result.json")
CHECKSUMS = REPORT_DIRECTORY / "checksums.txt"
REQUIRED_REPORTS = (
    "results.md",
    "historical-boundary.md",
    "resource-analysis.md",
    "recovery-decision.md",
    "hardware-authority.md",
    "configuration-change-policy.md",
    "next-boundary.md",
    "claims-and-nonclaims.md",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implied_capacity_mib(observed_use_mib: int, required_headroom_mib: int) -> int:
    return observed_use_mib + required_headroom_mib


def validate() -> dict[str, Any]:
    errors: list[str] = []
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    predecessor = json.loads(PREDECESSOR_RESULT.read_text(encoding="utf-8"))
    expected_result = {
        "stage": STAGE,
        "classification": CLASSIFICATION,
        "execution_class": "ZERO_TRAINING_WORK",
        "predecessor": PREDECESSOR,
        "recovery_mode": RECOVERY_MODE,
        "future_stage": FUTURE_STAGE,
        "scientific_design_change": "NONE",
        "configuration_change": "NONE",
    }
    for key, expected in expected_result.items():
        if result.get(key) != expected:
            errors.append(f"stage result mismatch: {key}")
    if predecessor.get("classification") != PREDECESSOR:
        errors.append("historical GPU-2B classification changed")
    memory = predecessor.get("memory_gate", {})
    immutable_memory = {
        "observed_physical_vram_used_mib": memory.get("maximum_observed_system_vram_used_mib"),
        "required_minimum_headroom_mib": memory.get("required_minimum_system_headroom_mib"),
        "implied_lower_bound_mib": implied_capacity_mib(
            memory.get("maximum_observed_system_vram_used_mib", 0),
            memory.get("required_minimum_system_headroom_mib", 0),
        ),
    }
    for key, expected in immutable_memory.items():
        if result["resource_authority"].get(key) != expected:
            errors.append(f"resource authority mismatch: {key}")
    if predecessor.get("formal_training_runs_launched") != 0:
        errors.append("historical formal launch count is not zero")
    if any(result["numerical_launches"].values()):
        errors.append("RD numerical launch count must remain zero")
    if result["frozen_scientific_design"].get("formal_runs") != 18:
        errors.append("frozen factorial run count changed")
    for filename in REQUIRED_REPORTS:
        if not (REPORT_DIRECTORY / filename).is_file():
            errors.append(f"missing required report: {filename}")
    if not CHECKSUMS.is_file():
        errors.append("missing checksums.txt")
    else:
        for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
            expected, path_text = line.split("  ", 1)
            path = Path(path_text)
            if not path.is_file() or file_sha256(path) != expected:
                errors.append(f"checksum mismatch: {path_text}")
    return {
        "stage": STAGE,
        "classification": "PASS" if not errors else "FAIL",
        "recovery_mode": RECOVERY_MODE,
        "future_stage": FUTURE_STAGE,
        "training_executed": False,
        "errors": errors,
    }
