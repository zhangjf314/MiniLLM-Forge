from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from minillm_forge.portfolio.verify import (
    REQUIRED_CLAIM_FIELDS,
    REQUIRED_EXPERIMENT_FIELDS,
    quick_verify,
)

REPO = Path(__file__).resolve().parents[1]
GPU4C = REPO / "artifacts/gpu4c"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict:
    return _json(GPU4C / "final_evidence_manifest.json")


@pytest.fixture(scope="module")
def claims() -> list[dict]:
    return _json(GPU4C / "claim_registry.json")["claims"]


def _claim(claims: list[dict], claim_id: str) -> dict:
    return next(item for item in claims if item["claim_id"] == claim_id)


def test_formal_checkpoint_exists(manifest: dict) -> None:
    assert (REPO / manifest["pretrained_checkpoint"]).is_file()


def test_formal_checkpoint_hash(manifest: dict) -> None:
    path = REPO / manifest["pretrained_checkpoint"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["pretrained_checkpoint_sha256"]


def test_tokenizer_hash(manifest: dict) -> None:
    path = REPO / manifest["tokenizer"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["tokenizer_sha256"]


def test_pretraining_dataset_hash(manifest: dict) -> None:
    path = REPO / "artifacts/data_manifests/minillm_pretrain_formal.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        manifest["major_dataset_hashes"]["native_pretraining_manifest"]
    )


def test_native_sft_checkpoint_hash(manifest: dict) -> None:
    path = REPO / manifest["native_sft_checkpoint"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["native_sft_checkpoint_sha256"]


def test_experiment_registry_schema() -> None:
    registry = _json(GPU4C / "experiment_registry.json")
    assert registry["schema_version"] == 1
    assert len(registry["experiments"]) >= 12
    assert all(REQUIRED_EXPERIMENT_FIELDS <= set(item) for item in registry["experiments"])


def test_claim_registry_schema(claims: list[dict]) -> None:
    assert len(claims) >= 20
    assert all(REQUIRED_CLAIM_FIELDS <= set(item) for item in claims)


def test_validated_claims_have_evidence(claims: list[dict]) -> None:
    assert all(
        (REPO / item["evidence_artifact"]).exists()
        for item in claims
        if item["status"] == "VALIDATED"
    )


def test_negative_results_are_not_success_claims(claims: list[dict]) -> None:
    negative = [item for item in claims if item["status"] == "NEGATIVE_RESULT"]
    assert negative
    assert all("success" not in item["claim_text"].lower() for item in negative)


def test_t1_subset_is_not_promoted_to_full_test(claims: list[dict]) -> None:
    claim = _claim(claims, "C007")
    assert "64/64" in claim["claim_text"]
    assert "subset" in claim["claim_text"].lower()
    assert "not full-test" in claim["limitations"].lower()


def test_gpu3b_is_evaluator_not_model_improvement(claims: list[dict]) -> None:
    claim = _claim(claims, "C013")
    assert claim["category"] == "evaluation_system"
    assert "not model-capability" in claim["limitations"]


def test_gpu3c_classification_remains_negative(manifest: dict) -> None:
    assert manifest["stage_classifications"]["GPU-3C"] == "GPU3C_CONTROLLED_NEGATIVE_RESULT"


def test_gpu4b_sdpa_is_not_performance_validated(claims: list[dict]) -> None:
    claim = _claim(claims, "C016")
    assert claim["status"] == "PARTIALLY_VALIDATED"
    assert "no qualified performance benchmark" in claim["limitations"]


def test_flash_is_blocked(claims: list[dict]) -> None:
    assert _claim(claims, "C018")["status"] == "BLOCKED"


def test_compile_is_blocked(claims: list[dict]) -> None:
    assert _claim(claims, "C019")["status"] == "BLOCKED"


def test_gpu4b_r1_does_not_rewrite_original_gate() -> None:
    result = _json(REPO / "artifacts/gpu4b_r1/gpu4b_r1_result.json")
    assert result["original_gate_reproduction"]["bf16_pass"] is False
    assert result["production_attention_changed"] is False


def test_readme_matches_registry_boundaries() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "0/64 to 64/64" in readme
    assert "fixed 64-example subsets" in readme
    assert "independent" in readme
    assert "test subset" in readme
    assert "no Flash, compile, SDPA speedup" in readme


def test_final_report_matches_registry_boundaries() -> None:
    report = (REPO / "reports/FINAL_REPORT.md").read_text(encoding="utf-8")
    assert "2/64" in report
    assert "GPU3C_CONTROLLED_NEGATIVE_RESULT" in report
    assert "No qualified speed" in report


def test_quick_verification_does_not_launch_long_work() -> None:
    result = quick_verify(REPO, write_output=False)
    assert result["classification"] == "PASS"
    assert result["long_training_executed"] is False
    assert result["network_accessed"] is False


def test_historical_artifact_hashes_remain_unchanged() -> None:
    protocol = _json(GPU4C / "protocol.json")
    for relative, expected in protocol["frozen_evidence_hashes"].items():
        assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == expected


def test_capability_matrix_exposes_unvalidated_and_blocked() -> None:
    text = (GPU4C / "CAPABILITY_MATRIX.md").read_text(encoding="utf-8")
    assert "IMPLEMENTED_BUT_NOT_EFFECTIVENESS_VALIDATED" in text
    assert "BLOCKED" in text
    assert "NOT_EXECUTED" in text
