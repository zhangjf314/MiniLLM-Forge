from __future__ import annotations

import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from minillm_forge.experiments_gpu3c.background import launch, pid_running, supervise
from minillm_forge.experiments_gpu3c.controlled import (
    IM_END_ID,
    ControlledEOSDataset,
    apply_eos_policy,
    audit_pair_identity,
)
from minillm_forge.experiments_gpu3c.reporting import summarize_generations

REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_position_based_eos_policy_masks_prompt_only() -> None:
    ids = torch.tensor([151643, 7, 151643, 8, IM_END_ID])
    labels = apply_eos_policy(ids, 2, 4, "C-EOS-SUPERVISED")
    assert labels.tolist() == [-100, -100, 151643, 8, IM_END_ID]


def test_masked_arm_changes_only_final_assistant_end() -> None:
    ids = torch.tensor([1, 2, 3, IM_END_ID])
    masked = apply_eos_policy(ids, 2, 3, "B-EOS-MASKED")
    supervised = apply_eos_policy(ids, 2, 3, "C-EOS-SUPERVISED")
    assert torch.nonzero(masked != supervised).flatten().tolist() == [3]
    assert masked[3] == -100
    assert supervised[3] == IM_END_ID


def test_policy_rejects_wrong_assistant_end_position() -> None:
    with pytest.raises(ValueError, match="does not point"):
        apply_eos_policy(torch.tensor([1, IM_END_ID, 2]), 1, 2, "C-EOS-SUPERVISED")


def test_assistant_content_tokens_participate_in_loss() -> None:
    labels = apply_eos_policy(torch.tensor([10, 11, 12, IM_END_ID]), 2, 3, "C-EOS-SUPERVISED")
    assert labels.tolist() == [-100, -100, 12, IM_END_ID]


def test_causal_lm_label_shift_predicts_token_t_from_logits_t_minus_one() -> None:
    labels = torch.tensor([[-100, 1, 2]])
    logits = torch.full((1, 3, 3), -10.0)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0
    shifted_loss = F.cross_entropy(logits[:, :-1].reshape(-1, 3), labels[:, 1:].reshape(-1))
    unshifted_loss = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), ignore_index=-100)
    assert shifted_loss < 1e-6
    assert unshifted_loss > 1.0


def test_complete_subset_count_and_final_token() -> None:
    dataset = ControlledEOSDataset(REPO, "C-EOS-SUPERVISED")
    assert len(dataset) == 12469
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        row = dataset[index]
        end = int(row["assistant_end_position"])
        assert row["input_ids"][end] == IM_END_ID
        assert row["labels"][end] == IM_END_ID


def test_pair_identity_audit_has_exactly_one_difference_per_sample() -> None:
    result = audit_pair_identity(REPO)
    assert result["status"] == "PASS"
    assert result["samples"] == 12469
    assert result["label_difference_count"] == result["samples"]
    assert result["target_token_difference"] == result["samples"]
    assert result["packing"] == "DISABLED"


def test_audit_excludes_incomplete_samples_from_controlled_dataset() -> None:
    details = [
        json.loads(line)
        for line in (REPO / "artifacts/gpu3c/data_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    complete = {row["source_index"] for row in details if row["has_supervised_assistant_end"]}
    incomplete = {row["source_index"] for row in details if not row["has_supervised_assistant_end"]}
    selected = {entry[1] for entry in ControlledEOSDataset(REPO, "C-EOS-SUPERVISED").entries}
    assert selected == complete
    assert not selected & incomplete
    incomplete_row = next(
        row for row in details if row["truncation_reason"] == "ASSISTANT_TRUNCATED_BEFORE_END"
    )
    assert incomplete_row["assistant_end_position"] >= 512
    assert incomplete_row["has_supervised_assistant_end"] is False


def test_protocol_freezes_training_and_evaluation_contract() -> None:
    protocol = json.loads((REPO / "artifacts/gpu3c/protocol.json").read_text(encoding="utf-8"))
    assert protocol["training"]["optimizer_steps"] == 64
    assert protocol["training"]["gradient_accumulation_steps"] == 16
    assert protocol["training"]["checkpoint_steps"] == [32, 64]
    assert protocol["evaluation"]["max_new_tokens"] == 512
    assert protocol["evaluation"]["problem_ids"] == [
        "gsm8k-1176",
        "gsm8k-0805",
        "math500-0076",
        "math500-0181",
    ]


def test_completed_smoke_saved_reloaded_and_generated() -> None:
    smoke = json.loads((REPO / "artifacts/gpu3c/smoke_result.json").read_text(encoding="utf-8"))
    assert smoke["status"] == "PASS"
    for arm in smoke["arms"]:
        checkpoint = REPO / "runs/gpu3c/smoke" / arm["group"] / "checkpoints/step-0001.pt"
        assert checkpoint.is_file()
        assert arm["adapter_changed"] is True
        assert arm["checkpoint_reload"] == "PASS"
        assert arm["generation_smoke"]["status"] == "COMPLETED"


def test_formal_pair_artifacts_are_complete_and_attributed() -> None:
    formal = json.loads(
        (REPO / "artifacts/gpu3c/formal_pair_result.json").read_text(encoding="utf-8")
    )
    assert formal["status"] == "COMPLETED"
    assert len(formal["arms"]) == 2
    assert len({arm["initial_adapter_sha256"] for arm in formal["arms"]}) == 1
    for arm in formal["arms"]:
        assert arm["steps"] == 64
        assert arm["adapter_changed"] is True
        assert (arm["oom_count"], arm["nan_count"], arm["inf_count"]) == (0, 0, 0)
        root = REPO / "runs/gpu3c/formal-64" / arm["group"]
        assert (root / "checkpoints/step-0032.pt").stat().st_size > 0
        assert (root / "checkpoints/step-0064.pt").stat().st_size > 0
        rows = [
            json.loads(line)
            for line in (root / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 8
        assert sum(row["decoding_path"] == "C0" for row in rows) == 4
        assert sum(row["decoding_path"] == "S3" for row in rows) == 4
        assert all(not row["external_answer_stop"] for row in rows if row["decoding_path"] == "C0")


def test_historical_inputs_match_frozen_hashes() -> None:
    protocol = json.loads((REPO / "artifacts/gpu3c/protocol.json").read_text(encoding="utf-8"))
    for relative, expected in protocol["historical_inputs"].items():
        assert _sha256(REPO / relative) == expected


def test_training_and_benchmark_content_hashes_are_isolated() -> None:
    audit = json.loads((REPO / "artifacts/gpu3c/data_audit.json").read_text(encoding="utf-8"))
    assert audit["data_isolation"]["exact_normalized_problem_hash_collisions"] == 0
    assert audit["data_isolation"]["train_validation_exact_problem_hash_collisions"] == 0


def test_generation_summary_keeps_autonomous_and_external_stops_separate() -> None:
    base = {
        "status": "SUCCESS",
        "premature_eos": False,
        "length_limit_stop": False,
        "first_valid": {"answer": "1", "status": "ANSWER"},
        "legacy_correct": True,
        "first_valid_correct": True,
        "conflict_aware_correct": True,
        "conflict_aware": {"status": "ANSWER"},
        "repeat_3gram": {"rate": 0.0},
        "repeat_4gram": {"rate": 0.0},
        "post_answer_token_count": 0,
        "generated_token_count": 4,
        "inference_wall_time": 0.1,
    }
    autonomous = {
        **base,
        "stop_reason": "ASSISTANT_END_151645",
        "autonomous_eos_stop": True,
        "external_answer_stop": False,
    }
    external = {
        **base,
        "stop_reason": "EXTERNAL_ANSWER",
        "autonomous_eos_stop": False,
        "external_answer_stop": True,
    }
    result = summarize_generations([autonomous, external])
    assert result["autonomous_eos_count"] == 1
    assert result["external_answer_stop_count"] == 1
    assert result["autonomous_im_end_count"] == 1


def test_generation_summary_separates_conflict_from_accuracy() -> None:
    row = {
        "status": "SUCCESS",
        "stop_reason": "LENGTH_LIMIT",
        "autonomous_eos_stop": False,
        "external_answer_stop": False,
        "premature_eos": False,
        "length_limit_stop": True,
        "first_valid": {"answer": "1", "status": "ANSWER"},
        "legacy_correct": True,
        "first_valid_correct": True,
        "conflict_aware_correct": False,
        "conflict_aware": {"status": "CONFLICT"},
        "repeat_3gram": {"rate": 0.5},
        "repeat_4gram": {"rate": 0.4},
        "post_answer_token_count": 3,
        "generated_token_count": 10,
        "inference_wall_time": 0.2,
    }
    result = summarize_generations([row])
    assert result["first_valid_correct_count"] == 1
    assert result["conflict_aware_correct_count"] == 0
    assert result["conflict_count"] == 1


def _job_config(name: str, seconds: float, exit_code: int, timeout: float) -> Path:
    job_dir = REPO / "runs/gpu3c/test-background" / f"{name}-{uuid.uuid4().hex}"
    job_dir.mkdir(parents=True)
    config = {
        "task_id": name,
        "command": [
            sys.executable,
            "-m",
            "minillm_forge.experiments_gpu3c.worker",
            "simulate",
            "--seconds",
            str(seconds),
            "--exit-code",
            str(exit_code),
        ],
        "working_directory": str(REPO),
        "max_wall_time_seconds": timeout,
        "status_file": str(job_dir / "status.json"),
    }
    path = job_dir / "job.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_background_supervisor_records_normal_completion() -> None:
    path = _job_config("normal", 0.01, 0, 5)
    assert supervise(path) == 0
    status = json.loads((path.parent / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "COMPLETED"
    assert status["exit_code"] == 0


def test_background_supervisor_records_failure() -> None:
    path = _job_config("failure", 0.01, 7, 5)
    assert supervise(path) == 1
    status = json.loads((path.parent / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "FAILED"
    assert status["exit_code"] == 7


def test_background_supervisor_enforces_timeout() -> None:
    path = _job_config("timeout", 2, 0, 0.05)
    assert supervise(path) == 1
    status = json.loads((path.parent / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "TIMED_OUT"
    assert not pid_running(int(status["worker_pid"]))


def test_background_launcher_rejects_duplicate_task_id() -> None:
    task_id = f"duplicate-test-{uuid.uuid4().hex}"
    command = [sys.executable, "-c", "pass"]
    launch(
        REPO,
        task_id=task_id,
        command=command,
        max_wall_time_seconds=5,
        estimated_duration_seconds=0.1,
        expected_artifacts=[],
    )
    with pytest.raises(RuntimeError, match="already launched"):
        launch(
            REPO,
            task_id=task_id,
            command=command,
            max_wall_time_seconds=5,
            estimated_duration_seconds=0.1,
            expected_artifacts=[],
        )
    time.sleep(0.1)
