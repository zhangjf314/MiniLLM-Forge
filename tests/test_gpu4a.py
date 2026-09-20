from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from minillm_forge.experiments_gpu4a.data import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    TASKS,
    SFTDataset,
    collate_rows,
    format_prompt,
    tokenize_record,
)
from minillm_forge.experiments_gpu4a.runtime import evaluation_ids
from minillm_forge.model import MiniLLM, MiniLLMConfig
from minillm_forge.model.model import causal_lm_loss
from minillm_forge.training import build_adamw

REPO = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_checkpoint_matches_frozen_inventory() -> None:
    inventory = _json(REPO / "artifacts/training/minillm_checkpoint_inventory.json")
    expected = next(item for item in inventory["checkpoints"] if item["path"].endswith("best.pt"))
    protocol = _json(REPO / "artifacts/gpu4a/protocol.json")
    assert protocol["initial_checkpoint"]["sha256"] == expected["sha256"]
    assert protocol["initial_checkpoint"]["global_step"] == 3052


def test_native_model_structure_is_the_frozen_37m_configuration() -> None:
    manifest = _json(REPO / "artifacts/training/minillm_formal_model.json")
    assert manifest["parameter_count"] == 37_462_528
    assert manifest["architecture"]["num_layers"] == 8
    assert manifest["architecture"]["num_attention_heads"] == 8
    assert manifest["architecture"]["num_key_value_heads"] == 4
    assert manifest["architecture"]["max_seq_len"] == 1024


def test_tokenizer_embedding_output_dimensions_and_special_tokens() -> None:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(REPO / "artifacts/tokenizers/minillm-tokenizer.json"))
    config = MiniLLMConfig()
    assert tokenizer.get_vocab_size() == config.vocab_size == 24_000
    assert tokenizer.token_to_id("<pad>") == config.pad_token_id == PAD_ID
    assert tokenizer.token_to_id("<bos>") == config.bos_token_id == BOS_ID
    assert tokenizer.token_to_id("<eos>") == config.eos_token_id == EOS_ID


def test_instruction_format_is_shared_by_training_and_inference() -> None:
    value = {"message": "Please route this invoice request."}
    prompt = format_prompt(TASKS[0], value)
    assert prompt.startswith("Instruction:\n")
    assert "\n\nInput:\n" in prompt
    assert prompt.endswith("\n\nResponse:\n")


def test_response_only_mask_and_eos_supervision() -> None:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(REPO / "artifacts/tokenizers/minillm-tokenizer.json"))
    row = {
        "sample_id": "unit",
        "task": TASKS[1],
        "split": "train",
        "input": {"a": 4, "b": 7},
        "target": "11",
        "semantic_key": "4+7",
    }
    encoded = tokenize_record(tokenizer, row)
    prompt = encoded["prompt_token_count"]
    assert encoded["input_ids"][0] == BOS_ID
    assert encoded["labels"][:prompt] == [-100] * prompt
    assert encoded["labels"][prompt:-1] == encoded["input_ids"][prompt:-1]
    assert encoded["input_ids"][-1] == encoded["labels"][-1] == EOS_ID


def test_padding_is_not_supervised_and_does_not_cross_samples() -> None:
    dataset = SFTDataset(REPO / "data/processed/gpu4a/train.jsonl")
    batch = collate_rows([dataset[0], dataset[1]])
    assert batch["input_ids"].shape[0] == 2
    assert torch.all(batch["labels"][~batch["attention_mask"]] == -100)
    assert torch.all(batch["input_ids"][~batch["attention_mask"]] == PAD_ID)
    for index, row in enumerate((dataset[0], dataset[1])):
        end = len(row["input_ids"]) - 1
        assert batch["labels"][index, end] == EOS_ID


def test_causal_label_shift_predicts_next_token() -> None:
    labels = torch.tensor([[-100, 1, 2]])
    logits = torch.full((1, 3, 3), -10.0)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0
    assert causal_lm_loss(logits, labels) < 1e-6


def test_dataset_has_complete_targets_without_truncation() -> None:
    audit = _json(REPO / "artifacts/gpu4a/dataset_audit.json")
    assert audit["all_complete"] is True
    assert audit["eos_present_count"] == audit["sample_count"] == 6200
    assert audit["eos_supervised_count"] == audit["sample_count"]
    assert audit["lengths"]["sequence_max"] <= audit["max_length"]


def test_train_validation_test_have_no_semantic_key_leakage() -> None:
    audit = _json(REPO / "artifacts/gpu4a/dataset_audit.json")
    assert audit["semantic_key_overlaps"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }


def test_frozen_split_counts_and_hashes_match_files() -> None:
    audit = _json(REPO / "artifacts/gpu4a/dataset_audit.json")
    assert audit["split_counts"] == {"train": 4800, "validation": 700, "test": 700}
    for split, expected in audit["split_sha256"].items():
        assert _sha256(REPO / f"data/processed/gpu4a/{split}.jsonl") == expected


def test_evaluation_partition_is_balanced_and_unique() -> None:
    selected = evaluation_ids(REPO)
    assert len(selected) == 3 * 2 * 64
    assert len(selected) == len(set(selected))


def test_optimizer_contains_every_and_only_trainable_parameter() -> None:
    model = MiniLLM(
        MiniLLMConfig(
            vocab_size=80,
            hidden_size=32,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
        )
    )
    optimizer = build_adamw(model, lr=1e-4)
    optimized = {id(value) for group in optimizer.param_groups for value in group["params"]}
    trainable = {id(value) for value in model.parameters() if value.requires_grad}
    assert optimized == trainable


def test_historical_frozen_inputs_still_match_protocol() -> None:
    protocol = _json(REPO / "artifacts/gpu4a/protocol.json")
    for relative, expected in protocol["historical_frozen_inputs"].items():
        assert _sha256(REPO / relative) == expected


def test_tiny_overfit_updates_and_restores_exact_model_state() -> None:
    result = _json(REPO / "artifacts/gpu4a/tiny_overfit_result.json")
    assert result["status"] == "PASS"
    assert result["parameters_updated"] is True
    assert result["checkpoint_model_exact"] is True
    assert result["restored_generation_matches"] is True
    assert result["trained_metrics"]["loss"] < result["initial_metrics"]["loss"]
    assert result["trained_metrics"]["target_token_accuracy"] == 1.0
    assert result["steps_after_resume"] == 61


def test_formal_full_sft_completed_and_selected_checkpoint_reloads() -> None:
    result = _json(REPO / "artifacts/gpu4a/formal_sft_result.json")
    assert result["status"] == "COMPLETED"
    assert result["steps"] == 300
    assert result["trainable_parameters"] == result["optimizer_parameter_count"] == 37_462_528
    assert result["parameters_updated"] is True
    assert result["selected_checkpoint"]["step"] == 300
    assert result["selected_checkpoint_reload"]["status"] == "PASS"
    assert result["selected_checkpoint_reload"]["matches_recorded_validation"] is True
    assert all((result[key] == 0) for key in ("nan_count", "inf_count", "oom_count"))


def test_baseline_and_sft_use_identical_evaluation_records_and_protocol() -> None:
    baseline = [
        json.loads(line)
        for line in (REPO / "runs/gpu4a/baseline/generations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    sft = [
        json.loads(line)
        for line in (REPO / "runs/gpu4a/sft-evaluation/generations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(baseline) == len(sft) == 384
    assert [row["sample_id"] for row in baseline] == [row["sample_id"] for row in sft]
    assert all(row["task_score"] == 0 for row in baseline)
    t1_test = [row for row in sft if row["task"] == TASKS[0] and row["split"] == "test"]
    t2_test = [row for row in sft if row["task"] == TASKS[1] and row["split"] == "test"]
    assert sum(row["task_score"] for row in t1_test) == 64
    assert sum(row["task_score"] for row in t2_test) == 2
    assert all(row["stop_reason"] == "EOS" for row in sft)
