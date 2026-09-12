from __future__ import annotations

import torch

from minillm_forge.cli.stage_gpu2a import classify_cpt_result
from minillm_forge.data.qwen_cpt import (
    ProbeContaminationFilter,
    freeze_general_partition,
    freeze_math_partitions,
)
from minillm_forge.training.qwen_cpt import FixedTokenBlockDataset


class CharacterTokenizer:
    eos_token_id = 255

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) % 255 for character in text]


def test_math_partitions_are_disjoint_and_remove_exact_probe_collision() -> None:
    rows = [
        {"text": "heldout"},
        {"text": "heldout"},
        {"text": "training-one"},
        {"text": "training-two"},
    ]
    train, validation, audit = freeze_math_partitions(
        rows,
        CharacterTokenizer(),
        text_field="text",
        math_validation_tokens=8,
        training_tokens=20,
        benchmark_texts=[],
        ngram_size=4,
        near_threshold=0.8,
    )
    assert validation.documents == ["heldout"]
    assert "heldout" not in train.documents
    assert audit["removed_exact"] == 1
    assert train.tokens.numel() == 20
    assert validation.tokens.numel() == 8


def test_general_partition_is_frozen_to_exact_token_budget() -> None:
    partition = freeze_general_partition(
        ["alpha", "alpha", "beta gamma"], CharacterTokenizer(), token_budget=10
    )
    assert partition.tokens.numel() == 10
    assert partition.documents == ["alpha", "beta gamma"]
    assert partition.duplicate_removals == 1


def test_probe_filter_and_fixed_blocks() -> None:
    contamination = ProbeContaminationFilter(["abcdefgh"], ngram_size=3, threshold=0.8)
    assert contamination.classify("abcdefgh")[0] == "normalized_exact"
    assert contamination.classify("abcdefghx")[0] == "ngram"
    assert contamination.classify("unrelated")[0] is None

    dataset = FixedTokenBlockDataset(torch.arange(11), sequence_length=5)
    assert len(dataset) == 2
    assert dataset[1]["input_ids"].tolist() == [5, 6, 7, 8, 9]
    assert dataset[1]["labels"].tolist() == [5, 6, 7, 8, 9]


def test_sustained_small_general_degradation_is_not_hidden() -> None:
    classification, hypotheses, sustained = classify_cpt_result(
        math_ppl_delta=-0.1,
        general_ppl_delta=0.01,
        base_general_ppl=17.0,
        post_general_ppls=[17.1, 17.08, 17.04, 17.01],
    )
    assert classification == "QWEN_MATH_CPT_VALIDATED_WITH_GENERAL_DEGRADATION"
    assert hypotheses["H-CPT-2"] == "supported"
    assert sustained
