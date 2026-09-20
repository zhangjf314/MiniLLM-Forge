from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from minillm_forge.evaluation.math_eval import extract_final_answer
from minillm_forge.experiments_gpu3b.core import (
    CONFIRMATION_IDS,
    DEVELOPMENT_IDS,
    answer_candidates,
    extract_protocol,
    has_complete_answer,
)
from minillm_forge.experiments_gpu3b.inference import generate_one


class _Tokenizer:
    values = {10: "ANSWER:", 11: " 42", 12: "\n", 13: "x", 14: "y"}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.values.get(int(value), "") for value in ids if int(value) < 151643)


class _Model(torch.nn.Module):
    def __init__(self, tokens, *, oom=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.tokens = list(tokens)
        self.index = 0
        self.oom = oom

    def forward(self, **kwargs):
        if self.oom:
            raise torch.OutOfMemoryError("synthetic")
        token = self.tokens[min(self.index, len(self.tokens) - 1)]
        self.index += 1
        logits = torch.full((1, 1, 151646), -1000.0)
        logits[0, 0, token] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=None)


def test_single_complete_answer() -> None:
    result = extract_protocol("work\nANSWER: 42\n", "first-valid-v1")
    assert result["answer"] == "42"


def test_trailing_empty_answer_is_not_a_candidate() -> None:
    result = extract_protocol("ANSWER: 42\nANSWER: ", "last-valid-v1")
    assert result["answer"] == "42"


def test_multiple_identical_answers_are_consensus() -> None:
    result = extract_protocol("ANSWER: 42\nANSWER: 42\n", "conflict-aware-v1")
    assert result["status"] == "ANSWER"
    assert result["conflict"] is False


def test_multiple_conflicting_answers_are_not_selected() -> None:
    result = extract_protocol("ANSWER: 41\nANSWER: 42\n", "conflict-aware-v1")
    assert result["status"] == "CONFLICT"
    assert result["answer"] is None


def test_cross_format_conflict_is_not_hidden_by_box_priority() -> None:
    result = extract_protocol("ANSWER: 41\nthen \\boxed{42}", "conflict-aware-v1")
    assert result["status"] == "CONFLICT"
    assert result["candidate_values"] == ["41", "42"]


def test_boxed_value_inside_answer_marker_is_semantically_deduplicated() -> None:
    result = extract_protocol("ANSWER: \\boxed{42}\n", "conflict-aware-v1")
    assert result["status"] == "ANSWER"
    assert set(result["candidate_values"]) == {"42"}


def test_first_and_last_valid_are_independent_of_reference() -> None:
    text = "ANSWER: 41\nANSWER: 42\n"
    assert extract_protocol(text, "first-valid-v1")["answer"] == "41"
    assert extract_protocol(text, "last-valid-v1")["answer"] == "42"


def test_nested_boxed_answer() -> None:
    result = extract_protocol(r"Thus \boxed{\frac{1}{2}}.", "first-valid-v1")
    assert result["answer"] == r"\frac{1}{2}"


def test_negative_decimal_and_fraction_formats() -> None:
    for value in ("-2", "3.14", "7/9"):
        assert extract_protocol(f"ANSWER: {value}\n", "first-valid-v1")["answer"] == value


def test_answer_marker_alone_does_not_trigger_external_stop() -> None:
    assert has_complete_answer("reasoning\nANSWER:") is False


def test_answer_content_without_boundary_does_not_trigger_external_stop() -> None:
    assert has_complete_answer("reasoning\nANSWER: 42") is False


def test_answer_line_boundary_triggers_external_stop() -> None:
    assert has_complete_answer("reasoning\nANSWER: 42\n") is True


def test_natural_answer_requires_sentence_boundary_and_following_space() -> None:
    assert has_complete_answer("The answer is 3.14.") is False
    assert has_complete_answer("The answer is 3.14. Next") is True


def test_complete_box_triggers_external_stop() -> None:
    assert has_complete_answer(r"reasoning \boxed{42}") is True


def test_incomplete_box_does_not_trigger_external_stop() -> None:
    assert has_complete_answer(r"reasoning \boxed{\frac{1}{2}") is False


def test_final_line_fallback() -> None:
    candidates = answer_candidates("reasoning\n42")
    assert candidates[-1].source == "legacy_fallback"
    assert candidates[-1].value == "42"


def test_empty_generation_has_no_valid_answer() -> None:
    assert extract_protocol("", "first-valid-v1")["status"] == "NO_VALID_ANSWER"


def test_development_and_confirmation_ids_do_not_overlap() -> None:
    assert not set(DEVELOPMENT_IDS) & set(CONFIRMATION_IDS)


def test_problem_partition_has_no_duplicates() -> None:
    assert len(DEVELOPMENT_IDS) == len(set(DEVELOPMENT_IDS))
    assert len(CONFIRMATION_IDS) == len(set(CONFIRMATION_IDS))


def test_model_stop_151643() -> None:
    result = generate_one(
        _Model([151643]), _Tokenizer(), [1], eos_token_ids=[151643], max_new_tokens=3
    )
    assert result["stop_reason"] == "EOS_151643"
    assert result["autonomous_eos_stop"] is True


def test_assistant_end_stop_151645() -> None:
    result = generate_one(
        _Model([151645]), _Tokenizer(), [1], eos_token_ids=[151645], max_new_tokens=3
    )
    assert result["stop_reason"] == "ASSISTANT_END_151645"
    assert result["assistant_end_stop"] is True


def test_generation_length_limit() -> None:
    result = generate_one(_Model([13]), _Tokenizer(), [1], eos_token_ids=[151643], max_new_tokens=3)
    assert result["stop_reason"] == "LENGTH_LIMIT"
    assert result["generated_token_count"] == 3


def test_external_answer_stop_after_complete_line() -> None:
    result = generate_one(
        _Model([10, 11, 12, 13]),
        _Tokenizer(),
        [1],
        eos_token_ids=[151643, 151645],
        max_new_tokens=8,
        external_answer_stop=True,
    )
    assert result["stop_reason"] == "EXTERNAL_ANSWER"
    assert result["generated_token_ids"] == [10, 11, 12]
    assert result["first_complete_answer_token_position"] == 2
    assert result["post_answer_token_count"] == 0


def test_external_stop_does_not_fire_on_bare_marker() -> None:
    result = generate_one(
        _Model([10]),
        _Tokenizer(),
        [1],
        eos_token_ids=[151643],
        max_new_tokens=2,
        external_answer_stop=True,
    )
    assert result["stop_reason"] == "LENGTH_LIMIT"


def test_oom_is_preserved_as_failure() -> None:
    result = generate_one(
        _Model([13], oom=True), _Tokenizer(), [1], eos_token_ids=[151643], max_new_tokens=2
    )
    assert result["status"] == "FAILED"
    assert result["stop_reason"] == "OOM"


def test_timeout_is_preserved_as_failure() -> None:
    result = generate_one(
        _Model([13]),
        _Tokenizer(),
        [1],
        eos_token_ids=[151643],
        max_new_tokens=2,
        wall_time_seconds=-1,
    )
    assert result["status"] == "FAILED"
    assert result["stop_reason"] == "TIMEOUT"


def test_missing_eos_is_reported_as_length_limit() -> None:
    result = generate_one(_Model([13]), _Tokenizer(), [1], eos_token_ids=[], max_new_tokens=2)
    assert result["first_151643_position"] is None
    assert result["first_151645_position"] is None
    assert result["length_limit_stop"] is True


def test_generation_budget_controls_exact_token_count() -> None:
    for budget in (1, 3, 7):
        result = generate_one(
            _Model([13]), _Tokenizer(), [1], eos_token_ids=[151643], max_new_tokens=budget
        )
        assert len(result["generated_token_ids"]) == budget


def test_no_repeat_ngram_changes_repeated_sequence() -> None:
    result = generate_one(
        _Model([13]),
        _Tokenizer(),
        [1],
        eos_token_ids=[151643],
        max_new_tokens=6,
        no_repeat_ngram_size=3,
    )
    triples = list(
        zip(
            result["generated_token_ids"],
            result["generated_token_ids"][1:],
            result["generated_token_ids"][2:],
            strict=False,
        )
    )
    assert len(triples) == len(set(triples))


def test_legacy_and_new_extractors_can_score_same_generation() -> None:
    text = "work\nANSWER: 42\nANSWER: "
    assert extract_final_answer(text) == ""
    assert extract_protocol(text, "last-valid-v1")["answer"] == "42"


def test_frozen_historical_inputs_still_match_hashes() -> None:
    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads((repo / "artifacts/gpu3b/protocol.json").read_text(encoding="utf-8"))
    for relative, expected in protocol["historical_inputs"].items():
        assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == expected
