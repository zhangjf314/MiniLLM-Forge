from __future__ import annotations

from minillm_forge.diagnostics.gpu3a import (
    answer_evidence,
    diagnose_record,
    max_consecutive_repeat_span,
    ngram_metrics,
)


def _row(ids=None, text="", answer="x"):
    return {
        "run_id": "B-LORA-s42",
        "benchmark": "gsm8k",
        "benchmark_scope": "GSM8K_FIXED_200",
        "problem_id": "gsm8k-0001",
        "generated_token_ids": ids,
        "generated_text": text,
        "extracted_answer": answer,
        "correct": False,
    }


def test_length_limit_without_eos() -> None:
    result = diagnose_record(_row([7] * 512), "evidence.jsonl")
    assert result["generation_stop_reason"] == "LENGTH_LIMIT"
    assert result["eos_observed"] is False


def test_eos_stop_is_distinct_from_length_limit() -> None:
    result = diagnose_record(_row([7, 151645]), "evidence.jsonl")
    assert result["generation_stop_reason"] == "EOS"
    assert result["first_eos_position"] == 1


def test_unknown_stop_when_token_ids_missing() -> None:
    result = diagnose_record(_row(None), "evidence.jsonl")
    assert result["generation_stop_reason"] == "UNKNOWN"
    assert result["diagnostic_status"] == "PARTIAL_MISSING_TOKEN_IDS"


def test_empty_text_ngram_is_explicitly_undefined() -> None:
    assert ngram_metrics([], 3)["rate"] is None


def test_short_text_ngram_is_explicitly_undefined() -> None:
    assert ngram_metrics([1, 2], 3)["rate"] is None


def test_ngram_occurrence_definition() -> None:
    metric = ngram_metrics([1, 2, 3, 1, 2, 3], 3)
    assert metric == {"n": 3, "total": 4, "unique": 3, "repeated_occurrences": 1, "rate": 0.25}


def test_no_answer_marker() -> None:
    result = diagnose_record(_row([1] * 512, "reasoning only", ""), "evidence.jsonl")
    assert "NO_ANSWER_MARKER" in result["failure_types"]


def test_one_complete_answer_marker() -> None:
    evidence = answer_evidence("work\nANSWER: 42\n")
    assert evidence["nonempty_marker_count"] == 1


def test_multiple_identical_answers() -> None:
    evidence = answer_evidence("ANSWER: 42\nANSWER: 42\n")
    assert evidence["multiple_identical_answers"] is True


def test_multiple_conflicting_answers() -> None:
    evidence = answer_evidence("ANSWER: 41\nANSWER: 42\n")
    assert evidence["multiple_conflicting_answers"] is True


def test_answer_then_repetition_then_empty_marker_is_rule_mismatch() -> None:
    text = "ANSWER: 42\n" * 10 + "ANSWER: "
    result = diagnose_record(_row([1] * 512, text, ""), "evidence.jsonl")
    assert "EXTRACTION_RULE_MISMATCH" in result["failure_types"]
    assert "REPETITION_AFTER_ANSWER" in result["failure_types"]


def test_periodic_repeat_span() -> None:
    assert max_consecutive_repeat_span([1, 2, 1, 2, 1, 2, 9]) == 6


def test_eos_at_length_cap_is_eos_stop() -> None:
    result = diagnose_record(_row([1] * 511 + [151643]), "evidence.jsonl")
    assert result["generation_stop_reason"] == "EOS"


def test_nonterminal_eos_is_not_counted_as_length_limit() -> None:
    result = diagnose_record(_row([151645] + [1] * 511), "evidence.jsonl")
    assert result["generation_stop_reason"] == "EOS_NOT_TERMINAL"


def test_answer_completeness_unknown_at_truncation() -> None:
    result = diagnose_record(_row([1] * 512, "ANSWER: 42"), "evidence.jsonl")
    assert result["answer_complete"] is None


def test_missing_post_answer_alignment_is_not_invented() -> None:
    result = diagnose_record(_row([1, 151645], "ANSWER: 42"), "evidence.jsonl")
    assert result["post_answer_token_count"] is None
    assert result["post_answer_token_count_missing_reason"]
