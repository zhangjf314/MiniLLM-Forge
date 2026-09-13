from minillm_forge.execution.gpu2d_formal import (
    EXECUTION_ORDER,
    _equivalence,
    _family_outcome,
    _overall_classification,
    _parse_run_id,
    _trim_generated,
)


def test_formal_execution_order_is_balanced_and_complete():
    parsed = [_parse_run_id(run_id) for run_id in EXECUTION_ORDER]
    assert len(parsed) == 12
    assert len({item["run_id"] for item in parsed}) == 12
    assert {item["seed"] for item in parsed} == {42, 31415, 271828}
    assert {item["family"] for item in parsed} == {"LORA", "QLORA"}
    for family in ("LORA", "QLORA"):
        for seed in (42, 31415, 271828):
            pair = [item for item in parsed if item["family"] == family and item["seed"] == seed]
            assert {item["initialization"] for item in pair} == {"BASE_INIT", "CPT_INIT"}


def test_generated_token_trimming_includes_first_stop():
    assert _trim_generated([7, 8, 151645, 151643], {151643, 151645}) == [7, 8, 151645]
    assert _trim_generated([7, 8], {151643, 151645}) == [7, 8]


def test_exact_equivalence_checks_all_frozen_fields():
    reference = [
        {
            "problem_id": "gsm8k-0001",
            "generated_token_ids": [1, 2],
            "generated_text": "2",
            "extracted_answer": "2",
            "correct": True,
        }
    ]
    assert _equivalence(reference, [dict(reference[0])])["all_match"]
    changed = dict(reference[0], generated_token_ids=[1, 3])
    comparison = _equivalence(reference, [changed])
    assert not comparison["all_match"]
    assert not comparison["field_matches"]["generated_token_ids"]


def test_frozen_transfer_rules_and_overall_classification():
    positive = {
        benchmark: {
            "mean_delta": 1.2,
            "positive_seeds": 2,
            "negative_seeds": 1,
            "paired_delta": [1.5, 2.0, 0.1],
        }
        for benchmark in ("gsm8k", "math500")
    }
    neutral = {
        benchmark: {
            "mean_delta": 0.2,
            "positive_seeds": 2,
            "negative_seeds": 1,
            "paired_delta": [0.4, 0.3, -0.1],
        }
        for benchmark in ("gsm8k", "math500")
    }
    assert _family_outcome(positive) == "POSITIVE"
    assert _family_outcome(neutral) == "NOT_SUPPORTED"
    assert (
        _overall_classification({"LORA": "POSITIVE", "QLORA": "NOT_SUPPORTED"})
        == "CPT_TO_PEFT_SFT_TRANSFER_METHOD_DEPENDENT"
    )
    assert (
        _overall_classification({"LORA": "MIXED", "QLORA": "NOT_SUPPORTED"})
        == "GPU2D_FORMAL_RESULTS_INCONCLUSIVE"
    )
