from minillm_forge.design.gpu2b import BenchmarkCollisionIndex, _candidate, _word_ngrams


def test_word_ngrams_normalize_case_and_spacing():
    assert _word_ngrams("One  TWO, three four five!") == {"one two three four five"}


def test_collision_index_finds_exact_near_and_containment():
    index = BenchmarkCollisionIndex(
        [
            {
                "benchmark": "probe",
                "benchmark_index": 0,
                "problem": "Find the sum of one two three four five six seven eight nine.",
            }
        ]
    )
    exact = index.collisions("Find the sum of one two three four five six seven eight nine!")
    assert exact[0]["match_type"] == "normalized_exact"
    overlap = index.collisions(
        "Context words. Find the sum of one two three four five six seven eight nine. Extra words."
    )
    assert any(item["match_type"].startswith("problem_statement_overlap") for item in overlap)


def test_candidate_requires_verified_complete_math_solution():
    row = {
        "problem": "What is 2+2?",
        "solution": "Reasoning. Final answer: 4",
        "answer": "4",
        "uuid": "x",
        "source": "unit",
        "is_reasoning_complete": [True],
        "correctness_math_verify": [True],
    }
    assert _candidate(row, 7)["source_index"] == 7
    row["correctness_math_verify"] = [False]
    assert _candidate(row, 7) is None
