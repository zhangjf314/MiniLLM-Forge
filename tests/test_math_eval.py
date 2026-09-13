from minillm_forge.evaluation.math_eval import exact_match, extract_final_answer, normalize_answer


def test_extracts_last_balanced_box_with_nested_latex():
    text = r"First \boxed{2}; therefore \boxed{\frac{1}{2}}."
    assert extract_final_answer(text) == r"\frac{1}{2}"


def test_extracts_gsm8k_final_marker():
    assert extract_final_answer("work\n#### 18") == "18"


def test_normalization_handles_latex_wrappers():
    assert normalize_answer(r"$\left(\mathrm{x}\right)$") == r"(x)"
    assert exact_match("Final answer: 1,200.", "#### 1200")
