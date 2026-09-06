from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from decimal import Decimal, InvalidOperation
from typing import Any

FINAL_PATTERNS = [
    re.compile(r"\\boxed\{([^{}]+)\}"),
    re.compile(r"(?:final answer|answer)\s*(?:is|:)?\s*([^\n.]+)", re.IGNORECASE),
    re.compile(r"####\s*([^\n]+)"),
]


def extract_final_answer(text: str) -> str:
    for pattern in FINAL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return str(matches[-1]).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_answer(answer: str) -> str:
    normalized = answer.strip().lower()
    normalized = normalized.replace(",", "").replace("$", "")
    normalized = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", normalized)
    normalized = normalized.rstrip(". ")
    try:
        return format(Decimal(normalized).normalize(), "f")
    except InvalidOperation:
        return re.sub(r"\s+", "", normalized)


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer(extract_final_answer(prediction)) == normalize_answer(
        extract_final_answer(reference)
    )


def evaluate_math(
    samples: Iterable[dict[str, Any]],
    generate: Callable[[str], str],
    *,
    problem_field: str = "problem",
    answer_field: str = "answer",
) -> dict[str, Any]:
    details = []
    correct = 0
    for sample in samples:
        problem = str(sample[problem_field])
        reference = str(sample[answer_field])
        prediction = generate(problem)
        matched = exact_match(prediction, reference)
        correct += matched
        details.append(
            {
                "problem": problem,
                "reference": extract_final_answer(reference),
                "prediction": extract_final_answer(prediction),
                "correct": matched,
            }
        )
    return {
        "accuracy": correct / max(len(details), 1),
        "correct": correct,
        "total": len(details),
        "details": details,
    }
