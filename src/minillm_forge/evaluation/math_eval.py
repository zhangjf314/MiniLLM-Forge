from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from decimal import Decimal, InvalidOperation
from typing import Any

FINAL_PATTERNS = [
    re.compile(r"(?:final answer|answer)\s*(?:is|:)?\s*([^\n.]+)", re.IGNORECASE),
    re.compile(r"####\s*([^\n]+)"),
]


def _balanced_boxed_answers(text: str) -> list[str]:
    answers: list[str] = []
    marker = r"\boxed{"
    start = 0
    while (marker_index := text.find(marker, start)) >= 0:
        content_start = marker_index + len(marker)
        depth = 1
        index = content_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            answers.append(text[content_start : index - 1].strip())
        start = content_start
    return answers


def extract_final_answer(text: str) -> str:
    boxed = _balanced_boxed_answers(text)
    if boxed:
        return boxed[-1]
    for pattern in FINAL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return str(matches[-1]).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_answer(answer: str) -> str:
    normalized = answer.strip().lower()
    normalized = normalized.replace(",", "").replace("$", "")
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
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
