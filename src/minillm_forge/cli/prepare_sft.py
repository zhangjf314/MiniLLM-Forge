from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from minillm_forge.cli.common import read_jsonl
from minillm_forge.data.contamination import audit_contamination
from minillm_forge.data.dedup import text_hash
from minillm_forge.data.manifest import write_data_manifest
from minillm_forge.data.sft import is_high_quality_math_sample


def extract_record(row: dict[str, Any]) -> dict[str, str] | None:
    problem = row.get("problem") or row.get("question")
    solution: Any = row.get("solution") or row.get("answer")
    if solution is None and row.get("solutions"):
        solutions = row["solutions"]
        solution = solutions[0] if isinstance(solutions, list) else solutions
    if isinstance(solution, list):
        solution = solution[0] if solution else None
    if (not problem or not solution) and isinstance(row.get("messages"), list):
        messages = row["messages"]
        problem = next(
            (item.get("content") for item in messages if item.get("role") == "user"), problem
        )
        solution = next(
            (item.get("content") for item in messages if item.get("role") == "assistant"),
            solution,
        )
    if not problem or not solution:
        return None
    return {"problem": str(problem).strip(), "solution": str(solution).strip()}


def source_rows(args: argparse.Namespace):
    if args.input:
        yield from read_jsonl(args.input)
        return
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset,
        args.subset,
        split=args.split,
        revision=args.revision,
        streaming=True,
    )
    yield from dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an audited assistant-only math SFT set")
    parser.add_argument("--input", help="Optional local JSONL instead of Hugging Face streaming")
    parser.add_argument("--dataset", default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--subset", default="default")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=20_000)
    parser.add_argument("--scan-limit", type=int, default=100_000)
    parser.add_argument("--max-characters", type=int, default=12_000)
    parser.add_argument("--benchmark", default="data/eval/controlled_math.jsonl")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--output", default="data/processed/openr1_math_sft.jsonl")
    parser.add_argument("--manifest", default="artifacts/data_manifests/openr1_math_sft.json")
    args = parser.parse_args()

    records: list[dict[str, str]] = []
    seen: set[str] = set()
    rejected = 0
    duplicates = 0
    for index, row in enumerate(source_rows(args)):
        if index >= args.scan_limit or len(records) >= args.max_samples:
            break
        record = extract_record(row)
        if record is None or not is_high_quality_math_sample(
            record, max_characters=args.max_characters
        ):
            rejected += 1
            continue
        digest = text_hash(record["problem"])
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        records.append(record)

    benchmark_rows = read_jsonl(args.benchmark)
    report = audit_contamination(
        [record["problem"] for record in records],
        [str(record["problem"]) for record in benchmark_rows],
    )
    contaminated = {collision["train_index"] for collision in report["collisions"]}
    records = [record for index, record in enumerate(records) if index not in contaminated]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    token_count = sum(
        len(tokenizer.encode(record["problem"] + "\n" + record["solution"])) for record in records
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = write_data_manifest(
        args.manifest,
        dataset=args.dataset if not args.input else str(Path(args.input)),
        revision=args.revision,
        records=records,
        token_count=token_count,
        filter_config={
            "quality_filter": True,
            "max_characters": args.max_characters,
            "rejected": rejected,
            "duplicates_removed": duplicates,
        },
        benchmark_overlap_removed=len(contaminated),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
