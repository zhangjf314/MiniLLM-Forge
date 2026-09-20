from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MAX_LENGTH = 128
SEED = 4204
TASKS = ("T1_SUPPORT_CLASSIFICATION", "T2_INTEGER_ADDITION")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def format_prompt(task: str, value: dict[str, Any]) -> str:
    if task == TASKS[0]:
        instruction = "Classify the support request. Reply with BILLING, TECHNICAL, or ACCOUNT."
        content = value["message"]
    elif task == TASKS[1]:
        instruction = "Add the two integers. Reply with only the integer sum."
        content = f"{value['a']} + {value['b']}"
    else:
        raise ValueError(f"unknown task: {task}")
    return f"Instruction:\n{instruction}\n\nInput:\n{content}\n\nResponse:\n"


def _support_rows(split: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    keywords = {
        "BILLING": ["invoice", "refund", "payment", "charge", "receipt"],
        "TECHNICAL": ["crashes", "freezes", "error", "offline", "loading"],
        "ACCOUNT": ["password", "username", "profile", "login", "email"],
    }
    templates = {
        "train": [
            "Please help: the {word} issue affects ticket {ticket}.",
            "Customer {name} reports a {word} problem on case {ticket}.",
            "Request {ticket}: I need assistance with {word}.",
            "The user says {word} is the reason for contacting support, ref {ticket}.",
        ],
        "validation": ["Regarding case {ticket}, {name} asks for help because of {word}."],
        "test": [
            "Support note {ticket} from {name}: concern is {word}.",
            "For reference {ticket}, the reported topic is {word}; please route it.",
        ],
    }
    names = ["Alex", "Blair", "Casey", "Drew", "Emery", "Finley", "Gray", "Harper"]
    prefixes = {"train": "TR", "validation": "VA", "test": "TE"}
    labels = list(keywords)
    rows = []
    for index in range(count):
        label = labels[index % len(labels)]
        word = rng.choice(keywords[label])
        template = rng.choice(templates[split])
        ticket = f"{prefixes[split]}{index:05d}"
        message = template.format(word=word, ticket=ticket, name=rng.choice(names))
        rows.append(
            {
                "sample_id": f"t1-{split}-{index:05d}",
                "task": TASKS[0],
                "split": split,
                "input": {"message": message},
                "target": label,
                "semantic_key": ticket,
            }
        )
    rng.shuffle(rows)
    return rows


def _addition_rows(split: str, pairs: list[tuple[int, int]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": f"t2-{split}-{index:05d}",
            "task": TASKS[1],
            "split": split,
            "input": {"a": a, "b": b},
            "target": str(a + b),
            "semantic_key": f"{a}+{b}",
        }
        for index, (a, b) in enumerate(pairs)
    ]


def build_records() -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(SEED)
    pairs = [(a, b) for a in range(100) for b in range(100)]
    rng.shuffle(pairs)
    sizes = {
        "train": {TASKS[0]: 1800, TASKS[1]: 3000},
        "validation": {TASKS[0]: 300, TASKS[1]: 400},
        "test": {TASKS[0]: 300, TASKS[1]: 400},
    }
    cursor = 0
    result = {}
    for split in ("train", "validation", "test"):
        addition_count = sizes[split][TASKS[1]]
        selected = pairs[cursor : cursor + addition_count]
        cursor += addition_count
        rows = _support_rows(split, sizes[split][TASKS[0]], rng)
        rows.extend(_addition_rows(split, selected))
        rng.shuffle(rows)
        result[split] = rows
    return result


def tokenize_record(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    prompt = format_prompt(row["task"], row["input"])
    prompt_ids = [BOS_ID, *tokenizer.encode(prompt, add_special_tokens=False).ids]
    target_ids = tokenizer.encode(row["target"], add_special_tokens=False).ids
    ids = [*prompt_ids, *target_ids, EOS_ID]
    if len(ids) > MAX_LENGTH:
        raise ValueError(f"complete sample exceeds context: {row['sample_id']}")
    labels = [-100] * len(prompt_ids) + target_ids + [EOS_ID]
    return {
        **row,
        "prompt": prompt,
        "input_ids": ids,
        "labels": labels,
        "prompt_token_count": len(prompt_ids),
        "target_token_count": len(target_ids) + 1,
        "total_sequence_length": len(ids),
        "answer_complete": True,
        "eos_present": ids[-1] == EOS_ID,
        "eos_supervised": labels[-1] == EOS_ID,
        "truncated_prompt_tokens": 0,
        "truncated_target_tokens": 0,
    }


def materialize(repo: str | Path) -> dict[str, Any]:
    from tokenizers import Tokenizer

    repo = Path(repo).resolve()
    tokenizer = Tokenizer.from_file(str(repo / "artifacts/tokenizers/minillm-tokenizer.json"))
    records = build_records()
    output = repo / "data/processed/gpu4a"
    output.mkdir(parents=True, exist_ok=True)
    tokenized: dict[str, list[dict[str, Any]]] = {}
    split_hashes = {}
    for split, rows in records.items():
        tokenized[split] = [tokenize_record(tokenizer, row) for row in rows]
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in tokenized[split]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    semantic = {
        split: {f"{row['task']}:{row['semantic_key']}" for row in rows}
        for split, rows in records.items()
    }
    overlaps = {
        "train_validation": len(semantic["train"] & semantic["validation"]),
        "train_test": len(semantic["train"] & semantic["test"]),
        "validation_test": len(semantic["validation"] & semantic["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"semantic leakage across GPU-4A splits: {overlaps}")
    all_rows = [row for rows in tokenized.values() for row in rows]
    audit = {
        "version": "gpu4a-sft-v1-frozen",
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "split_counts": {split: len(rows) for split, rows in records.items()},
        "task_split_counts": {
            split: dict(Counter(row["task"] for row in rows)) for split, rows in records.items()
        },
        "split_sha256": split_hashes,
        "semantic_key_overlaps": overlaps,
        "all_complete": all(row["answer_complete"] for row in all_rows),
        "eos_present_count": sum(row["eos_present"] for row in all_rows),
        "eos_supervised_count": sum(row["eos_supervised"] for row in all_rows),
        "sample_count": len(all_rows),
        "effective_supervised_tokens": {
            split: sum(row["target_token_count"] for row in rows)
            for split, rows in tokenized.items()
        },
        "lengths": {
            "prompt_max": max(row["prompt_token_count"] for row in all_rows),
            "target_max": max(row["target_token_count"] for row in all_rows),
            "sequence_max": max(row["total_sequence_length"] for row in all_rows),
            "sequence_min": min(row["total_sequence_length"] for row in all_rows),
        },
        "dataset_digest": _digest(
            {
                split: [(row["sample_id"], row["semantic_key"], row["target"]) for row in rows]
                for split, rows in records.items()
            }
        ),
        "limitations": [
            "Both tasks are deterministic synthetic controls, not broad language "
            "understanding benchmarks.",
            "T1 tests held-out phrasings but shares category keywords with training.",
            "T2 tests unseen operand pairs inside the same 0-99 range.",
        ],
    }
    artifact = repo / "artifacts/gpu4a"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return audit


class SFTDataset(Dataset):
    def __init__(self, path: str | Path, *, sample_ids: set[str] | None = None) -> None:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        self.rows = [row for row in rows if sample_ids is None or row["sample_id"] in sample_ids]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    width = max(len(row["input_ids"]) for row in rows)
    ids, masks, labels = [], [], []
    for row in rows:
        padding = width - len(row["input_ids"])
        ids.append(row["input_ids"] + [PAD_ID] * padding)
        masks.append([1] * len(row["input_ids"]) + [0] * padding)
        labels.append(row["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.bool),
        "labels": torch.tensor(labels, dtype=torch.long),
        "sample_ids": [row["sample_id"] for row in rows],
    }
