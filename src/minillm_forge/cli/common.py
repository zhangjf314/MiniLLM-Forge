from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, random_split

from minillm_forge.training.trainer import TrainingConfig


class StatefulRandomSampler(Sampler[int]):
    """Random sampler whose permutation and cursor can be checkpointed."""

    def __init__(self, data_source: Dataset, seed: int) -> None:
        self.data_source = data_source
        self.generator = torch.Generator().manual_seed(seed)
        self.order: list[int] = []
        self.position = 0

    def __iter__(self):
        if not self.order or self.position >= len(self.order):
            self.order = torch.randperm(len(self.data_source), generator=self.generator).tolist()
            self.position = 0
        while self.position < len(self.order):
            index = self.order[self.position]
            self.position += 1
            yield index

    def __len__(self) -> int:
        return len(self.data_source)

    def state_dict(self) -> dict[str, Any]:
        return {
            "order": self.order.copy(),
            "position": self.position,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.order = list(state["order"])
        self.position = int(state["position"])
        self.generator.set_state(state["generator_state"])


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def training_config(values: dict[str, Any]) -> TrainingConfig:
    allowed = set(TrainingConfig.__dataclass_fields__)
    return TrainingConfig(**{key: value for key, value in values.items() if key in allowed})


def split_loaders(
    dataset: Dataset,
    *,
    batch_size: int,
    validation_fraction: float,
    seed: int,
    collate_fn: Any = None,
) -> tuple[DataLoader, DataLoader | None]:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    validation_size = int(len(dataset) * validation_fraction)
    if validation_fraction > 0 and validation_size == 0 and len(dataset) > 1:
        validation_size = 1
    train_size = len(dataset) - validation_size
    if train_size <= 0:
        raise ValueError("dataset is too small for the requested validation split")
    if validation_size:
        train_set, validation_set = random_split(
            dataset,
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        train_set, validation_set = dataset, None
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=StatefulRandomSampler(train_set, seed),
        collate_fn=collate_fn,
    )
    validation_loader = (
        DataLoader(validation_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        if validation_set is not None
        else None
    )
    return train_loader, validation_loader


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record {line_number} is not an object")
            records.append(value)
    return records
