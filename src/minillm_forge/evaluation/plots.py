from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_metric_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def plot_training_curves(metrics_path: str | Path, output_path: str | Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires matplotlib") from exc
    records = read_metric_records(metrics_path)
    training = [record for record in records if record.get("event") == "train"]
    validation = [record for record in records if record.get("event") == "evaluation"]
    if not training:
        raise ValueError("metrics file contains no training records")

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    axes[0].plot(
        [record["global_step"] for record in training],
        [record["loss"] for record in training],
        label="train",
    )
    if validation:
        axes[0].plot(
            [record["global_step"] for record in validation],
            [record["validation_loss"] for record in validation],
            marker="o",
            label="validation",
        )
    axes[0].set(xlabel="Optimizer step", ylabel="Cross-entropy", title="Training loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        [record["global_step"] for record in training],
        [record["grad_norm"] for record in training],
        color="#c44e52",
        label="gradient norm",
    )
    axes[1].set(xlabel="Optimizer step", ylabel="L2 norm", title="Gradient stability")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination
