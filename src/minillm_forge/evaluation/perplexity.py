from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch


@torch.no_grad()
def evaluate_perplexity(
    model: Any,
    batches: Iterable[dict[str, torch.Tensor]],
    *,
    device: str | torch.device | None = None,
) -> dict[str, float | int]:
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(device or "cpu")
    if device is not None:
        model_device = torch.device(device)
    was_training = model.training
    model.eval()
    negative_log_likelihood = 0.0
    token_count = 0
    for raw_batch in batches:
        batch = {key: value.to(model_device) for key, value in raw_batch.items()}
        labels = batch.get("labels", batch["input_ids"])
        output = model(**{**batch, "labels": labels})
        loss = output["loss"] if isinstance(output, dict) else output.loss
        shifted = labels[:, 1:]
        valid_tokens = int(shifted.ne(-100).sum())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite loss during perplexity evaluation")
        negative_log_likelihood += float(loss) * valid_tokens
        token_count += valid_tokens
    model.train(was_training)
    mean_loss = negative_log_likelihood / max(token_count, 1)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "evaluated_tokens": token_count,
    }
