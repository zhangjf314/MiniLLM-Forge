from __future__ import annotations

from importlib.metadata import version
from typing import Any

import torch


def load_full_sft_model(
    model_name_or_path: str,
    *,
    precision: str = "bf16",
    gradient_checkpointing: bool = True,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("install 'transformers' to load a pretrained model") from exc
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(precision)
    if dtype is None:
        raise ValueError("precision must be fp32, fp16, or bf16")
    dtype_argument = (
        "dtype" if int(version("transformers").split(".", 1)[0]) >= 5 else "torch_dtype"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        **{dtype_argument: dtype},
        **kwargs,
    )
    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model
