from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from minillm_forge.finetuning.lora import add_lora_adapters


def load_qlora_model(
    model_name_or_path: str,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: str | Sequence[str] = "all-linear",
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    try:
        import bitsandbytes  # noqa: F401
        from peft import prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "QLoRA requires transformers, peft, bitsandbytes, and a supported CUDA GPU"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires CUDA in this project")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    return add_lora_adapters(
        model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )
