from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from minillm_forge.cli.common import read_jsonl
from minillm_forge.data.sft import SFTDataCollator, encode_sft_example
from minillm_forge.finetuning import add_lora_adapters, load_full_sft_model, load_qlora_model
from minillm_forge.finetuning.lora import trainable_parameter_summary


def _memory() -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mib": torch.cuda.memory_reserved() / 1024**2,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "free_mib": free_bytes / 1024**2,
        "total_mib": total_bytes / 1024**2,
    }


def _fixed_length_batch(
    tokenizer: Any,
    data_path: str | Path,
    sequence_length: int,
    *,
    objective: str,
) -> dict[str, torch.Tensor]:
    row = read_jsonl(data_path)[0]
    feature = encode_sft_example(
        tokenizer,
        user=str(row["problem"]),
        assistant=str(row["solution"]),
        max_length=sequence_length,
    )
    if len(feature["input_ids"]) > sequence_length:
        raise ValueError("smoke sample exceeds sequence length")
    padding = sequence_length - len(feature["input_ids"])
    feature["input_ids"].extend([tokenizer.pad_token_id] * padding)
    feature["attention_mask"].extend([0] * padding)
    feature["labels"].extend([-100] * padding)
    batch = SFTDataCollator(tokenizer.pad_token_id, pad_to_multiple_of=None)([feature])
    if objective == "cpt":
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][~batch["attention_mask"]] = -100
    return batch


def _load_model(mode: str, model_name: str, revision: str):
    if mode == "qlora":
        return load_qlora_model(
            model_name,
            revision=revision,
            rank=4,
            alpha=8,
            dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        )
    model = load_full_sft_model(
        model_name,
        revision=revision,
        precision="bf16",
        gradient_checkpointing=mode in {"full-cpt", "full-sft"},
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    if mode == "lora":
        model = add_lora_adapters(
            model,
            rank=4,
            alpha=8,
            dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        )
    return model


def qualify_qwen(
    *,
    mode: str,
    output_path: str | Path,
    model_name: str,
    revision: str,
    data_path: str | Path,
    sequence_length: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "model": model_name,
        "revision": revision,
        "sequence_length": sequence_length,
        "status": "FAIL",
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; Qwen qualification refuses CPU fallback")
        from transformers import AutoTokenizer

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        objective = "cpt" if mode == "full-cpt" else "sft"
        batch = _fixed_length_batch(tokenizer, data_path, sequence_length, objective=objective)
        model = _load_model(mode, model_name, revision)
        result.update(trainable_parameter_summary(model))
        result["after_load_memory"] = _memory()
        batch = {key: value.to("cuda") for key, value in batch.items()}

        if mode == "base":
            model.eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
            if logits.device.type != "cuda" or not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Qwen CUDA forward did not return finite CUDA logits")
            prompt_length = int(batch["attention_mask"].sum())
            prompt_ids = batch["input_ids"][:, :prompt_length]
            with torch.inference_mode():
                generated = model.generate(
                    prompt_ids,
                    max_new_tokens=4,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            result.update(
                {
                    "forward": "PASS",
                    "generation": "PASS",
                    "generated_tokens": generated.shape[1] - prompt_ids.shape[1],
                    "logits_device": str(logits.device),
                }
            )
        else:
            model.train()
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=1e-4)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**batch)
                loss = output.loss
            if loss is None or not bool(torch.isfinite(loss)):
                raise FloatingPointError("training smoke produced a non-finite loss")
            loss.backward()
            gradients_finite = all(
                bool(torch.isfinite(parameter.grad).all())
                for parameter in trainable
                if parameter.grad is not None
            )
            if not gradients_finite:
                raise FloatingPointError("training smoke produced non-finite gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            result.update(
                {
                    "forward": "PASS",
                    "backward": "PASS",
                    "optimizer_step": "PASS",
                    "loss": float(loss.detach()),
                    "grad_norm": float(grad_norm),
                    "gradients_finite": gradients_finite,
                }
            )
            if mode == "qlora":
                import bitsandbytes as bnb

                result["bitsandbytes_version"] = bnb.__version__
                result["quantization"] = {
                    "bits": 4,
                    "type": "nf4",
                    "double_quantization": True,
                    "compute_dtype": "bfloat16",
                }
        torch.cuda.synchronize()
        result["status"] = "PASS"
    except torch.OutOfMemoryError as exc:
        result["status"] = "OOM"
        result["error"] = str(exc)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if torch.cuda.is_available():
            try:
                result["final_memory"] = _memory()
            except Exception as exc:
                result["memory_error"] = str(exc)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
