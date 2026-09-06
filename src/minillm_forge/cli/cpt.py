from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from minillm_forge.cli.common import split_loaders, training_config, write_json
from minillm_forge.config import load_config
from minillm_forge.data.cpt import stream_huggingface_texts
from minillm_forge.data.pretrain import build_packed_dataset, iter_local_documents
from minillm_forge.finetuning.full_sft import load_full_sft_model
from minillm_forge.training import ForgeTrainer, build_adamw, build_cosine_scheduler
from minillm_forge.training.trainer import set_seed


def run(config_path: str, resume: str | None = None) -> dict[str, object]:
    config = load_config(config_path)
    data = config["data"]
    model_values = config["model"]
    training = config["training"]
    set_seed(training.get("seed", 42))
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("CPT requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_values["name_or_path"], revision=model_values.get("revision")
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if data.get("files"):
        texts = iter_local_documents(data["files"], data.get("text_field", "text"))
    else:
        texts = stream_huggingface_texts(
            data["dataset"],
            subset=data.get("subset"),
            split=data.get("split", "train"),
            text_field=data.get("text_field", "text"),
            revision=data.get("revision"),
            max_samples=data.get("max_samples"),
        )
    dataset = build_packed_dataset(
        texts,
        tokenizer,
        data.get("sequence_length", 1024),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        drop_remainder=True,
    )
    dataset_hash = hashlib.sha256(dataset.tokens.numpy().tobytes()).hexdigest()
    write_json(
        data.get("manifest", "artifacts/data_manifests/cpt.json"),
        {
            "dataset": data["dataset"],
            "subset": data.get("subset"),
            "revision": data.get("revision"),
            "sample_count": len(dataset),
            "token_count": dataset.tokens.numel(),
            "sha256": dataset_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sequence_length": data.get("sequence_length", 1024),
        },
    )
    train_loader, validation_loader = split_loaders(
        dataset,
        batch_size=training.get("micro_batch_size", 1),
        validation_fraction=data.get("validation_fraction", 0.02),
        seed=training.get("seed", 42),
    )
    model = load_full_sft_model(
        model_values["name_or_path"],
        precision=training.get("precision", "bf16"),
        gradient_checkpointing=training.get("gradient_checkpointing", True),
        revision=model_values.get("revision"),
    )
    optimizer = build_adamw(model, **config["optimizer"])
    scheduler = build_cosine_scheduler(
        optimizer, total_steps=training["max_steps"], **config.get("scheduler", {})
    )
    trainer = ForgeTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=training_config(training),
        run_config=config,
    )
    if resume:
        trainer.resume(resume)
    baseline = trainer.evaluate()
    state = trainer.train()
    export_path = Path(training["output_dir"]) / "final_model"
    model.save_pretrained(export_path, safe_serialization=True)
    tokenizer.save_pretrained(export_path)
    result = {
        "baseline": baseline,
        "final": trainer.evaluate(),
        "global_step": state.global_step,
        "peak_vram_mb": trainer.peak_vram_mb(),
        "peak_reserved_vram_mb": trainer.peak_reserved_vram_mb(),
        "device": str(trainer.device),
        "gpu_name": torch.cuda.get_device_name() if trainer.device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "model_path": str(export_path),
        "dataset_hash": dataset_hash,
    }
    write_json(f"{training['output_dir']}/summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue pretraining a base causal LM")
    parser.add_argument("--config", default="configs/cpt/qwen3_math.yaml")
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.resume), indent=2))


if __name__ == "__main__":
    main()
