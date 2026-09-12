from __future__ import annotations

import argparse
import hashlib
import json

import torch

from minillm_forge.cli.common import split_loaders, training_config, write_json
from minillm_forge.config import load_config
from minillm_forge.data.manifest import validate_file_manifest
from minillm_forge.data.pretrain import CausalLMDataset, build_packed_dataset, iter_local_documents
from minillm_forge.model import MiniLLM, MiniLLMConfig
from minillm_forge.training import ForgeTrainer, build_adamw, build_cosine_scheduler
from minillm_forge.training.trainer import set_seed


def synthetic_dataset(vocab_size: int, sequence_length: int, samples: int) -> CausalLMDataset:
    """Deterministic repeating sequences for CPU smoke and tiny-overfit gates."""
    sequences = []
    for sample in range(samples):
        start = 4 + sample % max(vocab_size - 20, 1)
        sequences.append(
            [1]
            + [4 + ((start + index) % (vocab_size - 4)) for index in range(sequence_length - 2)]
            + [2]
        )
    return CausalLMDataset(sequences)


def run(config_path: str, resume: str | None = None) -> dict[str, object]:
    config = load_config(config_path)
    if config.get("data", {}).get("frozen"):
        from minillm_forge.cli.formal_pretrain import run as run_formal

        return run_formal(config_path, resume=resume)
    model_values = dict(config["model"])
    data_values = config["data"]
    set_seed(config["training"].get("seed", 42))
    if data_values.get("synthetic", False):
        dataset = synthetic_dataset(
            model_values["vocab_size"],
            data_values["sequence_length"],
            data_values.get("samples", 128),
        )
    else:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("install tokenizers before pretraining on text") from exc
        tokenizer = Tokenizer.from_file(data_values["tokenizer"])
        if data_values.get("manifest"):
            files = data_values["files"]
            if len(files) != 1:
                raise ValueError(
                    "file-manifest validation currently requires exactly one corpus file"
                )
            validate_file_manifest(data_values["manifest"], files[0])
        model_values["vocab_size"] = tokenizer.get_vocab_size()
        dataset = build_packed_dataset(
            iter_local_documents(data_values["files"], data_values.get("text_field", "text")),
            tokenizer,
            data_values["sequence_length"],
            pad_token_id=tokenizer.token_to_id("<pad>"),
            eos_token_id=tokenizer.token_to_id("<eos>"),
            drop_remainder=data_values.get("drop_remainder", True),
        )
    train_values = config["training"]
    train_loader, validation_loader = split_loaders(
        dataset,
        batch_size=train_values.get("micro_batch_size", 2),
        validation_fraction=data_values.get("validation_fraction", 0.1),
        seed=train_values.get("seed", 42),
    )
    model = MiniLLM(MiniLLMConfig.from_dict(model_values))
    dataset_hash = hashlib.sha256(dataset.tokens.numpy().tobytes()).hexdigest()
    optimizer_values = config["optimizer"]
    optimizer = build_adamw(
        model,
        lr=float(optimizer_values.get("lr", 3e-4)),
        betas=tuple(optimizer_values.get("betas", [0.9, 0.95])),
        weight_decay=float(optimizer_values.get("weight_decay", 0.1)),
    )
    scheduler_values = config.get("scheduler", {})
    scheduler = build_cosine_scheduler(
        optimizer,
        total_steps=train_values["max_steps"],
        warmup_steps=scheduler_values.get("warmup_steps"),
        warmup_ratio=scheduler_values.get("warmup_ratio", 0.03),
        min_lr_ratio=scheduler_values.get("min_lr_ratio", 0.1),
    )
    trainer = ForgeTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=training_config(train_values),
        run_config=config,
    )
    if resume:
        trainer.resume(resume)
    initial = trainer.evaluate()
    state = trainer.train()
    final = trainer.evaluate()
    summary = {
        "parameters": model.num_parameters(),
        "trainable_parameters": model.num_parameters(trainable_only=True),
        "initial_evaluation": initial,
        "final_evaluation": final,
        "global_step": state.global_step,
        "tokens_seen": state.tokens_seen,
        "peak_vram_mb": trainer.peak_vram_mb(),
        "peak_reserved_vram_mb": trainer.peak_reserved_vram_mb(),
        "device": str(trainer.device),
        "gpu_name": torch.cuda.get_device_name() if trainer.device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "dataset_hash": dataset_hash,
        "training_smoke_pass": bool(
            state.global_step == train_values["max_steps"]
            and final
            and torch.isfinite(torch.tensor(final["validation_loss"]))
        ),
    }
    if config.get("experiment", {}).get("id") == "E01-tiny-overfit":
        summary["tiny_overfit_pass"] = bool(
            initial and final and final["validation_loss"] < initial["validation_loss"] * 0.7
        )
    write_json(f"{train_values['output_dir']}/summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the from-scratch MiniLLM")
    parser.add_argument("--config", default="configs/pretrain/smoke.yaml")
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.resume), indent=2))


if __name__ == "__main__":
    main()
