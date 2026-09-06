from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import Dataset

from minillm_forge.cli.common import read_jsonl, split_loaders, training_config, write_json
from minillm_forge.config import load_config
from minillm_forge.data.manifest import dataset_digest
from minillm_forge.data.sft import SFTDataCollator, encode_sft_example, is_high_quality_math_sample
from minillm_forge.finetuning import add_lora_adapters, load_full_sft_model, load_qlora_model
from minillm_forge.finetuning.lora import trainable_parameter_summary
from minillm_forge.training import ForgeTrainer, build_adamw, build_cosine_scheduler
from minillm_forge.training.trainer import set_seed


class EncodedSFTDataset(Dataset):
    def __init__(self, records: list[dict[str, list[int]]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.records[index]


def run(config_path: str, resume: str | None = None) -> dict[str, object]:
    config = load_config(config_path)
    model_values, data, training = config["model"], config["data"], config["training"]
    set_seed(training.get("seed", 42))
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("SFT requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_values["name_or_path"], revision=model_values.get("revision")
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = read_jsonl(data["file"])
    if data.get("quality_filter", True):
        records = [record for record in records if is_high_quality_math_sample(record)]
    records = records[: data.get("max_samples")]
    dataset_hash = dataset_digest(records)
    if data.get("manifest"):
        manifest = json.loads(Path(data["manifest"]).read_text(encoding="utf-8"))
        if manifest.get("sha256") != dataset_hash:
            raise ValueError(
                "prepared SFT records do not match the configured data manifest; "
                "regenerate the manifest before training"
            )
    encoded = [
        encode_sft_example(
            tokenizer,
            user=str(record[data.get("problem_field", "problem")]),
            assistant=str(record[data.get("answer_field", "solution")]),
            system=data.get("system_prompt", "You are a mathematical reasoning assistant."),
            max_length=data.get("max_length", 1024),
        )
        for record in records
    ]
    dataset = EncodedSFTDataset(encoded)
    collator = SFTDataCollator(tokenizer.pad_token_id)
    train_loader, validation_loader = split_loaders(
        dataset,
        batch_size=training.get("micro_batch_size", 1),
        validation_fraction=data.get("validation_fraction", 0.05),
        seed=training.get("seed", 42),
        collate_fn=collator,
    )
    method = model_values.get("method", "lora").lower()
    if method == "qlora":
        model = load_qlora_model(
            model_values["name_or_path"],
            revision=model_values.get("revision"),
            **config.get("lora", {}),
        )
    else:
        model = load_full_sft_model(
            model_values["name_or_path"],
            precision=training.get("precision", "bf16"),
            gradient_checkpointing=training.get("gradient_checkpointing", True),
            revision=model_values.get("revision"),
        )
        if method == "lora":
            model = add_lora_adapters(model, **config.get("lora", {}))
        elif method != "full":
            raise ValueError("model.method must be full, lora, or qlora")
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
        "method": method,
        **trainable_parameter_summary(model),
        "baseline": baseline,
        "final": trainer.evaluate(),
        "global_step": state.global_step,
        "peak_vram_mb": trainer.peak_vram_mb(),
        "model_path": str(export_path),
        "dataset_hash": dataset_hash,
    }
    write_json(f"{training['output_dir']}/summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full, LoRA, or QLoRA SFT")
    parser.add_argument("--config", default="configs/sft/qwen3_lora.yaml")
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.resume), indent=2))


if __name__ == "__main__":
    main()
