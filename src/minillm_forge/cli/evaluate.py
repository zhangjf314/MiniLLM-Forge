from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from typing import Any

import torch
from torch.utils.data import DataLoader

from minillm_forge.cli.common import read_jsonl, write_json
from minillm_forge.data.pretrain import build_packed_dataset, iter_local_documents
from minillm_forge.evaluation import evaluate_math, evaluate_perplexity
from minillm_forge.evaluation.efficiency import measure_efficiency, parameter_counts


def load_model_and_tokenizer(
    model_name_or_path: str, adapter: str | None = None, revision: str | None = None
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("evaluation requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dtype_argument = (
        "dtype" if int(version("transformers").split(".", 1)[0]) >= 5 else "torch_dtype"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, revision=revision, **{dtype_argument: dtype}
    )
    if adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("evaluating an adapter requires peft") from exc
        model = PeftModel.from_pretrained(model, adapter)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, device


def evaluate_ppl(args: argparse.Namespace, model: Any, tokenizer: Any, device: torch.device):
    dataset = build_packed_dataset(
        iter_local_documents(args.data, args.text_field),
        tokenizer,
        args.max_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        drop_remainder=False,
    )
    return evaluate_perplexity(
        model, DataLoader(dataset, batch_size=args.batch_size), device=device
    )


def evaluate_task(args: argparse.Namespace, model: Any, tokenizer: Any, device: torch.device):
    samples = read_jsonl(args.data[0])

    def generate(problem: str) -> str:
        messages = [
            {"role": "system", "content": "You are a mathematical reasoning assistant."},
            {"role": "user", "content": problem},
        ]
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"User:\n{problem}\n\nAssistant:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuation = output[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(continuation, skip_special_tokens=True)

    return evaluate_math(
        samples,
        generate,
        problem_field=args.problem_field,
        answer_field=args.answer_field,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate loss/PPL or mathematical exact match")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--revision")
    parser.add_argument("--mode", choices=["ppl", "math"], required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", default="artifacts/eval_manifests/evaluation.json")
    parser.add_argument("--dataset-version", default="local-unversioned")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--problem-field", default="problem")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.adapter, args.revision)
    with measure_efficiency(device) as efficiency:
        result = (
            evaluate_ppl(args, model, tokenizer, device)
            if args.mode == "ppl"
            else evaluate_task(args, model, tokenizer, device)
        )
    manifest = {
        "model": args.model,
        "adapter": args.adapter,
        "revision": args.revision,
        "dataset_version": args.dataset_version,
        "seed": args.seed,
        "mode": args.mode,
        **parameter_counts(model),
        **efficiency,
        "metrics": result,
    }
    write_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
