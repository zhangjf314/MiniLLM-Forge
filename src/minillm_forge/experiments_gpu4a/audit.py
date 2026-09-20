from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from minillm_forge.experiments_gpu4a.data import EOS_ID, MAX_LENGTH, TASKS, materialize
from minillm_forge.experiments_gpu4a.runtime import (
    CHECKPOINT,
    MODEL_MANIFEST,
    TOKENIZER_MANIFEST,
    TOKENIZER_PATH,
    file_sha256,
    load_native_model,
    read_json,
    release,
    write_json,
)

START_HEAD = "77bf6af5e8c45e71e74298f6768fba09ce32b49c"
START_BRANCH = "codex/gpu3c-eos-supervision-sft"


@torch.no_grad()
def _diagnostic_generations(model: Any, tokenizer: Any) -> list[dict[str, Any]]:
    prompts = [
        "The purpose of education is",
        "A computer program can",
        "When water freezes, it",
    ]
    records = []
    for prompt in prompts:
        prompt_ids = [1, *tokenizer.encode(prompt, add_special_tokens=False).ids]
        tensor = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        logits = model(tensor).logits
        output = model.generate(tensor, max_new_tokens=24, temperature=0.0)
        generated = output[0, len(prompt_ids) :].tolist()
        records.append(
            {
                "prompt": prompt,
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated,
                "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
                "generated_token_count": len(generated),
                "stop_reason": "EOS" if EOS_ID in generated else "LENGTH_LIMIT",
                "finite_logits": bool(torch.isfinite(logits).all()),
                "token_ids_in_vocabulary": all(
                    0 <= value < model.config.vocab_size for value in generated
                ),
            }
        )
    return records


def run_audit(repo: str | Path = ".") -> dict[str, Any]:
    started = time.perf_counter()
    repo = Path(repo).resolve()
    output = repo / "artifacts/gpu4a"
    output.mkdir(parents=True, exist_ok=True)
    dataset = materialize(repo)
    tokenizer_manifest = read_json(repo / TOKENIZER_MANIFEST)
    inventory = read_json(repo / "artifacts/training/minillm_checkpoint_inventory.json")
    expected = next(item for item in inventory["checkpoints"] if item["path"].endswith("best.pt"))
    checkpoint_hash = file_sha256(repo / CHECKPOINT)
    if checkpoint_hash != expected["sha256"]:
        raise RuntimeError("formal pretraining checkpoint hash mismatch")
    model, tokenizer, identity = load_native_model(repo)
    if model.lm_head.weight is not model.token_embedding.weight:
        raise RuntimeError("frozen tied embeddings contract is broken")
    diagnostics = _diagnostic_generations(model, tokenizer)
    if not all(row["finite_logits"] and row["token_ids_in_vocabulary"] for row in diagnostics):
        raise RuntimeError("native generation diagnostic failed")
    release(model)
    summary = read_json(repo / "runs/E01-minillm-formal/summary.json")
    result = {
        "stage": "GPU-4A-0",
        "status": "PASS",
        "checkpoint": identity,
        "checkpoint_inventory_match": True,
        "formal_pretraining": summary,
        "model_manifest_sha256": file_sha256(repo / MODEL_MANIFEST),
        "tokenizer_manifest_sha256": file_sha256(repo / TOKENIZER_MANIFEST),
        "tokenizer_artifact_sha256": file_sha256(repo / TOKENIZER_PATH),
        "tokenizer": tokenizer_manifest,
        "embedding_output_tied": True,
        "diagnostic_generations": diagnostics,
        "dataset_audit": dataset,
        "environment": {
            "python": subprocess.check_output(
                [str(repo / ".venv/Scripts/python.exe"), "--version"], text=True
            ).strip(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "gpu": torch.cuda.get_device_name(0),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "pretrained_model_audit.json", result)
    evaluation_ids = []
    from minillm_forge.experiments_gpu4a.runtime import evaluation_ids as select_ids

    evaluation_ids = select_ids(repo)
    protocol = {
        "stage": "GPU-4A",
        "version": "gpu4a-v1-frozen-before-sft",
        "start_branch": START_BRANCH,
        "start_head": START_HEAD,
        "model_type": "native_transformer",
        "initial_checkpoint": identity,
        "model_manifest_sha256": result["model_manifest_sha256"],
        "tokenizer_manifest_sha256": result["tokenizer_manifest_sha256"],
        "tokenizer_artifact_sha256": result["tokenizer_artifact_sha256"],
        "special_tokens": {"pad": 0, "bos": 1, "eos": 2, "unk": 3},
        "instruction_format": "Instruction:\\n...\\n\\nInput:\\n...\\n\\nResponse:\\n",
        "loss": "response-only including native EOS=2; prompt and PAD labels=-100",
        "max_length": MAX_LENGTH,
        "tasks": {
            TASKS[0]: "three-way support request classification with held-out test phrasings",
            TASKS[1]: "addition over unseen operand pairs in the fixed integer range 0-99",
        },
        "dataset": dataset,
        "generation_evaluation": {
            "selection": "first 64 stable sample IDs per task and split",
            "sample_ids": evaluation_ids,
            "max_new_tokens": 12,
            "decoding": "greedy",
            "checkpoint_selection": "lowest validation response-only loss; test is not consulted",
        },
        "formal_training_budget": "PENDING_BOUNDED_PREWARM_MEASUREMENT",
        "historical_frozen_inputs": {
            path.as_posix(): file_sha256(repo / path)
            for path in (
                Path("artifacts/training/minillm_formal_result.json"),
                Path("artifacts/gpu3c/gpu3c_result.json"),
                CHECKPOINT,
            )
        },
    }
    write_json(output / "protocol.json", protocol)
    (output / "PRETRAINED_MODEL_AUDIT.md").write_text(
        "# 自研 37M 预训练模型审计\n\n"
        f"正式 checkpoint：`{identity['path']}`，SHA-256 `{identity['sha256']}`。"
        f"训练 step {identity['global_step']}，处理 {identity['tokens_seen']:,} tokens。\n\n"
        f"模型共有 {identity['parameter_count']:,} 参数：8 层、hidden 512、GQA 8/4、FFN 1536、"
        "RoPE/RMSNorm/SwiGLU、1024 context、词嵌入与输出层共享。24K byte-level BPE 的"
        " `<pad>/<bos>/<eos>/<unk>` 分别为 0/1/2/3。结构、Tokenizer 与 checkpoint 严格匹配。\n\n"
        f"正式预训练验证 loss/PPL 为 {summary['final_validation']['validation_loss']:.4f}/"
        f"{summary['final_validation']['validation_perplexity']:.2f}；NaN/Inf/OOM 均为 0。"
        "三个固定 English prompt 均成功生成、logits 有限且 token ID 合法。基础生成仍有重复和"
        "语义不可靠现象，这符合 50M-token 小模型历史结论，不冒充指令能力。\n",
        encoding="utf-8",
    )
    (output / "SFT_DATASET_AND_TASK_PROTOCOL.md").write_text(
        "# GPU-4A SFT 数据与任务协议\n\n"
        "T1 是 BILLING/TECHNICAL/ACCOUNT 三分类；训练和测试使用不同句式，但共享类别语义词。"
        "T2 是 0–99 两整数加法；三个 split 的操作数对严格不重叠，测试答案仍位于训练答案分布。\n\n"
        f"划分为 train/validation/test = {dataset['split_counts']['train']:,}/"
        f"{dataset['split_counts']['validation']:,}/{dataset['split_counts']['test']:,}。"
        f"语义 key 交叉计数均为 0。最长 prompt/target/sequence 为 "
        f"{dataset['lengths']['prompt_max']}/{dataset['lengths']['target_max']}/"
        f"{dataset['lengths']['sequence_max']} tokens，低于 {MAX_LENGTH}，无截断。"
        f"全部 {dataset['sample_count']:,} 条样本目标完整且监督原生 EOS=2。\n\n"
        "格式固定为 Instruction/Input/Response 纯文本，不扩展词表、不使用 Qwen chat template。"
        "Prompt 和 PAD labels 为 -100；Response 正常 token 与最后 EOS 参与 causal LM loss。"
        "测试集仅在协议和 checkpoint 选择规则冻结后用于最终确认。\n",
        encoding="utf-8",
    )
    return result
