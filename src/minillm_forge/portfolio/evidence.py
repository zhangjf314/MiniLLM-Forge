# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _claim(
    claim_id: str,
    text: str,
    category: str,
    status: str,
    artifact: str,
    metric: str,
    limitation: str,
    *,
    resume: bool,
    interview: bool = True,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim_text": text,
        "category": category,
        "status": status,
        "evidence_artifact": artifact,
        "evidence_metric": metric,
        "limitations": limitation,
        "resume_allowed": resume,
        "interview_allowed": interview,
    }


def _claims() -> list[dict[str, Any]]:
    return [
        _claim("C001", "Implemented and checkpointed a 37,462,528-parameter, 8-layer decoder Transformer with GQA 8/4, RoPE, RMSNorm, SwiGLU, tied embeddings, and context 1024.", "native_model", "VALIDATED", "artifacts/training/minillm_formal_model.json", "parameter_count=37462528; architecture_frozen=true", "Architecture validation is not a claim of broad language ability.", resume=True),
        _claim("C002", "Trained a native 24K byte-level BPE tokenizer with zero validation UNKs and exact normalized round-trip success.", "tokenizer", "VALIDATED", "artifacts/tokenizers/minillm-tokenizer-manifest.json", "vocab_size=24000; unk_ratio=0; roundtrip_success_rate=1", "Tokenizer corpus and revision are bounded and pinned.", resume=True),
        _claim("C003", "Completed BF16 pretraining for 50,003,968 processed tokens and 3,052 optimizer steps; fixed validation loss improved 10.1964 to 4.6484 and PPL 26,805.55 to 104.42.", "pretraining", "VALIDATED", "artifacts/training/minillm_formal_result.json", "MINILLM_FORMAL_PRETRAINING_VALIDATED", "50.0M processed tokens are 1.554 corpus passes, not 50M unique tokens; no strong general-language claim.", resume=True),
        _claim("C004", "Verified exact checkpoint resume for model, optimizer, scheduler, and RNG state.", "training_system", "VALIDATED", "artifacts/training/minillm_gpu_resume_validation.json", "classification=EXACT_RESUME_CONFIRMED", "Bounded control on the qualified local environment.", resume=True),
        _claim("C005", "Built a leakage-controlled native SFT dataset with 4,800/700/700 train/validation/test rows, zero semantic-key overlap, 6,200 complete targets with supervised EOS, and no truncation.", "native_sft", "VALIDATED", "artifacts/gpu4a/dataset_audit.json", "dataset_digest=a21cf5...; eos_supervised_count=6200", "Two deterministic synthetic tasks; T1 shares category vocabulary and T2 stays in 0-99.", resume=True),
        _claim("C006", "Completed 300-step full-parameter BF16 response-only SFT of all 37,462,528 parameters; validation loss improved 5.9922 to 0.5943 in 48.87 seconds.", "native_sft", "VALIDATED", "artifacts/gpu4a/formal_sft_result.json", "selected checkpoint step=300; median=12188 input tok/s; peak allocated=1003.63 MiB", "One seed and one synthetic two-task mixture.", resume=True),
        _claim("C007", "Native SFT improved T1 support classification from 0/64 to 64/64 on a fixed independent test subset.", "native_sft_quality", "VALIDATED", "artifacts/gpu4a/sft_evaluation.json", "test:T1_SUPPORT_CLASSIFICATION correct=64 of 64", "This is a 64-example subset of the 300-row T1 test split, not full-test accuracy.", resume=True),
        _claim("C008", "Native SFT did not acquire reliable two-digit addition: T2 improved only from 0/64 to 2/64 on the fixed test subset.", "native_sft_quality", "NEGATIVE_RESULT", "artifacts/gpu4a/sft_evaluation.json", "test:T2_INTEGER_ADDITION correct=2 of 64", "Bounded synthetic task and subset, but clearly insufficient capability.", resume=False),
        _claim("C009", "All 384 native-SFT evaluation generations were format-valid, complete, EOS-terminated, non-repetitive, and avoided length-limit termination.", "generation_behavior", "VALIDATED", "artifacts/gpu4a/sft_evaluation.json", "384 records across 2 tasks x 3 splits x 64", "Generation behavior is not equivalent to task correctness.", resume=True),
        _claim("C010", "Completed the formal Qwen3-0.6B PEFT campaign: 12/12 LoRA/QLoRA runs, three seeds, 1,200 steps each, and 8,400 audited MATH-500/GSM8K outputs.", "peft", "VALIDATED", "artifacts/gpu2d_formal/final_result.json", "12 training runs; 24 evaluation jobs; 8400 outputs", "Frozen 512-context protocol; Qwen Full SFT and 1024-context formal study were not completed.", resume=True),
        _claim("C011", "Math CPT did not yield a general downstream reasoning improvement under the frozen PEFT protocol: LoRA transfer was negative and QLoRA was mixed.", "peft_quality", "NEGATIVE_RESULT", "artifacts/gpu2d_formal/final_result.json", "LORA=NEGATIVE; QLORA=MIXED; scientific=CPT_TO_PEFT_SFT_TRANSFER_METHOD_DEPENDENT", "Applies only to the frozen context, datasets, adapter settings, and three seeds.", resume=True),
        _claim("C012", "The formal PEFT generation campaign had 8,398/8,400 outputs reach max_new_tokens=512.", "generation_diagnostics", "NEGATIVE_RESULT", "artifacts/gpu3a/gpu3a_result.json", "length_limit numerator=8398 denominator=8400", "This observation alone does not prove why the models failed to stop.", resume=True),
        _claim("C013", "Improved evaluator extraction on frozen outputs: legacy 1,764/8,400; first-valid 2,008; last-valid 1,824; conflict-aware 1,793, with 607 conflicts exposed.", "evaluation_system", "VALIDATED", "artifacts/gpu3b/extraction_ablation.json", "36/36 legacy failures recovered; 16 correct", "Evaluator improvement, not model-capability improvement.", resume=True),
        _claim("C014", "External answer-boundary stopping reduced post-answer redundant generation in the bounded S3 confirmation without improving autonomous EOS learning.", "inference_system", "VALIDATED", "artifacts/gpu3b/gpu3b_result.json", "confirmation S3 reduced redundancy for 6/8 outputs; paired accuracy 3/8 unchanged", "Small confirmation set; external intervention, not a model improvement.", resume=True),
        _claim("C015", "A controlled 64-step EOS-label ablation did not validate autonomous-stop improvement; the EOS-supervised arm emitted no autonomous <|im_end|> and hit 512 tokens on all four C0 prompts.", "eos_ablation", "NEGATIVE_RESULT", "artifacts/gpu3c/gpu3c_result.json", "classification=GPU3C_CONTROLLED_NEGATIVE_RESULT", "One seed and four frozen confirmation problems.", resume=True),
        _claim("C016", "Integrated experimental Manual, SDPA Math, SDPA Auto, and forced Flash backend selection while retaining Manual as the formal default.", "attention_system", "PARTIALLY_VALIDATED", "artifacts/gpu4b/gpu4b_result.json", "classification=GPU4B_PARTIAL_VALIDATION; FP32 Math gate passed", "BF16 correctness gates failed and no qualified performance benchmark exists.", resume=True),
        _claim("C017", "SDPA Math and Auto failed the frozen BF16 correctness/gradient/update gates; no speedup claim is supported.", "attention_system", "NEGATIVE_RESULT", "artifacts/gpu4b/attention_correctness.json", "Math BF16 logits max abs=0.15625; Auto=0.20703125", "Fixed single-batch gate and frozen tolerances.", resume=True),
        _claim("C018", "Built-in FlashAttention was unavailable in the qualified Windows/PyTorch environment.", "attention_system", "BLOCKED", "artifacts/gpu4b/environment_audit.json", "FLASH_BACKEND_NOT_AVAILABLE", "Platform/build limitation; not a universal hardware claim.", resume=False),
        _claim("C019", "torch.compile qualification was blocked by missing Triton; no compile speedup was measured.", "compile", "BLOCKED", "artifacts/gpu4b/compile_qualification.json", "classification=TORCH_COMPILE_BLOCKED_TRITON_MISSING", "Environment limitation; eager execution remains formal.", resume=False),
        _claim("C020", "Localized the BF16 Manual/SDPA Math split to the first attention context boundary and supported a different intermediate-precision/rounding mechanism without finding a semantic implementation defect.", "numerical_diagnostics", "VALIDATED", "artifacts/gpu4b_r1/gpu4b_r1_result.json", "classification=GPU4B_R1_NUMERICAL_PATH_DIFFERENCE_IDENTIFIED", "Exact SDPA internal instruction ordering remains unknown.", resume=True),
        _claim("C021", "Qwen full-parameter SFT effectiveness was not formally evaluated in the completed campaign.", "qwen_full_sft", "NOT_VALIDATED", "artifacts/gpu2d_formal/final_result.json", "claim boundary excludes Full SFT", "Implementation and qualification history do not equal formal effectiveness evidence.", resume=False),
        _claim("C022", "Distributed DDP/FSDP training was not executed or validated.", "distributed_training", "NOT_VALIDATED", "artifacts/gpu4c/protocol.json", "explicitly outside GPU-4C and no formal artifact", "No performance or correctness claim is permitted.", resume=False),
        _claim("C023", "At context 512, 12,469/19,162 PEFT training samples retained a supervised assistant-end token (65.07%); 6,693 training samples lost it after truncation.", "eos_supervision", "VALIDATED", "artifacts/gpu3a/gpu3a_result.json", "context_512 coverage_over_training_samples=0.6507149567", "Coverage is a confirmed data fact, not proof that missing EOS supervision caused length-limit generation.", resume=True),
    ]


def _capabilities() -> list[dict[str, str]]:
    return [
        {"capability": "Native Transformer implementation", "status": "VALIDATED", "evidence": "C001", "limitation": "37.46M bounded research model"},
        {"capability": "Tokenizer training", "status": "VALIDATED", "evidence": "C002", "limitation": "24K train-only FineWeb-Edu tokenizer"},
        {"capability": "Pretraining", "status": "VALIDATED", "evidence": "C003", "limitation": "50.0M processed tokens; not converged general LLM"},
        {"capability": "BF16 training", "status": "VALIDATED", "evidence": "C003,C006", "limitation": "Qualified RTX 5060 Laptop environment"},
        {"capability": "Checkpoint exact resume", "status": "VALIDATED", "evidence": "C004", "limitation": "Bounded deterministic control"},
        {"capability": "Continued pretraining (CPT)", "status": "VALIDATED", "evidence": "artifacts/training/qwen_math_cpt_result.json", "limitation": "Math PPL improved while general PPL degraded"},
        {"capability": "Native Full SFT", "status": "VALIDATED", "evidence": "C005-C009", "limitation": "Synthetic tasks; generation evaluation used 64-row subsets"},
        {"capability": "Qwen Full SFT", "status": "IMPLEMENTED_BUT_NOT_EFFECTIVENESS_VALIDATED", "evidence": "C021", "limitation": "Formal arm not completed"},
        {"capability": "LoRA", "status": "VALIDATED", "evidence": "C010,C011", "limitation": "Training/evaluation valid; CPT transfer result negative"},
        {"capability": "QLoRA", "status": "VALIDATED", "evidence": "C010,C011", "limitation": "Training/evaluation valid; CPT transfer mixed"},
        {"capability": "Response-only loss", "status": "VALIDATED", "evidence": "C005,C006,C010", "limitation": "Validated in native and PEFT workflows"},
        {"capability": "Evaluation harness", "status": "VALIDATED", "evidence": "C009,C010,C013", "limitation": "Benchmark contamination remains model-history dependent"},
        {"capability": "Generation diagnostics", "status": "VALIDATED", "evidence": "C012-C015", "limitation": "Some causal explanations remain hypotheses"},
        {"capability": "Answer extraction", "status": "VALIDATED", "evidence": "C013", "limitation": "Evaluator improvement only"},
        {"capability": "External stopping", "status": "VALIDATED", "evidence": "C014", "limitation": "Bounded confirmation; not autonomous EOS"},
        {"capability": "EOS supervision ablation", "status": "NEGATIVE_RESULT", "evidence": "C015", "limitation": "Expected benefit was not observed"},
        {"capability": "SDPA integration", "status": "IMPLEMENTED_BUT_NOT_EFFECTIVENESS_VALIDATED", "evidence": "C016,C017,C020", "limitation": "FP32 pass; BF16 gate fail; no performance benchmark"},
        {"capability": "FlashAttention", "status": "BLOCKED", "evidence": "C018", "limitation": "Unavailable built-in kernel"},
        {"capability": "torch.compile", "status": "BLOCKED", "evidence": "C019", "limitation": "Triton missing; no speedup"},
        {"capability": "Distributed training", "status": "NOT_EXECUTED", "evidence": "C022", "limitation": "No DDP/FSDP experiment"},
    ]


def _experiment(
    experiment_id: str, stage: str, model: str, initialization: str, method: str,
    dataset: str, seed: int | None, steps: int | None, status: str, classification: str,
    metrics: dict[str, Any], artifact: str, checkpoint: str | None,
    *, historical: str = "CURRENT", resume_safe: bool = False, portfolio: str = "INCLUDE",
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id, "stage": stage, "model": model,
        "initialization": initialization, "training_method": method, "dataset": dataset,
        "seed": seed, "steps": steps, "status": status, "classification": classification,
        "primary_metrics": metrics, "artifact_path": artifact, "checkpoint_path": checkpoint,
        "historical_or_current": historical, "resume_safe": resume_safe,
        "portfolio_status": portfolio,
    }


def _experiments(gpu2d: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        _experiment("tiny-overfit", "GPU-1", "MiniLLM tiny", "random", "pretraining correctness gate", "bounded tiny corpus", 1337, 60, "COMPLETED", "PASS", {"validation_loss": "4.9014 -> 0.7506"}, "runs/tiny-overfit-verified/summary.json", "runs/tiny-overfit-verified/last.pt", historical="HISTORICAL_CONTROL", resume_safe=True),
        _experiment("E01", "GPU-1", "MiniLLM 37M", "random", "BF16 pretraining", "FineWeb-Edu pinned subset", 1337, 3052, "COMPLETED", "MINILLM_FORMAL_PRETRAINING_VALIDATED", {"tokens": 50003968, "validation_loss": 4.648381617334154, "validation_ppl": 104.41586393802609}, "artifacts/training/minillm_formal_result.json", "runs/E01-minillm-formal/best.pt", resume_safe=True),
        _experiment("E04", "GPU-2A", "Qwen3-0.6B-Base", "base", "full-parameter CPT", "FineMath-4+ bounded subset", 42, 2442, "COMPLETED", "QWEN_MATH_CPT_VALIDATED_WITH_GENERAL_DEGRADATION", {"tokens": 10002432, "math_ppl": "5.1715 -> 5.0348", "general_ppl": "17.0799 -> 17.2311"}, "artifacts/training/qwen_math_cpt_result.json", "runs/E04-qwen3-math-cpt/best_math_validation.pt", resume_safe=True),
        _experiment("GPU2B-memory", "GPU-2B", "Qwen3-0.6B", "base/CPT", "Full SFT qualification", "SFT math", 42, None, "BLOCKED", "STAGE_GPU_2B_BLOCKED_BY_MEMORY_QUALIFICATION", {}, "artifacts/training/gpu2b-stage-result.json", None, portfolio="DOCUMENT_LIMITATION"),
        _experiment("GPU2C-qualification", "GPU-2C", "Qwen3-0.6B", "CPT", "PEFT qualification", "SFT math", 42, None, "BLOCKED", "GPU2C_BLOCKED_BY_MEMORY_QUALIFICATION", {}, "artifacts/gpu2c/qualification-result.json", None, portfolio="DOCUMENT_LIMITATION"),
    ]
    for run_id in gpu2d["artifact_identities"]["training_summary_hashes"]:
        init, method, seed_text = run_id.split("-")
        rows.append(
            _experiment(
                run_id, "GPU-2D", "Qwen3-0.6B-Base", "base" if init == "B" else "math-CPT",
                method, "19,200-row math SFT; MATH-500 + GSM8K fixed-200", int(seed_text[1:]), 1200,
                "COMPLETED", "VALID_FORMAL_PEFT_RUN", {"evaluation_examples": 700},
                f"experiments/results/{run_id}.json", f"runs/gpu2d-formal/{run_id}/checkpoints/final.pt",
                resume_safe=True,
            )
        )
    rows.extend(
        [
            _experiment("GPU3A-audit", "GPU-3A", "Qwen3 PEFT adapters", "frozen GPU-2D", "offline generation diagnosis", "8400 frozen generations", None, None, "COMPLETED", "GPU3A_ROOT_CAUSES_IDENTIFIED", {"length_limit": "8398/8400", "legacy_extraction_failures": 36}, "artifacts/gpu3a/gpu3a_result.json", None),
            _experiment("GPU3B-evaluator", "GPU-3B", "Qwen3 PEFT adapters", "frozen weights", "extraction/stopping ablation", "8400 frozen + 34 regenerated outputs", 42, None, "COMPLETED", "GPU3B_DECODING_AND_EXTRACTION_IMPROVED", {"first_valid_correct": 2008, "conflicts": 607}, "artifacts/gpu3b/gpu3b_result.json", None),
            _experiment("GPU3C-eos", "GPU-3C", "Qwen3-0.6B LoRA", "same initial adapter", "controlled EOS supervision ablation", "1024 identical ordered examples per arm", 42, 64, "COMPLETED", "GPU3C_CONTROLLED_NEGATIVE_RESULT", {"supervised_autonomous_im_end": 0, "supervised_c0_length_limit": "4/4"}, "artifacts/gpu3c/gpu3c_result.json", "runs/gpu3c/formal-64/{arm}/checkpoints/step-0064.pt"),
            _experiment("gpu4a-native-full-sft-s42-v1", "GPU-4A", "MiniLLM 37M", "E01 best", "full-parameter BF16 response-only SFT", "T1/T2 synthetic controls", 4204, 300, "COMPLETED", "GPU4A_NATIVE_SFT_CAPABILITY_VALIDATED", {"validation_loss": "5.9922 -> 0.5943", "T1_test_subset": "0/64 -> 64/64", "T2_test_subset": "0/64 -> 2/64"}, "artifacts/gpu4a/gpu4a_result.json", "runs/gpu4a/formal-full-sft-s42-v1/checkpoints/step-0300.pt", resume_safe=True),
            _experiment("GPU4B-attention", "GPU-4B", "MiniLLM 37M", "E01/GPU4A", "attention/compile qualification", "fixed correctness batch", 4204, None, "PARTIAL", "GPU4B_PARTIAL_VALIDATION", {"sdpa_math_fp32": "PASS", "sdpa_math_bf16": "FAIL", "flash": "UNAVAILABLE", "compile": "TRITON_MISSING"}, "artifacts/gpu4b/gpu4b_result.json", None, portfolio="DOCUMENT_LIMITATION"),
            _experiment("GPU4B-R1-numerics", "GPU-4B-R1", "MiniLLM 37M", "E01", "BF16 numerical diagnosis", "fixed correctness batch", 4204, None, "COMPLETED", "GPU4B_R1_NUMERICAL_PATH_DIFFERENCE_IDENTIFIED", {"bf16_logits_max_abs": 0.15625, "earliest_boundary": "attention context"}, "artifacts/gpu4b_r1/gpu4b_r1_result.json", None),
        ]
    )
    return rows


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def build_release_artifacts(repo: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo).resolve()
    out = repo / "artifacts/gpu4c"
    protocol = read_json(out / "protocol.json")
    pretrain = read_json(repo / "artifacts/training/minillm_formal_result.json")
    model = read_json(repo / "artifacts/training/minillm_formal_model.json")
    tokenizer = read_json(repo / "artifacts/tokenizers/minillm-tokenizer-manifest.json")
    gpu2d = read_json(repo / "artifacts/gpu2d_formal/final_result.json")
    gpu3a = read_json(repo / "artifacts/gpu3a/gpu3a_result.json")
    gpu3b = read_json(repo / "artifacts/gpu3b/gpu3b_result.json")
    gpu3c = read_json(repo / "artifacts/gpu3c/gpu3c_result.json")
    gpu4a = read_json(repo / "artifacts/gpu4a/gpu4a_result.json")
    gpu4b = read_json(repo / "artifacts/gpu4b/gpu4b_result.json")
    gpu4br1 = read_json(repo / "artifacts/gpu4b_r1/gpu4b_r1_result.json")
    qwen_cpt = read_json(repo / "artifacts/training/qwen_math_cpt_result.json")
    claims = _claims()
    capabilities = _capabilities()
    experiments = _experiments(gpu2d)
    manifest = {
        "stage": "GPU-4C",
        "repository": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "source_head": _git(repo, "rev-parse", "HEAD"),
        "working_tree_at_generation": "clean" if not _git(repo, "status", "--porcelain") else "modified_by_gpu4c",
        "environment": pretrain["environment"],
        "pretrained_checkpoint": "runs/E01-minillm-formal/best.pt",
        "pretrained_checkpoint_sha256": protocol["frozen_evidence_hashes"]["runs/E01-minillm-formal/best.pt"],
        "native_sft_checkpoint": "runs/gpu4a/formal-full-sft-s42-v1/checkpoints/step-0300.pt",
        "native_sft_checkpoint_sha256": protocol["frozen_evidence_hashes"]["runs/gpu4a/formal-full-sft-s42-v1/checkpoints/step-0300.pt"],
        "tokenizer": "artifacts/tokenizers/minillm-tokenizer.json",
        "tokenizer_sha256": tokenizer["tokenizer_artifact_hash"],
        "major_dataset_hashes": {
            "native_pretraining_manifest": pretrain["dataset_manifest_hash"],
            "native_sft_dataset": gpu4a["dataset_and_task_protocol"]["dataset_digest"],
            "native_sft_splits": gpu4a["dataset_and_task_protocol"]["split_sha256"],
            "gpu2d_campaign_manifest": gpu2d["artifact_identities"]["campaign_manifest_sha256"],
        },
        "model_architecture": {"parameter_count": model["parameter_count"], **model["architecture"]},
        "stage_classifications": {
            "GPU-1": pretrain["classification"],
            "GPU-2A": qwen_cpt["classification"],
            "GPU-2B": "STAGE_GPU_2B_BLOCKED_BY_MEMORY_QUALIFICATION; recovery recommended PEFT transfer",
            "GPU-2C": "GPU2C_BLOCKED_BY_MEMORY_QUALIFICATION",
            "GPU-2D": {"primary": gpu2d["primary_classification"], "scientific": gpu2d["scientific_classification"]},
            "GPU-3A": gpu3a["classification"], "GPU-3B": gpu3b["classification"],
            "GPU-3C": gpu3c["classification"], "GPU-4A": gpu4a["classification"],
            "GPU-4B": gpu4b["classification"], "GPU-4B-R1": gpu4br1["classification"],
        },
        "frozen_evidence_hashes": protocol["frozen_evidence_hashes"],
    }
    write_json(out / "final_evidence_manifest.json", manifest)
    write_json(out / "claim_registry.json", {"schema_version": 1, "allowed_statuses": ["VALIDATED", "PARTIALLY_VALIDATED", "NEGATIVE_RESULT", "NOT_VALIDATED", "BLOCKED", "HISTORICAL_ONLY"], "claims": claims})
    write_json(out / "experiment_registry.json", {"schema_version": 1, "experiments": experiments})
    claim_md = "# Claim Registry\n\nEvery public claim is tied to repository evidence; status describes evidence, not code availability.\n\n" + _markdown_table(
        ["ID", "Claim", "Category", "Status", "Evidence", "Limitation", "Resume"],
        [[c["claim_id"], c["claim_text"], c["category"], c["status"], c["evidence_artifact"], c["limitations"], "yes" if c["resume_allowed"] else "no"] for c in claims],
    ) + "\n"
    (out / "CLAIM_REGISTRY.md").write_text(claim_md, encoding="utf-8")
    capability_md = "# Capability Matrix\n\n" + _markdown_table(
        ["Capability", "Status", "Evidence", "Limitation"],
        [[c["capability"], c["status"], c["evidence"], c["limitation"]] for c in capabilities],
    ) + "\n"
    (out / "CAPABILITY_MATRIX.md").write_text(capability_md, encoding="utf-8")
    experiment_md = "# Experiment Registry\n\nNegative, blocked, historical, and diagnostic runs are retained.\n\n" + _markdown_table(
        ["ID", "Stage", "Model", "Initialization", "Method", "Seed", "Steps", "Status", "Classification", "Artifact"],
        [[e["experiment_id"], e["stage"], e["model"], e["initialization"], e["training_method"], str(e["seed"]), str(e["steps"]), e["status"], e["classification"], e["artifact_path"]] for e in experiments],
    ) + "\n"
    (out / "EXPERIMENT_REGISTRY.md").write_text(experiment_md, encoding="utf-8")
    _write_narrative_documents(out, manifest, claims, capabilities, gpu2d, gpu3a, gpu3b, gpu3c, gpu4a, gpu4b, gpu4br1, pretrain)
    return {"manifest": manifest, "claims": len(claims), "capabilities": len(capabilities), "experiments": len(experiments)}


def _write_narrative_documents(
    out: Path, manifest: dict[str, Any], claims: list[dict[str, Any]], capabilities: list[dict[str, str]],
    gpu2d: dict[str, Any], gpu3a: dict[str, Any], gpu3b: dict[str, Any], gpu3c: dict[str, Any],
    gpu4a: dict[str, Any], gpu4b: dict[str, Any], gpu4br1: dict[str, Any], pretrain: dict[str, Any],
) -> None:
    truth = """# Documentation Truth Audit

## Scope

Reviewed `README.md`, `reports/FINAL_REPORT.md`, portfolio/resume evidence, CLI entry points, comments, and committed experiment artifacts for claims involving FlashAttention, torch.compile, Full SFT, LoRA/QLoRA, 50M tokens, 100%, EOS, DPO/GRPO, 1.5B models, DDP, and FSDP.

## Corrections applied

- Native T1 is stated only as **64/64 on a fixed independent test subset**, never full 300-row test accuracy.
- Native generation behavior retains its denominator: **384/384**.
- GPU-3B extraction and external stopping are evaluator/inference improvements, not model-capability improvements.
- GPU-3C is preserved as a controlled negative result; EOS supervision did not produce the expected autonomous stop.
- GPU-4B has no qualified SDPA/Flash/compile performance result. Manual + Eager remains formal.
- Qwen Full SFT code/qualification is distinguished from formal effectiveness evidence; the completed campaign is LoRA/QLoRA.
- Distributed training, DPO, and GRPO are not presented as validated capabilities.

## Outcome

`PASS_WITH_DOCUMENTED_LIMITATIONS`: no critical public claim conflicts with the claim registry after the GPU-4C edits. Historical experiment artifacts were not rewritten.
"""
    (out / "DOCUMENTATION_TRUTH_AUDIT.md").write_text(truth, encoding="utf-8")
    technical = f"""# MiniLLM-Forge Final Technical Report

## 1. Executive Summary

MiniLLM-Forge is a reproducible training and evaluation laboratory spanning a native 37.46M decoder Transformer and a separate Qwen3-0.6B adaptation line. It demonstrates the engineering loop from tokenizer and model implementation through pretraining, native full SFT, PEFT, evaluation diagnostics, controlled negative experiments, and BF16 numerical analysis. It is not a claim of a mature general-purpose LLM.

## 2. System Architecture

The repository contains a native 24K byte-level BPE tokenizer, pre-norm decoder blocks, explicit trainers, atomic full-state checkpoints, response-only SFT masking, LoRA/QLoRA workflows, exact-match evaluation, generation diagnostics, evidence manifests, and integrity tests. Native and Qwen experimental lines remain separate.

## 3. Native 37M Transformer

The frozen model has 37,462,528 parameters, 8 layers, hidden size 512, GQA 8/4 with head dimension 64, FFN 1536, RoPE, RMSNorm, SwiGLU, tied embeddings, and context 1024. Architecture code, frozen metadata, and checkpoint shapes agree.

## 4. Formal Pretraining

E01 completed 3,052 BF16 optimizer steps and 50,003,968 processed tokens. Fixed validation loss improved from {pretrain['initial_validation_loss']:.4f} to {pretrain['final_validation_loss']:.4f}; PPL improved from {pretrain['initial_validation_ppl']:.2f} to {pretrain['final_validation_ppl']:.2f}. Median throughput was {pretrain['median_tokens_per_second']:.0f} tokens/s and peak allocated VRAM was {pretrain['peak_allocated_vram_mib']:.1f} MiB. This validates trainability and the checkpoint/resume chain, not strong general language or reasoning capability.

## 5. Native Full SFT

GPU-4A used 4,800/700/700 train/validation/test rows across support classification and 0-99 addition. Splits have zero semantic-key overlap; all 6,200 targets are complete, EOS-supervised, and untruncated. Full-parameter BF16 response-only SFT used AdamW, learning rate 1e-4, micro batch 16, gradient accumulation 2, effective batch 32, and 300 steps. Validation loss improved 5.9922 to 0.5943 in 48.87 seconds. Median throughput was 12,188 input tokens/s and 850 target tokens/s; peak CUDA allocation/reservation was 1,003.63/1,318 MiB. On fixed independent 64-example test subsets, T1 improved 0/64 to 64/64 while T2 improved only 0/64 to 2/64. All 384 post-SFT generations were format-valid, complete, EOS-terminated, non-repetitive, and avoided length-limit termination.

## 6. PEFT

The Qwen3-0.6B campaign completed 12 LoRA/QLoRA runs: Base versus Math-CPT initialization, three seeds, and 1,200 steps. Each run evaluated 500 MATH-500 and a fixed 200-example GSM8K subset, totaling 8,400 outputs. LoRA CPT transfer was negative on both benchmarks; QLoRA was negative on MATH-500 and mixed on GSM8K. This does not support a general CPT-to-reasoning improvement claim. Qwen Full SFT effectiveness was not formally validated.

## 7. Evaluation and Generation Diagnostics

GPU-3A found 8,398/8,400 outputs reached 512 tokens and documented 36 legacy extraction failures. Frozen-cache audit showed that at context 512, 12,469/19,162 training samples retained supervised assistant-end tokens (65.07%) while 6,693 lost them after truncation. Real PEFT SFT training and effectiveness evaluation had executed, but these facts did not prove EOS-supervision causality. GPU-3B recovered all 36 extraction failures, 16 correct, and identified 607 conflicting outputs; first-valid increased the counted correct total from 1,764 to 2,008. External stopping reduced redundant generation in a bounded confirmation. These are evaluator/inference improvements. GPU-3C then held initialization, data, order, optimizer, and 64 steps fixed while changing final EOS labels; the EOS-supervised arm emitted no autonomous `<|im_end|>` and hit 512 tokens on all four C0 prompts, a controlled negative result.

## 8. Numerical and Training-System Study

GPU-4B retained Manual + Eager because SDPA Math passed FP32 but Math and Auto failed frozen BF16 gates; built-in Flash was unavailable and compile was blocked by missing Triton. No qualified performance benchmark exists. GPU-4B-R1 localized the earliest public BF16 difference to the first Attention×V context boundary after bit-identical Q/K/V and RoPE. Controlled FP32-intermediate reconstruction closely matched SDPA Math, supporting a precision-path mechanism; no semantic defect was found. Exact SDPA internal instruction ordering remains unknown.

## 9. Validated Capabilities

Validated capabilities are enumerated in `CAPABILITY_MATRIX.md`. They include native architecture/tokenizer construction, BF16 pretraining, exact resume, native full SFT, formal LoRA/QLoRA campaigns, evaluation diagnostics, extraction, and bounded external stopping.

## 10. Negative Results

- Native T2 arithmetic remained weak at 2/64 on the test subset.
- CPT-to-PEFT reasoning transfer was negative for LoRA and mixed for QLoRA.
- EOS supervision did not validate autonomous-stop improvement.
- Frozen SDPA BF16 correctness, gradient, and update gates failed.
- Built-in Flash was unavailable; compile was blocked; neither produced performance numbers.

## 11. Limitations

The native model is small and undertrained by modern LLM standards. T1/T2 are synthetic and generation scores use fixed 64-example subsets. PEFT findings are bound to three seeds and context 512. GPU-3C used one seed and four confirmation prompts. Hardware evidence comes from one RTX 5060 Laptop GPU Windows environment. Public benchmark contamination from base-model history cannot be excluded.

## 12. Reproducibility

The pretrained checkpoint SHA256 is `{manifest['pretrained_checkpoint_sha256']}`; native SFT is `{manifest['native_sft_checkpoint_sha256']}`; tokenizer is `{manifest['tokenizer_sha256']}`. Use `portfolio-verify quick` for artifact and metadata verification. Registries link every public claim to a structured artifact.

## 13. Conclusion

The portfolio's strongest result is the auditable end-to-end engineering and scientific workflow: positive evidence, negative evidence, blocked paths, and causal limits are retained together. The appropriate positioning is a reproducible LLM training/evaluation systems project, not a claim of state-of-the-art model quality.
"""
    (out / "MINILLM_FORGE_FINAL_TECHNICAL_REPORT.md").write_text(technical, encoding="utf-8")
    resume = """# Resume Facts

Only the following bounded fact atoms are approved for resume use.

## Native model

- Implemented a 37.46M-parameter, 8-layer decoder Transformer with GQA 8/4, RoPE, RMSNorm, SwiGLU, tied embeddings, context 1024, and a native 24K byte-level BPE tokenizer.

## Pretraining

- Completed 50.0M processed-token BF16 pretraining; fixed validation loss improved 10.1964→4.6484 and PPL 26,805.55→104.42; verified exact model/optimizer/scheduler/RNG resume.

## SFT

- Completed 300-step full-parameter response-only BF16 SFT; validation loss improved 5.9922→0.5943 in 48.87 s.
- On fixed independent 64-example test subsets: support classification improved 0/64→64/64; two-digit addition improved only 0/64→2/64.
- Verified 384/384 post-SFT generations were complete, format-valid, EOS-terminated, non-repetitive, and not length-limited.

## PEFT and evaluation

- Completed and audited 12 Qwen3-0.6B LoRA/QLoRA runs across Base/CPT initialization and three seeds, with 8,400 MATH-500/GSM8K outputs; retained negative/mixed CPT transfer findings.
- Diagnosed 8,398/8,400 length-limit outputs; improved extraction from 1,764 to 2,008 counted-correct outputs while explicitly identifying 607 conflicts—an evaluator improvement, not model improvement.

## Systems and numerical study

- Built reproducible manifests, exact checkpoint recovery, experiment/claim registries, and artifact hash verification.
- Localized a BF16 Manual-vs-SDPA numerical split to the first Attention×V context boundary; preserved failed BF16 gates and did not claim an unmeasured speedup.

Do not claim FlashAttention acceleration, torch.compile acceleration, full-test T1 accuracy, Qwen Full SFT effectiveness, distributed training, or strong general reasoning.
"""
    (out / "RESUME_FACTS.md").write_text(resume, encoding="utf-8")
    interview = """# Interview Facts

## Native model and scale

- **Fact:** 37.46M parameters and 50.0M processed pretraining tokens were chosen to close the full implementation/training/recovery loop on an 8GB laptop GPU.
- **Interpretation:** the scale is large enough to exercise realistic training systems but too small and undertrained for mature general-language claims.

## T1 versus T2

- **Fact:** T1 reached 64/64 on a fixed independent subset; T2 reached 2/64. All 384 generations had valid termination and format.
- **Interpretation:** SFT successfully taught a low-entropy classification mapping and output protocol but did not create robust arithmetic computation.
- **Hypothesis:** tokenization and memorization-friendly task structure help explain the difference; no dedicated causal ablation was run.

## PEFT and generation

- **Fact:** 12 LoRA/QLoRA runs and 8,400 outputs were completed. CPT transfer was negative for LoRA and mixed for QLoRA. 8,398 outputs hit 512 tokens.
- **Fact:** extraction changes recovered 36 failures and exposed 607 conflicts; external stopping reduced redundancy.
- **Interpretation:** evaluation and inference-system quality can improve without changing model weights or answer capability.

## EOS controlled negative

- **Fact:** GPU-3C changed only final EOS-label supervision over 64 steps and 1,024 ordered examples per arm; the supervised arm emitted no autonomous `<|im_end|>` on four C0 prompts.
- **Interpretation:** a plausible intervention failed under the bounded protocol; retaining this prevents post-hoc success narratives.
- **Hypothesis:** capacity, optimization horizon, decoding dynamics, and data distribution may matter, but none was causally isolated.

## SDPA and thresholds

- **Fact:** FP32 matched, BF16 frozen gates failed, and the first public difference was Attention×V context. FP32-intermediate reconstruction closely matched SDPA Math.
- **Interpretation:** different intermediate precision/rounding paths explain the dominant difference without a semantic attention bug.
- **Unknown:** exact SDPA internal instruction order.
- Thresholds were not relaxed because doing so after observing failure would invalidate the preregistered gate.

## Why no Triton/Flash continuation

- **Fact:** built-in Flash was unavailable and compile was blocked by missing Triton; no performance benchmark qualified.
- **Decision:** stop because environment repair and performance work would be a new research stage, while correctness evidence already required Manual + Eager as formal default.

## Engineering closure

The closed loop is: native model/tokenizer → formal pretraining → exact resume → native Full SFT → Qwen PEFT → unified evaluation → generation diagnostics → controlled negative ablation → BF16 numerical study → evidence registries and reproducible verification.
"""
    (out / "INTERVIEW_FACTS.md").write_text(interview, encoding="utf-8")
    readiness = """# Portfolio Release Readiness

Candidate version: `v2.0.0-portfolio` (not created by GPU-4C).

| Gate | Status | Evidence |
|---|---|---|
| Key checkpoints and hashes | PASS | final evidence manifest + quick verification |
| Key artifacts and schemas | PASS | claim/experiment registries |
| README and final report truthfulness | PASS_WITH_LIMITATIONS | documentation truth audit |
| Historical experiment artifacts | PASS | frozen SHA256 set |
| Quick verification | PENDING until executed | portfolio verification output |
| Full regression | PENDING until executed | validation output |
| No stale experimental path presented as production | PASS | Manual + Eager remains formal |

Release posture: `READY_WITH_DOCUMENTED_LIMITATIONS`. Subset evaluation, unavailable Flash, missing Triton compile path, unvalidated Qwen Full SFT effectiveness, and absent distributed training are explicitly visible. A portfolio tag may be recommended only after quick verification and full tests pass and the final commit leaves a clean worktree.
"""
    (out / "PORTFOLIO_RELEASE_READINESS.md").write_text(readiness, encoding="utf-8")
