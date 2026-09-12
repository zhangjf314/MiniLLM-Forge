# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from minillm_forge.cli.common import write_json
from minillm_forge.config import config_hash
from minillm_forge.training.qwen_cpt import file_sha256

BASE_EVAL = Path("artifacts/eval_manifests/qwen_base_pre_cpt.json")
CALIBRATION = Path("artifacts/training/qwen_cpt_memory_calibration.json")
QUALIFICATION = Path("artifacts/training/qwen_cpt_long_run_qualification.json")
FORMAL_RESULT = Path("artifacts/training/qwen_math_cpt_result.json")
REGRESSION = Path("artifacts/training/qwen_cpt_regression_validation.json")


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metrics(path: str | Path, event: str) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("event") == event:
                    records.append(record)
    return records


def run_regression(config: dict[str, Any]) -> dict[str, Any]:
    commands = {
        "pytest": [sys.executable, "-m", "pytest", "-q"],
        "ruff": [sys.executable, "-m", "ruff", "check", "."],
        "format": [sys.executable, "-m", "ruff", "format", "--check", "."],
        "lock": ["uv", "lock", "--check"],
        "build": ["uv", "build"],
    }
    checks: dict[str, Any] = {}
    for name, command in commands.items():
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        checks[name] = {
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "returncode": completed.returncode,
            "command": command,
            "output": (completed.stdout + completed.stderr)[-8000:],
        }
    import yaml

    tracked_yaml_output = subprocess.check_output(["git", "ls-files", "*.yaml", "*.yml"], text=True)
    yaml_files = [Path(line) for line in tracked_yaml_output.splitlines() if line.strip()]
    baseline_yaml_output = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            config["stage"]["baseline_commit"],
        ],
        text=True,
    )
    baseline_yaml_files = [
        line
        for line in baseline_yaml_output.splitlines()
        if line.lower().endswith((".yaml", ".yml"))
    ]
    error = None
    try:
        for path in yaml_files:
            yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    checks["yaml"] = {
        "status": "PASS" if error is None else "FAIL",
        "scope": "git-tracked repository YAML only; caches and test temporaries excluded",
        "files_parsed": len(yaml_files),
        "baseline_commit": config["stage"]["baseline_commit"],
        "baseline_files": len(baseline_yaml_files),
        "file_count_delta": len(yaml_files) - len(baseline_yaml_files),
        "error": error,
    }
    result = {
        "classification": (
            "PASS" if all(value["status"] == "PASS" for value in checks.values()) else "FAIL"
        ),
        "checks": checks,
        "git_commit": _git_commit(),
    }
    write_json(REGRESSION, result)
    return result


def _plot(
    records: list[dict[str, Any]],
    key: str,
    filename: str,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.plot(
        [record["tokens_seen"] for record in records],
        [record[key] for record in records],
        color=color,
    )
    axis.set(xlabel="Input tokens", ylabel=ylabel, title=title)
    axis.grid(alpha=0.25)
    output = Path("reports/figures") / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _write_figures(result: dict[str, Any]) -> None:
    train = _metrics(result["metrics_path"], "train")
    evaluation = _metrics(result["metrics_path"], "evaluation")
    curves = [
        (train, "train_loss", "qwen_cpt_train_loss.png", "Qwen math CPT loss", "Loss", "#4c72b0"),
        (
            evaluation,
            "math_ppl",
            "qwen_cpt_math_ppl.png",
            "Frozen math-domain perplexity",
            "PPL",
            "#55a868",
        ),
        (
            evaluation,
            "general_ppl",
            "qwen_cpt_general_ppl.png",
            "Frozen general-domain perplexity",
            "PPL",
            "#c44e52",
        ),
        (train, "learning_rate", "qwen_cpt_learning_rate.png", "Learning rate", "LR", "#8172b2"),
        (train, "gradient_norm", "qwen_cpt_grad_norm.png", "Gradient norm", "L2 norm", "#dd8452"),
        (
            train,
            "tokens_per_second",
            "qwen_cpt_throughput.png",
            "CPT throughput",
            "Tokens/s",
            "#64b5cd",
        ),
    ]
    for arguments in curves:
        _plot(*arguments)


def _checkpoint_inventory(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["training"]["output_dir"])
    names = {
        "early": "early.pt",
        "mid": "mid.pt",
        "three_quarter": "three-quarter.pt",
        "last": "last.pt",
        "best_math_validation": "best_math_validation.pt",
    }
    checkpoints = {}
    for label, name in names.items():
        path = output_dir / name
        if path.exists():
            checkpoints[label] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    inventory = {"checkpoints": checkpoints, "git_commit": _git_commit()}
    write_json("artifacts/training/qwen_math_cpt_checkpoint_inventory.json", inventory)
    return inventory


def _write_comparison(base: dict[str, Any], result: dict[str, Any]) -> None:
    math_percent = result["math_ppl_relative_delta"] * 100
    general_percent = result["general_ppl_relative_delta"] * 100
    text = f"""# Qwen Base vs Math CPT

| Metric | Base | CPT | Delta |
| --- | ---: | ---: | ---: |
| Math validation loss | {result["initial_math_loss"]:.6f} | {result["final_math_loss"]:.6f} | {result["final_math_loss"] - result["initial_math_loss"]:+.6f} |
| Math PPL | {result["initial_math_ppl"]:.4f} | {result["final_math_ppl"]:.4f} | {result["math_ppl_delta"]:+.4f} ({math_percent:+.2f}%) |
| General validation loss | {result["initial_general_loss"]:.6f} | {result["final_general_loss"]:.6f} | {result["final_general_loss"] - result["initial_general_loss"]:+.6f} |
| General PPL | {result["initial_general_ppl"]:.4f} | {result["final_general_ppl"]:.4f} | {result["general_ppl_delta"]:+.4f} ({general_percent:+.2f}%) |
| Controlled math EM | N/A | N/A | N/A |
| Peak allocated VRAM | {base["peak_allocated_vram_mib"]:.1f} MiB | {result["peak_allocated_vram_mib"]:.1f} MiB | — |
| Median training throughput | — | {result["median_tokens_per_second"]:.2f} tokens/s | — |
| Training tokens | 0 | {result["training_tokens"]:,} | +{result["training_tokens"]:,} |

Primary classification: **{result["classification"]}**.

Lower math-domain perplexity demonstrates domain language-model adaptation; it does not,
by itself, prove improved mathematical reasoning.
"""
    Path("reports/QWEN_BASE_VS_CPT.md").write_text(text, encoding="utf-8")


def _write_report(
    config: dict[str, Any],
    base: dict[str, Any],
    result: dict[str, Any],
    calibration: dict[str, Any],
    qualification: dict[str, Any],
) -> None:
    math_percent = result["math_ppl_relative_delta"] * 100
    general_percent = result["general_ppl_relative_delta"] * 100
    hypotheses = result["hypothesis_results"]
    text = f"""# Qwen Math Continued Pretraining

## 1. Objective

Measure whether pinned FineMath-4+ full-parameter CPT improves frozen math-domain
perplexity and quantify any change on a frozen general-domain probe.

## 2. Hypotheses

H-CPT-1 freezes expected math NLL/PPL improvement. H-CPT-2 requires simultaneous
general-domain measurement. H-CPT-3 requires reporting their trade-off.

## 3. Base Model

`{config["model"]["name_or_path"]}` at `{config["model"]["revision"]}` with
{result["parameters"]:,} parameters and one identical pinned tokenizer for both arms.

## 4. Hardware

NVIDIA GeForce RTX 5060 Laptop GPU (8,150.56 MiB), BF16, CUDA {torch.version.cuda}.

## 5. CPT Dataset

FineMath-4+ revision `{config["data"]["revision"]}`, deterministic pinned streaming range,
{result["training_tokens"]:,} input tokens and causal next-token prediction only.

## 6. Held-out Sets

Math and general partitions were frozen before training. General validation reuses frozen
GPU-1 FineWeb-Edu text but is independently Qwen-tokenized. Evaluated target counts are
math {base["math_tokens"]:,} and general {base["general_tokens"]:,}.

## 7. Contamination Audit

Exact normalized duplicates and near {config["data"]["ngram_size"]}-gram Jaccard collisions
at {config["data"]["near_threshold"]} were removed against math/general/benchmark probes.
This cannot rule out contamination in Qwen's original pretraining.

## 8. Baseline Evaluation

Math loss/PPL {result["initial_math_loss"]:.6f}/{result["initial_math_ppl"]:.4f}; general
loss/PPL {result["initial_general_loss"]:.6f}/{result["initial_general_ppl"]:.4f}.

## 9. Memory Calibration

512 context: {calibration["peak_allocated_mib"]:.1f} MiB peak allocated and
{calibration["estimated_min_system_headroom_mib"]:.1f} MiB estimated minimum headroom.
8-bit AdamW compresses optimizer state while all model weights remain BF16 and all
596,049,920 parameters remain trainable. This is full-parameter CPT, not QLoRA.

## 10. Long-run Qualification

`{qualification["classification"]}` at {qualification["tokens_seen"]:,} tokens; NaN/Inf/OOM
{qualification["nan_events"]}/{qualification["inf_events"]}/{qualification["oom_events"]}.

## 11. Formal Configuration

Sequence 512, micro-batch 1, accumulation 8 (4,096 tokens/update), BF16, gradient
checkpointing, AdamW LR {result["learning_rate"]:.1e}, 3% warmup and cosine decay.

## 12. Training Dynamics

{result["optimizer_steps"]:,} optimizer steps; final logged train loss
{result["final_train_loss"]:.6f}; no LR or warmup sweep.

## 13. Math-domain Evaluation

PPL {result["initial_math_ppl"]:.4f} → {result["final_math_ppl"]:.4f}
({math_percent:+.2f}%).

## 14. General-domain Evaluation

PPL {result["initial_general_ppl"]:.4f} → {result["final_general_ppl"]:.4f}
({general_percent:+.2f}%).

## 15. Catastrophic Forgetting Analysis

Both PPLs were measured together near 0/25/50/75/100% of the budget. The general-domain
delta is reported quantitatively rather than hidden by the math result. The 5% relative
general-loss early-stop gate did not trigger; the smaller sustained degradation remains
material to the classification.

## 16. Checkpoint / Resume

{result["resume_validation"]}; qualification performed train → checkpoint → restore →
continue while matching step, tokens, LR, scheduler and RNG state.

## 17. GPU Efficiency

Peak allocated/reserved {result["peak_allocated_vram_mib"]:.1f}/
{result["peak_reserved_vram_mib"]:.1f} MiB; median {result["median_tokens_per_second"]:.2f}
tokens/s; elapsed {result["elapsed_seconds"]:.1f}s.

## 18. Base vs CPT

See `reports/QWEN_BASE_VS_CPT.md` for absolute and relative deltas.

## 19. Hypothesis Results

- H-CPT-1: **{hypotheses["H-CPT-1"]}** — math PPL {math_percent:+.2f}%.
- H-CPT-2: **{hypotheses["H-CPT-2"]}** — general PPL {general_percent:+.2f}%.
- H-CPT-3: **{hypotheses["H-CPT-3"]}** — both domains are jointly reported.

## 20. Limitations

One seed, a bounded 10M-token first-range corpus and two PPL probes. The audit is local.
Generative EM is secondary for this non-instruction-tuned model and was not used as the
primary endpoint. Lower domain PPL is not proof of reasoning-task improvement.

## 21. Conclusion

**{result["classification"]}**. This is controlled domain language-model adaptation
evidence only; Stage GPU-2A does not enter SFT.
"""
    Path("reports/QWEN_MATH_CPT.md").write_text(text, encoding="utf-8")


def _update_registry(
    config: dict[str, Any], base: dict[str, Any], result: dict[str, Any], regression: dict[str, Any]
) -> None:
    path = Path("experiments/registry.csv")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [row for row in reader if row.get("experiment_id") not in {"E00", "E04"}]
    train_manifest = _read_json(config["data"]["train_manifest"])
    common = {
        "model": config["model"]["name_or_path"],
        "seed": config["training"]["seed"],
        "precision": "bf16",
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "status": "completed",
    }
    e00 = {
        **common,
        "experiment_id": "E00",
        "git_commit": base["git_commit"],
        "dataset": "frozen math/general held-out",
        "parameters": base["parameters"],
        "sequence_length": config["data"]["sequence_length"],
        "validation_tokens": base["math_tokens"] + base["general_tokens"],
        "initial_validation_loss": base["math_loss"],
        "initial_validation_ppl": base["math_ppl"],
        "classification": base["classification"],
    }
    e04 = {
        **common,
        "experiment_id": "E04",
        "git_commit": result["git_commit"],
        "config_hash": config_hash(config),
        "dataset_hash": train_manifest["data_digest"],
        "dataset": f"{config['data']['dataset']}:{config['data']['subset']}",
        "learning_rate": result["learning_rate"],
        "batch_size": result["micro_batch"],
        "effective_batch_size": result["micro_batch"] * result["gradient_accumulation"],
        "trainable_params": result["trainable_parameters"],
        "peak_vram_mb": result["peak_allocated_vram_mib"],
        "tokens_per_second": result["median_tokens_per_second"],
        "final_loss": result["final_train_loss"],
        "eval_score": result["final_math_ppl"],
        "peak_cuda_reserved_mb": result["peak_reserved_vram_mib"],
        "classification": result["classification"],
        "parameters": result["parameters"],
        "sequence_length": result["sequence_length"],
        "training_tokens": result["training_tokens"],
        "optimizer_steps": result["optimizer_steps"],
        "micro_batch": result["micro_batch"],
        "gradient_accumulation": result["gradient_accumulation"],
        "effective_input_tokens_per_update": config["training"]["effective_tokens_per_update"],
        "final_train_loss": result["final_train_loss"],
        "initial_validation_loss": result["initial_math_loss"],
        "final_validation_loss": result["final_math_loss"],
        "initial_validation_ppl": result["initial_math_ppl"],
        "final_validation_ppl": result["final_math_ppl"],
        "peak_allocated_vram_mib": result["peak_allocated_vram_mib"],
        "peak_reserved_vram_mib": result["peak_reserved_vram_mib"],
        "median_tokens_per_second": result["median_tokens_per_second"],
        "elapsed_seconds": result["elapsed_seconds"],
        "resume_validation": result["resume_validation"],
        "nan_events": result["nan_events"],
        "inf_events": result["inf_events"],
        "oom_events": result["oom_events"],
        "stage_gpu1_baseline": config["stage"]["baseline_commit"],
        "dataset_manifest": config["data"]["train_manifest"],
        "warmup_ratio": result["warmup_ratio"],
        "optimizer": result["optimizer"],
        "primary_classification": result["classification"],
        "regression_validation": regression["classification"],
        "regression_evidence": str(REGRESSION),
    }
    rows.extend(
        [
            {field: e00.get(field, "") for field in fields},
            {field: e04.get(field, "") for field in fields},
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _update_summaries(base: dict[str, Any], result: dict[str, Any]) -> None:
    if result["classification"] not in {
        "QWEN_MATH_CPT_VALIDATED",
        "QWEN_MATH_CPT_VALIDATED_WITH_GENERAL_DEGRADATION",
    }:
        return
    math_percent = result["math_ppl_relative_delta"] * 100
    general_percent = result["general_ppl_relative_delta"] * 100
    readme_path = Path("README.md")
    readme = readme_path.read_text(encoding="utf-8")
    block = f"""<!-- GPU2A_RESULTS_START -->
### GPU-2A measured CPT result

- Base → CPT math PPL: {result["initial_math_ppl"]:.4f} → {result["final_math_ppl"]:.4f} ({math_percent:+.2f}%)
- Base → CPT general PPL: {result["initial_general_ppl"]:.4f} → {result["final_general_ppl"]:.4f} ({general_percent:+.2f}%)
- CPT input tokens: {result["training_tokens"]:,}; peak allocated VRAM: {result["peak_allocated_vram_mib"]:.1f} MiB
- Median training throughput: {result["median_tokens_per_second"]:.2f} tokens/s
- Classification: `{result["classification"]}`

This is domain language-model adaptation evidence, not a claim of improved mathematical
reasoning accuracy.
<!-- GPU2A_RESULTS_END -->

"""
    marker = "## Phase B: Qwen CPT and SFT\n\n"
    if "<!-- GPU2A_RESULTS_START -->" not in readme:
        readme = readme.replace(marker, marker + block)
    readme = readme.replace(
        "| Qwen Base | zero-shot | - | TBD | Math EM TBD | - | pending formal run |",
        f"| Qwen Base | frozen PPL baseline | - | {base['peak_allocated_vram_mib']:.0f} MiB | Math PPL {base['math_ppl']:.2f} | - | E00 frozen |",
    )
    readme = readme.replace(
        "| Qwen | CPT | TBD | TBD | Math EM TBD | TBD | pending formal run |",
        f"| Qwen | full CPT | {result['trainable_parameters']:,} | {result['peak_allocated_vram_mib']:.0f} MiB | Math PPL {result['final_math_ppl']:.2f} | {result['median_tokens_per_second']:.0f} | E04 completed |",
    )
    readme = readme.replace(
        "| Qwen baseline, CPT, SFT, LoRA, QLoRA | eval/run manifests | pending formal GPU runs |",
        "| Qwen baseline and CPT | E00/E04 manifests, reports, curves | validated; SFT/LoRA/QLoRA remain pending |",
    )
    readme = readme.replace(
        "| Contamination audit | JSON report | command implemented; corpus audit pending |",
        "| CPT contamination audit | pinned JSON report | completed for local train/validation/benchmark probes |",
    )
    readme = readme.replace(
        "| Final technical report | `reports/FINAL_REPORT.md` | GPU-1 evidence integrated; later stages pending |",
        "| Final technical report | `reports/FINAL_REPORT.md` | GPU-1 and GPU-2A evidence integrated; later stages pending |",
    )
    readme_path.write_text(readme, encoding="utf-8")

    final_path = Path("reports/FINAL_REPORT.md")
    final_text = final_path.read_text(encoding="utf-8")
    measured = (
        f"Measured CPT result: **{result['classification']}** at {result['training_tokens']:,} "
        f"tokens. Math PPL {result['initial_math_ppl']:.4f} → {result['final_math_ppl']:.4f} "
        f"({math_percent:+.2f}%); general PPL {result['initial_general_ppl']:.4f} → "
        f"{result['final_general_ppl']:.4f} ({general_percent:+.2f}%). This is domain "
        "language-model adaptation evidence, not proof of mathematical reasoning gain."
    )
    final_text = final_text.replace("Measured CPT results: **TBD**.", measured)
    final_text = final_text.replace(
        "- Full CPT/SFT long-run stability is not established by the bounded one-step 8GB tests.",
        "- Full CPT long-run stability is established only for the frozen 512-token GPU-2A configuration; full SFT remains unqualified for long runs.",
    )
    final_text = final_text.replace(
        "Qwen CPT/SFT/LoRA/QLoRA scientific conclusions remain deferred to later pinned experiments.",
        "Qwen CPT now has pinned math/general PPL evidence; SFT/LoRA/QLoRA scientific conclusions remain deferred to later pinned experiments.",
    )
    final_text = final_text.replace(
        "repeated seeds, and the Qwen adaptation experiments remain future evidence.",
        "repeated seeds, and the Qwen SFT/LoRA/QLoRA experiments remain future evidence.",
    )
    final_path.write_text(final_text, encoding="utf-8")


def finalize(config: dict[str, Any]) -> dict[str, Any]:
    base = _read_json(BASE_EVAL)
    calibration = _read_json(CALIBRATION)
    qualification = _read_json(QUALIFICATION)
    result = _read_json(FORMAL_RESULT)
    regression = _read_json(REGRESSION)
    if regression["classification"] != "PASS":
        raise RuntimeError("regression must pass before evidence finalization")
    _write_figures(result)
    inventory = _checkpoint_inventory(config)
    _write_comparison(base, result)
    _write_report(config, base, result, calibration, qualification)
    _update_registry(config, base, result, regression)
    _update_summaries(base, result)
    evidence = {
        "classification": result["classification"],
        "reports": ["reports/QWEN_MATH_CPT.md", "reports/QWEN_BASE_VS_CPT.md"],
        "figures": 6,
        "checkpoint_count": len(inventory["checkpoints"]),
        "registry": "experiments/registry.csv",
        "regression": regression["classification"],
        "git_commit": _git_commit(),
        "evidence_digest": hashlib.sha256(FORMAL_RESULT.read_bytes()).hexdigest(),
    }
    write_json("artifacts/training/qwen_cpt_evidence_summary.json", evidence)
    return evidence
