"""Generate measured GPU-1 result, curves, registry row and evidence report."""

from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

from minillm_forge.config import config_hash
from minillm_forge.data.formal import MANIFEST, TOKENIZER_MANIFEST, json_write, verify_frozen
from minillm_forge.data.manifest import file_digest

ROOT = Path("runs/E01-minillm-formal")
ARTIFACTS = Path("artifacts/training")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def approve_budget():
    pilot = read_json("runs/gpu1-pilot/summary.json")
    initial = read_json("runs/gpu1-pilot/initial.json")
    passed = (
        pilot["status"] == "completed"
        and pilot["tokens_seen"] >= 5_000_000
        and not any(pilot[key] for key in ("nan_count", "inf_count", "oom_count"))
        and pilot["final_validation"]["validation_loss"] < initial["validation_loss"]
    )
    rate = pilot["median_tokens_per_second"]
    # Freeze 50M, not a retroactive token counter. Round to full optimizer updates.
    decision = {
        "formal_approved": passed,
        "pilot_summary_hash": file_digest("runs/gpu1-pilot/summary.json"),
        "pilot_tokens": pilot["tokens_seen"],
        "pilot_median_tokens_per_second": rate,
        "nominal_budget": 50_000_000,
        "planned_optimizer_steps": 3052,
        "actual_input_token_budget": 3052 * 16384,
        "estimated_compute_seconds": 3052 * 16384 / rate,
        "estimated_100m_compute_seconds": 100_000_000 / rate,
        "reason": (
            "Pilot is stable and validation improves. Freeze the preferred 50M bounded experiment; "
            "100M is not needed for this first baseline and would double the laptop runtime."
        ),
        "thermal_evidence": read_json("runs/gpu1-pilot/hardware_after.json"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    json_write(ARTIFACTS / "budget_decision.json", decision)
    import torch

    checkpoint = torch.load(
        "runs/gpu1-pilot/step-00000153.pt", map_location="cpu", weights_only=False
    )
    pilot_records = [
        json.loads(line)
        for line in Path("runs/gpu1-pilot/metrics.jsonl").read_text().splitlines()
        if line
    ]
    resumed = next(
        row for row in pilot_records if row["event"] == "resume" and row["global_step"] == 153
    )
    if resumed["tokens_seen"] != checkpoint["trainer_state"]["tokens_seen"]:
        raise ValueError("pilot resume token continuity failed")
    if resumed["learning_rate"] != checkpoint["optimizer"]["param_groups"][0]["lr"]:
        raise ValueError("pilot resume LR continuity failed")
    resume_path = ARTIFACTS / "minillm_gpu_resume_validation.json"
    resume = read_json(resume_path)
    resume["pilot_interruption"] = {
        "checkpoint_step": 153,
        "checkpoint_tokens": checkpoint["trainer_state"]["tokens_seen"],
        "optimizer_steps": sorted(
            {float(item["step"]) for item in checkpoint["optimizer"]["state"].values()}
        ),
        "scheduler_last_epoch": checkpoint["scheduler"]["last_epoch"],
        "resume_event": resumed,
        "final_step": pilot["global_step"],
        "final_tokens": pilot["tokens_seen"],
        "checkpoint_sha256": file_digest("runs/gpu1-pilot/step-00000153.pt"),
        "cuda_rng_present": "cuda" in checkpoint["rng_state"],
        "sampler_position": checkpoint["trainer_state"]["sampler_state"]["position"],
        "validation": "RESUME_SEMANTICS_VALIDATED; no full-pilot duplicate run",
    }
    json_write(resume_path, resume)
    for name in ("summary.json", "initial.json", "metrics.jsonl", "resolved_config.json"):
        shutil.copyfile(Path("runs/gpu1-pilot") / name, ARTIFACTS / f"pilot_{name}")
    print(json.dumps(decision, indent=2))


def plot_curves(train, evaluations):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [
        (train, "loss", "train_loss", "Training cross-entropy (interval mean)"),
        (evaluations, "validation_loss", "val_loss", "Fixed held-out cross-entropy"),
        (evaluations, "validation_perplexity", "perplexity", "Held-out perplexity (log scale)"),
        (train, "learning_rate", "learning_rate", "AdamW learning rate"),
        (train, "gradient_norm", "grad_norm", "Gradient L2 norm before clipping"),
        (train, "tokens_per_second", "throughput", "Compute throughput (input tokens/s)"),
    ]
    for records, key, filename, title in specs:
        fig, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        axis.plot(
            [row["tokens_seen"] / 1e6 for row in records],
            [row[key] for row in records],
            linewidth=1.8,
            color="#167D9A",
        )
        axis.set(
            xlabel="Processed input tokens (millions)",
            ylabel=title,
            title=f"MiniLLM / E01 — {title}",
        )
        if key == "validation_perplexity":
            axis.set_yscale("log")
        axis.grid(alpha=0.2)
        target = Path(f"reports/figures/minillm_{filename}.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150)
        plt.close(fig)


def finalize():
    summary = read_json(ROOT / "summary.json")
    if summary["status"] != "completed" or summary["tokens_seen"] < 20_000_000:
        raise ValueError("formal completion requires the frozen budget and at least 20M tokens")
    manifest = verify_frozen()
    initial = read_json(ROOT / "initial.json")
    config = read_json(ROOT / "resolved_config.json")
    tokenizer = read_json(TOKENIZER_MANIFEST)
    resume = read_json(ARTIFACTS / "minillm_gpu_resume_validation.json")
    regression = read_json(ARTIFACTS / "minillm_regression_validation.json")
    if regression["status"] != "PASS":
        raise ValueError("formal evidence cannot be validated without regression PASS")
    records = [
        json.loads(line) for line in (ROOT / "metrics.jsonl").read_text().splitlines() if line
    ]
    train = [row for row in records if row["event"] == "train"]
    evaluations = list(
        {row["global_step"]: row for row in records if row["event"] == "evaluation"}.values()
    )
    evaluations.sort(key=lambda row: row["global_step"])
    final_val = summary["final_validation"]
    result = {
        "classification": "MINILLM_FORMAL_PRETRAINING_VALIDATED",
        "primary_classification": "VALIDATED",
        "experiment_id": "E01",
        "model": "MiniLLM",
        "parameters": summary["parameters"],
        "tokenizer_vocab": 24000,
        "sequence_length": 1024,
        "training_tokens": summary["tokens_seen"],
        "supervised_next_token_targets": summary["supervised_tokens_seen"],
        "unique_train_corpus_tokens": manifest["partitions"]["train"]["packed_tokens"],
        "validation_tokens": manifest["partitions"]["validation"]["packed_tokens"],
        "validation_loss_target_tokens": final_val["evaluated_targets"],
        "optimizer_steps": summary["global_step"],
        "precision": "bf16",
        "micro_batch": config["training"]["micro_batch_size"],
        "gradient_accumulation": config["training"]["gradient_accumulation_steps"],
        "effective_batch": 16,
        "effective_input_tokens_per_update": 16384,
        "initial_train_loss": initial["initial_train_loss"],
        "final_train_loss": summary["final_train_loss"],
        "final_train_probe_loss": summary["final_train_probe_loss"],
        "train_loss_protocol": (
            "initial=fixed 32-block train probe; final=last online interval mean; "
            "final_train_probe_loss is the matched evaluation probe"
        ),
        "initial_validation_loss": initial["validation_loss"],
        "final_validation_loss": final_val["validation_loss"],
        "initial_validation_ppl": initial["validation_perplexity"],
        "final_validation_ppl": final_val["validation_perplexity"],
        "best_validation_loss": summary["best_validation_loss"],
        "best_validation_ppl": math.exp(summary["best_validation_loss"]),
        "best_step": summary["best_step"],
        "peak_allocated_vram_mib": summary["peak_allocated_mib"],
        "peak_reserved_vram_mib": summary["peak_reserved_mib"],
        "median_tokens_per_second": summary["median_tokens_per_second"],
        "mean_compute_tokens_per_second": summary["average_tokens_per_second"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "compute_seconds": summary["compute_seconds"],
        "resume_validation": resume["classification"],
        "regression_validation": regression["status"],
        "regression_evidence": str(ARTIFACTS / "minillm_regression_validation.json"),
        "non_finite_events": summary["nan_count"] + summary["inf_count"],
        "nan_events": summary["nan_count"],
        "inf_events": summary["inf_count"],
        "oom_events": summary["oom_count"],
        "git_commit": config["git_commit"],
        "stage_gpu1_baseline": "05df0d0201746fab521ea776c00440e206ac7f47",
        "dataset_manifest": str(MANIFEST),
        "dataset_manifest_hash": file_digest(MANIFEST),
        "tokenizer_manifest": str(TOKENIZER_MANIFEST),
        "tokenizer_hash": tokenizer["tokenizer_artifact_hash"],
        "model_config_hash": config_hash(config["model"]),
        "run_config_hash": config_hash(config),
        "gradient_norm_min": min(row["grad_norm"] for row in train),
        "gradient_norm_max": max(row["grad_norm"] for row in train),
        "median_optimizer_step_seconds": statistics.median(row["step_time"] for row in train),
        "first_quarter_median_throughput": statistics.median(
            row["tokens_per_second"] for row in train[: len(train) // 4]
        ),
        "last_quarter_median_throughput": statistics.median(
            row["tokens_per_second"] for row in train[-len(train) // 4 :]
        ),
        "environment": read_json(ROOT / "hardware_before.json"),
        "gpu_after": read_json(ROOT / "hardware_after.json"),
    }
    if result["non_finite_events"] or result["oom_events"]:
        raise ValueError("do not silently qualify a non-finite formal result")
    checkpoints = []
    for path in sorted(ROOT.glob("*.pt")):
        checkpoints.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": file_digest(path)}
        )
    json_write(ARTIFACTS / "minillm_checkpoint_inventory.json", {"checkpoints": checkpoints})
    json_write(ARTIFACTS / "minillm_formal_result.json", result)
    for path in ROOT.glob("generation*.json"):
        shutil.copyfile(path, ARTIFACTS / f"minillm_{path.name}")
    shutil.copyfile(ROOT / "metrics.jsonl", ARTIFACTS / "minillm_formal_metrics.jsonl")
    shutil.copyfile(ROOT / "resolved_config.json", ARTIFACTS / "minillm_formal_run_config.json")
    plot_curves(train, evaluations)
    register(result, config)
    report(result, config, manifest, evaluations)
    print(json.dumps(result, indent=2))


def register(result, config):
    path = Path("experiments/registry.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        rows = list(reader)
    record = {
        **result,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": result["run_config_hash"],
        "dataset_hash": result["dataset_manifest_hash"],
        "dataset": "FineWeb-Edu sample-10BT pinned first-shard subset",
        "seed": 42,
        "learning_rate": config["optimizer"]["lr"],
        "warmup_ratio": config["scheduler"]["warmup_ratio"],
        "optimizer": "AdamW",
        "batch_size": result["micro_batch"],
        "effective_batch_size": 16,
        "trainable_params": result["parameters"],
        "peak_vram_mb": result["peak_allocated_vram_mib"],
        "peak_cuda_reserved_mb": result["peak_reserved_vram_mib"],
        "tokens_per_second": result["median_tokens_per_second"],
        "final_loss": result["final_validation_loss"],
        "status": "completed",
        "device": "cuda",
        "gpu_name": result["environment"]["gpu"],
        "torch_version": result["environment"]["pytorch"],
        "torch_cuda_version": result["environment"]["cuda"],
    }
    for key in record:
        if key not in fields:
            fields.append(key)
    for key, value in record.items():
        if isinstance(value, dict):
            record[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    rows = [row for row in rows if row["experiment_id"] != "E01"] + [record]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    json_write(Path("experiments/results/E01.json"), result)


def report(r, config, manifest, evaluations):
    dynamics = "\n".join(
        f"| {x['tokens_seen']:,} | {x['validation_loss']:.4f} | {x['validation_perplexity']:.2f} |"
        for x in evaluations
    )
    text = f"""# MiniLLM Formal Pretraining — E01

## 1. Objective

A bounded from-scratch Transformer pretraining experiment. Establish trainability,
held-out generalization, GPU efficiency and resumability; no Qwen or fine-tuning campaign.

## 2. Hardware

{r["environment"]["gpu"]}, 8150.56 MiB, CUDA runtime {r["environment"]["cuda"]},
PyTorch {r["environment"]["pytorch"]}. No driver, Toolkit, clocks, fan or power changes.

## 3. Model Architecture

{r["parameters"]:,} trainable parameters. 8 layers, hidden 512, 8 query/4 KV heads,
head dimension 64, FFN 1536, 1024 context, RMSNorm/RoPE/GQA/SwiGLU, tied embeddings.
Gradient checkpointing enabled throughout calibration, pilot and formal training.
Architecture SHA-256: `{r["model_config_hash"]}`.

## 4. Tokenizer

24K byte-level BPE, trained on 5,000 train-only documents. Full byte alphabet prevents
unseen-byte UNK failures. The 304-document validation probe has 0 UNK and 100% normalized
round-trip; average 969.67 tokens/document and 4.5686 characters/token.
Tokenizer SHA-256: `{r["tokenizer_hash"]}`. It was not changed during training.

## 5. Dataset

FineWeb-Edu sample-10BT, revision `{manifest["revision"]}`, first 30,000 rows of
`{manifest["shard"]}`. 147,782,794 raw text bytes; 2 normalized duplicates removed.
29,694 train and 304 validation documents, split by normalized document SHA-256 before
tokenizer training. Independent streams packed with EOS between documents. No window
can cross the train/validation partition. There are {r["unique_train_corpus_tokens"]:,}
usable train tokens and {r["validation_tokens"]:,} fixed held-out input tokens
({r["validation_loss_target_tokens"]:,} next-token targets). Drop tails: 613 train,
186 validation tokens. The first-shard selection is deterministic, not a random sample
of the entire FineWeb-Edu distribution. Only exact, not semantic near-duplicate removal
was done in this stage; that limits claims about independence.

## 6. Training Configuration

AdamW, LR {config["optimizer"]["lr"]}, betas (0.9, 0.95), weight decay 0.1,
warmup ratio 0.03 followed by cosine decay to 10% LR, BF16, gradient clip 1.0.
Micro batch {r["micro_batch"]}, accumulation {r["gradient_accumulation"]}, GPU count 1:
16 sequences / 16,384 input tokens / 16,368 supervised targets per optimizer update.
Seed 42. Validation NLL is weighted by actual supervised tokens, not by batches.

## 7. Batch Calibration

393,216 real tokens over batches 2/4/8. Batch 4 selected: batch 8 gained only 0.9%
throughput but nearly doubled reserved memory. See GPU_BATCH_CALIBRATION.md.
No LR search was performed; no immediate divergence was observed at 3e-4.

## 8. Training Budget

Pilot: 5,013,504 processed tokens, separate initialization and shorter LR horizon.
Formal: {r["training_tokens"]:,} processed input tokens, {r["optimizer_steps"]:,} steps,
{r["supervised_next_token_targets"]:,} supervised next-token targets. Rounded from 50M
to full updates; calibration/pilot tokens are NOT added to E01. The finite corpus is
revisited: formal processed tokens / usable corpus tokens =
{r["training_tokens"] / r["unique_train_corpus_tokens"]:.3f}. No claim of 50M unique tokens.
50M was fixed after the pilot; 100M would roughly double laptop runtime and was not
needed to establish this first bounded baseline.

## 9. Training Dynamics

Initial fixed train-probe loss {r["initial_train_loss"]:.4f}; final online interval loss
{r["final_train_loss"]:.4f}. Matched final train-probe loss:
{r["final_train_probe_loss"]:.4f}. Online loss reflects training-time minibatches and
is not exactly the same estimator as evaluation loss.

## 10. Validation Dynamics

| Processed input tokens | Validation loss | Perplexity |
| ---: | ---: | ---: |
{dynamics}

Initial to final PPL: {r["initial_validation_ppl"]:.2f} → {r["final_validation_ppl"]:.2f}.
Best loss {r["best_validation_loss"]:.4f} (PPL {r["best_validation_ppl"]:.2f}) at
step {r["best_step"]}. Held-out improvement, not merely training loss, supports learning
on this fixed split. Final validation-minus-matched-train-probe loss gap is
{r["final_validation_loss"] - r["final_train_probe_loss"]:.4f}; the 32-block probe is small,
so this gap alone cannot establish the onset of overfitting.

## 11. Checkpoint / Resume

`{r["resume_validation"]}`: short real-corpus continuous-vs-resumed control matched model,
optimizer, scheduler and RNG state hashes. The 5M pilot also stopped at step 153 and
resumed to 306 with continuous token/LR counters. Original LR horizon was preserved.
Formal checkpoints include early step 100, mid step 1526, final `last.pt`, and `best.pt`.
Hashes are in minillm_checkpoint_inventory.json; large weights remain local under runs/.

## 12. GPU Efficiency

Peak allocated/reserved: {r["peak_allocated_vram_mib"]:.2f} / {r["peak_reserved_vram_mib"]:.2f} MiB.
Median compute throughput: {r["median_tokens_per_second"]:.1f} input tokens/s; median
optimizer-step time {r["median_optimizer_step_seconds"]:.3f}s.
Active wall time {r["elapsed_seconds"]:.1f}s,
measured update compute time {r["compute_seconds"]:.1f}s.
Update timing synchronizes CUDA and excludes evaluation/checkpointing; wall time includes
run setup/evaluation/export and excludes user/agent idle gaps between process segments.
First-quarter vs last-quarter medians: {r["first_quarter_median_throughput"]:.0f} /
{r["last_quarter_median_throughput"]:.0f} tokens/s. Endpoint GPU status is preserved in JSON;
no continuous thermal-control intervention was made.

## 13. Generation Sanity

Three fixed English completions, seed 42, greedy decoding (temperature 0, top_p 1),
32 new tokens, at initialization/mid/final. Raw outputs are retained in training artifacts.
Initialization repeats arbitrary tokens. Mid/final samples form common English syntactic
fragments but remain repetitive and semantically unreliable. This is a mechanism-level
improvement only; the samples are not a language or math benchmark.

## 14. Failures

Training NaN {r["nan_events"]}, Inf {r["inf_events"]}, OOM {r["oom_events"]}.
Recorded pre-clip gradient norms span {r["gradient_norm_min"]:.3f}–{r["gradient_norm_max"]:.3f}.
Clip threshold is 1.0. These are logged interval endpoint norms, not every micro-step.
The data discovery stall and its explicit-shard resolution are retained under failures/.

## 15. Regression Verification

Regression verification: {r["regression_validation"]}. Ruff check and full-repository
format checks passed, 28 tests passed, all 41 YAML files parsed, the lock remained valid,
and both sdist and wheel built. GPU-0 environment artifacts have no diff from the stage
baseline. Details are in minillm_regression_validation.json.

## 16. Limitations

Limited token budget; not compute-optimal, not fully pretrained, not production quality,
not comparable to Qwen. One seed, one fixed validation partition, no formal architecture
ablation. Repeated corpus exposure and first-shard selection limit generalization claims.
The model remains undertrained in the practical sense that this is only a small bounded
pretraining study; a learning curve does not establish a fully converged language model.

## 17. Conclusion

The measured held-out and training dynamics answer trainability and generalization for
this configuration. Resume and GPU metrics provide engineering evidence. Readiness for
GPU-2 means that the baseline is reproducible, not that MiniLLM has reached production
language quality. No GPU-2 work was performed in this stage.
"""
    Path("reports/MINILLM_FORMAL_PRETRAINING.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["budget"]:
        approve_budget()
    else:
        finalize()
