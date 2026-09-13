# Stage GPU-2C — PEFT Transfer Protocol

## 1. Scientific Question

Does validated math-CPT initialization improve downstream mathematical SFT under parameter-
efficient fine-tuning? GPU-2C separately studies CPT-to-LoRA and CPT-to-QLoRA transfer.

## 2. Claim Boundary

The only scientific scope is `CPT_TO_PEFT_SFT_TRANSFER`. GPU-2C cannot establish
`CPT_TO_FULL_SFT_TRANSFER`, complete GPU-2B, or replace the primary GPU-2B contrast
`C-FULL - B-FULL`.

The inherited boundary remains math-domain adaptation confirmed, reasoning improvement not
established, and CPT-to-Full-SFT transfer not assessed.

## 3. Frozen Initializations

`B` is `Qwen/Qwen3-0.6B-Base` revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, with 596,049,920 parameters and frozen
tokenizer digest `3fe32a8d...821cfe`.

`C` is the validated GPU-2A checkpoint `runs/E04-qwen3-math-cpt/final_model`, model SHA-256
`d9a30813...f2c4a`, descended from the same Base revision after 10,002,432 CPT tokens and
2,442 updates. No CPT retraining or checkpoint reselection is allowed.

## 4. Dataset

GPU-2C reuses the frozen GPU-2B payload only after fresh digest verification:

- 19,200 train and 800 validation records;
- 9,127,756 serialized input tokens;
- 7,024,493 supervised assistant targets;
- assistant-only labels and right truncation at 1,024 tokens;
- one complete pass and 1,200 optimizer updates at micro-batch 1 and accumulation 16.

The train, validation, derived token-cache, assistant-mask, prompt, scorer, and benchmark
hashes are binding. Any mismatch yields `GPU2C_FROZEN_DATA_IDENTITY_MISMATCH` and stops the
stage.

## 5. Seeds

Seeds are 42, 31415, and 271828. Their Torch permutations must exactly match the frozen
GPU-2B hashes. Base and CPT arms share the identical permutation within method and seed.
No replacement seed is permitted.

## 6. LoRA Configuration

GPU-2C reuses `configs/sft/gpu2b/lora-config.yaml` byte-for-byte: BF16 backbone, rank 16,
alpha 32, dropout 0.05, all-linear targets, no bias, no modules-to-save, AdamW at 2e-4,
cosine schedule, 3% warmup, context 1,024, micro-batch 1, accumulation 16, gradient
checkpointing, and 1,200 updates. Only adapter parameters may be trainable.

## 7. QLoRA Configuration

GPU-2C reuses `configs/sft/gpu2b/qlora-config.yaml` byte-for-byte: frozen true 4-bit NF4
backbone, double quantization, BF16 compute, the same rank-16 adapter, PagedAdamW8bit at
2e-4, and the same schedule/data budget. Q1 must prove `Linear4bit` module types, populated
4-bit quantization state, frozen backbone, and adapter-only trainability programmatically.

## 8. Qualification Gates

Q0-P verifies every static identity plus pytest, Ruff, format, all tracked YAML, lock, and
build. Q1 then runs exactly one Base/seed-42 control per method for 64 optimizer updates.
Each must complete with zero NaN, Inf, and OOM and at least 1,536 MiB minimum independently
measured physical headroom. CUDA allocated/reserved counters are recorded but are not the
WDDM physical-memory authority.

Qualifications are engineering controls, never formal runs or benchmark results.

## 9. Resume Contract

After both Q1 gates pass, LoRA and QLoRA independently compare a 16-update continuous
control with update 1–8, checkpoint, restore, and update 9–16. Required state includes the
model/adapters, optimizer, scheduler, sampler order/cursor/generator, DataLoader generator,
global counters, Python/NumPy/Torch/CUDA RNG, and quantized-backbone identity where relevant.

Exact equality earns `PEFT_EXACT_RESUME_CONFIRMED`. If GPU arithmetic prevents bitwise
identity while all state and continuation semantics satisfy frozen tolerances, use
`PEFT_RESUME_SEMANTICS_VALIDATED`. Sample/token order must always be exact.

## 10. Formal Matrix

The 12 run IDs use the existing repository convention:

```text
B-LORA-s42       C-LORA-s42
B-LORA-s31415    C-LORA-s31415
B-LORA-s271828   C-LORA-s271828
B-QLORA-s42      C-QLORA-s42
B-QLORA-s31415   C-QLORA-s31415
B-QLORA-s271828  C-QLORA-s271828
```

Each future run has an isolated directory containing config/environment snapshots, source
hashes, initialization identity, metrics, checkpoints, physical-memory logs, and summary.

## 11. Execution Order

The protocol JSON freezes a balanced seed-block order that alternates initialization and
method rather than running all Base arms before CPT arms. The exact 12-entry order is
immutable after protocol freeze; manual cherry-picking is forbidden.

## 12. Benchmark Protocol

All final adapters use direct PEFT adapter inference against their frozen initialization;
merged checkpoints are not the formal path. Both benchmarks reuse the frozen prompts,
tokenizer, scorer, records, and greedy generation: batch 1, one beam, no sampling,
temperature/top-p unset, 512 new-token maximum, EOS 151643/151645, pad 151643, and no manual
answer correction.

Formal evaluation is prohibited during GPU-2C-Q.

## 13. Checkpoint Selection

Every scientific comparison uses `FINAL_TOKEN_BUDGET_CHECKPOINT`. Validation is diagnostic
only. Benchmark or validation results cannot choose a checkpoint, extend training, or tune
an arm.

## 14. Primary Comparisons

For each benchmark and seed:

```text
delta_LORA(seed)  = accuracy(C-LORA, seed)  - accuracy(B-LORA, seed)
delta_QLORA(seed) = accuracy(C-QLORA, seed) - accuracy(B-QLORA, seed)
```

LoRA-versus-QLoRA is an engineering/secondary method comparison, not the primary question.

## 15. Secondary Metrics

Report training/validation loss, gradient norm, LR, physical/CUDA memory, target tokens/s,
samples/s, step and wall-clock time, trainable parameters, adapter/checkpoint size, and the
LoRA/QLoRA memory-quality tradeoff. Math PPL is optional only under a separately frozen
interpretation and cannot substitute for task accuracy.

## 16. Statistical Analysis

Report every per-seed delta, paired mean, paired standard deviation, range, and direction
consistency. Three training seeds do not justify overstated asymptotic significance.

Problem-level uncertainty uses 10,000 paired percentile-bootstrap resamples at 95%
confidence with seed 20260913. The unit is benchmark problem ID and each sampled index is
shared across arms and seeds. This interval does not replace training-seed variability.

The existing practical rule is reused within each method and benchmark: positive transfer
requires mean delta at least +1.0 point, at least two positive seeds, and none below -0.5;
negative transfer is symmetric. Mixed and remaining outcomes follow the frozen exhaustive
rules. Both methods positive yields `SUPPORTED`; both negative yields `NEGATIVE`; one
confirmed method or opposing confirmed directions yields `METHOD_DEPENDENT`; unresolved
mixed evidence yields `INCONCLUSIVE`; every remaining complete result is `NOT_SUPPORTED`.

## 17. Rerun Policy

Reruns are allowed only for infrastructure failure, hardware interruption, verified
software fault, or corrupted artifact, while retaining the failed evidence. Low accuracy,
unexpected loss, CPT loss, or an unfavorable seed never authorizes rerun or replacement.

## 18. Outcome Classifications

The only final scientific labels are:

- `CPT_TO_PEFT_SFT_TRANSFER_SUPPORTED`;
- `CPT_TO_PEFT_SFT_TRANSFER_METHOD_DEPENDENT`;
- `CPT_TO_PEFT_SFT_TRANSFER_NOT_SUPPORTED`;
- `CPT_TO_PEFT_SFT_TRANSFER_NEGATIVE`;
- `GPU2C_FORMAL_RESULTS_INCONCLUSIVE`.

The ordered numerical rules in the protocol JSON are authoritative.

## 19. Forbidden Claims

Before the formal campaign, and regardless of qualification results, do not claim CPT
improves GSM8K, MATH-500, or reasoning; LoRA/QLoRA superiority; validated CPT transfer;
GPU-2B completion; or CPT-to-Full-SFT transfer.

## 20. Formal Authorization Rule

Formal authorization requires protocol freeze, Q0-P PASS, LoRA Q1 PASS, QLoRA Q1 PASS,
LoRA Q2 PASS, QLoRA Q2 PASS, physical-memory authority PASS, and final regression PASS.

GPU-2C-Q stops after recording that authorization decision. It must not start any of the 12
formal runs. The next stage after a complete qualification is `STAGE_GPU_2C_F`.
