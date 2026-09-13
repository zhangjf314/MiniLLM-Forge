# Stage GPU-2C-Q — Qualification Report

## 1. Authorization Boundary

GPU-2C-Q was authorized only to freeze the PEFT transfer study and execute Q0-P, bounded
LoRA/QLoRA memory qualification, and bounded resume qualification. It was not authorized
to launch any of the 12 formal runs or either formal benchmark. The sequential stop rule
applied as soon as LoRA Q1 failed its frozen memory-authority gate.

## 2. Frozen Scientific Design

The secondary study remains 2 initializations (`B`, `C`) × 2 methods (`LORA`, `QLORA`) ×
3 seeds (42, 31415, 271828), for 12 future formal runs. Its only claim scope is
`CPT_TO_PEFT_SFT_TRANSFER`. The historical GPU-2B Full-SFT study remains frozen and
blocked; GPU-2C neither completes nor replaces it.

## 3. Data Identity

Q0-P revalidated 19,200 train records, 800 validation records, 9,127,756 input tokens, and
7,024,493 assistant target tokens. Train, validation, token-cache, assistant-mask, prompt,
scorer, contamination, and benchmark snapshot hashes exactly matched the frozen protocol.

## 4. Seed Identity

The regenerated permutations for seeds 42, 31415, and 271828 exactly matched the frozen
GPU-2B SHA-256 values. Q0-P recorded each permutation's sample count and first/last eight
indices.

## 5. Base/CPT Initialization

Base revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, its model configuration,
596,049,920 parameter count, tokenizer revision, and tokenizer digest all passed. The
validated Math-CPT checkpoint hash, Base lineage, 10,002,432-token budget, 2,442 updates,
configuration hash, and GPU-2A validation classification also passed.

## 6. LoRA Contract

The byte-exact GPU-2B LoRA configuration was retained: BF16, rank 16, alpha 32, dropout
0.05, all-linear targets, AdamW, 2e-4 learning rate, cosine schedule, 3% warmup, context
1,024, micro-batch 1, accumulation 16, and a future formal budget of 1,200 updates. The Q1
model contained 10,092,544 trainable adapter parameters (1.6650%) and no unexpected
trainable backbone parameters.

## 7. QLoRA Contract

The byte-exact 4-bit NF4/double-quantized GPU-2B QLoRA design remains frozen. Its runtime
quantization and adapter-only gates were not executed because the preceding LoRA memory
gate failed. No QLoRA qualification claim is made.

## 8. Q0-P

`GPU2C_Q0_P_PASS`. All scientific identities, both configuration hashes, seed order,
prompt/scorer and benchmark identities, pytest, Ruff, format, YAML parsing, lock validation,
and package build passed. Q0-P launched no training or benchmark.

## 9. LoRA Q1

The Base/seed-42 control completed exactly 64 optimizer updates and 1,024 samples through
real forward, backward, optimizer, and scheduler operations with assistant-only loss.
NaN=0, Inf=0, OOM=0, losses were finite, and gradient norms ranged from 0.380363 to
2.544876. Nevertheless, physical VRAM peaked at 7,786 MiB and minimum physical headroom
was only 25 MiB. This fails the prospectively frozen requirement of at least 1,536 MiB and
therefore classifies as `GPU2C_LORA_MEMORY_QUALIFICATION_FAILED`.

## 10. QLoRA Q1

`NOT_RUN`. The sequential stop rule prohibited QLoRA Q1 after LoRA Q1 failed. This is not a
QLoRA failure and supplies no QLoRA stability, quantization-integrity, or memory evidence.

## 11. LoRA Resume Q2

`NOT_RUN`. Both Q1 memory gates were required before any Q2 execution.

## 12. QLoRA Resume Q2

`NOT_RUN`. Both Q1 memory gates were required before any Q2 execution.

## 13. Physical VRAM Evidence

Independent physical monitoring collected 1,686 samples during LoRA Q1, with no monitor
errors. The physical peak was 7,786 MiB and minimum headroom was 25 MiB, compared with the
1,536 MiB authority threshold. Framework peaks were 3,447.914 MiB allocated and 7,840 MiB
reserved. The allocator/physical difference reinforces why physical monitoring is binding.

## 14. Regression

The final regression passed: pytest 45 passed and 0 failed; Ruff, formatting, all 29 tracked
YAML files, `uv lock --check`, and `uv build` all passed. Frozen scientific artifacts were
revalidated by Q0-P before the qualification launch.

## 15. Formal Campaign Authorization

`formal_campaign_authorized = NO`. Q0-P passed, LoRA Q1 failed, and QLoRA Q1 plus both Q2
paths were not run. Formal training launches=0 and benchmark launches=0. `STAGE_GPU_2C_F`
is not authorized.

## 16. Remaining Risks

The LoRA control's numerical completion does not establish robust 1,200-update memory
authority. QLoRA may use less memory, but that unmeasured possibility cannot rescue a
two-method study whose LoRA arm is required. Any recovery would need a new prospective
authorization; the frozen 1,536 MiB gate must not be relaxed post hoc.

## 17. Final Classification

`GPU2C_BLOCKED_BY_MEMORY_QUALIFICATION`.

No downstream scientific endpoint was measured. The stage does not establish CPT transfer,
reasoning improvement, LoRA/QLoRA quality ordering, GPU-2B completion, or
CPT-to-Full-SFT transfer.
