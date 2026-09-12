# Qwen Math Continued Pretraining

## 1. Objective

Measure whether pinned FineMath-4+ full-parameter CPT improves frozen math-domain
perplexity and quantify any change on a frozen general-domain probe.

## 2. Hypotheses

H-CPT-1 freezes expected math NLL/PPL improvement. H-CPT-2 requires simultaneous
general-domain measurement. H-CPT-3 requires reporting their trade-off.

## 3. Base Model

`Qwen/Qwen3-0.6B-Base` at `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` with
596,049,920 parameters and one identical pinned tokenizer for both arms.

## 4. Hardware

NVIDIA GeForce RTX 5060 Laptop GPU (8,150.56 MiB), BF16, CUDA 12.8.

## 5. CPT Dataset

FineMath-4+ revision `e92b25a616738fe95dc186b64dfb19f9c8525594`, deterministic pinned streaming range,
10,002,432 input tokens and causal next-token prediction only.

## 6. Held-out Sets

Math and general partitions were frozen before training. General validation reuses frozen
GPU-1 FineWeb-Edu text but is independently Qwen-tokenized. Evaluated target counts are
math 65,408 and general 65,408.

## 7. Contamination Audit

Exact normalized duplicates and near 8-gram Jaccard collisions
at 0.8 were removed against math/general/benchmark probes.
This cannot rule out contamination in Qwen's original pretraining.

## 8. Baseline Evaluation

Math loss/PPL 1.643171/5.1715; general
loss/PPL 2.837904/17.0799.

## 9. Memory Calibration

512 context: 4406.9 MiB peak allocated and
2410.0 MiB estimated minimum headroom.
8-bit AdamW compresses optimizer state while all model weights remain BF16 and all
596,049,920 parameters remain trainable. This is full-parameter CPT, not QLoRA.

## 10. Long-run Qualification

`FULL_CPT_LONG_RUN_QUALIFIED` at 503,808 tokens; NaN/Inf/OOM
0/0/0.

## 11. Formal Configuration

Sequence 512, micro-batch 1, accumulation 8 (4,096 tokens/update), BF16, gradient
checkpointing, AdamW LR 2.0e-05, 3% warmup and cosine decay.

## 12. Training Dynamics

2,442 optimizer steps; final logged train loss
0.836129; no LR or warmup sweep.

## 13. Math-domain Evaluation

PPL 5.1715 → 5.0348
(-2.64%).

## 14. General-domain Evaluation

PPL 17.0799 → 17.2311
(+0.89%).

## 15. Catastrophic Forgetting Analysis

Both PPLs were measured together near 0/25/50/75/100% of the budget. The general-domain
delta is reported quantitatively rather than hidden by the math result. The 5% relative
general-loss early-stop gate did not trigger; the smaller sustained degradation remains
material to the classification.

## 16. Checkpoint / Resume

CPT_RESUME_SEMANTICS_VALIDATED; qualification performed train → checkpoint → restore →
continue while matching step, tokens, LR, scheduler and RNG state.

## 17. GPU Efficiency

Peak allocated/reserved 4406.9/
4536.0 MiB; median 1972.91
tokens/s; elapsed 5172.9s.

## 18. Base vs CPT

See `reports/QWEN_BASE_VS_CPT.md` for absolute and relative deltas.

## 19. Hypothesis Results

- H-CPT-1: **supported** — math PPL -2.64%.
- H-CPT-2: **supported** — general PPL +0.89%.
- H-CPT-3: **supported** — both domains are jointly reported.

## 20. Limitations

One seed, a bounded 10M-token first-range corpus and two PPL probes. The audit is local.
Generative EM is secondary for this non-instruction-tuned model and was not used as the
primary endpoint. Lower domain PPL is not proof of reasoning-task improvement.

## 21. Conclusion

**QWEN_MATH_CPT_VALIDATED_WITH_GENERAL_DEGRADATION**. This is controlled domain language-model adaptation
evidence only; Stage GPU-2A does not enter SFT.
