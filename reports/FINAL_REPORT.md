# MiniLLM-Forge Technical Report

## 1. Problem

The project demonstrates two distinct capabilities: understanding Transformer training
by implementing a decoder-only model directly in PyTorch, and operating modern LLM
adaptation workflows using a pretrained base model. It targets mathematical reasoning so
continued pretraining and instruction tuning have a coherent domain.

## 2. Objectives

1. Validate the architecture, causal objective, numerical behavior, and resumable
   training loop of a small model.
2. Compare Qwen3-0.6B-Base, math CPT, direct SFT, CPT-to-SFT, LoRA, and QLoRA under
   controlled data and evaluation conditions.
3. Measure quality, memory, throughput, stability, and reproducibility without invented
   results.

## 3. Transformer Architecture

MiniLLM is an 8-layer pre-norm decoder by default. Query projections use eight heads;
keys and values use four heads and are repeated within GQA groups. RoPE is applied to
queries and keys before the attention product. A lower-triangular boolean mask prevents
every position from reading future keys. RMSNorm and a three-projection SwiGLU complete
each block. Input embeddings and the language-model head share weights.

Tests perturb a future token and assert all earlier logits remain unchanged. Separate
tests cover shapes, KV repetition, rotary norm preservation, RMS normalization, and
weight tying.

## 4. Pretraining

The objective is next-token prediction. The model returns logits aligned with inputs;
`causal_lm_loss` compares `logits[:, :-1]` to `labels[:, 1:]`. The custom trainer handles
gradient accumulation, clipping, AdamW, warmup/cosine scheduling, AMP, evaluation,
finite checks, OOM records, and atomic checkpoints.

The measured E01 run trained the 37,462,528-parameter model from scratch for 3,052
optimizer steps: 50,003,968 input tokens and 49,955,136 supervised next-token targets.
Fixed held-out loss fell from 10.1964 to 4.6484 and perplexity from 26,805.55 to 104.42.
The matched 32-block train probe ended at loss 4.5320, for a validation-minus-probe gap
of 0.1164. The best validation result occurred at the final step; this supports stable
learning over the bounded run, not full language-model convergence. Complete dynamics
and six measured curves are in `reports/MINILLM_FORMAL_PRETRAINING.md`.

The CPU Tiny Overfit gate completed 60 optimizer steps on 15,360 tokens. Validation loss
fell from 4.9014 to 0.7506 and perplexity from 134.48 to 2.118. Resuming the saved step-30
checkpoint reproduced the same final loss. This is a correctness gate, not a formal
pretraining quality result.

## 5. Data

Track A trains a byte-level BPE tokenizer with `<pad>`, `<bos>`, `<eos>`, and `<unk>`.
Validation records vocabulary size, average document length, unknown ratio, character
compression ratio, and normalized round-trip success.

Data governance normalizes Unicode, hashes normalized text for exact deduplication, and
uses character n-gram Jaccard similarity for near-collision detection. Upstream revision,
filter settings, counts, corpus digest, and removed benchmark overlaps belong in every
data manifest.

E01 used the pinned FineWeb-Edu sample-10BT revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, with the first 30,000 rows of the first
parquet shard. Exact normalized deduplication removed two documents before a hash-based
train/validation split, leaving 29,694 train and 304 validation documents. The packed
training stream contains 32,179,200 input tokens, so the 50.004M formal budget is
1.554 corpus passes—not 50M unique tokens. The formal tokenizer is a train-only 24K
byte-level BPE with zero validation UNKs and a frozen artifact hash.

## 6. Continued Pretraining

The CPT path streams a bounded FineMath-4+ subset and retains the causal-LM objective.
The planned test compares both math and general held-out perplexity before and after CPT.
A math improvement accompanied by general degradation will be reported as possible
catastrophic forgetting, not as an unconditional gain.

Measured CPT result: **QWEN_MATH_CPT_VALIDATED_WITH_GENERAL_DEGRADATION** at 10,002,432 tokens. Math PPL 5.1715 → 5.0348 (-2.64%); general PPL 17.0799 → 17.2311 (+0.89%). This is domain language-model adaptation evidence, not proof of mathematical reasoning gain.

## 7. SFT

SFT examples contain system, user, and assistant roles. Prefix length is computed with
the loaded tokenizer's own chat template. System/user/padding labels are `-100`; only
assistant response tokens are trained. This masking path is independently tested.

Full-SFT formal transfer was not completed. GPU-2D instead completed the prospectively
qualified 512-context LoRA/QLoRA study. This distinction is retained: PEFT evidence must
not be presented as Full-SFT evidence.

GPU-2B-D design status: **STAGE_GPU_2B_D_COMPLETE**. The preregistered numerical stage
uses Base/CPT initialization paired within Full SFT, LoRA and QLoRA, three shared seeds,
one frozen 19,200-example/7,024,493-target-token SFT budget, and GSM8K plus MATH-500
generated-answer accuracy as the primary endpoints. No SFT training was run in GPU-2B-D.

## 8. LoRA and QLoRA

LoRA uses PEFT adapters with configurable rank, alpha, dropout, and target modules.
The rank experiment is bounded to 4, 8, 16, and 32. Placement compares `q_proj/v_proj`,
all attention projections, and `all-linear`. QLoRA loads the base in NF4, enables double
quantization, computes in BF16, prepares k-bit training, and attaches LoRA adapters.

GPU-2D completed Base/CPT initialization paired within LoRA and QLoRA for seeds 42,
31415, and 271828. All 12 runs completed 1,200 updates with zero NaN, Inf, or OOM events.
LoRA used 3428-3686 MiB physical VRAM and averaged 735.05 target tokens/s; QLoRA used
5792-6116 MiB and averaged 587.58 target tokens/s on the qualified Windows/WDDM stack.
Thus QLoRA was neither faster nor more memory-efficient in this specific environment.

## 9. Evaluation

Training metrics are loss, PPL, gradient norm, and parameter norm. Task metrics are exact
match and accuracy after final-answer extraction. Efficiency metrics are actual peak
allocated CUDA memory, elapsed time, throughput, total/trainable parameters, and tokens
processed. Stability records non-finite values, OOM events, and resume success.

Public benchmark results and the controlled held-out set are reported separately.

The final GPU-2D audit validated 24/24 evaluation jobs and 8,400/8,400 problem-level
records: 12 full MATH-500 jobs (6,000 generations) and 12 jobs on the outcome-blind
GSM8K fixed-200 subset (2,400 generations), with zero missing or duplicate IDs. Accuracy
was recomputed from raw correctness records, and every checkpoint, adapter, prompt,
scorer, subset, repository, and completion identity was verified.

MATH-500 paired CPT-minus-Base deltas were negative for every seed under both methods:
LoRA [-0.2, -0.6, -2.2] percentage points and QLoRA [-1.2, -2.0, -0.6]. GSM8K
fixed-200 was negative for every LoRA seed [-3.0, -12.0, -1.0] but mixed for QLoRA
[+1.5, -6.0, +9.5]. The classification is
`CPT_TO_PEFT_SFT_TRANSFER_METHOD_DEPENDENT`, not a claim of reasoning improvement.

## 10. Ablation

The predefined 15-run matrix covers architecture, optimization, training path, LoRA
rank and placement, quantization, and repeated seed. Every completed experiment must use
`reports/ablations/EXPERIMENT_TEMPLATE.md` and be registered automatically.

The 12-run GPU-2D Base/CPT-by-LoRA/QLoRA multi-seed matrix is complete. Other planned
architecture, rank, placement, and Full-SFT ablations remain outside the completed formal
evidence and are not inferred from GPU-2D.

## 11. Training Failures

Failures are retained with symptom, evidence, root cause, fix, and before/after behavior.
No hypothetical failure is presented as observed evidence.

The generic streaming dataset enumerator stalled before writing corpus bytes during
GPU-1 preparation. The zero-byte attempt and diagnosis are retained; preparation was
resolved with the same pinned source/revision and an explicit first parquet shard. E01
training itself recorded zero NaN, Inf, and OOM events.

## 12. Efficiency

The reference design remains a single 24 GB GPU for broader full-fine-tuning margin. The
actual local qualification hardware is an NVIDIA GeForce RTX 5060 Laptop GPU with 8151
MiB, driver 577.02, and compute capability 12.0. It was qualified using torch
2.11.0+cu128 and BF16.

At sequence length 1024 and micro-batch one, measured peak allocated/reserved memory was
927/1058 MiB for MiniLLM, 2184/2358 MiB for Qwen evaluation, 5021/5050 MiB for LoRA,
3315/3584 MiB for QLoRA, and 5999/6032 MiB for both full CPT and full SFT single-step
tests. QLoRA is the preferred local formal path. Full CPT/SFT are technically
single-step feasible but retain less than 1 GiB of measured system-level headroom.

For E01, calibration selected micro-batch 4 with accumulation 4. The formal run measured
1,849/2,394 MiB peak allocated/reserved memory, 19,573 input tokens/s median compute
throughput, 2,549.1 seconds of synchronized update compute, and 2,684.8 seconds active
wall time. First/last-quarter throughput medians were 19,557/19,485 tokens/s. A real-data
continuous-vs-resumed control matched model, optimizer, scheduler, and RNG hashes;
the separate 5M-token pilot also resumed across an intentional midpoint interruption.

## 13. Limitations

- MiniLLM receives a deliberately limited token budget and is not claimed to be fully
  pretrained or compute-optimal.
- A 0.6B model has limited capacity; conclusions should not be extrapolated to larger
  models without validation.
- Public benchmarks may exist in upstream pretraining data. Collision auditing can
  protect local fine-tuning data but cannot prove the base model was uncontaminated.
- QLoRA depends on bitsandbytes and supported CUDA hardware. The local Windows backend
  passed NF4 forward/backward, but other platforms require independent qualification.
- Full CPT long-run stability is established only for the frozen 512-token GPU-2A configuration; full SFT remains unqualified for long runs.
- E01 is one seed and one bounded first-shard corpus run. GPU-2D adds three paired seeds
  for LoRA/QLoRA but does not complete the separate architecture or Full-SFT ablations.
- GPU-2D applies only to the frozen 512-context exposure, Qwen3-0.6B, one SFT corpus,
  LoRA/QLoRA, and three seeds. The original 1024-context protocol remained blocked.
- GSM8K is a frozen 200-problem subset, not the full test set.
- 8,398/8,400 formal generations reached the 512-token ceiling; repetition and truncation
  materially limit interpretation of the absolute benchmark scores.
- Problem-level paired bootstrap intervals quantify problem sampling, not training-seed
  or end-to-end pipeline uncertainty.

## 14. Conclusions

The training stack is code-complete and its RTX 5060 CUDA, MiniLLM, Qwen, LoRA, and QLoRA
paths are qualified. GPU-1 validates stable bounded MiniLLM pretraining and exact resume;
GPU-2A validates math-domain language-model adaptation with a small general-domain cost;
GPU-2D validates an auditable 12-run PEFT campaign and 8,400-answer formal evaluation.

The downstream result is deliberately non-promotional: Math-CPT did not produce a
consistent positive transfer under the frozen 512-context protocol. LoRA was negative on
both benchmarks; QLoRA was negative on MATH-500 and mixed on GSM8K fixed-200. This is a
reproducible engineering and scientific-audit portfolio, not a production model or a
general mathematical-reasoning claim. See `reports/GPU2D_FORMAL_PEFT_TRANSFER.md` for
the complete matrix, bootstrap intervals, efficiency evidence, and error analysis.
