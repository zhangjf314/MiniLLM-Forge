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

Measured formal-run results: **TBD**.

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

## 6. Continued Pretraining

The CPT path streams a bounded FineMath-4+ subset and retains the causal-LM objective.
The planned test compares both math and general held-out perplexity before and after CPT.
A math improvement accompanied by general degradation will be reported as possible
catastrophic forgetting, not as an unconditional gain.

Measured CPT results: **TBD**.

## 7. SFT

SFT examples contain system, user, and assistant roles. Prefix length is computed with
the loaded tokenizer's own chat template. System/user/padding labels are `-100`; only
assistant response tokens are trained. This masking path is independently tested.

Measured full-SFT results: **TBD**.

## 8. LoRA and QLoRA

LoRA uses PEFT adapters with configurable rank, alpha, dropout, and target modules.
The rank experiment is bounded to 4, 8, 16, and 32. Placement compares `q_proj/v_proj`,
all attention projections, and `all-linear`. QLoRA loads the base in NF4, enables double
quantization, computes in BF16, prepares k-bit training, and attaches LoRA adapters.

Measured quality/memory/cost trade-off: **TBD**.

## 9. Evaluation

Training metrics are loss, PPL, gradient norm, and parameter norm. Task metrics are exact
match and accuracy after final-answer extraction. Efficiency metrics are actual peak
allocated CUDA memory, elapsed time, throughput, total/trainable parameters, and tokens
processed. Stability records non-finite values, OOM events, and resume success.

Public benchmark results and the controlled held-out set are reported separately.

## 10. Ablation

The predefined 15-run matrix covers architecture, optimization, training path, LoRA
rank and placement, quantization, and repeated seed. Every completed experiment must use
`reports/ablations/EXPERIMENT_TEMPLATE.md` and be registered automatically.

Results and conclusions: **TBD after controlled runs**.

## 11. Training Failures

Failures are retained with symptom, evidence, root cause, fix, and before/after behavior.
No hypothetical failure is presented as observed evidence.

Observed cases: **none recorded yet**.

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

## 13. Limitations

- MiniLLM receives a deliberately limited token budget and is not claimed to be fully
  pretrained or compute-optimal.
- A 0.6B model has limited capacity; conclusions should not be extrapolated to larger
  models without validation.
- Public benchmarks may exist in upstream pretraining data. Collision auditing can
  protect local fine-tuning data but cannot prove the base model was uncontaminated.
- QLoRA depends on bitsandbytes and supported CUDA hardware. The local Windows backend
  passed NF4 forward/backward, but other platforms require independent qualification.
- Full CPT/SFT long-run stability is not established by the bounded one-step 8GB tests.
- Code completion is not portfolio completion. GPU training, repeated runs, frozen
  manifests, curves, and written conclusions remain required evidence.

## 14. Conclusions

The training stack is code-complete and its RTX 5060 CUDA, MiniLLM, Qwen, LoRA, and QLoRA
paths are qualified. Scientific quality conclusions remain deferred until pinned formal
experiments populate the registry. See `reports/GPU_ENVIRONMENT_QUALIFICATION.md` for the
environment evidence and measured feasibility boundary.
