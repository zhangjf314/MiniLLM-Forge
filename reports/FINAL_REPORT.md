# MiniLLM-Forge Final Report

This is the repository-facing summary. The complete evidence-audited report is
[`artifacts/gpu4c/MINILLM_FORGE_FINAL_TECHNICAL_REPORT.md`](../artifacts/gpu4c/MINILLM_FORGE_FINAL_TECHNICAL_REPORT.md),
with machine-readable claims in
[`artifacts/gpu4c/claim_registry.json`](../artifacts/gpu4c/claim_registry.json).

## Project position

MiniLLM-Forge is a reproducible LLM training, evaluation, and numerical-diagnostics
laboratory. It is not a production chat service and does not claim state-of-the-art
language or reasoning quality.

## Native model

The formal MiniLLM is a 37,462,528-parameter, 8-layer decoder with hidden size 512,
GQA 8/4, FFN 1536, RoPE, RMSNorm, SwiGLU, tied embeddings, context 1024, and a native
24K byte-level BPE tokenizer.

Formal BF16 pretraining completed 3,052 optimizer steps and 50,003,968 processed tokens.
Fixed validation loss improved from 10.1964 to 4.6484 and perplexity from 26,805.55 to
104.42. Exact model, optimizer, scheduler, and RNG resume was verified. This establishes
trainability and systems correctness, not mature general-language capability.

## Native Full SFT

GPU-4A completed 300-step, full-parameter, response-only BF16 SFT on two deterministic
synthetic controls. Validation loss improved from 5.9922 to 0.5943 in 48.87 seconds.

- T1 support classification, fixed independent test subset: `0/64 → 64/64`.
- T2 addition, fixed independent test subset: `0/64 → 2/64`.
- Generation behavior: `384/384` format-valid, complete, EOS-terminated,
  non-repetitive, and not length-limited.

The T1 result is not a full 300-row test-set accuracy claim. The T2 result is retained as
a negative capability result.

## Qwen3-0.6B PEFT

The formal 512-context campaign completed 12/12 LoRA/QLoRA runs, three seeds, 1,200
steps per run, 24/24 evaluation jobs, and 8,400 audited MATH-500/GSM8K outputs. Math-CPT
transfer was negative for LoRA and mixed for QLoRA. Qwen Full SFT effectiveness and the
planned 1024-context formal campaign were not completed.

## Generation and evaluation diagnostics

- GPU-3A: `GPU3A_ROOT_CAUSES_IDENTIFIED`; 8,398/8,400 outputs hit the 512-token limit.
  At context 512, 12,469/19,162 training samples retained supervised assistant-end
  tokens (65.07%), while 6,693 lost them after truncation. Real PEFT SFT and effectiveness
  evaluation had executed, but these observations did not prove that missing EOS
  supervision caused the behavior.
- GPU-3B: `GPU3B_DECODING_AND_EXTRACTION_IMPROVED`; first-valid extraction counted
  2,008/8,400 correct versus legacy 1,764/8,400, recovered 36/36 legacy failures with
  16 correct, and exposed 607 conflicts. This is evaluator improvement, not model
  improvement. External stopping reduced redundancy but did not teach autonomous EOS.
- GPU-3C: `GPU3C_CONTROLLED_NEGATIVE_RESULT`; a controlled 64-step EOS-label ablation
  did not validate the expected autonomous-stop improvement.

## Attention and numerical study

GPU-4B is `GPU4B_PARTIAL_VALIDATION`: SDPA Math passed FP32 but Math/Auto failed frozen
BF16 gates. Built-in FlashAttention was unavailable and compile was blocked by missing
Triton. No qualified speed, throughput, or memory comparison exists.

GPU-4B-R1 is `GPU4B_R1_NUMERICAL_PATH_DIFFERENCE_IDENTIFIED`. Q/K/V and RoPE were
bit-identical; the earliest public BF16 difference appeared at the first Attention×V
context. Controlled FP32-intermediate reconstruction closely matched SDPA Math and no
semantic implementation defect was found. Exact internal SDPA instruction order remains
unknown. Manual Attention + Eager remains the formal default.

## Reproducibility

- Pretraining checkpoint SHA256:
  `ee306384fcdef646f557cf9aa1165c464b2ca373c6d64d81f5df3b19b8f859fc`
- Native SFT checkpoint SHA256:
  `9c33b1c5a781c4ab1d9832a0aad1349516fafb011d24ff6c67be3b1d3885b96e`
- Tokenizer SHA256:
  `e9578dfd6d2a0f4c3137d7cc223a7b8a782afba8190ae1ad105e1b8ef37e9a22`

Use `portfolio-verify quick` to verify hashes, checkpoint metadata, tokenizer loading,
registry schemas, and the critical claim boundaries without training or network access.

## Release boundary

The portfolio is suitable for an evidence-preserving release with documented
limitations. It must not advertise FlashAttention or compile acceleration, full-test T1
accuracy, Qwen Full SFT effectiveness, DDP/FSDP, DPO, or GRPO as validated capabilities.
