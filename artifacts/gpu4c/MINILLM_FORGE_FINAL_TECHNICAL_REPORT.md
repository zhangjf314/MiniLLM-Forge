# MiniLLM-Forge Final Technical Report

## 1. Executive Summary

MiniLLM-Forge is a reproducible training and evaluation laboratory spanning a native 37.46M decoder Transformer and a separate Qwen3-0.6B adaptation line. It demonstrates the engineering loop from tokenizer and model implementation through pretraining, native full SFT, PEFT, evaluation diagnostics, controlled negative experiments, and BF16 numerical analysis. It is not a claim of a mature general-purpose LLM.

## 2. System Architecture

The repository contains a native 24K byte-level BPE tokenizer, pre-norm decoder blocks, explicit trainers, atomic full-state checkpoints, response-only SFT masking, LoRA/QLoRA workflows, exact-match evaluation, generation diagnostics, evidence manifests, and integrity tests. Native and Qwen experimental lines remain separate.

## 3. Native 37M Transformer

The frozen model has 37,462,528 parameters, 8 layers, hidden size 512, GQA 8/4 with head dimension 64, FFN 1536, RoPE, RMSNorm, SwiGLU, tied embeddings, and context 1024. Architecture code, frozen metadata, and checkpoint shapes agree.

## 4. Formal Pretraining

E01 completed 3,052 BF16 optimizer steps and 50,003,968 processed tokens. Fixed validation loss improved from 10.1964 to 4.6484; PPL improved from 26805.55 to 104.42. Median throughput was 19573 tokens/s and peak allocated VRAM was 1849.0 MiB. This validates trainability and the checkpoint/resume chain, not strong general language or reasoning capability.

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

The pretrained checkpoint SHA256 is `ee306384fcdef646f557cf9aa1165c464b2ca373c6d64d81f5df3b19b8f859fc`; native SFT is `9c33b1c5a781c4ab1d9832a0aad1349516fafb011d24ff6c67be3b1d3885b96e`; tokenizer is `e9578dfd6d2a0f4c3137d7cc223a7b8a782afba8190ae1ad105e1b8ef37e9a22`. Use `portfolio-verify quick` for artifact and metadata verification. Registries link every public claim to a structured artifact.

## 13. Conclusion

The portfolio's strongest result is the auditable end-to-end engineering and scientific workflow: positive evidence, negative evidence, blocked paths, and causal limits are retained together. The appropriate positioning is a reproducible LLM training/evaluation systems project, not a claim of state-of-the-art model quality.
