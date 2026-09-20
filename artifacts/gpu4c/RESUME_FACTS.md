# Resume Facts

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
