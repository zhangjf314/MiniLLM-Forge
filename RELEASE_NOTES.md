# MiniLLM-Forge v1.0.0 Portfolio Release

MiniLLM-Forge is a reproducible training and scientific-audit portfolio for a hand-written
decoder-only Transformer and Qwen3-0.6B mathematical-domain adaptation.

## Main Capabilities

- From-scratch PyTorch Transformer with RMSNorm, RoPE, GQA/MHA, SwiGLU, tied embeddings,
  causal masking, and an explicit next-token training loop
- Formal 37,462,528-parameter MiniLLM pretraining with checkpoint/resume and data controls
- Qwen3-0.6B-Base mathematical-domain continued pretraining (CPT)
- Full SFT, LoRA, and QLoRA implementation paths, with LoRA/QLoRA used in the formal study
- Experiment registration, resource qualification, failure records, automated generative
  evaluation, and independent result recomputation

## Formal Experiment Scope

- 12/12 valid formal training runs
- 24/24 valid benchmark jobs
- 8,400/8,400 valid problem-level generation records
- 6,000 records from full MATH-500 and 2,400 records from a frozen GSM8K 200-problem subset

Under the frozen 512-context Base/CPT-by-LoRA/QLoRA protocol, Math-CPT did not produce a
stable positive downstream mathematical-task benefit. LoRA was negative on both benchmarks
for all three seeds; QLoRA was negative on MATH-500 and mixed on the GSM8K fixed-200 subset.

## Resources and Reproduction Boundary

The formal campaign ran on an 8 GB RTX 5060 Laptop GPU. The 12 training runs accumulated
27.28 run-hours; the 24 evaluation jobs accumulated 80.13 job-hours. Full SFT was
resource-qualified only and was not completed as a formal comparison. GSM8K results are
fixed-200 subset accuracy, not full-test accuracy.

This release contains code, configurations, tests, reports, manifests, plots, and compact
structured evidence. It contains no base-model weights, trained checkpoints/adapters, raw
datasets, full generation JSONL files, or machine-local logs. External models and datasets
must be obtained from their publishers under their own terms. See `PUBLIC_RELEASE.md` for
pinned sources, licenses, hardware requirements, and exact reproduction boundaries.
