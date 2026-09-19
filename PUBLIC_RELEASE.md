# Public Release and Reproducibility Guide

This repository publishes the MiniLLM-Forge source code, frozen configurations, tests,
reports, plots, manifests, and compact structured evidence. It does not publish a trained
model package. The release is intended for code review, scientific audit, and bounded
reproduction—not as a zero-download reproduction bundle for the full campaign.

## What Runs Directly

After installing one explicit dependency extra from `README.md`, a source checkout can
run the unit/regression suite, lint and format checks, YAML validation, dependency-lock
validation, package builds, and the CPU-compatible smoke workflows. These checks do not
repeat the formal training campaign.

The model implementation, training and evaluation entry points, data-governance tools,
frozen YAML configurations, experiment registry, and report-generation/audit code are
included. The committed `artifacts/gpu2d_formal/final_result.json` is the canonical compact
result for the formal transfer study.

## External Assets

The following assets must be obtained from their publishers at the pinned revisions. Their
licenses and terms remain those of their respective publishers.

| Asset | Use | Pinned revision | Published license/status |
| --- | --- | --- | --- |
| [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd) | Base model and tokenizer | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | Apache-2.0 |
| [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9) | MiniLLM pretraining/general validation source | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-By; source content is also subject to Common Crawl terms |
| [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath/tree/e92b25a616738fe95dc186b64dfb19f9c8525594) | Math CPT source | `e92b25a616738fe95dc186b64dfb19f9c8525594` | ODC-By |
| [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/tree/e4e141ec9dea9f8326f4d347be56105859b2bd68) | SFT source | `e4e141ec9dea9f8326f4d347be56105859b2bd68` | Apache-2.0 |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866) | Secondary evaluation source | `740312add88f781978c0658806c59bc2815b9866` | MIT |
| [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be) | Primary evaluation source | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | The pinned snapshot does not declare a license; its stated upstream [`openai/prm800k`](https://github.com/openai/prm800k) repository is MIT |

Raw dataset files are not redistributed here. Compact qualification evidence does retain
prompt token IDs, hashes, reference answers, and generated outputs needed for the audit;
it is not a substitute for the source benchmarks. In particular, users must obtain
MATH-500 from the publisher and independently confirm that their intended use is allowed;
this project does not infer a license for the Hugging Face snapshot from its upstream.
The tracked MiniLLM tokenizer is a derived training artifact; FineWeb-Edu attribution and
source terms still apply.

## Intentionally Not Distributed

- Qwen base-model weights and tokenizer cache
- E01 MiniLLM, E04 CPT, LoRA, QLoRA, and qualification checkpoints/adapters
- raw or processed training and evaluation datasets
- full per-problem generation JSONL files and machine-local execution logs
- dependency, Hugging Face, and build caches

These items remain local and are ignored by Git. The final structured evidence records
relative paths, sizes, and cryptographic digests for the local checkpoints and formal
outputs, but those records are not downloadable model artifacts.

## Formal Campaign Boundary

The completed downstream study used one 8 GB RTX 5060 Laptop GPU, a frozen 512-token
context protocol, Base/CPT initializations, LoRA/QLoRA, and three seeds. It contains 12
valid training runs (27.28 aggregate run-hours) and 24 valid benchmark jobs (80.13
aggregate job-hours): full MATH-500 for each run and an outcome-blind fixed 200-problem
GSM8K subset for each run. The 8,400 problem records were audited with zero missing or
duplicate IDs.

Full SFT was resource-qualified in a bounded single-step test but was not run as a formal
comparison. The study does not establish a full-GSM8K score, 1024-context transfer, or a
general QLoRA memory advantage. Math-CPT did not provide stable positive downstream
transfer under the frozen protocol.

## Recomputing the Final Result

`scripts/finalize_gpu2d.py` verifies the complete local evidence package and recomputes the
formal aggregate. A fresh public checkout cannot run that full recomputation because the
large checkpoints, raw generations, and machine-local supervisor records are deliberately
excluded. It can inspect the committed aggregate, run its regression tests, and review the
hashes and integrity findings in the formal report. Repeating the entire campaign requires
downloading the external assets, recreating the frozen datasets, and paying the stated GPU
time cost; it is not part of a normal installation or release smoke test.

## Canonical Evidence

- `reports/FINAL_REPORT.md`
- `reports/GPU2D_FORMAL_PEFT_TRANSFER.md`
- `reports/MINILLM_FORMAL_PRETRAINING.md`
- `reports/QWEN_MATH_CPT.md`
- `reports/PORTFOLIO_RESUME_EVIDENCE.md`
- `artifacts/gpu2d_formal/final_result.json`
- `artifacts/data_manifests/`
- `artifacts/eval_manifests/`

Repository source code is released under the root `LICENSE`. Third-party assets are not
relicensed by MiniLLM-Forge.
