# MiniLLM-Forge

**A reproducible LLM training, evaluation, and evidence-audit portfolio.**

MiniLLM-Forge implements a decoder-only Transformer directly in PyTorch and follows it
through tokenizer training, BF16 pretraining, exact checkpoint resume, full-parameter SFT,
generation evaluation, and numerical diagnostics. A separate Qwen3-0.6B study covers CPT,
LoRA, QLoRA, benchmark evaluation, answer extraction, and stopping behavior.

This is a research and engineering portfolio, not a production-grade general LLM, chat
product, or claim of state-of-the-art model quality. Positive, negative, blocked, and
unvalidated results are all retained. The release classification is
`GPU4C_READY_WITH_DOCUMENTED_LIMITATIONS`.

## Key Results

### Native Transformer

- 37,462,528 parameters; 8 decoder layers; hidden size 512
- GQA with 8 query heads / 4 key-value heads; head dimension 64
- FFN size 1536; RoPE, RMSNorm, SwiGLU, and tied embeddings
- Context length 1024 and a native 24K byte-level BPE tokenizer

### Formal Pretraining

- BF16 training for 3,052 optimizer steps and 50,003,968 processed tokens
- Fixed-validation loss: 10.1964 -> 4.6484
- Fixed-validation perplexity: 26,805.55 -> 104.42
- Exact model, optimizer, scheduler, and RNG-state resume verified

The token count represents 1.554 passes over a deliberately limited corpus, not 50M
unique tokens and not a claim of a fully converged general-purpose model.

### Native Full SFT

- All 37,462,528 parameters trained for 300 BF16 response-only steps
- Validation loss: 5.9922 -> 0.5943
- T1 support classification, fixed independent 64-example test subset: 0/64 to 64/64
- T2 two-digit addition, fixed independent 64-example test subset: 0/64 to 2/64
- 384/384 audited generations were complete, format-valid, EOS-terminated,
  non-repetitive, and not length-limited

The T1 result is not a claim of 100% accuracy on the full 300-row test split. T2 remained
weak and is preserved as a negative result.

## PEFT Study

The Qwen3-0.6B campaign completed 12 LoRA/QLoRA runs: Base and Math-CPT initializations,
both adapter methods, three seeds, and 1,200 steps per run. Evaluation covered full
MATH-500 and an outcome-blind fixed 200-example GSM8K subset, totaling 8,400 audited
outputs.

| Method | MATH-500 Base | MATH-500 CPT | Paired delta | GSM8K fixed-200 Base | GSM8K fixed-200 CPT | Paired delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 21.67% | 20.67% | -1.00 pp | 30.33% | 25.00% | -5.33 pp |
| QLoRA | 18.40% | 17.13% | -1.27 pp | 21.17% | 22.83% | +1.67 pp |

Math-CPT transfer was negative for LoRA and mixed/negative for QLoRA under this frozen
protocol. The study does not establish a general reasoning improvement. Qwen
full-parameter SFT effectiveness was not formally validated.

## Evaluation and Generation Diagnostics

- 8,398/8,400 formal PEFT outputs reached the 512-token generation ceiling.
- On the same frozen outputs, legacy extraction counted 1,764 correct; first-valid
  extraction counted 2,008, last-valid counted 1,824, and conflict-aware counted 1,793.
- All 36 legacy extraction failures were recovered, including 16 correct answers.
- The audit exposed 607 conflicting-answer outputs.

These are evaluator/extraction protocol findings, not model-training accuracy gains.
External answer-boundary stopping reduced redundant generation in a bounded confirmation
set, but did not improve autonomous EOS learning.

## Numerical Study

- SDPA Math passed the FP32 reference gate.
- SDPA Math and Auto failed the frozen BF16 correctness gates.
- R1 localized the earliest public BF16 divergence to the first Attention x V context
  boundary after bit-identical Q/K/V and RoPE values.
- FP32-intermediate reconstruction closely matched SDPA Math, supporting a numerical
  precision-path mechanism; no semantic Attention defect was found.
- Built-in FlashAttention was unavailable, and `torch.compile` was blocked by missing
  Triton. No acceleration claim or qualified performance result is made.

Manual Attention with eager execution remains the formal native-model default.
The repository makes no Flash, compile, SDPA speedup claim.

## Quick Start

Python 3.10+ and [uv](https://docs.astral.sh/uv/) are required. This repository has not
been published to PyPI; install from the checkout using the real project metadata.

```powershell
git clone https://github.com/zhangjf314/MiniLLM-Forge.git
Set-Location MiniLLM-Forge
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv sync --extra dev --extra cpu
uv run portfolio-verify quick
```

The quick verifier is bounded and offline after installation: it downloads no model,
starts no training, requires no API key, and verifies 29 evidence, hash, schema, and truth
boundary checks. If the environment was created before the command entry point was added,
the equivalent no-resync invocation is:

```powershell
uv run --no-sync python -m minillm_forge.cli.portfolio_verify quick
```

A low-cost configuration inspection requires no checkpoint or dataset:

```powershell
uv run --no-sync python -c "from minillm_forge.model.config import MiniLLMConfig; print(MiniLLMConfig())"
```

CUDA and QLoRA environments use mutually exclusive extras:

```powershell
uv sync --extra dev --extra cuda
uv sync --extra dev --extra qlora
```

## Evidence Navigation

- [Final evidence manifest](artifacts/gpu4c/final_evidence_manifest.json)
- [Claim Registry](artifacts/gpu4c/CLAIM_REGISTRY.md)
- [Capability Matrix](artifacts/gpu4c/CAPABILITY_MATRIX.md)
- [Experiment Registry](artifacts/gpu4c/EXPERIMENT_REGISTRY.md)
- [Portfolio verification](artifacts/gpu4c/PORTFOLIO_VERIFICATION.md)
- [Documentation truth audit](artifacts/gpu4c/DOCUMENTATION_TRUTH_AUDIT.md)
- [Final technical report](artifacts/gpu4c/MINILLM_FORGE_FINAL_TECHNICAL_REPORT.md)
- [Resume facts](artifacts/gpu4c/RESUME_FACTS.md)
- [Interview facts](artifacts/gpu4c/INTERVIEW_FACTS.md)
- [Public distribution boundaries](PUBLIC_RELEASE.md)
- [v2.0.0 portfolio release notes](RELEASE_NOTES_v2.0.0-portfolio.md)

## Data and Artifact Policy

The native SFT tasks are deterministic synthetic tasks generated by repository code.
FineWeb-Edu, FineMath, OpenR1-Math-220k, MATH-500, and GSM8K are third-party sources or
benchmarks pinned to upstream revisions and governed by their publishers' terms. The
repository does not redistribute raw or processed training corpora, Qwen weights, or full
benchmark datasets. The small repository-owned controlled-math fixture and compact
evidence required to audit reported results are tracked.

Formal checkpoints are intentionally local and untracked. Their SHA256 digests are
published for verification; they are not GitHub Release assets. See
[`PUBLIC_RELEASE.md`](PUBLIC_RELEASE.md) for exact source, license, and redistribution
boundaries.

## Known Limitations

- The native model is small and undertrained by modern LLM standards.
- Native task scores use fixed 64-example subsets selected independently from the test
  splits.
- T2 arithmetic remained weak at 2/64.
- PEFT findings are limited to three seeds and the frozen 512-context protocol.
- The EOS ablation did not validate autonomous-stop improvement.
- Qwen full-parameter SFT effectiveness was not formally validated.
- FlashAttention was unavailable; `torch.compile` was blocked by missing Triton.
- No qualified SDPA/Flash/compile performance comparison exists.
- Distributed DDP/FSDP training, DPO, and GRPO were not executed.
- Hardware evidence comes from one RTX 5060 Laptop GPU on Windows.
- Public benchmark contamination from the base model's pretraining history cannot be
  excluded.

## Repository Map

```text
configs/                 frozen model and experiment configurations
src/minillm_forge/       model, data, training, evaluation, and audit code
tests/                   unit, regression, and release-evidence gates
experiments/             experiment matrix and registry
reports/                 historical and final reports
artifacts/gpu4c/         canonical claim/evidence/verification system
artifacts/release_v2/    public release audit and machine-readable result
```

Source code is released under the root [MIT license](LICENSE). Third-party assets retain
their original licenses and terms.
