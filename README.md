# MiniLLM-Forge

**Transformer Pretraining & LLM Fine-tuning Laboratory**

MiniLLM-Forge is a reproducible training portfolio project for mathematical reasoning. It
contains a decoder-only Transformer implemented directly in PyTorch and a separate
Qwen3-0.6B-Base pipeline for continued pretraining (CPT), full SFT, LoRA, and QLoRA.
The project is an experiment laboratory, not a chat application or inference service.

> MiniLLM uses a deliberately limited corpus to validate the complete Transformer
> training mechanism and controlled architecture/optimization experiments. The E01
> formal run completed 50,003,968 processed tokens with fixed-validation perplexity
> 104.42. This does not claim compute-optimal or fully converged pretraining.

## What Is Implemented

- From-scratch RMSNorm, RoPE, causal attention, GQA/MHA, SwiGLU, Transformer blocks,
  tied embeddings, and next-token label shifting
- 24K byte-level BPE training, normalization, round-trip checks, and token statistics
- Text/JSONL ingestion, fixed-length packing, exact deduplication, n-gram benchmark
  collision auditing, and versioned manifest helpers
- Explicit PyTorch loop with AdamW, warmup/cosine decay, gradient accumulation,
  clipping, FP32/FP16/BF16 AMP, gradient checkpointing, numerical checks, evaluation,
  OOM records, JSONL metrics, and full-state checkpoint/resume
- Qwen Base CPT and assistant-only Full SFT, LoRA, and NF4 double-quantized QLoRA
- Perplexity, mathematical exact match, parameters, throughput, elapsed time, and actual
  peak CUDA memory measurement
- A 15-experiment controlled matrix, automatic CSV registration, fixed held-out math
  evaluation set, failure template, and technical report structure

No benchmark score or GPU metric is fabricated in this repository. Result cells remain
`TBD` until the corresponding pinned run has completed.

## Architecture

```mermaid
flowchart TD
    A[UTF-8 corpus] --> B[24K BPE tokenizer]
    B --> C[Packed causal-LM sequences]
    C --> D[From-scratch MiniLLM]
    D --> E[Pretraining and architecture ablations]

    F[Qwen3-0.6B-Base] --> G[Baseline evaluation]
    G --> H[FineMath CPT]
    G --> I[Direct SFT]
    H --> J[CPT then SFT]
    I --> K[Full / LoRA / QLoRA]
    J --> K

    E --> L[Unified metrics and registry]
    K --> L
    M[Contamination audit] --> L
    L --> N[Technical report]
```

Each MiniLLM block is pre-norm:

```text
x -> RMSNorm -> RoPE GQA causal attention -> residual
  -> RMSNorm -> SwiGLU -> residual
```

The default configuration is approximately 35M-40M parameters: 512 hidden dimensions,
8 layers, 8 query heads, 4 key/value heads, a 1536-wide MLP, and a 24K vocabulary.

## Install

Python 3.10+ is required. Choose exactly one hardware path; uv keeps the environment and
cache inside the repository when `UV_CACHE_DIR` is set as shown.

CPU development and unit tests:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv sync --extra dev --extra cpu
uv run pytest
```

NVIDIA CUDA training (official PyTorch cu128 wheel):

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv sync --extra dev --extra cuda
uv run minillm-qualify-gpu
```

QLoRA selects the same cu128 torch build and adds the qualified bitsandbytes backend:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv sync --extra dev --extra qlora
```

The CPU extra conflicts with CUDA and QLoRA by design. Do not run a hardware workflow
without selecting its explicit extra. `requirements-cpu.lock`, `requirements-cuda.lock`,
and `requirements.lock` (QLoRA) are exported from the same `uv.lock` resolution.

Blackwell RTX 50-series GPUs require a PyTorch CUDA build with CUDA runtime 12.8 or
newer. The locally qualified combination is torch 2.11.0+cu128 on an RTX 5060 Laptop GPU
(8151 MiB, compute capability 12.0) with driver 577.02. This is a measured configuration,
not the only supported configuration. CUDA 13.x was not selected because NVIDIA requires
a 580+ driver for that runtime family.

### GPU-0 Qualification

The bounded qualification commands do not run a formal training campaign:

```powershell
uv run minillm-pretrain --config configs/pretrain/gpu_smoke.yaml
uv run minillm-qualify-qwen --mode base --sequence-length 1024 `
  --output artifacts/environment/qwen_base_1024.json
uv run minillm-qualify-qwen --mode lora --sequence-length 1024 `
  --output artifacts/environment/qwen_lora_1024.json
uv run minillm-qualify-qwen --mode qlora --sequence-length 1024 `
  --output artifacts/environment/qwen_qlora_1024.json
```

Set `HF_HOME` and `HF_HUB_CACHE` under the repository before downloading models when
strict workspace-local caching is required. Full evidence and the 8GB decision are in
`reports/GPU_ENVIRONMENT_QUALIFICATION.md`.

## Phase A: From Scratch

Run the fast CPU-compatible model tests and Tiny Overfit gate:

```powershell
uv run pytest
uv run minillm-pretrain --config configs/pretrain/smoke.yaml
uv run minillm-plot --metrics runs/tiny-overfit-verified/metrics.jsonl `
  --output reports/figures/tiny-overfit.png
```

Train a tokenizer from line-oriented UTF-8 corpus files:

```powershell
uv run minillm-prepare-corpus `
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 `
  --max-documents 100000 `
  --output data/processed/fineweb_edu_train.txt
uv run minillm-tokenizer data/processed/fineweb_edu_train.txt `
  --output artifacts/tokenizers/minillm-tokenizer.json `
  --vocab-size 24000
```

For an exact formal token budget, rerun corpus preparation with `--tokenizer` and
`--target-tokens` after the tokenizer is trained. This writes the pinned data manifest.

The frozen FineWeb-Edu subset, 24K tokenizer, calibration decision, and exact resume
control are recorded in versioned manifests. Run or resume the formal configuration with:

```powershell
uv run --no-sync minillm-pretrain --config configs/pretrain/formal.yaml
uv run --no-sync minillm-pretrain --config configs/pretrain/formal.yaml `
  --resume runs/E01-minillm-formal/last.pt
```

The formal run must not start unless unit tests, Tiny Overfit, and finite-gradient checks
pass. E01 has completed; its measured protocol, limitations, curves, and checkpoint
inventory are in `reports/MINILLM_FORMAL_PRETRAINING.md`. The MHA and no-RoPE controls
remain in `configs/pretrain/` and were not run during GPU-1.

![MiniLLM validation loss](reports/figures/minillm_val_loss.png)

![MiniLLM held-out perplexity](reports/figures/minillm_perplexity.png)

## Phase B: Qwen CPT and SFT

<!-- GPU2A_RESULTS_START -->
### GPU-2A measured CPT result

- Base → CPT math PPL: 5.1715 → 5.0348 (-2.64%)
- Base → CPT general PPL: 17.0799 → 17.2311 (+0.89%)
- CPT input tokens: 10,002,432; peak allocated VRAM: 4406.9 MiB
- Median training throughput: 1972.91 tokens/s
- Classification: `QWEN_MATH_CPT_VALIDATED_WITH_GENERAL_DEGRADATION`

This is domain language-model adaptation evidence, not a claim of improved mathematical
reasoning accuracy.
<!-- GPU2A_RESULTS_END -->

<!-- GPU2D_FORMAL_RESULTS_START -->
### GPU-2D measured downstream PEFT transfer

The formal 512-context campaign completed 12/12 valid training runs and 24/24 valid
evaluation jobs. An independent audit recomputed all 8,400 problem-level records:
6,000 full MATH-500 generations and 2,400 generations on a frozen GSM8K 200-problem
subset, with zero missing or duplicate IDs.

| Method | MATH-500 Base mean | MATH-500 CPT mean | Paired delta | GSM8K fixed-200 Base mean | GSM8K fixed-200 CPT mean | Paired delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 21.67% | 20.67% | -1.00 pp | 30.33% | 25.00% | -5.33 pp |
| QLoRA | 18.40% | 17.13% | -1.27 pp | 21.17% | 22.83% | +1.67 pp |

The scientific classification is `CPT_TO_PEFT_SFT_TRANSFER_METHOD_DEPENDENT`: LoRA was
negative on both benchmarks across all three seeds; QLoRA was negative on MATH-500 but
strongly mixed on the GSM8K fixed-200 subset. This does not support a general claim that
Math-CPT improved downstream reasoning. The result is limited to the frozen 512-context
PEFT-SFT protocol; Full SFT and the original 1024-context study were not completed.

Formal generation also exposed a material limitation: 8,398/8,400 records reached the
512-token ceiling. See `reports/GPU2D_FORMAL_PEFT_TRANSFER.md` for the integrity audit,
paired bootstrap intervals, error analysis, and measured LoRA/QLoRA efficiency.
<!-- GPU2D_FORMAL_RESULTS_END -->

### GPU-2B-D transfer design freeze

`STAGE_GPU_2B_D_COMPLETE` freezes a 2-initialization × 3-adaptation × 3-seed design:
Base/CPT initialization paired within Full SFT, LoRA and QLoRA. The identical 19,200-row
training partition provides 7,024,493 assistant target tokens to every arm. GSM8K and
MATH-500 generated-answer accuracy are primary; frozen math/general PPL are diagnostics.
See `reports/GPU2B_DESIGN.md` for the preregistered rules and execution gates.

This is a zero-training design result. SFT, LoRA and QLoRA formal results remain pending.

Freeze the baseline before any training:

```powershell
uv run minillm-evaluate `
  --model Qwen/Qwen3-0.6B-Base `
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd `
  --mode math `
  --data data/eval/controlled_math.jsonl `
  --dataset-version controlled-math-v1 `
  --output artifacts/eval_manifests/qwen3-base-controlled.json
```

Then run CPT and SFT independently:

```powershell
uv run minillm-prepare-sft `
  --revision e4e141ec9dea9f8326f4d347be56105859b2bd68 `
  --tokenizer-revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd
uv run minillm-experiment --config configs/cpt/qwen3_math.yaml --id E04
uv run minillm-experiment --config configs/sft/qwen3_full.yaml --id E05
uv run minillm-experiment --config configs/sft/qwen3_lora.yaml --id E09
uv sync --extra qlora
uv run minillm-experiment --config configs/sft/qwen3_qlora.yaml --id E11
```

SFT labels are constructed before collation. System, user, and padding tokens receive
`-100`; only assistant tokens contribute to causal loss. The behavior is covered by an
independent test.

## Data Contract

Raw and processed training data are intentionally excluded from version control. Local
SFT JSONL uses this schema:

```json
{"problem": "Solve 2x + 4 = 10.", "solution": "...\nFinal answer: 3"}
```

Before training, pin the upstream revision, filter malformed/incomplete solutions,
deduplicate normalized text, remove any benchmark collisions, and write a manifest.
Audit a prepared set against the held-out set with:

```powershell
uv run minillm-audit `
  --train data/processed/openr1_math_sft.jsonl `
  --benchmark data/eval/controlled_math.jsonl `
  --output artifacts/data_manifests/contamination_report.json
```

Public GSM8K, SVAMP, ASDiv, and MATH-500 scores must be reported separately from the
repository's controlled held-out set because public benchmark contamination cannot be
ruled out from model pretraining history alone.

## Checkpoint and Resume

Each checkpoint contains model, optimizer, scheduler, AMP scaler, global step, epoch,
batches/tokens seen, metric history, Python/NumPy/PyTorch/CUDA RNG states, and the run
configuration. Checkpoints are written atomically.

```powershell
uv run minillm-pretrain `
  --config configs/pretrain/formal.yaml `
  --resume runs/E01-minillm-baseline/step-00000500.pt
```

For strict continuous-vs-resumed comparisons, keep loader ordering, seed, hardware,
software lock, and dataset manifest identical.

## Experiment Protocol

The bounded matrix is in `experiments/matrix.yaml`. Each run must state a hypothesis,
controlled variables, one independent variable, metrics, interpretation, and one of
`supported`, `partially-supported`, or `rejected`. Runs append environment and config
metadata to `experiments/registry.csv`.

The central comparisons are GQA/MHA, RoPE/no-RoPE, learning rate, warmup, CPT/direct
SFT/CPT-to-SFT, LoRA rank 4/8/16/32, adapter placement, LoRA/QLoRA, and repeated seeds.

| Model | Method | Trainable params | Peak allocated VRAM | Quality metric | Tokens/s | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen Base | frozen PPL baseline | - | 2148 MiB | Math PPL 5.17 | - | E00 frozen |
| MiniLLM | from-scratch pretrain | 37,462,528 | 1,849 MiB | fixed-val PPL 104.42 | 19,573 | E01 completed |
| Qwen | full CPT | 596,049,920 | 4407 MiB | Math PPL 5.03 | 1973 | E04 completed |
| Qwen | LoRA SFT | 10,092,544 | 3428-3686 MiB physical | MATH-500 20.67-21.67% means | 735 target tok/s | 6/6 valid |
| Qwen | QLoRA SFT | 10,092,544 | 5792-6116 MiB physical | MATH-500 17.13-18.40% means | 588 target tok/s | 6/6 valid |

## Repository Map

```text
configs/                 model and controlled-run YAML
data/eval/               immutable held-out examples
src/minillm_forge/model  hand-written Transformer
src/minillm_forge/data   packing, SFT masking, governance
src/minillm_forge/training explicit trainer and checkpointing
src/minillm_forge/finetuning Full SFT, LoRA, QLoRA
src/minillm_forge/evaluation PPL, exact match, efficiency
scripts/                 direct Python wrappers
tests/                   architecture and training gates
experiments/             matrix, registry, measured outputs
reports/                 ablations, failures, final report
artifacts/               data/evaluation manifests
```

## Definition of Done

Code-complete means the paths and tests exist. Portfolio-complete additionally requires
real training evidence. The current repository deliberately distinguishes them:

| Gate | Evidence | Current state |
| --- | --- | --- |
| Transformer unit tests | `tests/` | implemented; see latest test run |
| Tiny overfit | `runs/tiny-overfit-verified/summary.json` | passed locally; see generated summary |
| Checkpoint resume | exact control plus pilot interruption | exact resume confirmed |
| MiniLLM formal pretraining | E01 result, metrics, curves, checkpoints | validated at 50,003,968 tokens |
| Qwen baseline and CPT | E00/E04 manifests, reports, curves | validated; domain adaptation does not imply reasoning gain |
| GPU-2D formal PEFT training | training matrix and 12 run summaries | 12/12 valid at 512 context |
| GPU-2D formal evaluation | final result plus raw local generations | 24/24 jobs and 8400/8400 records validated |
| 10+ controlled experiments | registry plus per-run result JSON | completed formal runs registered |
| CPT contamination audit | pinned JSON report | completed for local train/validation/benchmark probes |
| Final technical report | `reports/FINAL_REPORT.md` | GPU-1, GPU-2A, and GPU-2D evidence integrated |

See `reports/FINAL_REPORT.md` for assumptions, expected evidence, and limitations.
