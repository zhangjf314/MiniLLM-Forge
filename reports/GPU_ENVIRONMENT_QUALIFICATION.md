# GPU Environment Qualification

## 1. Initial Failure

The repository environment used `torch 2.14.0+cpu`. `torch.version.cuda` was `None`,
CUDA availability was false, and no CUDA device was visible to PyTorch despite working
NVIDIA hardware, driver, and toolkit installations.

## 2. Root Cause

The project declared only `torch>=2.5,<3` from the default PyPI index. On Windows, uv
therefore selected the CPU wheel. The QLoRA extra added bitsandbytes but did not select a
CUDA torch source. The old lock reproduced that CPU-only resolution exactly.

## 3. Hardware

| Item | Measured value |
| --- | --- |
| OS | Windows 10 build 26100 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| VRAM | 8150.5625 MiB reported by PyTorch; 8151 MiB by nvidia-smi |
| Compute capability | 12.0 (`sm_120`, Blackwell) |
| Driver | 577.02 |
| CUDA Toolkit | 12.9, nvcc V12.9.41 |

## 4. Previous PyTorch Environment

Python was 3.11.15 inside `.venv`; torch was `2.14.0+cpu`. The system Python remained
unchanged and had no torch package.

## 5. Selected CUDA/PyTorch Combination

The qualified environment uses Python 3.11.15, `torch 2.11.0+cu128`, PyTorch CUDA
runtime 12.8, Transformers 5.16.1, PEFT 0.20.0, and Accelerate 1.14.0. QLoRA adds
bitsandbytes 0.50.2.

## 6. Why This Combination

PyTorch publishes an official Windows CPython 3.11 cu128 wheel for 2.11.0. Its compiled
architecture list includes `sm_120`. NVIDIA's compatibility table permits CUDA 12.x on
drivers 525 through the pre-580 range, which includes 577.02. CUDA 13.x requires driver
580 or newer, so cu130 was deliberately rejected. Sources:

- https://pytorch.org/get-started/previous-versions/
- https://download.pytorch.org/whl/cu128/torch/
- https://docs.astral.sh/uv/guides/integration/pytorch/
- https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html

## 7. Dependency Changes

The base package no longer chooses an implicit torch wheel. Mutually exclusive `cpu` and
`cuda` extras route torch through explicit official indexes. The `qlora` extra selects
the same cu128 torch and pins the tested Windows bitsandbytes release. `uv.lock` captures
all paths. No driver, Toolkit, system Python, Conda, or global package was modified.

## 8. CUDA Kernel Qualification

CUDA allocation, 1024 by 1024 matmul, an MLP forward/backward/AdamW step, and CUDA memory
accounting all passed. Kernel-test peak was 35.9927 MiB allocated and 46 MiB reserved.
This proves executable `sm_120` kernels, not only successful device enumeration.

## 9. BF16 Qualification

`torch.cuda.is_bf16_supported()` returned true. A real BF16-autocast forward, backward,
and optimizer step completed with finite loss.

## 10. MiniLLM GPU Smoke

The default 37,462,528-parameter MiniLLM completed BF16 training, accumulation, clipping,
scheduler steps, evaluation, checkpointing, and metric logging at sequence lengths 512
and 1024. Peaks were 809.7861/874 MiB at 512 resume and 927.4097/1058 MiB at 1024
(allocated/reserved). Step-2 resume continued to step 4 and reproduced the continuous
final loss of 9.1051859856.

## 11. Qwen Base Smoke

The pinned Qwen3-0.6B-Base revision loaded 596,049,920 parameters on CUDA. A real forward
and four-token generation passed at lengths 512 and 1024. Peak memory was
1664.3354/1770 MiB and 2183.6563/2358 MiB respectively.

## 12. LoRA Smoke

Rank-4 adapters on q/v projections exposed 573,440 trainable parameters (0.0961%). One
assistant-only SFT forward/backward/AdamW step passed at both lengths. Peaks were
3096.6646/3122 MiB at 512 and 5020.5479/5050 MiB at 1024.

## 13. QLoRA Smoke

bitsandbytes loaded its native CUDA backend and detected CUDA 12.8 and capability 12.0.
NF4, double quantization, BF16 compute, PEFT attachment, forward, backward, and optimizer
step all passed. Peaks were 2215.7866/2248 MiB at 512 and 3315.1758/3584 MiB at 1024.

## 14. Peak VRAM

| Path | Seq | Peak allocated MiB | Peak reserved MiB | Result |
| --- | ---: | ---: | ---: | --- |
| MiniLLM train/resume | 512 | 809.7861 | 874 | PASS |
| MiniLLM train | 1024 | 927.4097 | 1058 | PASS |
| Qwen evaluation | 512 | 1664.3354 | 1770 | PASS |
| Qwen evaluation | 1024 | 2183.6563 | 2358 | PASS |
| Qwen LoRA train | 512 | 3096.6646 | 3122 | PASS |
| Qwen LoRA train | 1024 | 5020.5479 | 5050 | PASS |
| Qwen QLoRA train | 512 | 2215.7866 | 2248 | PASS |
| Qwen QLoRA train | 1024 | 3315.1758 | 3584 | PASS |
| Qwen full CPT train | 1024 | 5999.2119 | 6032 | PASS, low headroom |
| Qwen full SFT train | 1024 | 5999.2119 | 6032 | PASS, low headroom |

## 15. Full CPT/SFT Feasibility

Both full CPT and full SFT completed one real full-parameter optimizer step at lengths
512 and 1024. At 1024, only about 878 MiB of system-level GPU memory remained at the end
of measurement. They are single-step feasible, but not qualified as safe long-running
8GB defaults. CPU offload and swap were not used.

## 16. Regression

Final regression passed: 26 pytest tests, zero failures, Ruff pass, format pass, 25 YAML
files parsed, uv lock check pass, and wheel/source-distribution build pass.

## 17. Remaining Limitations

- Full CPT/SFT long-run stability was not tested and has less than 1 GiB measured margin.
- Smoke inputs are deliberately tiny and do not establish model quality.
- The upstream `torch.utils.collect_env` command exits on this Chinese Windows locale due
  to an OEM Unicode decode error. Its failure is preserved, followed by output from the
  same collector with decode errors replaced.
- No formal training or benchmark campaign ran in this stage.

## 18. Final Classification

Primary: `CUDA_GPU_QUALIFIED`. Additional: `QLORA_GPU_QUALIFIED`.

## 19. Recommended Next Stage

Use QLoRA at micro-batch 1 and sequence length 1024 as the preferred 8GB formal path.
LoRA 512 is the conservative non-quantized alternative. MiniLLM formal pretraining is
ready with micro-batch 1 and accumulation. Keep full CPT/SFT optional until a longer
stability run confirms that the narrow memory margin is operationally acceptable.
