# Stage GPU-2B execution result

## Classification

```text
STAGE_GPU_2B_BLOCKED_BY_MEMORY_QUALIFICATION
```

GPU-2B stopped at `GPU2B-Q1`. No formal SFT training run, benchmark generation, PPL
evaluation, or transfer analysis was launched.

## Q0 preflight

`GPU2B-Q0` passed on commit `52cb008ce6e56aa4001fc1c9460275520c154ecc`:

- pytest: 40 passed;
- Ruff and formatting: passed;
- `uv lock --check` and build: passed;
- all 29 version-controlled YAML files parsed;
- all sealed design-artifact checksums matched;
- the 20,000 / 19,200 / 800 frozen dataset identities matched;
- all three frozen seed-permutation digests matched;
- exact token recount matched 9,127,756 input tokens and 7,024,493 assistant targets;
- known remaining contamination conflicts: zero.

Formal training remained unauthorized and the launch count remained zero.

## Q1 Full-SFT memory qualification

`FULL_SFT_Q64` used the frozen Base initialization, length 1,024, micro-batch 1,
accumulation 16, BF16, full trainability, and AdamW8bit configuration. The final measured
attempt completed all 64 optimizer updates over 1,024 frozen examples:

| Metric | Result |
| --- | ---: |
| Input tokens | 490,855 |
| Assistant target tokens | 378,911 |
| Peak allocated VRAM | 5,647.37 MiB |
| Maximum observed physical VRAM used | 7,776 MiB |
| Minimum observed physical headroom | **35 MiB** |
| Frozen minimum-headroom requirement | **1,536 MiB** |
| Physical-memory samples | 1,423 |
| Median assistant tokens/s | 1,093.25 |
| Gradient norm range | 3.609375 to 19.75 |
| NaN / Inf / OOM | 0 / 0 / 0 |

The run was numerically finite and did not OOM, but it failed the frozen system-headroom
gate by 1,501 MiB. The independent physical-memory monitor is authoritative for this gate.
Raw CUDA allocator counters are retained in the JSON evidence because Windows WDDM
reported virtual reservations above physical capacity and therefore could not serve as the
system-headroom measurement.

Two earlier attempts are retained in the qualification JSON. The first established severe
allocator growth. A pre-formal implementation change released unused CUDA cache blocks at
optimizer boundaries without changing any scientific or training hyperparameter. The
second exposed WDDM allocator-counter incompatibility. The final attempt added independent
0.2-second `nvidia-smi` sampling; it did not alter training and produced the blocking
physical-headroom result above.

## Stop discipline

The sealed memory plan requires at least 1,536 MiB of minimum system-level headroom. It
forbids reducing sequence length, changing micro-batch or accumulation, changing optimizer
or precision, lowering LoRA rank, or disabling required modules under the same stage
identity. Therefore the remaining Q1 families, Q2 resume controls, all 18 formal runs, and
all evaluations are `NOT_RUN_BLOCKED_BY_Q1`.

This is an engineering qualification result, not evidence for or against CPT-to-SFT
transfer. The predecessor conclusion remains unchanged:

```text
QWEN_MATH_CPT_VALIDATED_WITH_GENERAL_DEGRADATION
MATH_DOMAIN_ADAPTATION = CONFIRMED
MATH_REASONING_IMPROVEMENT = NOT_ESTABLISHED
```

Continuing GPU-2B would require a new design revision or hardware with sufficient frozen-
configuration headroom. It must not reuse the current stage identity while silently
changing the frozen configuration.
