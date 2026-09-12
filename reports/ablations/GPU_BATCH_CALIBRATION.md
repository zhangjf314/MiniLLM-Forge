# GPU-1 batch calibration (hardware tuning, not a model-quality ablation)

Same 37,462,528-parameter model, 1024 tokens, BF16, gradient checkpointing, AdamW LR
3e-4, seed 42, and 16 sequences/update. Each candidate ran 8 updates (131,072 real
corpus input tokens); the first two steps were excluded from timing medians. The total
calibration budget was 393,216 tokens. No synthetic data and no LR search were used.

| Micro batch | Accumulation | Allocated peak MiB | Reserved peak MiB | Median tokens/s | Step seconds | Measured free/total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 8 | 1226.17 | 1580 | 18,254 | 0.898 | 65.39% |
| 4 | 4 | 1848.08 | 2138 | 19,795 | 0.828 | 58.55% |
| 8 | 2 | 3042.06 | 3990 | 19,972 | 0.820 | 35.83% |

Selected **micro batch 4, accumulation 4**. Batch 8 gained only 0.9% (within short-run
timing noise) while nearly doubling reserved memory. Batch 16 was not tested: throughput
had plateaued, and more memory pressure offered little demonstrated benefit. All tested
candidates exceeded the required 15% free VRAM margin including runtime/desktop usage.
GPU temperatures after these short tests were 55–57 C; no power, fan, or clock controls
were changed. Baseline LR remained 3e-4; no NaN, Inf, or OOM occurred.

The calibration loader used the frozen token streams identified by `data_identity` in
each JSON artifact. The legacy text paths in the captured parent config were not used;
the formal config now explicitly names these token streams.

Effective batch: 4 × 4 × 1 = 16 sequences; input tokens/update = 16 × 1024 = 16,384;
supervised next-token targets/update = 16 × 1023 = 16,368.
