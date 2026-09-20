# Final optimization comparison

| Configuration | Correctness | Performance evidence | Recommendation |
|---|---|---|---|
| Manual Eager | GPU-4A full baseline retained | 48.9s formal SFT; 12,188 input tokens/s | Default |
| SDPA Math Eager | FP32 passed; frozen BF16/gradient/update gate failed | Not benchmarked after gate failure | Do not enable |
| SDPA Auto Eager | Efficient Attention dispatched; frozen BF16/gradient/update gate failed | Not benchmarked after gate failure | Do not enable |
| SDPA Flash Eager | Kernel unavailable | None | Not available |
| Manual + compile | Inductor failed with TritonMissing | No compiled steady state | Not available |
| SDPA + compile | Preconditions not met | None | Not run |

Speedups are intentionally not reported: adding percentages from missing or correctness-ineligible runs would be invalid. The current recommended configuration remains Manual Eager.
