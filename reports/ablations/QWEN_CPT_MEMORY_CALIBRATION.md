# Qwen CPT Memory Calibration

## Decision

**CALIBRATION_PASS**

The frozen long-run candidate is sequence length 512, micro-batch 1, gradient
accumulation 8, BF16, gradient checkpointing, and full-parameter 8-bit AdamW. Quantizing
optimizer state reduces memory without quantizing model weights or freezing parameters.
Earlier FP32/BF16-state AdamW attempts remain in the JSON evidence.

## Measured calibration

- Optimizer steps: 8
- Input tokens: 32768
- Peak allocated: 4406.92 MiB
- Peak reserved: 4536.00 MiB
- Estimated minimum system headroom: 2410.00 MiB
- Median throughput: 2004.60 tokens/s
- NaN / Inf / OOM: 0 / 0 / 0

The gate target is at least 1536 MiB practical system-level headroom. A short calibration
does not itself establish long-run stability; the 503,808-token qualification is separate.
