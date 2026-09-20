# Formal Full SFT training report

Configuration: formal 50M-token checkpoint; Full SFT; AdamW with LR `1e-4`, betas `0.9/0.95`, weight decay `0.01`; BF16; micro-batch 16; gradient accumulation 2; effective batch 32; 300 steps; validation and checkpointing every 50 steps.

| Step | Validation loss | Target-token accuracy |
|---:|---:|---:|
| 50 | 1.0087 | 73.9% |
| 100 | 0.7872 | 80.1% |
| 150 | 0.7007 | 81.0% |
| 200 | 0.6571 | 81.5% |
| 250 | 0.6179 | 82.2% |
| 300 | 0.5943 | 82.9% |

The lowest validation-loss rule selected step 300 without consulting test data. Training processed 9,600 examples, 461,254 input tokens, and 32,174 supervised tokens. Final train loss was 0.5784.

Training took 48.9 seconds. Median throughput was 12188 input tokens/s and 850 supervised tokens/s. Peak CUDA allocated/reserved memory was 1003.6/1318.0 MiB. No OOM, NaN, or Inf occurred. Reloading the selected checkpoint reproduced the recorded validation metrics exactly.
