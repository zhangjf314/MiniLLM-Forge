# Recovery decision

## Selected branch

```text
RECOVERY_MODE = HARDWARE_MIGRATION
FUTURE_STAGE = STAGE_GPU_2B_H1
```

Branch H is selected prospectively because it preserves every scientific and training
configuration from GPU-2B-D while changing only the physical execution environment.

The following remain unchanged: sequence length 1,024; micro-batch 1; accumulation 16;
BF16; AdamW8bit Full SFT; frozen LoRA and QLoRA configurations; 19,200 training examples;
7,024,493 assistant targets; 1,200 updates; prompts; generation; evaluation; and decision
rules.

Adequate Branch-H hardware has not yet been frozen or qualified. Selection of migration is
not a claim that a particular device is available or will pass. A single environment with
at least 12 GiB physical VRAM is the preferred provisioning target.

Branch C is not selected and no `STAGE_GPU_2B_D2` is authorized. It may be reconsidered
only in a new recovery decision after adequate Branch-H hardware is prospectively recorded
as unavailable; there is no automatic H-to-C fallthrough.
