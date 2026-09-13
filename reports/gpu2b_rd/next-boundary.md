# Next boundary

This completed RD decision authorizes creation of `STAGE_GPU_2B_H1`, but it authorizes no
H1 training by itself.

H1 is a fresh execution identity. Historical qualification attempts count as context only.
It must rerun, in order:

1. repository and evidence Q0;
2. Full-SFT, LoRA, and QLoRA Q64 memory qualifications;
3. Full-SFT, LoRA, and QLoRA resume qualifications.

Every Q64 must retain the 1,536 MiB physical-headroom rule and finish 64/64 updates with
zero NaN, Inf, and OOM events. Physical monitoring and framework allocator counters must
both be recorded.

Only seven PASS results—Q0, all three Q1 families, and all three Q2 families—authorize the
unchanged 18 formal runs. H1 must then use the original GPU-2B-D data, seeds, training
budgets, generation protocol, evaluation protocol, and decision rules.

`STAGE_GPU_2B_D2` is not authorized by this Branch-H completion. It requires a separate
Branch-C recovery decision after hardware unavailability is recorded.
