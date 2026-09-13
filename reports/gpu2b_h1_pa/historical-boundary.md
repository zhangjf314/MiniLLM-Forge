# Historical boundary

The following results remain immutable:

```text
STAGE_GPU_2B_RD_COMPLETE
recovery_mode = HARDWARE_MIGRATION

STAGE_GPU_2B_BLOCKED_BY_MEMORY_QUALIFICATION
```

The historical Full-SFT qualification completed 64/64 updates with zero NaN, Inf, and OOM
events. Maximum observed physical occupancy was 7,776 MiB, minimum physical headroom was
35 MiB, and the frozen requirement was 1,536 MiB. Formal training launches were zero.

This availability audit neither changes the RD decision to configuration revision nor
relabels GPU-2B. The 8 GB qualification executed; it failed an engineering safety-headroom
contract. No downstream-transfer endpoint was measured.
