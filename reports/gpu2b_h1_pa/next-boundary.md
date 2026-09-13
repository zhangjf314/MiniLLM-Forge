# Next boundary

The only work authorized by this audit is a new zero-training recovery decision:

```text
STAGE_GPU_2B_RD2_DESIGN_ONLY
```

RD2 may ask whether the project should select `CONFIGURATION_REVISION` now that the
hardware-preserving branch is unavailable. It must not launch training.

If RD2 selects configuration revision, it may authorize a separate exploratory
`STAGE_GPU_2B_MC8`. MC8 should prospectively freeze a bounded first-hit sequence of
Full-SFT-only memory candidates: first implementation-level reduction at context 1,024,
then bounded optimizer/state offload at context 1,024, and only if those fail a shorter-
context candidate. MC8 may run no CPT arm, benchmark, or transfer analysis.

Only after MC8 selects a memory-feasible candidate may `STAGE_GPU_2B_D2` freeze a revised
scientific execution contract. Neither D2 nor revised training is authorized by H1-PA.
