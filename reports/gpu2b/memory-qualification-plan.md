# GPU-2B memory qualification plan

Status: `PLANNED_NOT_RUN` — this design stage performs no training.

Before any formal arm, run one method-level qualification for `FULL_SFT`, `LORA`, and
`QLORA` at the frozen maximum length 1,024, micro-batch 1, accumulation 16 and the exact
family optimizer. Use `BASE_INIT` for the measured qualification; separately verify that
the CPT checkpoint has identical architecture, tokenizer, parameter shapes and BF16 dtype.
No Base/CPT-specific batch or sequence search is allowed.

Each qualification must cover at least 64 optimizer updates and 1,024 frozen SFT examples.
It must record peak allocated/reserved VRAM, minimum system headroom, median target and input
tokens/s, NaN/Inf/OOM counts, gradient norms, clipping events and package versions.

Pass gates:

- no NaN, Inf or OOM;
- at least 1,536 MiB minimum system-level headroom;
- all gradient norms finite;
- exact QLoRA proof: backbone parameters are 4-bit and frozen, only adapters are trainable;
- trainable-parameter counts match between Base/CPT initializations within each family.

If a family fails, GPU-2B execution is blocked. A revised method configuration requires a
new design-freeze commit; it must change both Base and CPT arms identically.

Historical GPU-0 one-step values are context only, not this qualification: full SFT
5,999/6,032 MiB, LoRA 5,021/5,050 MiB, QLoRA 3,315/3,584 MiB at length 1,024.
