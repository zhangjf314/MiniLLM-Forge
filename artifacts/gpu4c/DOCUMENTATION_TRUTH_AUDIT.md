# Documentation Truth Audit

## Scope

Reviewed `README.md`, `reports/FINAL_REPORT.md`, portfolio/resume evidence, CLI entry points, comments, and committed experiment artifacts for claims involving FlashAttention, torch.compile, Full SFT, LoRA/QLoRA, 50M tokens, 100%, EOS, DPO/GRPO, 1.5B models, DDP, and FSDP.

## Corrections applied

- Native T1 is stated only as **64/64 on a fixed independent test subset**, never full 300-row test accuracy.
- Native generation behavior retains its denominator: **384/384**.
- GPU-3B extraction and external stopping are evaluator/inference improvements, not model-capability improvements.
- GPU-3C is preserved as a controlled negative result; EOS supervision did not produce the expected autonomous stop.
- GPU-4B has no qualified SDPA/Flash/compile performance result. Manual + Eager remains formal.
- Qwen Full SFT code/qualification is distinguished from formal effectiveness evidence; the completed campaign is LoRA/QLoRA.
- Distributed training, DPO, and GRPO are not presented as validated capabilities.

## Outcome

`PASS_WITH_DOCUMENTED_LIMITATIONS`: no critical public claim conflicts with the claim registry after the GPU-4C edits. Historical experiment artifacts were not rewritten.
