# Numerical path audit

## Confirmed from production code and public runtime boundaries

- Both paths use identical bias-free Q/K/V projections, RoPE placement, scale `1/sqrt(64)`, causal + right-padding mask semantics, dropout=0, and GQA 8/4 grouping.
- Manual explicitly expands each K/V head twice, computes QK under BF16 autocast, scales, applies the mask, runs softmax in FP32, rounds probabilities to BF16, then performs Attention x V under BF16 autocast.
- SDPA Math receives BF16 Q/K/V, the same Boolean mask and explicit scale, with native `enable_gqa=True`. Its public attention context is BF16.
- The installed PyTorch documentation states: for Math, all intermediates are kept in float when inputs are half or bfloat16. This is **documented backend behavior**, not a hook observation.
- SDPA internal QK scores, softmax probabilities, and AV tensors are **UNKNOWN** because the kernel does not expose hookable module boundaries.

## Semantic checks

- Q/K/V and post-RoPE values are bit-identical through the first layer in BF16.
- Replacing Manual's finite-min mask sentinel with negative infinity produces context max error `0.0`.
- FP32 logits reproduce the frozen gate: `True`.
- No duplicate scaling, duplicate causal mask, GQA mapping error, RoPE displacement, or dropout mismatch was found.

Evidence levels are recorded in `precision_experiment.json`; controlled arithmetic variants are labeled `DIAGNOSTIC_RECONSTRUCTION` and are not represented as SDPA internals.
