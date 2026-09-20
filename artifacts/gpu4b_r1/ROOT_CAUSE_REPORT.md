# GPU-4B-R1 root-cause report

## Classification

`GPU4B_R1_NUMERICAL_PATH_DIFFERENCE_IDENTIFIED`

1. **Original failure reproduced:** yes. FP32 passes and BF16 logits still fail the frozen GPU-4B gate (`max_abs=0.15625`, allclose fraction `0.9999606013298035`).
2. **Why FP32 passes:** both implementations share the same mathematical semantics and FP32 suppresses the BF16 intermediate-rounding split; observed max logits error is `1.5735626220703125e-05`.
3. **Earliest observable BF16 difference:** layer-1 attention context at the `o_proj` input, after bit-identical Q/K/V projections and RoPE.
4. **Supported mechanism:** Manual rounds QK results and FP32-softmax probabilities back through BF16 before AV; documented SDPA Math retains float intermediates. Controlled precision reconstructions move toward the Math result when intermediates are retained in FP32.
5. **Implementation defect:** none found in scaling, masks, GQA, RoPE, dropout, or training/eval routing. Exact SDPA internal instruction order is still unknown.
6. **Gradient/update mechanism:** forward differences propagate to backward; tied gradient combines a sparse input-embedding contribution (dominant max/RMS error) with a widespread output-head contribution (dominant sign-disagreement count). AdamW's first-step sign normalization amplifies near-zero sign flips to about `2*lr`, explaining threshold-edge update differences.
7. **Gate status:** the original GPU-4B BF16, gradient, and optimizer gates remain failed and are not rewritten.
8. **Production changes:** none. Manual + Eager remains the formal default; no checkpoint or historical artifact changed.
9. **Recommendation:** if SDPA work continues, first establish a separately frozen BF16 equivalence protocol based on distributions, normalized gradients, multiple shapes, bounded trajectories, checkpoint compatibility, and generation behavior. Do not relabel GPU-4B's old gate.
10. **Performance work:** not justified in this stage; no benchmark or `torch.compile` run was performed.
