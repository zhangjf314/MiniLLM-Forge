# Gradient and update audit

- BF16 loss difference is `0.017578125`.
- Tied weight total gradient max/RMS error: `0.05410867929458618` / `0.00012423696171026677`; cosine `0.999960437697542`.
- Output-head contribution max/RMS error: `0.0078125` / `2.4877801479306072e-05`.
- Input-embedding contribution max/RMS error: `0.05410867929458618` / `0.00012175626761745661`.
- The decomposition recomposes total gradients with max residual `0.0`.

The tied result is therefore not a mysterious third path: it is the sum of the input lookup path and output vocabulary projection path. The sparse input-embedding contribution dominates the recorded max and RMS discrepancy, while the output-head contribution is widespread and accounts for most sign disagreements. Component values at the maximum-update location are recorded in the JSON.

For the tied parameter, the AdamW delta max error is `0.00020000338554382324` at `[1, 65]`. The corresponding Manual/Math gradients are `0.0003250003792345524` and `-0.0003556914161890745`; sign flip=`True`. With zero-initialized moments, first-step Adam normalization approaches a sign update, so a tiny gradient sign disagreement can produce approximately `2 * lr = 0.0002`. Full per-parameter gradients, update norms, sign-disagreement counts, and `exp_avg`/`exp_avg_sq` differences are in `gradient_update_diagnostics.json`.
