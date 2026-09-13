# GPU-2B preregistered decision rules

Status: `FROZEN_BEFORE_FORMAL_RESULTS`

All accuracy deltas are absolute percentage points and are paired by seed. For each family
and benchmark, define `delta_i = accuracy(CPT_INIT, seed_i) - accuracy(BASE_INIT, seed_i)`.
The practical-effect margin is 1.0 point; the conflict margin is -0.5 point. These margins
are larger than one-item score resolution on GSM8K (0.0758 point) and MATH-500 (0.2 point)
and are fixed before training. With only three seeds, raw deltas and directional consistency
take priority over asymptotic p-values.

## Primary Full-SFT transfer

Apply these rules in order:

1. `CPT_TO_SFT_NEGATIVE_TRANSFER`: on both benchmarks, mean delta is at most -1.0,
   at least two of three deltas are negative, and no delta exceeds +0.5.
2. `CPT_TO_SFT_TRANSFER_CONFIRMED`: on both benchmarks, mean delta is at least +1.0,
   at least two of three deltas are positive, and no delta is below -0.5.
3. `CPT_TO_SFT_TRANSFER_MIXED`: either benchmark reaches an absolute mean delta of at
   least 1.0, or benchmark directions conflict, without satisfying rule 1 or 2.
4. `CPT_TO_SFT_TRANSFER_NOT_CONFIRMED`: every remaining outcome.

The classification uses `C-FULL` versus `B-FULL`; LoRA or QLoRA cannot override it.
Report all six per-benchmark paired deltas, means and ranges regardless of label.

## General-domain cost

For each family compute paired relative general-PPL deltas. Declare
`GENERAL_DEGRADATION` when the mean is at least +5% and at least two of three deltas are
positive. Declare `GENERAL_IMPROVEMENT` symmetrically at -5%. Otherwise use
`GENERAL_CHANGE_BELOW_FROZEN_MARGIN`. Append `WITH_GENERAL_DEGRADATION` only when the
transfer label and this independent rule both hold.

## Method interaction

Apply the same transfer rule within LoRA and QLoRA, then classify:

- `METHOD_INVARIANT`: Full, LoRA and QLoRA are all confirmed with the same direction.
- `FULL_ONLY`: only Full is confirmed.
- `PEFT_ONLY`: Full is not confirmed while both LoRA and QLoRA are confirmed.
- `NOT_CONFIRMED`: no family is confirmed and none is mixed.
- `METHOD_DEPENDENT`: every other pattern, including mixed or conflicting family results.

Engineering metrics are reported separately and cannot change a scientific label.
