# GPU-3A Root Cause Report

## Executive result

Classification: **GPU3A_ROOT_CAUSES_IDENTIFIED**.

The historical counts are reproducible: 8,400 unique formal outputs; 8,398 length-limit contacts
(99.976190%); 2 recognized EOS stops; and 36 empty legacy extractions.

## Root-cause evidence

| ID | Classification | Finding |
|---|---|---|
| RC-A | CONFIRMED | Training truncation leaves 6,693/19,162 actual context-512 samples without a supervised assistant `<|im_end|>`. |
| RC-G | CONFIRMED | Generation continues after answer-like content and develops repeated spans; 8,398/8,400 outputs reach the ceiling. |
| RC-H | CONFIRMED | The last-match extractor returns empty on 36 records even though diagnostic parsing finds earlier non-empty answer content. |
| RC-C | REFUTED for static stop recognition | The decoder accepts both model EOS 151643 and chat end 151645. |
| RC-D | REFUTED for ignored EOS | Both observed stop tokens terminate generation; no stored record contains an earlier ignored stop token. |
| RC-F | HYPOTHESIS | Model capacity/objective/optimization may explain failure to emit stop tokens, but current evidence is not causal. |

The confirmed defects interact: partial EOS supervision makes termination learning weaker; repeated
post-answer generation consumes the fixed budget; and the final-match extractor converts a trailing
empty marker into an extraction failure. Correctness, answer completeness and extraction success
remain distinct outcomes.

## Unresolved

- The archive cannot establish causality between missing EOS supervision and each individual output.
- Answer-token boundaries were not stored, so post-answer token counts are UNKNOWN rather than re-tokenized estimates.
- No comparable Base/CPT-only or Full-SFT outputs exist.
- Stored records cannot reveal logits or whether 151645 was nearly selected.

## GPU-3B

Run a controlled, explicitly authorized ablation that preserves full assistant endings (or filters
truncated targets), logs raw stop reasons/logits and compares 151643 versus 151645 termination under
identical checkpoints and prompts. Do not change the historical evaluator.

## GPU-3C

Evaluate a versioned multi-answer-aware extractor separately from model quality. Report first-valid,
last-valid and conflict-aware outcomes side by side, and add paired decoding controls for repetition
penalty/stopping only after GPU-3B isolates supervision effects.
