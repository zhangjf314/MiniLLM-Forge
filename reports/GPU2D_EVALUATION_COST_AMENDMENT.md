# GPU-2D Outcome-Blind Evaluation Cost Amendment

## 1. Original Evaluation Protocol

The original frozen protocol remains preserved and unmodified as
`ORIGINAL_GPU2D_EVALUATION_PROTOCOL = PRESERVED_NOT_EXECUTED`. It specified full
GSM8K (1,319 problems) and full MATH-500 (500 problems) for all 12 formal runs,
for 21,828 generations. Its measured projection is 227.1305 hours. The reason it
was not executed is `COMPUTE_COST`, not scientific failure.

## 2. Runtime Evidence

The reference evaluator uses batch size 1 and a 512-token generation ceiling.
Measured rates are approximately 15.40 generated tokens/s for LoRA and 12.35
generated tokens/s for QLoRA. Runtime recovery confirmed active KV caching,
correct eval/inference mode, disabled gradient checkpointing, SDPA, and no safe
prefix-monotonic early-stop rule. All 64 calibration generations reached 512
tokens, so 227.13 hours is a real checkpoint-specific compute projection.

## 3. Why 227 h Is Operationally Excessive

The original plan requires approximately 9.46 continuous days. That duration
creates material laptop power, thermal, interruption, and resume-management risk.
The amendment reduces problem coverage only for the secondary GSM8K endpoint; it
does not reduce seeds, methods, initializations, or MATH-500 coverage.

## 4. Outcome-Blind Amendment Timing

At freeze time there were zero formal evaluation jobs, zero formal problem
outputs, and zero formal accuracy results. No arm or benchmark outcome informed
the amendment or subset selection. Therefore:

`OUTCOME_BLIND_AMENDMENT = TRUE`

## 5. Preserved 12-Run Design

The full paired matrix remains Base/CPT × LoRA/QLoRA × seeds 42, 31415, and
271828, totaling 12 runs. Primary contrasts remain `C-LORA - B-LORA` and
`C-QLORA - B-QLORA`, paired within training seed. No experimental arm was
removed.

## 6. MATH-500 Full Evaluation

MATH-500 remains the full 500-problem primary benchmark for all 12 runs. The
primary endpoint is generated-answer accuracy and the paired Base/CPT delta.
Analysis includes accuracy, paired per-problem comparisons, paired bootstrap
confidence intervals, and separately reported variation across the three
training seeds.

## 7. GSM8K-200 Sampling

The frozen GSM8K snapshot has no predeclared scientific stratification field, so
no strata were invented. A uniform sample without replacement was drawn using
`random.Random(20260916).sample` over the 1,319 IDs in snapshot order. The 200
selected IDs were then sorted by benchmark index for execution. Their canonical
SHA-256 is:

`fb9635d80b6210da35a9603da67f16148b5f7e4ca2ba140638865878c1c4cfc8`

Every formal run must use this exact subset.

## 8. Statistical Boundary

GSM8K analysis reports fixed-200 subset accuracy, paired per-problem comparison,
and paired bootstrap confidence intervals. Its uncertainty is limited by the
fixed subset size. Problem bootstrap uncertainty and training-seed variation are
reported separately. The result must never be labeled full GSM8K accuracy.

## 9. Runtime Reduction

Using the formal calibration values, the amended 8,400-generation plan projects
to 87.3158 hours (3.64 continuous days): 39.0638 LoRA hours and 48.2521 QLoRA
hours. This saves 139.8146 hours, a 61.5570% reduction from 227.1305 hours.
Full MATH-500 accounts for 62.0439 generation hours, GSM8K-200 for 24.9892 hours,
and repeated model loading for 0.2827 hours.

## 10. Claim Boundary

The overall claim remains
`CPT_TO_PEFT_SFT_TRANSFER_UNDER_FROZEN_512_CONTEXT`. Full MATH-500 accuracy may
be reported. GSM8K must be named `GSM8K fixed-200 subset accuracy`. Directional
agreement strengthens consistency; disagreement must be disclosed. A future
full-GSM8K run may be separate optional validation, but cannot alter the frozen
primary result.

## 11. Formal Authorization

The amended order freezes 12 full MATH-500 jobs followed by 12 GSM8K fixed-200
jobs. Every job is unconditional on observed outcomes, resumable by problem ID,
incremental, and fail-closed. Prompt, chat template, decoding, checkpoint,
extractor, scorer, EOS semantics, batch size 1, and `max_new_tokens=512` remain
unchanged.

`GPU2D_EVALUATION_COST_AMENDMENT_FROZEN`

`FORMAL_EVALUATION_AUTHORIZED = YES`

Authorization applies only to an amendment-aware supervisor consuming the frozen
subset and 24-job order. This stage did not launch the evaluator supervisor.
