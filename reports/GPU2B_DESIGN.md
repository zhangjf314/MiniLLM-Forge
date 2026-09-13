# Stage GPU-2B-D — CPT-to-SFT transfer design freeze

## Stage result

```text
STAGE_GPU_2B_D_COMPLETE
execution_class = ZERO_TRAINING_WORK
factorial_design = 2_INIT × 3_ADAPTATION
formal_arms = 6
formal_seeds_per_arm = 3
maximum_formal_training_runs = 18
primary_science = CPT_TO_DOWNSTREAM_SFT_TRANSFER
primary_metric_class = GENERATED_MATH_TASK_PERFORMANCE
secondary_metrics = MATH_PPL, GENERAL_PPL
SFT_execution = NOT_RUN
```

The design baseline is commit `5479eaea03432532af6d4eb7ded4cda97176b3d7`.
No model training or model-generated benchmark answer was performed in this stage.

## Frozen predecessor

Both initialization arms share Qwen3-0.6B architecture and tokenizer. `BASE_INIT` is
`Qwen/Qwen3-0.6B-Base` revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. `CPT_INIT` is the local validated
`runs/E04-qwen3-math-cpt/final_model` checkpoint produced after 10,002,432 tokens and
2,442 updates.

The predecessor classification remains
`QWEN_MATH_CPT_VALIDATED_WITH_GENERAL_DEGRADATION`: math PPL 5.171540 → 5.034781
(-2.64%), general PPL 17.079928 → 17.231117 (+0.89%). This is domain adaptation, not
evidence of improved mathematical reasoning.

## Causal factorial design

| Arm | Initialization | Adaptation | Family config |
| --- | --- | --- | --- |
| B-FULL | BASE_INIT | FULL_SFT | `full-sft-config.yaml` |
| C-FULL | CPT_INIT | FULL_SFT | `full-sft-config.yaml` |
| B-LORA | BASE_INIT | LORA | `lora-config.yaml` |
| C-LORA | CPT_INIT | LORA | `lora-config.yaml` |
| B-QLORA | BASE_INIT | QLORA | `qlora-config.yaml` |
| C-QLORA | CPT_INIT | QLORA | `qlora-config.yaml` |

Seeds are 42, 31415 and 271828 in every arm. Formal run IDs are the Cartesian product of
the six arm IDs and the three seeds, for example `B-FULL-s42` and `C-FULL-s42`. No failed
seed replacement is allowed.

The primary causal contrast is `C-FULL - B-FULL`, paired by seed. LoRA and QLoRA contrasts
answer method-interaction questions and cannot override a contradictory Full-SFT result.

## Frozen SFT_MATH_V1 data

The only formal SFT corpus is `open-r1/OpenR1-Math-220k`, subset `default`, split `train`,
revision `e4e141ec9dea9f8326f4d347be56105859b2bd68`, Apache-2.0. Selection scans pinned
source order and retains the first 20,000 items passing correctness/completeness, length,
answer, serialization, exact-deduplication and contamination rules.

- source rows scanned: 29,619;
- accepted: 20,000;
- training: 19,200; validation: 800;
- training input tokens after 1,024 truncation: 9,127,756;
- training assistant target tokens: 7,024,493;
- training untruncated tokens: 9,619,652;
- right-truncated training records: 1,624 (8.46%);
- training record SHA-256: `681e9f003926aa038b9bb0ee96a249dd7ece833e1ab5a502166084ae21ef0be2`.

The validation partition is the 800 smallest
`sha256("gpu2b-validation-v1:" + uuid)` values; both partitions preserve source order.
Each seed applies a frozen Torch permutation to the same 19,200 rows. The budget is one
complete pass, 7,024,493 assistant targets and exactly 1,200 updates at micro-batch 1 and
accumulation 16 for every arm.

## Contamination protection

The audit covers GSM8K, MATH-500, controlled math, and both frozen GPU-2A PPL probes.
It combines normalized exact match, word-5-gram Jaccard ≥0.80, and word-5-gram containment
≥0.60 with at least eight shared n-grams. Four candidates were removed: one normalized
exact MATH-500 collision and three MATH-500 problem-statement overlaps. No known collision
remains in SFT_MATH_V1. Lexical auditing cannot rule out semantic paraphrases or upstream
pretraining contamination.

## Primary evaluation

Generated-answer accuracy/exact match is primary on both frozen public tests:

- GSM8K `main/test`, 1,319 problems, revision
  `740312add88f781978c0658806c59bc2815b9866`, MIT, digest
  `ee22a458cb44be48e782d5150027ba530911efbfcaa71fe3fb01fa67a48952cf`;
- HuggingFaceH4/MATH-500 `default/test`, 500 problems, revision
  `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, digest
  `f850b2b7554370eb31b6ea5dba6f893ce6f8462912c9d0edf658802feda5a9ec`.

The MATH-500 HF snapshot does not declare a license; its stated upstream OpenAI `prm800k`
repository is MIT. This provenance limitation is explicit rather than silently assigning
the HF snapshot a license.

Generation is greedy, batch 1, one beam, 512 maximum new tokens, with the same prompt,
tokenizer and EOS policy for every arm. Scoring is automatic; manual answer correction is
forbidden. The existing frozen math and general PPL probes are secondary continuity and
retention diagnostics, never substitutes for task accuracy.

## Family configurations

All families use length 1,024, BF16 compute, one epoch, micro-batch 1, accumulation 16,
gradient clip 1.0, 3% warmup and cosine decay. Full SFT uses AdamW8bit at 2e-5. LoRA uses
rank 16, alpha 32, dropout 0.05, no bias, no modules-to-save, all-linear targets and AdamW
at 2e-4. QLoRA uses the identical adapter plus truly frozen NF4 4-bit backbone, BF16
compute, double quantization and PagedAdamW8bit at 2e-4.

Configurations are frozen by family, not initialization. Changing a family config requires
a new design revision applied identically to Base and CPT arms.

## Checkpoint and selection policy

Formal comparison uses `FINAL_TOKEN_BUDGET_CHECKPOINT`. Formal test benchmarks cannot
select checkpoints or tune LR, rank, data, prompt, stopping, sequence length or optimizer.
The frozen 800-example SFT validation partition is diagnostic only. Early stopping is off.

## Decision rules

The complete numerical rules are in `reports/gpu2b/decision-rules.md`. Strong Full-SFT
transfer requires both benchmarks to average at least +1.0 absolute accuracy point, at
least two of three positive paired seed deltas on each, and no delta below -0.5 point.
Negative, mixed and not-confirmed labels are prospectively exhaustive. A separate +5%
paired general-PPL rule governs general degradation. Engineering efficiency cannot change
the scientific label.

## Execution gates for STAGE_GPU_2B

Before formal training, the numerical stage must:

1. pass pytest, Ruff, format, lock, build and all version-controlled YAML validation;
2. verify all hashes in `checksums.txt` and the ignored local data files against manifests;
3. implement seed-invariant pre-split loading, exact target-token accounting and all 18
   paired arm specifications without per-initialization tuning;
4. pass separate FULL_SFT, LORA and QLORA 64-update memory qualifications;
5. pass continuous-versus-resumed SFT controls for all three families;
6. prove QLoRA backbone weights are actually 4-bit and frozen;
7. run formal benchmarks only after every training checkpoint is final.

Failure of any gate blocks formal training; it does not authorize a silent configuration
change.

## Boundaries

This completed design authorizes creation of the numerical `STAGE_GPU_2B`. It does not
itself authorize skipping its memory/resume/regression gates, and it does not claim any
CPT-to-SFT transfer result.
