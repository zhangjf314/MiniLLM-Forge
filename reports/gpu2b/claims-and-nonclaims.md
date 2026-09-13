# GPU-2B-D claims and nonclaims

## Claims frozen from predecessors

- Qwen math CPT completed 10,002,432 tokens and 2,442 updates.
- Frozen math PPL changed 5.171540 → 5.034781 (-2.64%).
- Frozen general PPL changed 17.079928 → 17.231117 (+0.89%).
- This establishes controlled domain language-model adaptation with general degradation.

## Claims made by this stage

- A 2-initialization × 3-adaptation × 3-seed causal design is prospectively fixed.
- Data, tokenizer, serialization, generation, scoring, budgets and decision rules are shared
  across paired initialization arms.
- The stage performs zero model training and zero model-generated benchmark answers.

## Explicit nonclaims

This stage does not establish improved math reasoning, better GSM8K or MATH-500 accuracy,
CPT-to-SFT transfer, LoRA or QLoRA superiority, CPT/PEFT synergy, long-run SFT stability or
SFT resume correctness. Those claims require the separate numerical `STAGE_GPU_2B` gates
and formal runs.
