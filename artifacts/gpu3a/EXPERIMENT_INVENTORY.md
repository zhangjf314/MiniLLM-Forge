# GPU-3A Experiment Inventory

## Provenance

- Audit baseline: `main` at `91c7efe6a646b0b8a9c6592390abbf36ed6f58af`.
- Frozen GPU-2D protocol baseline: `9dbd99ee6b92103ae2af742bca8e9ecbd9a216c8`.
- Formal campaign: 12 runs, 24 evaluation jobs, 8400 problem outputs.
- Generation records: `runs/gpu2d-formal/*/evaluation-amended/{math500,gsm8k-fixed-200}.jsonl`.

## Experiment attribution

The 8,400 records are the sum of 12 independently trained PEFT adapters. Every adapter has
500 MATH-500 outputs and 200 frozen GSM8K-subset outputs. The grid is Base/CPT initialization
× LoRA/QLoRA × seeds 42/31415/271828. All use Qwen3-0.6B, assistant-only SFT, context 512,
1,200 updates, greedy decoding, `max_new_tokens=512`, stop IDs 151643 and 151645, and pad ID
151643. The training corpus has 19,200 source records; 19,162 remain eligible at context 512.

## Availability

| Arm | Training | Frozen generation records | Comparability |
|---|---:|---:|---|
| Base-init LoRA | 3 completed runs | 2,100 | Paired with CPT-init LoRA |
| CPT-init LoRA | 3 completed runs | 2,100 | Paired with Base-init LoRA |
| Base-init QLoRA | 3 completed runs | 2,100 | Paired with CPT-init QLoRA |
| CPT-init QLoRA | 3 completed runs | 2,100 | Paired with Base-init QLoRA |
| Base model without SFT | NOT_AVAILABLE | NOT_AVAILABLE | No new inference authorized |
| CPT model without SFT | NOT_AVAILABLE | NOT_AVAILABLE | No new inference authorized |
| Full SFT | Configured but not formally run | NOT_AVAILABLE | Not comparable |

Checkpoint and adapter identities are recorded per run in `artifacts/gpu2d_formal/final_result.json`.
Dataset revisions, hashes, prompt contract, evaluator hash and decode contract are in
`artifacts/gpu2d_formal/protocol.json`. Unknown fields are not inferred.
