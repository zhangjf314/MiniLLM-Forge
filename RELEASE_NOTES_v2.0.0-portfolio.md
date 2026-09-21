# MiniLLM-Forge v2.0.0 Portfolio Release

Release classification: `GPU4C_READY_WITH_DOCUMENTED_LIMITATIONS`.

This release packages the completed research as an auditable portfolio. It does not
publish model weights and does not position the native model as a production general LLM.

## Highlights

- Native 37,462,528-parameter Transformer with GQA 8/4, RoPE, RMSNorm, SwiGLU,
  tied embeddings, context 1024, and a 24K byte-level BPE tokenizer.
- Formal BF16 pretraining for 50,003,968 processed tokens and 3,052 optimizer steps.
- Native 300-step full-parameter response-only SFT with controlled task evaluation.
- Twelve Qwen3-0.6B LoRA/QLoRA runs and 8,400 audited benchmark outputs.
- Generation-length, answer-extraction, stopping, EOS-supervision, and BF16 Attention
  diagnostics.
- Evidence manifest plus claim, capability, and experiment registries.

## Validated

- Formal pretraining reduced fixed-validation loss from 10.1964 to 4.6484 and perplexity
  from 26,805.55 to 104.42; exact full-state resume was verified.
- Native Full SFT reduced validation loss from 5.9922 to 0.5943.
- T1 support classification improved from 0/64 to 64/64 on a fixed independent
  64-example test subset; this is not full-test accuracy.
- All 384 native post-SFT audit generations were complete, format-valid, EOS-terminated,
  non-repetitive, and not length-limited.
- Formal LoRA/QLoRA execution and evaluation completed for all 12 runs and 8,400 outputs.
- First-valid extraction counted 2,008 correct on the same frozen outputs versus 1,764
  with the legacy extractor. This is an evaluator change, not a model accuracy gain.
- SDPA Math passed the FP32 reference gate, and R1 localized the BF16 numerical-path
  divergence without finding a semantic Attention defect.

## Negative Results Retained

- Native T2 two-digit addition remained weak: 0/64 to 2/64 on the fixed test subset.
- Math-CPT transfer was negative for LoRA and mixed/negative for QLoRA.
- 8,398/8,400 formal PEFT generations reached the 512-token ceiling.
- The controlled EOS-supervision study did not validate autonomous stopping.
- SDPA Math and Auto failed the frozen BF16 correctness gates.

## Blocked or Unavailable

- Built-in FlashAttention was unavailable in the qualified environment.
- `torch.compile` was blocked by missing Triton.
- No distributed DDP/FSDP training was executed.
- Qwen full-parameter SFT effectiveness, DPO, and GRPO were not validated.

## Reproducibility

After installing the source checkout, run:

```powershell
uv run portfolio-verify quick
```

The verifier performs 29 bounded checks without model download, network access, or
training. Canonical local artifact digests are:

| Asset | SHA256 |
| --- | --- |
| Native pretrained checkpoint | `ee306384fcdef646f557cf9aa1165c464b2ca373c6d64d81f5df3b19b8f859fc` |
| Native SFT checkpoint | `9c33b1c5a781c4ab1d9832a0aad1349516fafb011d24ff6c67be3b1d3885b96e` |
| Native tokenizer | `e9578dfd6d2a0f4c3137d7cc223a7b8a782afba8190ae1ad105e1b8ef37e9a22` |

Checkpoints are intentionally not tracked or attached to this release. The final evidence
manifest records their local paths, metadata, and hashes.

## Known Limitations

- The native model is small and undertrained by modern standards.
- Native capability scores use fixed 64-example subsets; arithmetic remained weak.
- PEFT conclusions are bounded to three seeds and a 512-context protocol.
- Hardware evidence comes from one RTX 5060 Laptop GPU on Windows.
- Public benchmark contamination from base-model history cannot be excluded.
- Flash, compile, Qwen Full SFT effectiveness, and distributed-training claims are absent
  because those capabilities were blocked or not formally validated.

See the [Claim Registry](artifacts/gpu4c/CLAIM_REGISTRY.md),
[Capability Matrix](artifacts/gpu4c/CAPABILITY_MATRIX.md), and
[final technical report](artifacts/gpu4c/MINILLM_FORGE_FINAL_TECHNICAL_REPORT.md) for the
complete evidence boundaries.
