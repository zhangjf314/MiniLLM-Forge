# Capability Matrix

| Capability | Status | Evidence | Limitation |
| --- | --- | --- | --- |
| Native Transformer implementation | VALIDATED | C001 | 37.46M bounded research model |
| Tokenizer training | VALIDATED | C002 | 24K train-only FineWeb-Edu tokenizer |
| Pretraining | VALIDATED | C003 | 50.0M processed tokens; not converged general LLM |
| BF16 training | VALIDATED | C003,C006 | Qualified RTX 5060 Laptop environment |
| Checkpoint exact resume | VALIDATED | C004 | Bounded deterministic control |
| Continued pretraining (CPT) | VALIDATED | artifacts/training/qwen_math_cpt_result.json | Math PPL improved while general PPL degraded |
| Native Full SFT | VALIDATED | C005-C009 | Synthetic tasks; generation evaluation used 64-row subsets |
| Qwen Full SFT | IMPLEMENTED_BUT_NOT_EFFECTIVENESS_VALIDATED | C021 | Formal arm not completed |
| LoRA | VALIDATED | C010,C011 | Training/evaluation valid; CPT transfer result negative |
| QLoRA | VALIDATED | C010,C011 | Training/evaluation valid; CPT transfer mixed |
| Response-only loss | VALIDATED | C005,C006,C010 | Validated in native and PEFT workflows |
| Evaluation harness | VALIDATED | C009,C010,C013 | Benchmark contamination remains model-history dependent |
| Generation diagnostics | VALIDATED | C012-C015 | Some causal explanations remain hypotheses |
| Answer extraction | VALIDATED | C013 | Evaluator improvement only |
| External stopping | VALIDATED | C014 | Bounded confirmation; not autonomous EOS |
| EOS supervision ablation | NEGATIVE_RESULT | C015 | Expected benefit was not observed |
| SDPA integration | IMPLEMENTED_BUT_NOT_EFFECTIVENESS_VALIDATED | C016,C017,C020 | FP32 pass; BF16 gate fail; no performance benchmark |
| FlashAttention | BLOCKED | C018 | Unavailable built-in kernel |
| torch.compile | BLOCKED | C019 | Triton missing; no speedup |
| Distributed training | NOT_EXECUTED | C022 | No DDP/FSDP experiment |
