# GPU-2D Evaluation Runtime Recovery

## 1. Reference Runtime

The preserved batch-size-one evaluator projects to **227.13 h**.
A fresh bounded recheck projects to 218.36 h.

## 2. Why 227 h Is Real

All 64 calibration generations reached the frozen 512-token ceiling. The projection is
therefore based on measured checkpoint-specific decode time, not a token-accounting error.

## 3. KV Cache Audit

Both LoRA and QLoRA created a 28-layer DynamicCache. Its sequence length increased by one
on a one-token decode step, so `KV_CACHE_ACTIVE = YES`.

## 4. Evaluation Mode Audit

Both paths use eval mode, `torch.inference_mode()`, disabled gradient checkpointing,
`use_cache=True`, and the SDPA attention implementation.

## 5. EOS / Termination Audit

Classification: `MODEL_DID_NOT_GENERATE_TERMINATION_TOKEN`. Tokenizer EOS appeared in 0/64 outputs and Qwen `<|im_end|>` appeared in 0/64 outputs.

## 6. Candidate Implementations

Static cache was rejected because this environment lacks a working Triton runtime. LoRA
merge was rejected after a token-1 mismatch. Serial greedy was retained only for LoRA;
QLoRA keeps the reference `generate()` path because the manual loop was slower.

## 7. Exact-Equivalence Results

All four LoRA/QLoRA × GSM8K/MATH-500 cells: **True** for token
IDs, decoded text, extracted answer, and correctness.

## 8. Physical VRAM

Minimum measured headroom was 5701 MiB; the 1536 MiB gate passed.

## 9. Runtime Results

| Cell | Reference tok/s | Selected tok/s | Selected s/problem |
| --- | ---: | ---: | ---: |
| LORA_gsm8k | 15.328 | 15.129 | 33.843 |
| LORA_math500 | 15.248 | 16.478 | 31.073 |
| QLORA_gsm8k | 13.306 | 13.306 | 38.480 |
| QLORA_math500 | 13.193 | 13.193 | 38.809 |

Best rejected candidate projection: **217.23 h** (1.046× versus 227.13 h). This is below the predeclared 5% material-improvement threshold, so the reference evaluator remains selected.

## 10. Prefix Answer Diagnostic

A parseable non-empty fallback existed by prefix 64 in 64/64 cases. The extracted answer changed
after first becoming parseable in 62/64 cases.
This is diagnostic evidence only; `max_new_tokens=512` remains frozen.

## 11. Scorer Monotonicity

No prefix-monotonic terminal condition can be proved. The extractor uses the last boxed
expression, last final-answer marker, or final non-empty line, all of which later tokens
can replace.

## 12. Selected Evaluator

`REFERENCE_EVALUATOR`. Evaluation code freeze: `True`.

## 13. Remaining Cost

Selected reference projection: LoRA 100.84 h; QLoRA 126.01 h; GSM8K 164.80 h; MATH-500 62.04 h; model-load overhead 0.28 h.

## 14. Formal Launch Recommendation

`FORMAL_EVAL_HIGH_COST_REQUIRES_EXPLICIT_DECISION`. Expected continuous runtime is 9.05 days. Do not start the supervisor without an
explicit next-stage decision; resumable append-only per-problem output remains available.
