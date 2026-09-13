# GPU-2D Formal Evaluation Executor Qualification

## Frozen identity

The GSM8K and MATH-500 snapshots, prompt/chat template, greedy decoding, 512-token
generation ceiling, stop semantics, extraction, and scorer were reused without change.
The deterministic qualification subset contains 16 frozen-hash-selected problems per
benchmark. The formal checkpoint is the final update-1200 checkpoint.

## Exact-equivalence and memory results

| Family | Batch | Gate | Exact token/text/answer/score | Min physical headroom | Problems/s | Generated tokens/s |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| LORA | 1 | PASS | True | 6299 MiB | 0.0477 | 15.66 |
| LORA | 2 | FAIL | False | 6199 MiB | 0.0861 | 29.28 |
| LORA | 4 | FAIL | False | 5939 MiB | 0.1546 | 52.61 |
| LORA | 8 | FAIL | False | 5243 MiB | 0.2968 | 102.88 |
| QLORA | 1 | PASS | True | 6381 MiB | 0.0400 | 12.59 |
| QLORA | 2 | FAIL | False | 6275 MiB | 0.0664 | 21.13 |
| QLORA | 4 | FAIL | False | 5707 MiB | 0.1132 | 37.59 |
| QLORA | 8 | FAIL | False | 4013 MiB | 0.2079 | 65.13 |

Classification: `EVAL_EXECUTOR_QUALIFIED_SERIAL`. The common selected batch size is
**1**. The selection is the largest tested batch that is
exactly equivalent to serial and retains at least 1536 MiB physical headroom
for both LoRA and QLoRA execution paths.

## Measured planning update

At the selected batch, aggregate qualification throughput was
0.0436 problems/s and
13.99 generated tokens/s, with
321.33 mean generated tokens/problem.
The resulting extrapolation for all 12 × 1,819 formal evaluations is
139.22 hours. This is still an estimate;
the full benchmarks determine actual runtime.

## Resumability

Formal evaluation writes one append-only JSONL row per completed problem and resumes by
the frozen problem ID. Completeness requires unique IDs, no missing IDs, exact frozen
counts, and checkpoint/prompt identity on every row.
