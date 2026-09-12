# MiniLLM Formal Pretraining — E01

## 1. Objective

A bounded from-scratch Transformer pretraining experiment. Establish trainability,
held-out generalization, GPU efficiency and resumability; no Qwen or fine-tuning campaign.

## 2. Hardware

NVIDIA GeForce RTX 5060 Laptop GPU, 8150.56 MiB, CUDA runtime 12.8,
PyTorch 2.11.0+cu128. No driver, Toolkit, clocks, fan or power changes.

## 3. Model Architecture

37,462,528 trainable parameters. 8 layers, hidden 512, 8 query/4 KV heads,
head dimension 64, FFN 1536, 1024 context, RMSNorm/RoPE/GQA/SwiGLU, tied embeddings.
Gradient checkpointing enabled throughout calibration, pilot and formal training.
Architecture SHA-256: `596a434d917d99d05d41b55b74446dc561f0fa082c4dec6397f9301bd6bd8592`.

## 4. Tokenizer

24K byte-level BPE, trained on 5,000 train-only documents. Full byte alphabet prevents
unseen-byte UNK failures. The 304-document validation probe has 0 UNK and 100% normalized
round-trip; average 969.67 tokens/document and 4.5686 characters/token.
Tokenizer SHA-256: `e9578dfd6d2a0f4c3137d7cc223a7b8a782afba8190ae1ad105e1b8ef37e9a22`. It was not changed during training.

## 5. Dataset

FineWeb-Edu sample-10BT, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, first 30,000 rows of
`sample/10BT/000_00000.parquet`. 147,782,794 raw text bytes; 2 normalized duplicates removed.
29,694 train and 304 validation documents, split by normalized document SHA-256 before
tokenizer training. Independent streams packed with EOS between documents. No window
can cross the train/validation partition. There are 32,179,200
usable train tokens and 294,912 fixed held-out input tokens
(294,624 next-token targets). Drop tails: 613 train,
186 validation tokens. The first-shard selection is deterministic, not a random sample
of the entire FineWeb-Edu distribution. Only exact, not semantic near-duplicate removal
was done in this stage; that limits claims about independence.

## 6. Training Configuration

AdamW, LR 0.0003, betas (0.9, 0.95), weight decay 0.1,
warmup ratio 0.03 followed by cosine decay to 10% LR, BF16, gradient clip 1.0.
Micro batch 4, accumulation 4, GPU count 1:
16 sequences / 16,384 input tokens / 16,368 supervised targets per optimizer update.
Seed 42. Validation NLL is weighted by actual supervised tokens, not by batches.

## 7. Batch Calibration

393,216 real tokens over batches 2/4/8. Batch 4 selected: batch 8 gained only 0.9%
throughput but nearly doubled reserved memory. See GPU_BATCH_CALIBRATION.md.
No LR search was performed; no immediate divergence was observed at 3e-4.

## 8. Training Budget

Pilot: 5,013,504 processed tokens, separate initialization and shorter LR horizon.
Formal: 50,003,968 processed input tokens, 3,052 steps,
49,955,136 supervised next-token targets. Rounded from 50M
to full updates; calibration/pilot tokens are NOT added to E01. The finite corpus is
revisited: formal processed tokens / usable corpus tokens =
1.554. No claim of 50M unique tokens.
50M was fixed after the pilot; 100M would roughly double laptop runtime and was not
needed to establish this first bounded baseline.

## 9. Training Dynamics

Initial fixed train-probe loss 10.1976; final online interval loss
4.5591. Matched final train-probe loss:
4.5320. Online loss reflects training-time minibatches and
is not exactly the same estimator as evaluation loss.

## 10. Validation Dynamics

| Processed input tokens | Validation loss | Perplexity |
| ---: | ---: | ---: |
| 0 | 10.1964 | 26805.55 |
| 2,506,752 | 6.8903 | 982.71 |
| 5,013,504 | 6.2821 | 534.93 |
| 7,520,256 | 5.9586 | 387.07 |
| 10,027,008 | 5.7200 | 304.90 |
| 12,533,760 | 5.5445 | 255.83 |
| 15,040,512 | 5.3964 | 220.62 |
| 17,547,264 | 5.2771 | 195.80 |
| 20,054,016 | 5.1756 | 176.90 |
| 22,560,768 | 5.0879 | 162.05 |
| 25,001,984 | 5.0098 | 149.88 |
| 25,067,520 | 5.0110 | 150.05 |
| 27,574,272 | 4.9417 | 140.01 |
| 30,081,024 | 4.8852 | 132.31 |
| 32,587,776 | 4.8258 | 124.68 |
| 35,094,528 | 4.7841 | 119.59 |
| 37,601,280 | 4.7468 | 115.21 |
| 40,108,032 | 4.7189 | 112.04 |
| 42,614,784 | 4.6944 | 109.33 |
| 45,121,536 | 4.6752 | 107.25 |
| 47,628,288 | 4.6583 | 105.45 |
| 50,003,968 | 4.6484 | 104.42 |

Initial to final PPL: 26805.55 → 104.42.
Best loss 4.6484 (PPL 104.42) at
step 3052. Held-out improvement, not merely training loss, supports learning
on this fixed split. Final validation-minus-matched-train-probe loss gap is
0.1164; the 32-block probe is small,
so this gap alone cannot establish the onset of overfitting.

## 11. Checkpoint / Resume

`EXACT_RESUME_CONFIRMED`: short real-corpus continuous-vs-resumed control matched model,
optimizer, scheduler and RNG state hashes. The 5M pilot also stopped at step 153 and
resumed to 306 with continuous token/LR counters. Original LR horizon was preserved.
Formal checkpoints include early step 100, mid step 1526, final `last.pt`, and `best.pt`.
Hashes are in minillm_checkpoint_inventory.json; large weights remain local under runs/.

## 12. GPU Efficiency

Peak allocated/reserved: 1849.02 / 2394.00 MiB.
Median compute throughput: 19573.0 input tokens/s; median
optimizer-step time 0.837s.
Active wall time 2684.8s,
measured update compute time 2549.1s.
Update timing synchronizes CUDA and excludes evaluation/checkpointing; wall time includes
run setup/evaluation/export and excludes user/agent idle gaps between process segments.
First-quarter vs last-quarter medians: 19557 /
19485 tokens/s. Endpoint GPU status is preserved in JSON;
no continuous thermal-control intervention was made.

## 13. Generation Sanity

Three fixed English completions, seed 42, greedy decoding (temperature 0, top_p 1),
32 new tokens, at initialization/mid/final. Raw outputs are retained in training artifacts.
Initialization repeats arbitrary tokens. Mid/final samples form common English syntactic
fragments but remain repetitive and semantically unreliable. This is a mechanism-level
improvement only; the samples are not a language or math benchmark.

## 14. Failures

Training NaN 0, Inf 0, OOM 0.
Recorded pre-clip gradient norms span 0.453–1.823.
Clip threshold is 1.0. These are logged interval endpoint norms, not every micro-step.
The data discovery stall and its explicit-shard resolution are retained under failures/.

## 15. Regression Verification

Regression verification: PASS. Ruff check and full-repository
format checks passed, 28 tests passed, all 41 YAML files parsed, the lock remained valid,
and both sdist and wheel built. GPU-0 environment artifacts have no diff from the stage
baseline. Details are in minillm_regression_validation.json.

## 16. Limitations

Limited token budget; not compute-optimal, not fully pretrained, not production quality,
not comparable to Qwen. One seed, one fixed validation partition, no formal architecture
ablation. Repeated corpus exposure and first-shard selection limit generalization claims.
The model remains undertrained in the practical sense that this is only a small bounded
pretraining study; a learning curve does not establish a fully converged language model.

## 17. Conclusion

The measured held-out and training dynamics answer trainability and generalization for
this configuration. Resume and GPU metrics provide engineering evidence. Readiness for
GPU-2 means that the baseline is reproducible, not that MiniLLM has reached production
language quality. No GPU-2 work was performed in this stage.
