# Stage GPU-2B-RD2 — Recovery Design and Scientific Re-scope

## 1. Current Scientific Boundary

```text
MATH_DOMAIN_ADAPTATION = CONFIRMED
MATH_REASONING_IMPROVEMENT = NOT_ESTABLISHED
CPT_TO_FULL_SFT_TRANSFER = NOT_ASSESSED
```

GPU-2A showed a math-domain language-model benefit and slight general-domain PPL
degradation. It did not establish a reasoning improvement. No downstream SFT-transfer
endpoint has since been measured.

RD2 is design-only. It launched zero training, benchmark, PPL, or generation jobs.

## 2. Frozen GPU-2B Design

The original study remains a 2-initialization × 3-method × 3-seed factorial experiment:

- Base and math-CPT initialization;
- Full SFT, LoRA, and QLoRA;
- seeds 42, 31415, and 271828;
- 19,200 train and 800 validation records;
- 9,127,756 input tokens and 7,024,493 assistant targets per run;
- 1,200 optimizer updates per run;
- GSM8K and MATH-500 generated-answer accuracy;
- primary paired contrast `C-FULL - B-FULL`.

The 18-run matrix, data, configurations, prompts, scorer, evaluation protocol, and decision
rules remain frozen. RD2 neither edits nor supersedes them.

## 3. Memory Qualification Failure

GPU2B-Q0 passed. Full-SFT Q64 processed 1,024 examples and completed 64/64 optimizer
updates with zero NaN, Inf, and OOM events. The training path was numerically finite.

The independent physical monitor nevertheless measured 7,776 MiB maximum occupancy and
only 35 MiB minimum headroom. The preregistered requirement was 1,536 MiB, so the immutable
classification remains:

```text
STAGE_GPU_2B_BLOCKED_BY_MEMORY_QUALIFICATION
```

LoRA Q64, QLoRA Q64, resume qualification, all 18 formal runs, and all formal evaluations
are `NOT_EXECUTED_AFTER_GATE_FAILURE`, not failed runs.

## 4. Why This Is Not a Scientific Failure

The qualification did not compare Base and CPT initialization and did not evaluate GSM8K
or MATH-500. It therefore provides no evidence for or against CPT transfer, Full-SFT task
quality, or adaptation-method effectiveness.

It also does not prove that Full SFT is physically impossible on 8 GB: the sampled run did
execute. It proves that this platform did not satisfy the frozen operational safety margin.

## 5. Recovery Objectives

RD2 separates three objectives:

1. preserve the original Full-SFT-primary scientific study for future adequate hardware;
2. acquire valuable downstream PEFT evidence now on the authorized 8 GB platform;
3. retain a bounded research path for implementation-only Full-SFT memory recovery without
   disguising a new training configuration as the original experiment.

The largest current portfolio gap is no longer basic model or CUDA infrastructure. The
repository already demonstrates a from-scratch Transformer, 50M-token pretraining, Qwen
CPT, CUDA training, data governance, and checkpoint/resume machinery. The missing evidence
is real downstream SFT, LoRA, QLoRA, generated benchmarks, and a controlled multi-seed
comparison.

## 6. Track H — Hardware Migration

Track H preserves GPU-2B exactly and remains the only route to its original primary
question without an implementation amendment.

### Capacity analysis

The estimates below subtract the observed 7,776 MiB physical peak from nominal capacity.
They are planning arithmetic, not qualification evidence.

| Nominal VRAM | Estimated headroom |
| --- | ---: |
| Current 8 GB platform | 35 MiB measured |
| 12 GiB | 4,512 MiB estimated |
| 16 GiB | 8,608 MiB estimated |
| 24 GiB | 16,800 MiB estimated |

Twelve GiB remains the minimum recommendation with meaningful margin. Driver, OS,
allocator, display, and workload variation mean a nominal capacity cannot guarantee the
1,536 MiB measured-headroom gate.

### H1 execution chain

```text
hardware authority
→ environment reconstruction
→ frozen-artifact verification
→ environment qualification
→ fresh Q0
→ fresh FULL / LORA / QLORA Q1
→ fresh FULL / LORA / QLORA Q2 resume
→ 18 formal runs
→ evaluation
→ paired analysis
```

Dataset payloads, hashes, token cache, seed permutations, family configs, prompts, and
decision rules may be copied only after digest verification. Hardware metadata, physical
VRAM evidence, throughput, environment manifest, and resume qualifications must be freshly
generated. Historical local qualification is not an H1 PASS.

Current H1 platform authority is `UNAVAILABLE`, so Track H is preserved but cannot be the
immediate execution route.

## 7. Track P — PEFT Transfer Study

RD2 recommends a separately identified secondary study:

```text
STAGE_GPU_2C_PEFT_TRANSFER
```

### Candidate matrix

```text
2 initializations × 2 PEFT methods × 3 seeds = 12 formal runs
```

The arms are B-LORA, C-LORA, B-QLORA, and C-QLORA at seeds 42, 31415, and 271828. The
candidate preserves the existing 19,200/800 split, 7,024,493 assistant targets, 1,200
updates, prompts, benchmarks, and seeds, but omits Full SFT under a new stage identity.

Primary paired comparisons are:

```text
C-LORA - B-LORA
C-QLORA - B-QLORA
```

Primary metrics are GSM8K and MATH-500 accuracy. Held-out SFT loss, training loss, math PPL
where justified, VRAM, throughput, elapsed time, and LoRA/QLoRA efficiency are secondary.

Three seeds support paired means, ranges, standard deviation, and direction consistency;
they do not justify overstated asymptotic significance claims. Problem-level bootstrap
intervals may be preregistered for benchmark accuracy in the future design.

### Resource budget

The candidate formal matrix contains 14,400 optimizer updates, 230,400 micro-batches,
109,533,072 input tokens, and 84,293,916 assistant targets. Evaluation contains 21,828
checkpoint/benchmark problem generations and at most 11,175,936 generated tokens under the
existing 512-token cap. These are design quantities, not executed work.

Older GPU qualification smokes reached one optimizer step at context 1,024 with LoRA and
QLoRA, and QLoRA used substantially less memory. Those tests are feasibility references
only: they are not the frozen GPU-2C configuration, a 64-update memory PASS, or long-run
evidence. GPU-2C must freshly run Q0-P, LoRA Q1, QLoRA Q1, and PEFT resume Q2 before formal
authorization.

### Claim boundary and portfolio value

A complete GPU-2C could support `CPT_TO_PEFT_SFT_TRANSFER`, real LoRA/QLoRA training,
multi-seed paired design, benchmark evaluation, and measured memory/quality tradeoffs. It
cannot support `CPT_TO_FULL_SFT_TRANSFER`, replace `C-FULL - B-FULL`, or complete GPU-2B.

Within that boundary, it directly fills the project's largest employment-evidence gap and
is valuable even while Full SFT remains pending.

## 8. Track F — Full-SFT Memory Recovery

### Static memory composition

The Full-SFT model has 596,049,920 parameters. The following values are estimates from
parameter count and dtype, not profiler attribution:

| Component | Static estimate |
| --- | ---: |
| BF16 model weights | 1,136.875 MiB |
| BF16 gradients | 1,136.875 MiB |
| Two nominal 8-bit moment arrays | 1,136.875 MiB |
| Nominal persistent subtotal | 3,410.625 MiB |
| Peak CUDA allocation minus nominal subtotal | 2,236.745 MiB |
| Physical peak minus peak CUDA allocation | 2,128.630 MiB |

The residuals combine activations, temporary tensors, optimizer metadata or higher-
precision small-tensor state, CUDA workspaces, allocator behavior, driver/WDDM occupancy,
and system use. They are not uniquely attributable from existing evidence and must not be
presented as profiler measurements.

### Existing baseline mechanisms

- Non-reentrant gradient checkpointing is already enabled, and cache use is disabled.
- The loaded Qwen3 model resolves its attention implementation to PyTorch SDPA.
- Micro-batch is already the minimum value of one.
- Full SFT already uses AdamW8bit; switching to an 8-bit optimizer is not a new recovery.
- Unused CUDA cache is already released at optimizer boundaries in the qualified executor.
- Frozen data is processed as one variable-length sample per micro-batch, not sequence
  packing.

### Candidate classification

| Technique | Memory gain | Same objective | Same optimizer | Same effective batch | Same sample order | Numerical difference | Classification |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Existing gradient checkpointing | Already realized | Yes | Yes | Yes | Yes | Existing recomputation | `NOT_USEFUL` as an additional lever |
| Activation checkpoint segmentation | Unknown, potentially moderate | Yes | Yes | Yes | Yes | Recompute/RNG ordering may differ | `POTENTIALLY_COMPATIBLE_REQUIRES_EQUIVALENCE_TEST` |
| Current PyTorch SDPA | Already realized | Yes | Yes | Yes | Yes | Frozen baseline | `NOT_USEFUL` as an additional lever |
| Alternative attention backend | Unknown | Yes | Yes | Yes | Yes | Kernel/reduction order differs | `POTENTIALLY_COMPATIBLE_REQUIRES_EQUIVALENCE_TEST` |
| CPU optimizer-state offload | Up to roughly optimizer-state footprint | Yes | Intended yes | Yes | Yes | Transfer timing/update implementation may differ | `POTENTIALLY_COMPATIBLE_REQUIRES_EQUIVALENCE_TEST` |
| Gradient offload | Up to roughly gradient footprint | Yes | Intended yes | Yes | Yes | Transfer and accumulation ordering risk | `POTENTIALLY_COMPATIBLE_REQUIRES_EQUIVALENCE_TEST` |
| CUDA allocator configuration | Unknown; fragmentation only | Yes | Yes | Yes | Yes | Normally none in objective | `POTENTIALLY_COMPATIBLE_REQUIRES_EQUIVALENCE_TEST`, low-confidence adjunct |
| Sequence packing | Potentially lower padding, but batch is already one | Not assured | Yes | Not assured | No | Token weighting/boundaries change | `SCIENTIFIC_CONFIG_CHANGE` |
| Shorter context | Potentially substantial | No: exposure/truncation changes | Yes | Yes | Record order only | Training data changes | `SCIENTIFIC_CONFIG_CHANGE` |
| Smaller micro-batch + larger accumulation | None: micro-batch already one | Yes | Yes | Could match | Numerical order changes | RNG/reduction changes | `NOT_FEASIBLE`; any accumulation change is configuration change |
| Different 8-bit optimizer | Unknown | Yes | No | Yes | Yes | Update semantics change | `SCIENTIFIC_CONFIG_CHANGE` |
| FP16 instead of BF16 | Unknown | Yes | Yes | Yes | Yes | Precision and scaling change | `SCIENTIFIC_CONFIG_CHANGE` |
| Single-GPU ZeRO/FSDP sharding | No genuine sharding gain | Yes | Potentially | Yes | Yes | Implementation differs | `NOT_FEASIBLE`; CPU-offload variants belong to the offload candidate |

Allocator tuning alone is not the primary solution: the authoritative observation is only
35 MiB physical headroom. It can be tested only as a bounded adjunct, not assumed to recover
1,501 MiB of missing safety margin.

## 9. Semantic Equivalence Analysis

Semantic equivalence means unchanged examples, tokens, objective, optimizer equations,
effective batch, sample order, scheduler, update count, and Base/CPT symmetry. Numerical
equivalence means differences stay inside prospectively declared tolerances. Bitwise
equivalence means every relevant state bit matches; it is stronger and is not automatically
expected across different kernels or offload implementations.

Every potentially compatible candidate requires a future bounded test using one frozen
initialization, seed, small dataset, and update count. Continuous baseline and candidate
must compare:

- sample and token identity;
- per-update loss and gradient-norm trajectories;
- scheduler and learning-rate state;
- final parameter differences by tensor and global norm;
- optimizer states;
- RNG and sampler states;
- checkpoint/resume continuation.

Acceptance tolerances must be frozen before results. Passing semantic equivalence does not
imply bitwise identity.

An eventual implementation-only recovery may enter a `GPU2B-AMENDMENT-v1` feasibility path
only if the original files and blocked result remain intact, the candidate receives a new
hash, equivalence evidence passes, Q0/Q1/Q2 are rerun, and all 18 arms use the same amended
implementation symmetrically. RD2 does not authorize that amendment today.

## 10. Portfolio Value Analysis

Scores use 1–5, where higher is better; for time and hardware, 5 means lower cost or lower
dependency.

| Criterion | Track H | Track P | Track F |
| --- | ---: | ---: | ---: |
| Scientific rigor | 5 | 4 | 4 |
| LLM training relevance | 5 | 5 | 4 |
| PyTorch depth | 4 | 4 | 5 |
| Fine-tuning evidence | 5 | 5 | 2 |
| LoRA/QLoRA evidence | 5 | 5 | 1 |
| Experiment-design evidence | 5 | 5 | 4 |
| Resume value | 5 | 4 | 5 |
| Interview value | 5 | 5 | 4 |
| Time-cost score | 1 | 3 | 2 |
| Hardware-independence score | 1 | 5 | 5 |
| **Total / 50** | **41** | **45** | **36** |

Track H has the broadest scientific scope but is unavailable. Track F shows deep systems
work but may consume substantial time without producing downstream SFT evidence. Track P
has a narrower scientific claim yet best addresses the immediate portfolio gap on current
hardware.

## 11. Decision Matrix

| Track | Answers original Full-SFT question | Current feasibility | Scientific risk | Time/hardware risk | Immediate portfolio return | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| H — migration | Yes | Unavailable | Low after fresh gates | Very high dependency | Highest if completed | Preserve for future |
| P — PEFT study | No; secondary scope | Promising, unqualified | Low with explicit boundary | Moderate | High | **Recommend next** |
| F — Full recovery | Conditional on equivalence | Unproven | Medium | High implementation risk | Medium | Retain as bounded research option |

## 12. Recommended Immediate Stage

```text
Primary classification:
GPU2B_RD2_PEFT_TRANSFER_RECOMMENDED

RECOMMENDED_NEXT_AUTHORIZED_STAGE:
STAGE_GPU_2C_PEFT_TRANSFER
```

The authorization is to design and freshly qualify the separate GPU-2C study. It is not
authorization to launch 12 formal runs immediately. GPU-2C must freeze its own identity,
qualification contracts, exact run IDs, analysis rules, and evidence boundaries first.

## 13. Preserved Scientific Claims

- Math CPT achieved validated domain adaptation under GPU-2A.
- The original GPU-2B scientific design remains frozen and unanswered.
- GPU-2B Full Q64 was numerically finite but failed memory authority.
- Hardware migration remains the future primary-study route.
- A future GPU-2C may ask only whether CPT initialization transfers through LoRA and QLoRA.

## 14. Forbidden Claims

RD2 and any future GPU-2C may not claim:

- mathematical-reasoning improvement from PPL alone;
- CPT-to-Full-SFT transfer;
- completion or replacement of GPU-2B;
- Full-SFT, LoRA, or QLoRA superiority before measurement;
- 8 GB Full-SFT impossibility;
- 12 GiB qualification sufficiency;
- equivalence of an offload/backend implementation before testing.

## 15. Future Execution Gates

GPU-2C preliminary gates are Q0-P, LoRA Q1, QLoRA Q1, and PEFT resume Q2. Frozen GPU-2B
data, token cache, sampler, and seeds may be reused only after fresh hash, example, token,
and permutation verification. Formal training begins only after all GPU-2C gates pass.

If authorized hardware later appears, GPU-2B-H1 must reconstruct and freeze its environment
and rerun all original Q0/Q1/Q2 gates before the 18-run campaign. If Track F later yields a
candidate, it must pass the separate equivalence and amendment governance described above.

The original GPU-2B remains preserved in all cases.
