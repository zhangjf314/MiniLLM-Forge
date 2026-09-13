# GPU-2D-R0 — 8GB-Native PEFT Recovery Design and Qualification

## 1. Historical Boundary

This is the prospective `STAGE_GPU_2D_R0` recovery stage. It does not amend or
continue either historical stage:

- GPU-2B remains `STAGE_GPU_2B_BLOCKED_BY_MEMORY_QUALIFICATION`.
- GPU-2C remains `GPU2C_BLOCKED_BY_MEMORY_QUALIFICATION`.
- No GPU-2B or GPU-2C evidence artifact was changed.

The current scientific boundary is unchanged: math-domain adaptation is confirmed,
while math-reasoning improvement, CPT-to-Full-SFT transfer, and CPT-to-PEFT-SFT
transfer are not established.

## 2. Why GPU-2C Is Frozen

GPU-2C froze a 1024-context, 12-run LoRA/QLoRA transfer design. Its LoRA Q1
completed all 64 updates without NaN, Inf, or OOM, but used 7,786 MiB physical VRAM
and left only 25 MiB headroom against the preregistered 1,536 MiB minimum. GPU-2C
therefore remains blocked by its engineering memory contract. Completing a sampled
run is not authority to remove that contract, and QLoRA was correctly left NOT_RUN
under that stage's sequential stop rule.

## 3. Recovery Objective

The objective was to prospectively identify the highest-value real downstream PEFT
transfer study that is native to the only available platform: an NVIDIA GeForce RTX
5060 Laptop GPU with 8,151 MiB nominal VRAM, Windows/WDDM, driver 577.02, CUDA 12.8,
and PyTorch 2.11.0+cu128. This stage permitted only static analysis, exact truncation
analysis, bounded 64-update Q1 trials, and bounded resume tests. It launched zero
formal training and zero benchmark evaluations.

## 4. Sequence-Length Distribution

The pinned Qwen tokenizer, production chat template, and production
`encode_sft_example` path were applied to all 19,200 training and 800 validation
examples. Nearest-rank training-set percentiles were:

| Measure | P50 | P75 | P90 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Total tokens | 400 | 633 | 973 | 1,175 | 1,673 | 4,011 |
| Assistant tokens | 289 | 514 | 842 | 1,050 | 1,532 | 3,860 |

Validation total-token percentiles were 385, 583, 930, 1,155, 1,660, and 4,035;
assistant-token percentiles were 271, 463, 805, 1,002, 1,599, and 3,560.

## 5. Context Retention

Right truncation exactly matched production semantics. Final-answer retention means
that the normalized current-scorer extraction after truncation equals the same
extraction from the full solution. It is deliberately not reported as reference
accuracy: only 23.66% of full training solutions matched the frozen reference under
the current scorer, and reference availability is recorded separately in the JSON.

| Context | Truncated examples | Eligible examples | Assistant tokens retained | Assistant retention vs untruncated | Assistant retention vs 1024 | Final answers retained | Final-answer rate | Final-answer retention vs 1024 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 6,748 | 19,162 | 5,177,783 | 68.89% | 73.71% | 12,952 | 67.46% | 73.22% |
| 768 | 3,329 | 19,191 | 6,405,009 | 85.21% | 91.18% | 16,116 | 83.94% | 91.11% |
| 1024 | 1,624 | 19,200 | 7,024,493 | 93.46% | 100.00% | 17,689 | 92.13% | 100.00% |

Both 1024→768 and 1024→512 are `SCIENTIFIC_DATA_EXPOSURE_CHANGE`. Separate token
caches and manifests were therefore created; no GPU-2C token count or data identity
is reused as GPU-2D evidence. The 512 loss is material, but it retains a majority of
the 1024 assistant and final-answer signal. It is accepted only after both higher-
context resource paths failed.

## 6. Candidate A

Candidate A is the unified Base/CPT × LoRA/QLoRA × three-seed study. Context 768
could not qualify both methods: LoRA passed, but QLoRA failed the unchanged physical
headroom gate. At context 512, both methods passed Q1 and exact Q2. Candidate A at
512 is therefore `QUALIFIED`, with 12 future formal runs and the scoped question
`CPT_TO_PEFT_SFT_TRANSFER_UNDER_FROZEN_512_CONTEXT`.

The shared adapter configuration remains rank 16, alpha 32, dropout 0.05, and
all-linear targets. It has 10,092,544 trainable parameters (1.6650% of the logical
606,142,464-parameter adapted model), 20,185,088 bytes of BF16 adapter parameters,
and no unexpected non-adapter trainables.

## 7. Candidate B

Candidate B would preserve context 1024 and run Base/CPT × QLoRA × three seeds. Its
Q1 completed 64/64 updates with no NaN, Inf, or OOM and proved a real frozen 4-bit
backbone, but physical peak was 7,784 MiB and minimum headroom was only 27 MiB.
Consequently `QLORA_R16_ALL_LINEAR_1024` is `FAIL`, Candidate B is
`BLOCKED_BY_MEMORY_QUALIFICATION`, and its Q2 correctly remains NOT_RUN.

## 8. Candidate C

The bounded fallback was statically defined but not GPU-tested because Candidate A
at 512 qualified and has higher scientific and portfolio value:

- C1: rank 8, alpha 16, all-linear; 5,046,272 trainable parameters (0.8395%).
- C2: rank 16, alpha 32, q/k/v/o projections; 4,587,520 trainable parameters
  (0.7638%).

Both remain `BOUNDED_FALLBACK_NOT_RUN`. Either would be a
`NEW_PEFT_CONFIGURATION`, could support only a newly contracted
`CPT_TO_RESOURCE_CONSTRAINED_LORA_SFT_TRANSFER` claim, and could not be compared as
the same LoRA configuration as GPU-2C.

## 9. Physical VRAM Analysis

Independent `nvidia-smi` sampling remained authoritative; the gate was never changed
from minimum physical headroom ≥1,536 MiB. CUDA allocator evidence was retained but
was not substituted for physical occupancy.

| Test | Physical peak | Minimum headroom | CUDA allocated peak | CUDA reserved peak | Inactive-split peak | Physical samples | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LoRA 768 | 5,810 MiB | 2,001 MiB | 2,915.7 MiB | 5,698 MiB | 1,172.0 MiB | 1,681 | PASS |
| LoRA 512 | 3,430 MiB | 4,381 MiB | 2,376.5 MiB | 3,294 MiB | 981.2 MiB | 1,590 | PASS |
| QLoRA 1024 | 7,784 MiB | 27 MiB | 3,418.0 MiB | 9,480 MiB | 1,557.1 MiB | 2,358 | FAIL |
| QLoRA 768 | 6,852 MiB | 959 MiB | 2,871.3 MiB | 6,714 MiB | 1,088.8 MiB | 2,324 | FAIL |
| QLoRA 512 | 5,050 MiB | 2,761 MiB | 2,320.0 MiB | 4,914 MiB | 1,089.7 MiB | 1,978 | PASS |

The evidence shows substantial and context-dependent allocator reservation and
inactive-split bytes, plus a gap between allocator counters and independently sampled
physical occupancy under WDDM. It does not uniquely apportion the cause among live
tensors, reservation/fragmentation, driver/runtime, and WDDM behavior. No allocator
tuning was performed, and `PYTORCH_CUDA_ALLOC_CONF` remained unset.

## 10. LoRA Qualification

Both Base/seed-42 LoRA trials completed 64/64 optimizer updates and 1,024 samples with
NaN=0, Inf=0, OOM=0, finite loss, and adapter-only trainables. Context 768 passed with
2,001 MiB headroom and median 831.91 target tokens/s. Context 512 passed with 4,381
MiB headroom and median 716.40 target tokens/s. The 512 trial was necessary after the
matched QLoRA 768 trial failed.

## 11. QLoRA Qualification

All three Base/seed-42 QLoRA trials completed 64/64 updates with NaN=0, Inf=0, OOM=0.
Each proved NF4 with double quantization and BF16 compute, 392 `Linear4bit` modules,
220,200,960 quantized backbone parameters, a frozen backbone, and adapter-only
trainables. Contexts 1024 and 768 failed only the physical headroom gate; context 512
passed with 2,761 MiB headroom. An externally interrupted first 512 attempt stopped
at update 48 with no numerical or OOM event; its log and SHA-256 were preserved, and
the authorized from-scratch rerun produced the PASS evidence.

## 12. Resume Qualification

LoRA 512 and QLoRA 512 each passed an exact 16-update control versus 8-update
checkpoint plus resume test. Save/restore boundary, independent control boundary,
and final-state comparisons matched for adapter/model, optimizer, scheduler, sample
order, sampler cursor, loader generator, Python/NumPy/Torch/CUDA RNG, counters, and
trajectory. QLoRA additionally matched the 4-bit identity state.

Two verified software defects were found and fixed before clean reruns: scalar
optimizer tensors initially broke evidence hashing, and bitsandbytes NF4 auxiliary
serialization keys initially conflicted with PyTorch's ordinary strict loader. The
failed-attempt metrics and hashes are preserved in the resume artifact. Neither event
is classified as a resource, numerical, or scientific failure.

## 13. Scientific Trade-offs

Candidate A retains the controlled within-method Base/CPT pairing and a matched
context across LoRA and QLoRA, so method-specific transfer contrasts remain valid.
The cost is a new 512-context exposure with 26.29% fewer assistant tokens and 26.78%
less final-answer retention than the 1024 exposure. Future results will apply only to
the frozen 512-context design and cannot be relabeled as GPU-2C or generalized to
Full SFT. Candidate B would have preserved more data but is not resource-qualified;
Candidate C would introduce adapter-capacity confounding.

## 14. Portfolio Trade-offs

The qualified Candidate A provides the strongest available portfolio evidence: real
LoRA and real QLoRA training, paired Base/CPT comparisons, three seeds, GSM8K and
MATH-500, and reproducible resume evidence. Candidate B is narrower and unavailable;
Candidate C would add more low-value memory search after a dual-method route has
already qualified. The reduced context must remain prominent in every presentation
of future results.

## 15. Compute Budget

All values below are `ESTIMATE`, not measured formal runtime. Training extrapolates
the measured 64-update wall clock linearly to 1,200 updates. Evaluation uses the
frozen 1,819 problems × 512 max-new-token ceiling and an explicit, unmeasured planning
range of 20–40 generated tokens/s. EOS termination should reduce this ceiling; model
load, checkpoint, scoring, and bootstrap overhead are not measured here.

| Candidate | Training/run | Total training | Estimated evaluation | Estimated campaign |
| --- | ---: | ---: | ---: | ---: |
| A 512, 6 LoRA + 6 QLoRA | LoRA 2.05 h; QLoRA 2.59 h | 27.85 h | 77.61–155.22 h | 105.46–183.07 h |
| B 1024, 6 QLoRA (blocked) | 3.05 h | 18.33 h | 38.81–77.61 h | 57.13–95.94 h |
| C 1024, 6 LoRA (not run) | 2.20 h proxy | 13.20 h proxy | 38.81–77.61 h | 52.01–90.81 h proxy |

Candidate C uses the unqualified historical GPU-2C LoRA-1024 Q1 as a planning proxy;
it is not a measured Candidate C runtime.

## 16. Recommended Formal Study

The recommended next stage is `STAGE_GPU_2D_PEFT_TRANSFER` with context 512:

- initializations: Base and frozen math-CPT;
- methods: LoRA and QLoRA with the same rank-16 all-linear adapter configuration;
- seeds: 42, 31415, and 271828;
- 12 total formal runs, 1,200 updates each;
- frozen GSM8K and MATH-500 prompt, scorer, decoding, and answer extraction;
- per-seed paired delta, mean, SD, range, direction consistency, and retained
  problem-level paired bootstrap.

The new 512 data manifest and token cache are authoritative for this future stage.

## 17. Claim Boundary

This stage supports only the engineering/design claim that an 8GB-native unified
512-context LoRA/QLoRA transfer study is qualified for a new formal stage. If that
future stage is completed, its maximum transfer claim scope is
`CPT_TO_PEFT_SFT_TRANSFER_UNDER_FROZEN_512_CONTEXT`. Current allowed scientific claims
remain `MATH_DOMAIN_ADAPTATION=CONFIRMED` and the previously validated math-PPL
benefit with general degradation.

No downstream scientific endpoint was measured here.

## 18. Forbidden Claims

This evidence does not establish math-reasoning improvement, CPT-to-SFT transfer,
CPT-to-Full-SFT transfer, LoRA superiority, QLoRA superiority, benchmark accuracy,
or formal-run stability. It does not show that GPU-2B or GPU-2C passed, that their
1,536 MiB gates should be lowered, that 8GB cannot physically execute Full SFT, or
that reduced-context results are equivalent to 1024-context results. Q1 numerical
stability is not a downstream result.

## 19. Formal Authorization Decision

`GPU2D_REDUCED_CONTEXT_PEFT_RECOMMENDED`

`recommended_formal_stage = STAGE_GPU_2D_PEFT_TRANSFER`

`formal_campaign_authorized = YES_FOR_NEXT_STAGE_ONLY`

`STAGE_GPU_2D_R0` now stops. It performed zero formal training launches, zero full
GSM8K/MATH-500 launches, and no 1,200-update run.
