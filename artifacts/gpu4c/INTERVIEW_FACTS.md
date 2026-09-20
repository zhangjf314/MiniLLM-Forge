# Interview Facts

## Native model and scale

- **Fact:** 37.46M parameters and 50.0M processed pretraining tokens were chosen to close the full implementation/training/recovery loop on an 8GB laptop GPU.
- **Interpretation:** the scale is large enough to exercise realistic training systems but too small and undertrained for mature general-language claims.

## T1 versus T2

- **Fact:** T1 reached 64/64 on a fixed independent subset; T2 reached 2/64. All 384 generations had valid termination and format.
- **Interpretation:** SFT successfully taught a low-entropy classification mapping and output protocol but did not create robust arithmetic computation.
- **Hypothesis:** tokenization and memorization-friendly task structure help explain the difference; no dedicated causal ablation was run.

## PEFT and generation

- **Fact:** 12 LoRA/QLoRA runs and 8,400 outputs were completed. CPT transfer was negative for LoRA and mixed for QLoRA. 8,398 outputs hit 512 tokens.
- **Fact:** extraction changes recovered 36 failures and exposed 607 conflicts; external stopping reduced redundancy.
- **Interpretation:** evaluation and inference-system quality can improve without changing model weights or answer capability.

## EOS controlled negative

- **Fact:** GPU-3C changed only final EOS-label supervision over 64 steps and 1,024 ordered examples per arm; the supervised arm emitted no autonomous `<|im_end|>` on four C0 prompts.
- **Interpretation:** a plausible intervention failed under the bounded protocol; retaining this prevents post-hoc success narratives.
- **Hypothesis:** capacity, optimization horizon, decoding dynamics, and data distribution may matter, but none was causally isolated.

## SDPA and thresholds

- **Fact:** FP32 matched, BF16 frozen gates failed, and the first public difference was Attention×V context. FP32-intermediate reconstruction closely matched SDPA Math.
- **Interpretation:** different intermediate precision/rounding paths explain the dominant difference without a semantic attention bug.
- **Unknown:** exact SDPA internal instruction order.
- Thresholds were not relaxed because doing so after observing failure would invalidate the preregistered gate.

## Why no Triton/Flash continuation

- **Fact:** built-in Flash was unavailable and compile was blocked by missing Triton; no performance benchmark qualified.
- **Decision:** stop because environment repair and performance work would be a new research stage, while correctness evidence already required Manual + Eager as formal default.

## Engineering closure

The closed loop is: native model/tokenizer → formal pretraining → exact resume → native Full SFT → Qwen PEFT → unified evaluation → generation diagnostics → controlled negative ablation → BF16 numerical study → evidence registries and reproducible verification.
