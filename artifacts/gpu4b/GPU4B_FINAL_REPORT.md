# GPU-4B final report

Classification: `GPU4B_PARTIAL_VALIDATION`

The environment is Windows with Python 3.11.15, PyTorch 2.11.0+cu128, CUDA runtime 12.8, driver 577.02, and an RTX 5060 Laptop GPU (compute capability 12.0, BF16 supported). No software component was installed or upgraded.

The production Attention audit confirmed Manual GQA 8/4 with RoPE-before-attention, `1/sqrt(64)` scaling, causal plus key-padding masking, FP32 softmax, dropout semantics, and unchanged Q/K/V/O parameter shapes. GPU-4B preserved Manual as the historical default and added explicit experimental backends.

SDPA Math passed the strict FP32 reference: maximum logits error 1.57e-05, maximum first-layer Attention-output error 2.98e-08, and identical loss. In BF16, Math and Auto retained small mean logits errors and identical greedy answers/EOS behavior, compatible checkpoints, finite gradients, and matching 10-step improvement direction. They nevertheless exceeded the pre-frozen full-logit allclose, tied-embedding gradient, and one-step update thresholds, so both failed B2 and were excluded from performance testing.

Forced built-in Flash returned `FLASH_BACKEND_NOT_AVAILABLE`; the PyTorch build reports no available Flash kernel. Auto with explicit equivalent K/V expansion actually dispatched Efficient Attention, verified from operator traces, but that does not override its correctness failure.

Accordingly, 256/512/768/1024 attention-only and full-model benchmarks, as well as the 300-step real-SFT replay, are explicitly `NOT_RUN_CORRECTNESS_GATE_FAILED`. GPU-4A Manual Eager remains the only accepted real workload result: about 48.9 seconds, 12,188 input tokens/s, and 1003.6/1318.0 MiB peak allocated/reserved memory. Generation/T1/T2 regression testing for optimized candidates was not expanded to the full 384 records because the earlier numerical gate already failed; the fixed probes retained identical token IDs, answers, format, and EOS.

`torch.compile` exists at the API level, as do Dynamo and Inductor, but both `default` and `reduce-overhead` failed actual CUDA compilation with `TritonMissing`. The first uncached default attempt cost 17.216s; the formal repeat then failed after 3.484s with caches already populated, while `reduce-overhead` failed after 0.084s. `max-autotune` was skipped because neither prerequisite mode worked. No eager fallback was mislabeled, so there is no steady-state or end-to-end compile speedup and no compiled checkpoint claim.

The recommended current configuration is `manual + eager`. SDPA Math/Auto should not be enabled under the frozen acceptance contract, Flash is unavailable, and compile is unsupported by the unchanged environment. Historical GPU-4A artifacts and checkpoints remain unchanged: `True`.

GPU-4C should isolate reproducible Windows fused-kernel and compiler-stack qualification. If a supported build is introduced in a separately controlled environment, rerun the frozen correctness gates before measuring performance; do not mix an environment upgrade with the current GPU-4B baseline.
