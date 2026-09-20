# Attention performance benchmark

Performance execution was stopped by the pre-declared correctness gate. SDPA Math and Auto did not satisfy the frozen BF16 full-logit, selected-gradient, and one-step update tolerances; built-in Flash is unavailable. Therefore no optimized backend was eligible for the 256/512/768/1024 attention-only, full-model, or 300-step real-SFT benchmarks.

`attention_benchmarks.json` retains an explicit backend/sequence-length matrix with `NOT_RUN` reasons and configuration hashes. No microbenchmark, model throughput, memory, or speedup number is fabricated from an ineligible implementation. The unchanged GPU-4A Manual Eager measurement remains the only formal real-SFT baseline.
