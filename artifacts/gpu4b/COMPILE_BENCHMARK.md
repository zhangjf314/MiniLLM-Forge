# torch.compile benchmark

No compile performance benchmark was run. Both required qualification modes failed before producing a compiled CUDA graph because the unchanged environment has no working Triton installation. Cold failure time and peak memory are retained in `compile_qualification.json`; steady-state, end-to-end, graph-break, and recompilation metrics are `NOT_RUN_NO_QUALIFIED_MODE`. Running Eager and calling it a compile fallback would violate the benchmark protocol.
