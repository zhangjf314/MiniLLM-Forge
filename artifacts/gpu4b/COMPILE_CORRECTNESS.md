# torch.compile qualification and correctness

The fixed Attention implementation was the validated historical `manual` path. `torch.compile` was applied only to a tiny CUDA model forward/training module for actual environment qualification; DataLoader, logging, checkpointing, evaluation, and generation control flow were not compiled.

- `default`: `NOT_SUPPORTED` after 3.5s, `TritonMissing`.
- `reduce-overhead`: `NOT_SUPPORTED` after 0.1s, `TritonMissing`.
- `max-autotune`: `SKIPPED_PREREQUISITE` because neither prerequisite mode established a stable path.

The first uncached default attempt failed after 17.216s. The formal runner then observed cache-affected failure times shown above. Both attempted modes failed with the current Windows Inductor stack because no working Triton installation exists. The failures were surfaced and recorded; neither result silently fell back to Eager. Consequently logits/loss, backward, optimizer, checkpoint-to-Eager, generation, graph-break, and recompile experiments were not mislabeled as compiled results.
