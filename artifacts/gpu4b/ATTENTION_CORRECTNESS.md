# Attention correctness gates

Tolerances were frozen in `protocol.json` before this run. Tests use the real 37M checkpoint, frozen GPU-4A tokenized samples, identical weights/data order, BF16 autocast where specified, and forced backend contexts where applicable.

FP32 manual versus SDPA Math: `True`. BF16 Math/Auto: `False` / `False`.
Backward Math/Auto: `False` / `False`. Optimizer step Math/Auto: `False` / `False`.
Ten-step trajectory Math/Auto: `True` / `True`. Checkpoint compatibility: `True`.
Greedy generation Math/Auto: `True` / `True`; generated IDs, extracted answers, and EOS behavior are identical to manual.

Flash was excluded from correctness/performance acceptance because forced built-in Flash qualification failed. It is not represented by a fallback result.
