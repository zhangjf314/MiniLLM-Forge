# Model Comparison

Only the 12 frozen PEFT runs are directly comparable: they share datasets, prompts, 512-token
training context, decoding parameters, extraction rules and evaluation samples. Base-init versus
CPT-init comparisons are paired within method, seed and benchmark. LoRA versus QLoRA is descriptive
because quantization changes the training method, although the evaluation contract is shared.

The historical paired accuracy result is LoRA NEGATIVE and QLoRA MIXED for CPT-to-PEFT transfer.
GPU-3A additionally confirms ceiling contact is effectively invariant (8,398/8,400 overall), so
accuracy differences cannot be interpreted as evidence that any arm learned reliable termination.
No Base-only, CPT-only or Full-SFT generation records exist; those comparisons are NOT_AVAILABLE.
Per-run termination, repetition and extraction counts are in `generation_statistics.json`.
