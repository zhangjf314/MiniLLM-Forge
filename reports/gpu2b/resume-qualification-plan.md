# GPU-2B resume qualification plan

Status: `PLANNED_NOT_RUN`.

Run a continuous-versus-interrupted control before formal training for each adaptation
family. Use `BASE_INIT`, seed 42, the frozen training partition and family configuration.
Compare a 16-update continuous run against 8 updates, checkpoint, restore and continue to
update 16.

Required restore identity at update 8:

- model state; for LoRA/QLoRA this explicitly includes adapter state;
- optimizer and scheduler state, current LR and gradient-accumulation boundary;
- global step, consumed examples, input tokens and assistant target tokens;
- Python, NumPy, CPU Torch and every CUDA RNG state;
- sampler permutation, generator state and exact data cursor;
- quantization configuration and frozen-backbone identity for QLoRA.

At update 16, require identical model/adapter, optimizer, scheduler, RNG, sampler and metric
digests between continuous and resumed controls. Any mismatch blocks all formal arms in the
affected family. A failed seed may not be replaced.

The earlier CPT resume result does not satisfy this SFT-specific gate.
