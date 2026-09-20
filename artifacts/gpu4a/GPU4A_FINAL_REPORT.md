# GPU-4A final report

Classification: `GPU4A_NATIVE_SFT_CAPABILITY_VALIDATED`

The formal native 50M-token checkpoint `ee306384fcdef646f557cf9aa1165c464b2ca373c6d64d81f5df3b19b8f859fc` loaded successfully. It contains 37,462,528 parameters in 8 layers with hidden size 512, GQA 8/4, FFN size 1536, RoPE, RMSNorm, SwiGLU, tied embeddings, context length 1024, and a native 24K byte-level BPE tokenizer. Full SFT updated every parameter and completed save, reload, exact validation, and resume checks.

The frozen synthetic dataset contains two tasks: three-way support-request classification and integer addition for operands 0-99. Train/validation/test counts are 4,800/700/700. Operand pairs or classification semantic IDs never cross splits; all targets and EOS markers are complete and supervised. This protocol supports controlled input-dependent tests, not broad capability claims.

Response-only Full SFT ran for 300 steps in 48.9 seconds. Validation loss improved from 5.992 to 0.594. Independent T1 test correctness improved from 0/64 to 64/64; T2 improved only from 0/64 to 2/64. Every SFT output was well-formed, complete, EOS-terminated, non-repetitive, and below the length limit.

The evidence validates the native training path, stopping behavior, and controlled classification adaptation. It does not validate general arithmetic or broad instruction understanding. The most likely limitation for T2 is the combination of a small, only 50M-token pretrained model and an SFT task requiring systematic digit-level generalization; simply increasing SFT steps has no current evidence base.

Peak CUDA allocated/reserved memory was 1003.6/1318.0 MiB, median training throughput was 12188 input tokens/s, and no numerical or memory failure occurred. No GPU-4A task required a background process. Historical checkpoint and GPU-2D through GPU-3C frozen hashes remain unchanged: `True`.

Optional LoRA was not run because Full SFT already established the requested functional result. GPU-4B should use this path as its correctness baseline when testing SDPA/FlashAttention and `torch.compile`, with logits, loss, checkpoint, and generation equivalence checked before accepting speedups. More pretraining should be a separate hypothesis-driven study if broader language or arithmetic capability is required.
