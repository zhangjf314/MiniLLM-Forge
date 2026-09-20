# GPU-4B environment and backend audit

- OS: `Windows-10-10.0.26100-SP0`
- Python: `3.11.15`
- PyTorch: `2.11.0+cu128`
- CUDA runtime / driver: `12.8` / `577.02`
- GPU: `NVIDIA GeForce RTX 5060 Laptop GPU`, compute capability `[12, 0]`, 8150.6 MiB
- BF16: `True`
- `torch.compile` / Dynamo / Inductor: `True` / `True` / `True`
- Triton Python package: `False`

Forced kernel qualification used BF16, causal masking, forward and backward, and the production GQA adapter. SDPA Math passed with native `enable_gqa`. Expanded-GQA Efficient Attention and cuDNN Attention passed. The production `sdpa_auto` path selected `efficient_attention` according to recorded dispatcher operators.

PyTorch built-in Flash qualification: `FLASH_BACKEND_NOT_AVAILABLE`. Forced Flash execution returned `RuntimeError: No available kernel. Aborting execution.`; no fallback result is labeled as Flash. No packages, drivers, CUDA components, or PyTorch builds were changed.
