# BF16 precision experiment

All variants reuse the real first-layer Q/K/V, RoPE outputs, mask, scale, and checkpoint. They are **DIAGNOSTIC_RECONSTRUCTION**, not observations of SDPA kernel internals.

| Variant | RMS error vs actual Manual | RMS error vs actual SDPA Math |
|---|---:|---:|
| production_equivalent_manual | 0 | 0.000184238874 |
| fp32_qk_bf16_probability_bf16_av | 0.000160087395 | 0.000168602463 |
| bf16_qk_fp32_probability_fp32_av_then_bf16 | 0.000168373605 | 0.00011393764 |
| full_fp32_then_bf16 | 0.000184238845 | 1.47937893e-08 |
| negative_infinity_mask_manual | 0 | 0.000184238874 |

The production-equivalent reconstruction matches actual Manual exactly: `True`. The closest tested reconstruction to SDPA Math is `full_fp32_then_bf16`. Moving QK/softmax/AV retention toward FP32 reduces context RMS error versus actual Math from `0.0001842388737713918` to `1.4793789304690108e-08`. Combined with PyTorch's documented Math float-intermediate behavior, this supports a BF16 rounding-path mechanism. It does not prove the precise C++ instruction ordering, which remains UNKNOWN.
