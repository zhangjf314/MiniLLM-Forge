# Forward error localization

The frozen batch and checkpoint were reproduced. FP32 passes; BF16 fails the original logits gate.

- FP32 logits max/mean absolute error: `1.5735626220703125e-05` / `7.755662636554916e-07`; allclose `True`.
- BF16 logits max/mean absolute error: `0.15625` / `0.008180638775229454`; allclose fraction `0.9999606013298035`.
- BF16 loss Manual/Math/difference: `5.5029296875` / `5.4853515625` / `0.017578125`.
- First nonzero public boundary: `block.0.attention.context`. First boundary failing the frozen BF16 allclose rule: `logits`.
- At the max-logit-error index `[1, 52, 171]`, Manual=`0.625`, Math=`0.46875`.

## Propagation by block output

| Block | max abs | error RMS | relative RMS | allclose fraction |
|---:|---:|---:|---:|---:|
| 1 | 0.0080566406 | 0.00098530645 | 0.0038942856 | 1.00000000 |
| 2 | 0.012695312 | 0.0015637559 | 0.0045724434 | 1.00000000 |
| 3 | 0.013305664 | 0.0020549251 | 0.0049774407 | 1.00000000 |
| 4 | 0.013809204 | 0.0026290386 | 0.0054277732 | 1.00000000 |
| 5 | 0.013839722 | 0.0030054722 | 0.0054669619 | 1.00000000 |
| 6 | 0.016906738 | 0.0037134683 | 0.0057505755 | 1.00000000 |
| 7 | 0.027984619 | 0.0051536248 | 0.0066307318 | 1.00000000 |
| 8 | 0.046203613 | 0.007160163 | 0.0065726637 | 1.00000000 |

The first error is observable at layer-1 attention context (the input to `o_proj`), after bit-identical Q/K/V and RoPE boundaries. It then propagates through residual and MLP paths. Final-normalized hidden error RMS is `0.010453247465193272` and logits error RMS is `0.013104256242513657`, an observed RMS gain of `1.253606239223513` through the tied vocabulary projection. The largest output-row L2 norm is `1.9835994243621826`. The FP32 linearized hidden delta is separately compared with observed BF16 logits delta in the JSON, avoiding inference from maxima alone.
