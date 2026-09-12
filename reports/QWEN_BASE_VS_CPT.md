# Qwen Base vs Math CPT

| Metric | Base | CPT | Delta |
| --- | ---: | ---: | ---: |
| Math validation loss | 1.643171 | 1.616370 | -0.026800 |
| Math PPL | 5.1715 | 5.0348 | -0.1368 (-2.64%) |
| General validation loss | 2.837904 | 2.846717 | +0.008813 |
| General PPL | 17.0799 | 17.2311 | +0.1512 (+0.89%) |
| Controlled math EM | N/A | N/A | N/A |
| Peak allocated VRAM | 2148.3 MiB | 4406.9 MiB | — |
| Median training throughput | — | 1972.91 tokens/s | — |
| Training tokens | 0 | 10,002,432 | +10,002,432 |

Primary classification: **QWEN_MATH_CPT_VALIDATED_WITH_GENERAL_DEGRADATION**.

Lower math-domain perplexity demonstrates domain language-model adaptation; it does not,
by itself, prove improved mathematical reasoning.
