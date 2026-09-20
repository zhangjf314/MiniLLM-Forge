# Capability comparison before and after SFT

| Split | Task | Base correct | SFT correct | SFT format | SFT EOS |
|---|---|---:|---:|---:|---:|
| train | T1_SUPPORT_CLASSIFICATION | 0/64 | 64/64 | 100% | 100% |
| train | T2_INTEGER_ADDITION | 0/64 | 4/64 | 100% | 100% |
| validation | T1_SUPPORT_CLASSIFICATION | 0/64 | 64/64 | 100% | 100% |
| validation | T2_INTEGER_ADDITION | 0/64 | 5/64 | 100% | 100% |
| test | T1_SUPPORT_CLASSIFICATION | 0/64 | 64/64 | 100% | 100% |
| test | T2_INTEGER_ADDITION | 0/64 | 2/64 | 100% | 100% |

T1 reached 64/64 on all three generated subsets. Its test split uses sentence phrasings absent from training, and the test confusion matrix is diagonal. T2 reached only 4/64 on train, 5/64 on validation, and 2/64 on independent test. The model therefore learned valid numeric formatting and EOS behavior but did not reliably learn two-digit addition.

Every SFT generation was valid, complete, EOS-terminated, below the length bound, and free of detected repetition. T1 supports a narrow claim of input-dependent classification generalization: the category keywords and semantics were present during training, so this is not evidence of broad language understanding. The T2 result also shows why lower validation loss and corrected output format must be reported separately from task correctness.
