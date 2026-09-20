# EOS and SFT Audit

## Token contract

- Tokenizer EOS/PAD: `<|endoftext|>` = 151643; BOS and UNK are unset in tokenizer metadata.
- Chat message terminator: `<|im_end|>` = 151645. The chat template appends it to assistant replies.
- Formal decoding accepts both 151643 and 151645, so no static decoder EOS-recognition defect was found.
- Decoded text was stored with `skip_special_tokens=True`; token IDs, not text, are authoritative for EOS.

## Frozen training-batch evidence

The frozen cache is the actual input to `ContextSFTDataset`; labels clone input IDs and mask only
the prompt prefix. Micro-batch size is one, so the GPU-2D collator adds no padding. Hugging Face
causal-LM forward computes the causal shift; the project does not perform a second shift.

| Context | Actual training samples | Supervised assistant end | Missing over actual samples | Coverage |
|---:|---:|---:|---:|---:|
| 512 | 19162 | 12469 | 6693 | 65.071% |
| 768 | 19191 | 15880 | 3311 | 82.747% |

At context 512, 38 source samples lose every assistant token and are excluded. Of the 19,162
actual samples, 6,693 lack a supervised `<|im_end|>` after truncation. This is direct evidence of
partial—not absent—end-token supervision. PAD is not masked by token identity; labels are masked by
position, avoiding the usual `pad_token_id == eos_token_id` masking bug.

## SFT truth status

| Claim | Status | Evidence |
|---|---|---|
| SFT_IMPLEMENTED | PASS | Chat formatting, assistant-only labels, collator and causal-LM loss path exist. |
| SFT_TRAINING_EXECUTED | PASS | 12 LoRA/QLoRA runs reached 1,200 updates with checkpoints/adapters. |
| SFT_SUPERVISION_VALIDATED | PASS | Frozen batches prove assistant supervision; EOS coverage is only 65.071%. |
| SFT_EFFECTIVENESS_VALIDATED | PASS | Paired benchmark results exist for PEFT only; no Full-SFT arm and ceiling contact limits interpretation. |

This is genuine instruction SFT for the completed PEFT arms, not merely an implemented SFT module.
Full SFT remains `NOT_EXECUTED`.
