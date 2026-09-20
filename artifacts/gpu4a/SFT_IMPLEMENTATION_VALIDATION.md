# Native-model SFT implementation validation

The implementation reuses `MiniLLM.forward`, `causal_lm_loss`, `ForgeTrainer`, AdamW, BF16, checkpointing, and `StatefulRandomSampler`. The GPU-4A adapter only removes non-model metadata before forward and aggregates loss by valid target tokens; it does not introduce a Hugging Face or Qwen model path.

The audited batches use `-100` labels for prompt and padding positions. Response tokens and native EOS token 2 are supervised. All 6,200 examples preserve their complete targets without truncation. The optimizer and trainable sets both contain 37,462,528 parameters.

Tiny Overfit used the real formal pretrained initialization. Loss fell from 5.7065 to 0.001684; target-token accuracy rose from 21.9% to 100%, and exact generation rose from 0/8 to 8/8. The step-60 checkpoint restored an identical model hash and identical generation, then resumed successfully to step 61.
