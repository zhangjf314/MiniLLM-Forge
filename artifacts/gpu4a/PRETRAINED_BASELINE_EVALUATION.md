# Pretrained-model instruction baseline

The frozen formal checkpoint `ee306384fcdef646f557cf9aa1165c464b2ca373c6d64d81f5df3b19b8f859fc` was evaluated on 384 frozen generation examples: 64 examples per task and split. Evaluation used the same instruction template, greedy decoding, native EOS token 2, and a 12-token bound used after SFT.

Every task/split cell scored 0/64 for correctness and 0% for valid output format. All generations reached the token limit and none emitted EOS. Repetition rates varied by cell. This is the actual instruction-following baseline and does not indicate a checkpoint-loading failure: logits and token IDs were finite and valid in the separate model audit.
