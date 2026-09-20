# 自研 37M 预训练模型审计

正式 checkpoint：`D:\Agents\Codex\MiniLLM-Forge\runs\E01-minillm-formal\best.pt`，SHA-256 `ee306384fcdef646f557cf9aa1165c464b2ca373c6d64d81f5df3b19b8f859fc`。训练 step 3052，处理 50,003,968 tokens。

模型共有 37,462,528 参数：8 层、hidden 512、GQA 8/4、FFN 1536、RoPE/RMSNorm/SwiGLU、1024 context、词嵌入与输出层共享。24K byte-level BPE 的 `<pad>/<bos>/<eos>/<unk>` 分别为 0/1/2/3。结构、Tokenizer 与 checkpoint 严格匹配。

正式预训练验证 loss/PPL 为 4.6484/104.42；NaN/Inf/OOM 均为 0。三个固定 English prompt 均成功生成、logits 有限且 token ID 合法。基础生成仍有重复和语义不可靠现象，这符合 50M-token 小模型历史结论，不冒充指令能力。
