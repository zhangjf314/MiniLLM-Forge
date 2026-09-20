# GPU-4A SFT 数据与任务协议

T1 是 BILLING/TECHNICAL/ACCOUNT 三分类；训练和测试使用不同句式，但共享类别语义词。T2 是 0–99 两整数加法；三个 split 的操作数对严格不重叠，测试答案仍位于训练答案分布。

划分为 train/validation/test = 4,800/700/700。语义 key 交叉计数均为 0。最长 prompt/target/sequence 为 66/6/71 tokens，低于 128，无截断。全部 6,200 条样本目标完整且监督原生 EOS=2。

格式固定为 Instruction/Input/Response 纯文本，不扩展词表、不使用 Qwen chat template。Prompt 和 PAD labels 为 -100；Response 正常 token 与最后 EOS 参与 causal LM loss。测试集仅在协议和 checkpoint 选择规则冻结后用于最终确认。
