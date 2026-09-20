# GPU-3C 最终报告

Classification: `GPU3C_CONTROLLED_NEGATIVE_RESULT`

## 执行摘要

GPU-3C 完成了数据/模板审计、B/C 单变量训练、D1 静态完整性方案、正式 64-step 配对训练及 C0/S3 评测。核心假设未得到支持：C-EOS-SUPERVISED 没有生成任何 `<|im_end|>`，并从 B 的 1/4 触顶恶化为 4/4 触顶。

## 数据与监督

19,162 条实际 512-context 样本中，12,469 条监督 `<|im_end|>`，覆盖率 65.071%；6,693 条在 assistant 结束标记前截断。另有 38 条 prompt 耗尽 context、未进入实际训练。D1 保留 12,469 条完整样本并达到 100% EOS 覆盖，但有效监督 token 从 5,177,783 降至 2,583,463，且来源/长度分布改变，因此没有把 D1 当作 EOS 单变量效果。

## 受控训练与自主终止

B/C 的 12,469 条输入和顺序完全一致，只在每条最终 EOS label 处有一个差异。两臂各完成 64 steps、1,024 examples；B/C 最终 loss 分别为 1.0788/1.1724，但因有效监督 token 分母不同，不以 loss 判定效果。

C0 的 B→C：`<|im_end|>` 0→0，`<|endoftext|>` 3→0，触顶 1→4，提前 EOS 2→0，生成 token 均值 175.0→512.0，3-gram 重复率 0.387→0.858。C 组虽消除了提前 EOS，却以完全不终止和严重答案后重复为代价。

## 答案质量与推理系统

C0 First-valid 正确数为 1→3，Conflict-aware 正确数为 1→2；样本仅 4 道，不能据此声称任务能力提升。C 组 math500-0181 在 C0 产生冲突。S3 对 C 组 3/4 成功外部停止，并将平均长度降至 184.25 token，同时消除了该冲突；这是推理系统效果，不计入自主 EOS。

## 成本、限制与建议

两臂训练加评测总耗时 1007.8s；后台任务端到端 1012.6s；峰值 CUDA allocated/reserved 为 1291.2/2218.0 MiB，无 OOM、NaN 或 Inf。最终回归 120 passed。

本轮只有 seed 42 和 4 道既有 benchmark，未运行 D1 与原始数据的独立 A/D 训练，也未扩展至 1,200 steps。鉴于预期 EOS 效果方向未出现，当前不建议直接追加多 seed 或更大训练预算。建议保留 S3 作为工程止损，并转入第二阶段 37M Transformer SFT、SDPA/FlashAttention 与 torch.compile；只有提出新的、可检验的结束 token/模板机制假设后，才值得恢复 Qwen3-0.6B 终止优化。
