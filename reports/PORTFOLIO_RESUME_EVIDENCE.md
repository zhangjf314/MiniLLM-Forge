# MiniLLM-Forge：中文简历事实材料

## 项目名称

MiniLLM-Forge｜Transformer 预训练、数学领域微调与训练优化实验平台

本文只提供可选事实材料，不替换现有简历。所有数字都能追溯到仓库内报告、结构化
artifact 或本地哈希登记的原始实验证据。

## Agent / AI 应用开发岗候选亮点

1. 从零实现 decoder-only Transformer 训练栈，覆盖 RMSNorm、RoPE、GQA、SwiGLU、
   causal mask、梯度累积、AMP、checkpoint/resume、数据去重与污染审计；完成
   50,003,968 token 的 MiniLLM 正式预训练，并用固定验证集和精确恢复对照闭环。
   证据：`reports/MINILLM_FORMAL_PRETRAINING.md`、
   `artifacts/training/minillm_final_metrics.json`。
2. 搭建 Qwen3-0.6B 数学领域 CPT 与 PEFT-SFT 实验流水线，在 8GB RTX 5060 Laptop
   GPU 上完成 Base/CPT × LoRA/QLoRA × 3 seeds 的 12-run 正式训练，所有 run 均
   完成 1200 updates，训练期 NaN/Inf/OOM 为 0。证据：
   `artifacts/gpu2d_formal/training_matrix.json`、12 个
   `experiments/results/*.json`。
3. 设计可脱离交互会话运行的 Windows Task Scheduler 监督器与 append-only 恢复式
   评测流程；完成 24 个正式 job、8400 条生成记录，并对 checkpoint、adapter、
   prompt、scorer、problem IDs 和恢复事件逐项审计。证据：
   `artifacts/gpu2d_formal/final_result.json`、
   `reports/GPU2D_FORMAL_PEFT_TRANSFER.md`。
4. 面对不符合预期的结果，坚持 outcome-blind 协议与失败留痕：MATH-500 主评测中
   LoRA、QLoRA 的 CPT-vs-Base paired delta 均为 3/3 seeds 负向，最终据实分类为
   `CPT_TO_PEFT_SFT_TRANSFER_METHOD_DEPENDENT`，没有包装为能力提升。证据：
   `artifacts/gpu2d_formal/final_result.json` 的 `paired_transfer` 与
   `scientific_classification`。

## 算法工程师岗候选亮点

1. 预注册并执行 Base/CPT × LoRA/QLoRA × 3 seeds 的配对实验，使用完整 MATH-500
   与 outcome-blind 固定 GSM8K 200 题子集；从 problem-level correctness vector
   重算准确率、paired delta、seed SD 与 10,000 次 paired bootstrap CI。证据：
   `scripts/finalize_gpu2d.py`、`artifacts/gpu2d_formal/final_result.json`。
2. 在同一真实硬件上量化质量—资源权衡：LoRA/QLoRA 平均训练吞吐分别为
   735.05/587.58 target tok/s，训练物理显存峰值范围分别为 3428–3686/
   5792–6116 MiB；没有预设 QLoRA 必然省显存。证据：
   `reports/GPU2D_FORMAL_PEFT_TRANSFER.md` 第 14 节。
3. 对 8400 条生成进行完整性与协议身份审计：6000 条 MATH-500、2400 条 GSM8K
   fixed-200，missing=0、duplicates=0；重新验证冻结 tokenizer 解码、答案抽取与
   exact match。证据：`artifacts/gpu2d_formal/final_result.json` 的
   `completion_audit` 与 `problem_integrity_audit`。
4. 识别并量化推理阶段的关键威胁：8398/8400 条生成达到 512-token ceiling，
   36 条答案抽取失败；通过原始 generation 人工核查题意错误、推理错误、重复生成与
   Base/CPT correctness flip，不把 final correctness 自动解释为错误原因。证据：
   `reports/GPU2D_FORMAL_PEFT_TRANSFER.md` 第 13 节。

## 面试可讲的代表性技术问题与解决过程

### 1. 8GB GPU 无法承载原始 1024-context 协议

- 问题：GPU-2B/GPU-2C 的原始路径无法通过长期稳定性与物理显存余量门禁。
- 处理：保留失败记录，前瞻性冻结 512-context 科学暴露变更；重新资格验证 LoRA、
  QLoRA、batch size、恢复语义与物理显存阈值。
- 边界：不把 512-context 结果外推到 1024 context，也不声称 Full SFT 已完成。
- 证据：`reports/GPU2D_8GB_NATIVE_RECOVERY.md`、
  `artifacts/gpu2d_formal/validation_context_512.json`。

### 2. 预计 227 小时的评测成本不可接受

- 问题：原始全量双 benchmark 方案预计约 227.13 小时。
- 处理：在未观察正式结果前，将辅助 GSM8K 冻结为 outcome-blind 200 题子集，
  保留 MATH-500 全量主评测；冻结顺序、subset hash、job IDs 与恢复合同。
- 结果：24 个正式 job 实际累计约 80.13 小时，仍保持 8400 条可配对记录。
- 证据：`reports/GPU2D_EVALUATION_COST_AMENDMENT.md`、
  `artifacts/gpu2d_formal/evaluation_order_amended.json`。

### 3. 长时 Windows 任务如何避免依赖交互会话

- 问题：约三天的评测不能依赖 Codex shell 存活，也不能因断开会话丢失进度。
- 处理：使用 Windows Task Scheduler 托管单实例 supervisor，每题 append-only 写入，
  每 job 用 completion artifact 和 output hash 封口；以冻结 ID 集合恢复缺失题而非重写
  已完成题。
- 结果：单次 supervisor 启动完成 24/24 job，计划任务结果为 0，无重复 job 或恢复
  混用身份。
- 证据：`artifacts/gpu2d_formal/final_result.json` 的 campaign/recovery audit。

### 4. 结果不支持预期时如何做科学收尾

- 问题：Math-CPT 的领域 PPL 改善不等于下游数学推理提升。
- 处理：以 MATH-500 为主评测，按相同 seed 配对 Base/CPT；同时报告 GSM8K
  fixed-200、problem bootstrap 与 seed 离散程度，并严格区分二者。
- 结果：LoRA 为负向、QLoRA 为混合，最终分类为方法依赖，未捏造正向提升。
- 证据：`reports/GPU2D_FORMAL_PEFT_TRANSFER.md` 第 8–15 节。

### 5. 生成式评测准确率为什么可能不稳定

- 问题：资格测试 64/64 触顶后，不能直接假设正式 8400 条也全部触顶。
- 处理：从每条 `generated_token_ids` 重算长度并核查实际生成；抽样阅读生成与参考答案，
  分离 extraction failure、incorrect extracted answer 与 correctness flip。
- 结果：正式记录中 8398/8400 触顶，明确把异常终止与重复生成列为解释限制。
- 证据：`artifacts/gpu2d_formal/final_result.json` 的 `error_analysis`。

## 使用边界

简历中可以写“完成多 seed 正式训练与可审计生成式评测”“识别方法依赖/负向 transfer”
以及真实显存和吞吐数字；不能写“Math-CPT 提升数学推理”“GSM8K 全量准确率”“QLoRA
更省显存”“Full SFT 已完成”或“1024-context 已验证”。
