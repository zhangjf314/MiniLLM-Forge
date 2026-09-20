# GPU-3C 训练数据与模板审计

## 数据长度与截断

对冻结训练源的 19,200 条样本使用实际 Qwen chat template 重新序列化。512 context 下有 19,162 条进入原训练，12,469 条保留并监督目标 assistant `<|im_end|>`，覆盖率为 65.071%；6,693 条实际训练样本缺少目标结束监督。

| 指标 | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Prompt tokens | 98 | 169 | 201 | 305 | 1008 |
| Assistant tokens | 289 | 842 | 1050 | 1532 | 3860 |
| Full sequence | 400 | 973 | 1175 | 1673 | 4011 |

截断分类：`{'ASSISTANT_TRUNCATED_BEFORE_END': 6693, 'NONE': 12469, 'PROMPT_EXHAUSTS_CONTEXT': 38}`。右截断不会追加或伪造 EOS；prompt 已耗尽 context 的样本不进入实际训练，其余 EOS 缺失来自 assistant 在目标结束标记前被截断。完整答案边界使用最终答案 token 子序列定位；无法可靠定位的样本标记 UNKNOWN，不用 `ANSWER:` 的存在性冒充完整性。

本次语义完整答案计数为 12,951；GPU-2D 旧统计多 1 条，是因为旧逻辑把“完整答案为空且截断答案也为空”判为相等。本审计要求非空最终答案，因此不计该样本。

## 模板与特殊 token

- `<|endoftext|>` / tokenizer EOS / PAD：151643。
- Assistant 消息结束 `<|im_end|>`：151645；这是本阶段直接控制的训练 label。
- 训练和推理均使用冻结 Qwen chat template，系统提示一致。
- 冻结 Base tokenizer chat template 没有显式 thinking-mode 开关；本阶段不改变回答结构或引入 thinking 配置。
- Prompt labels 按位置设为 -100；真实 EOS 不按 token identity 批量屏蔽，避免 PAD==EOS 时误伤。

## 数据隔离与 D1 静态方案

训练题与 GSM8K/MATH-500 共 1,819 道评测题进行规范化内容哈希检查，精确碰撞 0 条。GPU-3C 固定 4 道既有 benchmark 题，只用于冻结确认，不宣称完全未接触 holdout。

D1 完整样本过滤保留 12,469/19,200 条，过滤 6,731 条，EOS 覆盖率变为 100%。这同时改变样本长度和来源分布，因此只能作为数据处理综合方案，不能用来估计 EOS 的单变量因果效应。B/C 对照使用相同的这 12,469 条样本，唯一主动差异是最终 `<|im_end|>` label 是否为 -100。
