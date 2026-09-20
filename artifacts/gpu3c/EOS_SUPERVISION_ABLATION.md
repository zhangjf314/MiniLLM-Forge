# EOS 单变量实验

B/C 共用 12,469 条样本和相同顺序；input_ids、attention_mask 完全一致，共 12,469 个 label 差异，均为每条样本末尾 `<|im_end|>`。

| C0 指标 | B EOS-MASKED | C EOS-SUPERVISED |
|---|---:|---:|
| `<|im_end|>` 自主终止 | 0/4 | 0/4 |
| `<|endoftext|>` 自主终止 | 3/4 | 0/4 |
| 512-token 触顶 | 1/4 | 4/4 |
| 提前 EOS | 2/4 | 0/4 |
| First-valid 正确 | 1/4 | 3/4 |
| Conflict-aware 正确 | 1/4 | 2/4 |
| 3-gram 重复率均值 | 0.387 | 0.858 |
| 生成 token 均值 | 175.0 | 512.0 |

两臂均完成 64 个 optimizer steps，初始 adapter 哈希相同，且均无 OOM、NaN 或 Inf。Loss 按有效 label token 取平均，B/C 有效监督 token 数不同，因此 loss 不作为效果结论。

结论：监督目标 `<|im_end|>` 没有使模型在冻结题目上生成该 token；反而 C 组 4/4 触顶，且重复率显著升高。本轮不支持预期的 EOS 监督收益。正确率上升来自仅 4 道题的单 seed 结果，不能抵消自主终止失败，也不足以形成能力提升结论。
