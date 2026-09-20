# EOS 单变量实验

状态：`FORMAL_BACKGROUND_PENDING`。

训练前 B/C 审计已通过：12,469 条完整样本的 `input_ids`、`attention_mask` 和顺序完全一致；labels 共 12,469 个差异，每条仅有最终 assistant `<|im_end|>` 一个位置。B 有效监督 token 为 2,570,994，C 为 2,583,463。

两臂 1-step smoke 均完成，无 OOM、NaN 或 Inf；LoRA 权重更新、checkpoint 保存/加载及短生成均通过。正式 64-step 配对训练完成后，本文件由后台工作进程写入实际 C0/S3 结果。
