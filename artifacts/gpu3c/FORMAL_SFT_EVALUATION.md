# 正式 SFT 与评测

状态：`FORMAL_BACKGROUND_PENDING`。

冻结方案为 Qwen3-0.6B Base-init、LoRA、seed 42、context 512；B/C 各 64 optimizer steps，在 step 32/64 保存 checkpoint。评测固定 4 道既有 benchmark，每臂分别运行 C0 自主生成和 S3 外部答案停止，共 16 条生成。

正式后台任务完成后，本文件由工作进程写入实际训练成本、C0/S3 自主停止、正确率、完整率、重复率和冲突指标；计划值不会被当作结果。
