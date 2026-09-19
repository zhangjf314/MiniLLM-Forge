# GPU-2D Formal PEFT Transfer

## 1. Research Question

在冻结的 512-context 数学 SFT 暴露和 PEFT 协议下，Math-CPT 初始化是否改善
下游数学任务表现？研究分别考察 LoRA 与 QLoRA，并把 MATH-500 全量准确率作为
主评测、GSM8K fixed-200 subset accuracy 作为辅助评测。

## 2. Historical Experimental Boundary

历史证据 `MATH_DOMAIN_ADAPTATION = CONFIRMED` 仅说明 Math-CPT 改变了数学领域
语言建模表现，不能自动推出数学推理提升。GPU-2B/GPU-2C 的 1024-context 路径因
8GB GPU 内存资格门禁失败而保留为失败记录；本研究是经预先授权的 512-context
科学暴露变更。Full SFT 正式 transfer study 未完成。

## 3. Frozen 512-Context Design

- 2 initializations（Base、Math-CPT）× 2 methods（LoRA、QLoRA）× 3 seeds
  （42、31415、271828）= 12 个训练 run。
- 每个 run 1200 updates；相同的冻结训练/验证数据与 seed 配对规则。
- generation 使用 `REFERENCE_EVALUATOR`、batch size 1、greedy decoding、
  `max_new_tokens = 512`。
- MATH-500 使用完整 500 题；GSM8K 使用 seed 20260916 冻结的 200 题子集，
  ID hash 为 `fb9635d80b6210da35a9603da67f16148b5f7e4ca2ba140638865878c1c4cfc8`。
- claim boundary：`CPT_TO_PEFT_SFT_TRANSFER_UNDER_FROZEN_512_CONTEXT`。

## 4. Formal Training Campaign

12/12 run 均为 `VALID_COMPLETED_RUN`，每个完成 1200 updates，NaN、Inf、OOM
均为 0。训练累计 wall time 为 27.28 小时。两次历史中断被原样保留：
`C-LORA-s42` 的 PowerShell launcher 在模型加载前失败；`B-QLORA-s271828`
在 update 213 前后发生无 Python traceback 的进程树终止。二者均在明确授权后从
update 0 以同一冻结身份重跑，没有覆盖已完成的有效 run。

## 5. Evaluation Cost Amendment

原始全量 GSM8K 方案投影约 227.13 小时。正式执行前、未观察正式结果时，协议修改为
MATH-500 全量与 GSM8K 固定 200 题子集，投影 87.3158 小时。实际 24 个 job 的
累计 job wall time 为 80.13 小时；该修改没有改变 prompt、decoding、scorer、
answer extractor 或 `max_new_tokens`。

## 6. Evaluation Protocol and Integrity Audit

审计从原始 problem-level JSONL 重新计算，而不是复制 completion summary：

- Windows Scheduled Task 最终返回 0；supervisor 为 `COMPLETED`，24 次 start 与
  24 次 valid completion 一一对应，无 supervisor restart。
- 24/24 completion artifact 的 output hash、checkpoint、adapter、config、HEAD、
  source-tree、problem-ID set 和物理显存日志均匹配。
- 12 个最终 checkpoint 及 12 个 adapter tree 已重新计算 SHA-256 并匹配训练
  summary；Base/CPT、LoRA/QLoRA 与 seed 身份无错配。
- 6000/6000 MATH-500 与 2400/2400 GSM8K fixed-200 记录有效；missing = 0，
  duplicates = 0。GSM8K 不仅数量为 200，其 ID 集合也与冻结 manifest 完全一致。
- 每条记录重新验证 prompt hash/token IDs、生成 token 解码、答案抽取与 exact match。

因此 F0 gate 为 `GPU2D_FORMAL_EVALUATION_VALIDATED`。

## 7. Complete 12-Run Result Matrix

| Run | MATH-500 | Accuracy | GSM8K fixed-200 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| B-LORA-s42 | 105/500 | 21.0% | 57/200 | 28.5% |
| C-LORA-s42 | 104/500 | 20.8% | 51/200 | 25.5% |
| B-LORA-s31415 | 102/500 | 20.4% | 65/200 | 32.5% |
| C-LORA-s31415 | 99/500 | 19.8% | 41/200 | 20.5% |
| B-LORA-s271828 | 118/500 | 23.6% | 60/200 | 30.0% |
| C-LORA-s271828 | 107/500 | 21.4% | 58/200 | 29.0% |
| B-QLORA-s42 | 92/500 | 18.4% | 42/200 | 21.0% |
| C-QLORA-s42 | 86/500 | 17.2% | 45/200 | 22.5% |
| B-QLORA-s31415 | 96/500 | 19.2% | 43/200 | 21.5% |
| C-QLORA-s31415 | 86/500 | 17.2% | 31/200 | 15.5% |
| B-QLORA-s271828 | 88/500 | 17.6% | 42/200 | 21.0% |
| C-QLORA-s271828 | 85/500 | 17.0% | 61/200 | 30.5% |

## 8. MATH-500 Primary Results

LoRA 的 Base/CPT 三 seed 平均准确率为 21.67%/20.67%，平均 paired delta 为
-1.00 个百分点。QLoRA 为 18.40%/17.13%，平均 delta 为 -1.27 个百分点。
两个方法的三个 seed 均为负向；主评测没有提供 Math-CPT 正向 transfer 的证据。

## 9. GSM8K Fixed-200 Secondary Results

LoRA 的 Base/CPT 平均 subset accuracy 为 30.33%/25.00%，平均 delta 为
-5.33 个百分点。QLoRA 为 21.17%/22.83%，平均 delta 为 +1.67 个百分点，
但三个 seed 的方向强烈冲突。以上均是固定 200 题子集结果，不是 GSM8K 全量
test accuracy。

## 10. LoRA Paired Transfer

| Benchmark | Seed | Base | CPT | Correct Δ | Accuracy Δ (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MATH-500 | 42 | 21.0% | 20.8% | -1 | -0.2 |
| MATH-500 | 31415 | 20.4% | 19.8% | -3 | -0.6 |
| MATH-500 | 271828 | 23.6% | 21.4% | -11 | -2.2 |
| GSM8K fixed-200 | 42 | 28.5% | 25.5% | -6 | -3.0 |
| GSM8K fixed-200 | 31415 | 32.5% | 20.5% | -24 | -12.0 |
| GSM8K fixed-200 | 271828 | 30.0% | 29.0% | -2 | -1.0 |

MATH-500 delta sample SD 为 1.06 pp、范围 [-2.2, -0.2]；GSM8K fixed-200
为 5.86 pp、范围 [-12.0, -1.0]。两个 benchmark 均为 3/3 负向，LoRA family
outcome 为 `NEGATIVE`。

## 11. QLoRA Paired Transfer

| Benchmark | Seed | Base | CPT | Correct Δ | Accuracy Δ (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MATH-500 | 42 | 18.4% | 17.2% | -6 | -1.2 |
| MATH-500 | 31415 | 19.2% | 17.2% | -10 | -2.0 |
| MATH-500 | 271828 | 17.6% | 17.0% | -3 | -0.6 |
| GSM8K fixed-200 | 42 | 21.0% | 22.5% | +3 | +1.5 |
| GSM8K fixed-200 | 31415 | 21.5% | 15.5% | -12 | -6.0 |
| GSM8K fixed-200 | 271828 | 21.0% | 30.5% | +19 | +9.5 |

MATH-500 delta sample SD 为 0.70 pp、范围 [-2.0, -0.6]，3/3 负向；
GSM8K fixed-200 为 7.75 pp、范围 [-6.0, +9.5]，方向为 2 正 1 负。
QLoRA family outcome 为 `MIXED`。

## 12. Seed Variability and Paired Bootstrap

每个区间使用相同题目的 Base/CPT correctness vector、10,000 次 percentile paired
bootstrap；seed 规则为 `20260913 + training_seed`。

| Method | Benchmark | Seed | Δ (pp) | 95% CI (pp) |
| --- | --- | ---: | ---: | ---: |
| LoRA | MATH-500 | 42 | -0.2 | [-3.2, 2.6] |
| LoRA | MATH-500 | 31415 | -0.6 | [-3.8, 2.6] |
| LoRA | MATH-500 | 271828 | -2.2 | [-5.4, 0.8] |
| LoRA | GSM8K fixed-200 | 42 | -3.0 | [-9.0, 3.0] |
| LoRA | GSM8K fixed-200 | 31415 | -12.0 | [-18.0, -6.0] |
| LoRA | GSM8K fixed-200 | 271828 | -1.0 | [-6.0, 4.0] |
| QLoRA | MATH-500 | 42 | -1.2 | [-4.2, 1.8] |
| QLoRA | MATH-500 | 31415 | -2.0 | [-5.2, 1.2] |
| QLoRA | MATH-500 | 271828 | -0.6 | [-3.8, 2.4] |
| QLoRA | GSM8K fixed-200 | 42 | +1.5 | [-4.5, 7.5] |
| QLoRA | GSM8K fixed-200 | 31415 | -6.0 | [-11.5, -1.0] |
| QLoRA | GSM8K fixed-200 | 271828 | +9.5 | [3.0, 16.5] |

单 seed 的题目层面区间不等同于训练流程的 seed 不确定性。只有三个训练 seed，故不做
事后显著性检验，也不把个别不跨零的区间包装为稳定的训练收益。

## 13. Error and Truncation Analysis

8400 条生成中有 8398 条（99.976%）到达 512-token ceiling；仅两条提前终止。
答案抽取失败 36 条；另有 6600 条成功抽取但 exact match 错误。该异常终止行为是解释
准确率时的主要限制，不能把所有错误都归因为数学推理能力。

对原始生成做了有边界的人工检查：

- `B-LORA-s42 / math500-0447`：模型错误地把 13 的倍数约束解释为倍数系数也需
  为 13，随后重复 `ANSWER: 18 boys` 直至截断；末尾残缺的 `ANSWER:` 使冻结
  extractor 返回空字符串。这同时是逻辑错误、重复生成与 extraction failure 示例。
- `B-LORA-s42 / math500-0005`：把正六边形边长与小等边三角形周长混淆，计算
  `6 × 21 = 126`，参考答案为 42；这是可追溯的题意/几何推理错误。
- `math500-0018`：同题同 seed 下，Base-LoRA 正确抽取 28，而 CPT-LoRA 将
  124° 错当作等腰三角形两个底角并得到不可能的 -68° 剩余角，最终输出 124；这是
  `Base correct / CPT wrong` 的可追溯推理错误。

`final_result.json` 还保留 `Base wrong / CPT correct`、反向 flip、抽取失败和 ceiling
示例的 run ID、problem ID 与 generation artifact 路径。未人工查看的错误仅按
“抽取失败”或“抽取后错误”统计，不自动臆测为算术/推理/格式类别。

## 14. Training and Inference Efficiency

| Metric | LoRA | QLoRA |
| --- | ---: | ---: |
| Mean final train loss | 0.9047 | 0.9249 |
| Mean validation loss | 0.9271 | 0.9472 |
| Cumulative training wall time | 12.25 h | 15.03 h |
| Mean training throughput | 735.05 target tok/s | 587.58 target tok/s |
| Training physical VRAM range | 3428–3686 MiB | 5792–6116 MiB |
| Minimum training headroom | 4125 MiB | 1695 MiB |
| Trainable parameters | 10,092,544 | 10,092,544 |
| Adapter size | ≈40.43 MB | ≈40.43 MB |
| Evaluation generated-token throughput | 16.88 tok/s | 13.48 tok/s |
| Evaluation physical VRAM range | 1494–1752 MiB | 2102–2256 MiB |
| Evaluation job wall time | 35.62 h | 44.51 h |

在这台真实 8GB RTX 5060 Laptop GPU 和 Windows/WDDM 软件栈上，QLoRA 并未
更省物理显存，反而训练物理峰值更高、吞吐更低、耗时更长。这个结论只适用于本次
量化后端和冻结配置，不能泛化为 QLoRA 的普遍性质。质量上，LoRA 的 MATH-500
绝对准确率整体高于 QLoRA；但这不是跨方法随机化因果比较。

## 15. Scientific Classification

最终分类为：

`CPT_TO_PEFT_SFT_TRANSFER_METHOD_DEPENDENT`

理由是：LoRA 在主、辅 benchmark 的三个 seed 均负向，符合冻结规则的
`NEGATIVE`；QLoRA 的 MATH-500 三 seed 均负向，但 GSM8K fixed-200 在 -6.0 至
+9.5 pp 间强烈冲突，只能判为 `MIXED`。因此 Math-CPT 没有获得一致的正向支持，
也不能把两个方法合并成无条件的单一负向结论。

## 16. Limitations and Threats to Validity

- 结论仅适用于 Qwen3-0.6B、冻结的 512-context 数据暴露、LoRA/QLoRA 与三个 seed。
- GSM8K 结果来自固定 200 题子集，不是全量 test set。
- 99.976% 的生成触顶，说明终止/重复/截断问题可能压低并扰动准确率。
- 公共 benchmark 可能出现在基础模型预训练语料中；本地去重不能证明基础模型无污染。
- Full SFT、1024 context、其他模型规模、其他领域与一般数学推理能力均不在结论范围。
- 问题层面 bootstrap 不估计训练 seed 或完整训练流程的不确定性。

## 17. Portfolio Interpretation

可信的 Portfolio 叙述重点是：构建了可审计的 12-run 多 seed PEFT 训练与 24-job
生成式评测系统；在结果不支持预期时保留失败与恢复证据，并通过原始 8400 条记录、
checkpoint hash、paired analysis 和误差追踪给出诚实结论。不能宣称 Math-CPT 提升了
数学推理，也不能宣称 QLoRA 在本机更省显存。

## 18. Conclusion

实验闭合且证据完整，但科学结论不是“CPT 有效”。主评测 MATH-500 在 LoRA、QLoRA
下均为三个 seed 全部负向；辅助 GSM8K fixed-200 对 LoRA 负向、对 QLoRA 高度混合。
在冻结 512-context PEFT-SFT 协议下，Math-CPT transfer 呈方法依赖，未获得稳定正向
支持。完整机器可读证据见 `artifacts/gpu2d_formal/final_result.json`。
