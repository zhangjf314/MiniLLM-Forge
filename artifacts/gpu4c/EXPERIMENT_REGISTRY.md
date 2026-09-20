# Experiment Registry

Negative, blocked, historical, and diagnostic runs are retained.

| ID | Stage | Model | Initialization | Method | Seed | Steps | Status | Classification | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiny-overfit | GPU-1 | MiniLLM tiny | random | pretraining correctness gate | 1337 | 60 | COMPLETED | PASS | runs/tiny-overfit-verified/summary.json |
| E01 | GPU-1 | MiniLLM 37M | random | BF16 pretraining | 1337 | 3052 | COMPLETED | MINILLM_FORMAL_PRETRAINING_VALIDATED | artifacts/training/minillm_formal_result.json |
| E04 | GPU-2A | Qwen3-0.6B-Base | base | full-parameter CPT | 42 | 2442 | COMPLETED | QWEN_MATH_CPT_VALIDATED_WITH_GENERAL_DEGRADATION | artifacts/training/qwen_math_cpt_result.json |
| GPU2B-memory | GPU-2B | Qwen3-0.6B | base/CPT | Full SFT qualification | 42 | None | BLOCKED | STAGE_GPU_2B_BLOCKED_BY_MEMORY_QUALIFICATION | artifacts/training/gpu2b-stage-result.json |
| GPU2C-qualification | GPU-2C | Qwen3-0.6B | CPT | PEFT qualification | 42 | None | BLOCKED | GPU2C_BLOCKED_BY_MEMORY_QUALIFICATION | artifacts/gpu2c/qualification-result.json |
| B-LORA-s42 | GPU-2D | Qwen3-0.6B-Base | base | LORA | 42 | 1200 | COMPLETED | VALID_FORMAL_PEFT_RUN | experiments/results/B-LORA-s42.json |
| C-LORA-s42 | GPU-2D | Qwen3-0.6B-Base | math-CPT | LORA | 42 | 1200 | COMPLETED | VALID_FORMAL_PEFT_RUN | experiments/results/C-LORA-s42.json |
| B-LORA-s31415 | GPU-2D | Qwen3-0.6B-Base | base | LORA | 31415 | 1200 | COMPLETED | VALID_FORMAL_PEFT_RUN | experiments/results/B-LORA-s31415.json |
| C-LORA-s31415 | GPU-2D | Qwen3-0.6B-Base | math-CPT | LORA | 31415 | 1200 | COMPLETED | VALID_FORMAL_PEFT_RUN | experiments/results/C-LORA-s31415.json |
| B-LORA-s271828 | GPU-2D | Qwen3-0.6B-Base | base | LORA | 271828 | 1200 | COMPLETED | VALID_FORMAL_PEFT_RUN | experiments/results/B-LORA-s271828.json |
| C-LORA-s271828 | GPU-2D | Qwen3-0.6B-Base | math-CPT | LORA | 271828 | 1200 | COMPLETED | VALID_FORMAL_PEFT_RUN | experiments/results/C-LORA-s271828.json |
| B-QLORA-s42 | GPU-2D | Qwen3-0.6B-Base | base | QLORA | 42 | 1200 | COMPLETED | VALID_FORMAL_PEFT_RUN | experiments/results/B-QLORA-s42.json |
| C-QLORA-s42 | GPU-2D | Qwen3-0.6B-Base | math-CPT | QLORA | 42 | 1200 | COMPLETED | VALID_FORMAL_PEFT_RUN | experiments/results/C-QLORA-s42.json |
| B-QLORA-s31415 | GPU-2D | Qwen3-0.6B-Base | base | QLORA | 31415 | 1200 | COMPLETED | VALID_FORMAL_PEFT_RUN | experiments/results/B-QLORA-s31415.json |
| C-QLORA-s31415 | GPU-2D | Qwen3-0.6B-Base | math-CPT | QLORA | 31415 | 1200 | COMPLETED | VALID_FORMAL_PEFT_RUN | experiments/results/C-QLORA-s31415.json |
| B-QLORA-s271828 | GPU-2D | Qwen3-0.6B-Base | base | QLORA | 271828 | 1200 | COMPLETED | VALID_FORMAL_PEFT_RUN | experiments/results/B-QLORA-s271828.json |
| C-QLORA-s271828 | GPU-2D | Qwen3-0.6B-Base | math-CPT | QLORA | 271828 | 1200 | COMPLETED | VALID_FORMAL_PEFT_RUN | experiments/results/C-QLORA-s271828.json |
| GPU3A-audit | GPU-3A | Qwen3 PEFT adapters | frozen GPU-2D | offline generation diagnosis | None | None | COMPLETED | GPU3A_ROOT_CAUSES_IDENTIFIED | artifacts/gpu3a/gpu3a_result.json |
| GPU3B-evaluator | GPU-3B | Qwen3 PEFT adapters | frozen weights | extraction/stopping ablation | 42 | None | COMPLETED | GPU3B_DECODING_AND_EXTRACTION_IMPROVED | artifacts/gpu3b/gpu3b_result.json |
| GPU3C-eos | GPU-3C | Qwen3-0.6B LoRA | same initial adapter | controlled EOS supervision ablation | 42 | 64 | COMPLETED | GPU3C_CONTROLLED_NEGATIVE_RESULT | artifacts/gpu3c/gpu3c_result.json |
| gpu4a-native-full-sft-s42-v1 | GPU-4A | MiniLLM 37M | E01 best | full-parameter BF16 response-only SFT | 4204 | 300 | COMPLETED | GPU4A_NATIVE_SFT_CAPABILITY_VALIDATED | artifacts/gpu4a/gpu4a_result.json |
| GPU4B-attention | GPU-4B | MiniLLM 37M | E01/GPU4A | attention/compile qualification | 4204 | None | PARTIAL | GPU4B_PARTIAL_VALIDATION | artifacts/gpu4b/gpu4b_result.json |
| GPU4B-R1-numerics | GPU-4B-R1 | MiniLLM 37M | E01 | BF16 numerical diagnosis | 4204 | None | COMPLETED | GPU4B_R1_NUMERICAL_PATH_DIFFERENCE_IDENTIFIED | artifacts/gpu4b_r1/gpu4b_r1_result.json |
