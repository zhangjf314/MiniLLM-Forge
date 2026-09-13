# GPU-2B prompt and serialization contract

Status: `FROZEN_BEFORE_TRAINING`

All six arms use the tokenizer from `Qwen/Qwen3-0.6B-Base` revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Its artifact digest is
`3fe32a8d9deb69b4b17efbc1af1bbde2dc345949efdc975ff315a3d066821cfe` and its chat-template
SHA-256 is `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5`.

## SFT serialization

The literal rendered form is:

```text
<|im_start|>system
You are a mathematical reasoning assistant.<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
<think>

</think>

{solution}<|im_end|>
```

The model's pinned chat template produces this text. No arm may substitute a hand-built
template. Maximum serialized length is 1,024 tokens. Truncation is from the right.

Loss is assistant-only: system and user tokens are `-100`; assistant tokens, including
the template-inserted think delimiters and final `<|im_end|>`, are supervised. Padding is
`-100`. Every retained record has at least 16 supervised assistant tokens after truncation.

## Benchmark rendering

Benchmark prompts contain the same system message and the benchmark problem as the user
message. `add_generation_prompt=True` renders:

```text
<|im_start|>system
You are a mathematical reasoning assistant.<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
```

Generation is greedy (`do_sample=false`, one beam), batch size 1, at most 512 new tokens.
Stop on token 151643 (`<|endoftext|>`) or 151645 (`<|im_end|>`); pad with 151643.
No model-specific prompt or decoding change is permitted.

## Scoring

The primary scorer is the checked-in deterministic implementation in
`src/minillm_forge/evaluation/math_eval.py`. It selects the last balanced boxed expression,
then a final-answer marker, then GSM8K's `####` marker, and finally the last non-empty line.
Normalization is frozen and no manual correction is allowed.
