from __future__ import annotations

import argparse
import json

from minillm_forge.qualification.qwen import qualify_qwen


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one bounded Qwen CUDA qualification")
    parser.add_argument(
        "--mode", choices=["base", "lora", "qlora", "full-cpt", "full-sft"], required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--revision", default="da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
    parser.add_argument("--data", default="data/smoke/math_sft.jsonl")
    parser.add_argument("--sequence-length", type=int, choices=[512, 1024], default=512)
    args = parser.parse_args()
    result = qualify_qwen(
        mode=args.mode,
        output_path=args.output,
        model_name=args.model,
        revision=args.revision,
        data_path=args.data,
        sequence_length=args.sequence_length,
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
