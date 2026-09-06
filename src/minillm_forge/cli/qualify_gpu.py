from __future__ import annotations

import argparse
import json

from minillm_forge.qualification import qualify_cuda_environment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run strict CUDA identity and kernel qualification"
    )
    parser.add_argument("--output", default="artifacts/environment/gpu_qualification.json")
    parser.add_argument("--collect-env", default="artifacts/environment/torch_collect_env.txt")
    args = parser.parse_args()
    result = qualify_cuda_environment(args.output, args.collect_env)
    print(json.dumps(result, indent=2))
    if result["qualification"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
