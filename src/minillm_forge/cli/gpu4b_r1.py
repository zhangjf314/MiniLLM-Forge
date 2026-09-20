from __future__ import annotations

import argparse
import json

from minillm_forge.experiments_gpu4b_r1.diagnostics import run_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-4B-R1 BF16 numerical diagnostics")
    parser.add_argument("stage", choices=("run",))
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    result = run_diagnostics(args.repo)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
