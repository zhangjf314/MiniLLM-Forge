from __future__ import annotations

import argparse
import json

from minillm_forge.diagnostics.gpu3a import run_gpu3a_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild GPU-3A diagnostics without training/inference"
    )
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output-dir", default="artifacts/gpu3a")
    parser.add_argument("--experiment", action="append", dest="experiments")
    args = parser.parse_args()
    result = run_gpu3a_audit(args.repo, args.output_dir, set(args.experiments or []) or None)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
