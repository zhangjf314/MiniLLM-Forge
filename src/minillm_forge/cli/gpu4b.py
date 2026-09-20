from __future__ import annotations

import argparse
import json

from minillm_forge.experiments_gpu4b.audit import run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-4B attention and compile experiments")
    parser.add_argument("stage", choices=("audit", "correctness", "compile", "finalize"))
    parser.add_argument("--repo", default=".")
    parser.add_argument("--test-count", type=int)
    args = parser.parse_args()
    if args.stage == "audit":
        result = run_audit(args.repo)
    elif args.stage == "correctness":
        from minillm_forge.experiments_gpu4b.correctness import run_correctness

        result = run_correctness(args.repo)
    elif args.stage == "compile":
        from minillm_forge.experiments_gpu4b.compile_eval import run_compile_qualification

        result = run_compile_qualification(args.repo)
    else:
        if args.test_count is None:
            parser.error("--test-count is required for finalize")
        from minillm_forge.experiments_gpu4b.reporting import write_final_reports

        result = write_final_reports(args.repo, test_count=args.test_count)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
