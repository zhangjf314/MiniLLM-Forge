from __future__ import annotations

import argparse
import json

from minillm_forge.experiments_gpu3b.analysis import build_reports
from minillm_forge.experiments_gpu3b.core import freeze_protocol, run_extraction_ablation
from minillm_forge.experiments_gpu3b.inference import config, run_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen-weight GPU-3B experiment stages")
    parser.add_argument("stage", choices=("freeze", "extract", "infer", "report"))
    parser.add_argument("--repo", default=".")
    parser.add_argument("--run-id")
    parser.add_argument("--partition", choices=("development", "confirmation"))
    parser.add_argument("--sample", action="append", dest="samples")
    parser.add_argument("--config-id", default="S0-R0-L512")
    parser.add_argument("--stop", choices=("S0", "S1", "S2", "S3"), default="S0")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--tests-passed", type=int)
    args = parser.parse_args()
    if args.stage == "freeze":
        result = freeze_protocol(args.repo)
    elif args.stage == "extract":
        result = run_extraction_ablation(args.repo)
    elif args.stage == "report":
        result = build_reports(args.repo, tests_passed=args.tests_passed)
    else:
        if not args.run_id or not args.partition or not args.samples:
            parser.error("infer requires --run-id, --partition and at least one --sample")
        generation_config = config(
            args.config_id,
            stop=args.stop,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            max_new_tokens=args.max_new_tokens,
        )
        result = run_jobs(
            args.repo,
            run_id=args.run_id,
            sample_ids=args.samples,
            configs=[generation_config],
            partition=args.partition,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
