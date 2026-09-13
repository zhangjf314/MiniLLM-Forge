from __future__ import annotations

import argparse
import json

from minillm_forge.execution.gpu2d_formal import (
    EXECUTION_ORDER,
    analyze,
    evaluate_run,
    freeze_protocol,
    preflight,
    qualify_evaluator,
    train_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the frozen GPU-2D formal campaign")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("qualify-evaluator")
    subparsers.add_parser("preflight")
    train = subparsers.add_parser("train-run")
    train.add_argument("--run-id", choices=EXECUTION_ORDER, required=True)
    evaluate = subparsers.add_parser("evaluate-run")
    evaluate.add_argument("--run-id", choices=EXECUTION_ORDER, required=True)
    subparsers.add_parser("analyze")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_protocol()
    elif args.command == "qualify-evaluator":
        result = qualify_evaluator()
    elif args.command == "preflight":
        result = preflight()
    elif args.command == "train-run":
        result = train_run(args.run_id)
    elif args.command == "evaluate-run":
        result = evaluate_run(args.run_id)
    else:
        result = analyze()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
