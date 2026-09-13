from __future__ import annotations

import argparse
import json

from minillm_forge.design.gpu2d import analyze_context_retention
from minillm_forge.execution.gpu2d import (
    memory_qualification,
    parameter_analysis,
    resume_qualification,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute prospective GPU-2D recovery design")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("analyze-context")
    subparsers.add_parser("analyze-parameters")
    q1 = subparsers.add_parser("qualify-memory")
    q1.add_argument("--family", choices=("LORA", "QLORA"), required=True)
    q1.add_argument("--context", choices=(512, 768, 1024), type=int, required=True)
    q1.add_argument("--variant", default="R16_ALL_LINEAR")
    q2 = subparsers.add_parser("qualify-resume")
    q2.add_argument("--family", choices=("LORA", "QLORA"), required=True)
    q2.add_argument("--context", choices=(512, 768, 1024), type=int, required=True)
    q2.add_argument("--variant", default="R16_ALL_LINEAR")
    args = parser.parse_args()
    if args.command == "analyze-context":
        result = analyze_context_retention()
    elif args.command == "analyze-parameters":
        result = parameter_analysis()
    elif args.command == "qualify-memory":
        result = memory_qualification(args.family, args.context, args.variant)
    else:
        result = resume_qualification(args.family, args.context, args.variant)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
