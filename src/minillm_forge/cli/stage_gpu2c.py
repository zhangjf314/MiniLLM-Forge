from __future__ import annotations

import argparse
import json

from minillm_forge.execution.gpu2c import (
    finalize,
    memory_qualification,
    preflight,
    resume_qualification,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute frozen GPU-2C qualification gates")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    q1 = subparsers.add_parser("qualify-memory")
    q1.add_argument("--family", choices=("LORA", "QLORA"), required=True)
    q2 = subparsers.add_parser("qualify-resume")
    q2.add_argument("--family", choices=("LORA", "QLORA"), required=True)
    subparsers.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight()
    elif args.command == "qualify-memory":
        result = memory_qualification(args.family)
    elif args.command == "qualify-resume":
        result = resume_qualification(args.family)
    else:
        result = finalize()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
