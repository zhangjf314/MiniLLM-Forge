from __future__ import annotations

import argparse
import json

from minillm_forge.execution.gpu2b import build_token_cache, memory_qualification, preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute frozen STAGE_GPU_2B gates")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    cache = subparsers.add_parser("build-token-cache")
    cache.add_argument("--force", action="store_true")
    memory = subparsers.add_parser("qualify-memory")
    memory.add_argument("--family", required=True, choices=("FULL", "LORA", "QLORA"))
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight()
    elif args.command == "build-token-cache":
        result = build_token_cache(force=args.force)
    else:
        result = memory_qualification(args.family)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
