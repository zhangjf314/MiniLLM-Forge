from __future__ import annotations

import argparse
import json

from minillm_forge.design.gpu2b import finalize, prepare, recount, run_regression, validate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze or validate the zero-training GPU-2B design"
    )
    parser.add_argument(
        "command", choices=("prepare", "recount", "validate", "regression", "finalize")
    )
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "recount":
        result = recount()
    elif args.command == "regression":
        result = run_regression()
    elif args.command == "finalize":
        result = finalize()
    else:
        result = validate()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
