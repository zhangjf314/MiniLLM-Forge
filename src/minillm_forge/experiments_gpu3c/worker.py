from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from minillm_forge.experiments_gpu3c.background import supervise
from minillm_forge.experiments_gpu3c.controlled import run_formal_pair, run_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("supervise", "simulate", "smoke", "formal"))
    parser.add_argument("--config")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()
    if args.action == "supervise":
        if not args.config:
            parser.error("supervise requires --config")
        raise SystemExit(supervise(args.config))
    if args.action == "simulate":
        time.sleep(args.seconds)
        print(json.dumps({"status": "SIMULATED", "seconds": args.seconds}))
        raise SystemExit(args.exit_code)
    if args.action == "smoke":
        result = run_smoke(Path(args.repo))
    else:
        result = run_formal_pair(Path(args.repo))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
