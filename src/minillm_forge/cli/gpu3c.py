from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from minillm_forge.experiments_gpu3c.audit import run_data_audit
from minillm_forge.experiments_gpu3c.background import launch
from minillm_forge.experiments_gpu3c.controlled import audit_pair_identity, run_smoke


def _status(repo: Path, task_id: str) -> dict:
    from minillm_forge.experiments_gpu3c.reporting import refresh_background_reports

    refresh_background_reports(repo)
    path = repo / "runs/gpu3c/background" / task_id / "status.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPU-3C controlled SFT stages")
    parser.add_argument(
        "stage",
        choices=(
            "audit",
            "pair-audit",
            "smoke",
            "simulate-background",
            "formal-background",
            "status",
        ),
    )
    parser.add_argument("--repo", default=".")
    parser.add_argument("--task-id")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    if args.stage == "audit":
        result = run_data_audit(repo)
    elif args.stage == "pair-audit":
        result = audit_pair_identity(repo)
    elif args.stage == "smoke":
        result = run_smoke(repo)
    elif args.stage == "status":
        if not args.task_id:
            parser.error("status requires --task-id")
        result = _status(repo, args.task_id)
    else:
        task_id = args.task_id or (
            "gpu3c-background-simulation"
            if args.stage == "simulate-background"
            else f"gpu3c-formal-{int(time.time())}"
        )
        if args.stage == "simulate-background":
            worker_args = ["simulate", "--seconds", "2"]
            max_wall, estimate = 15, 2.0
            expected = []
        else:
            worker_args = ["formal", "--repo", str(repo)]
            max_wall, estimate = 2700, 1426.0
            expected = [
                "artifacts/gpu3c/formal_pair_result.json",
                "runs/gpu3c/formal-64/B-EOS-MASKED/result.json",
                "runs/gpu3c/formal-64/C-EOS-SUPERVISED/result.json",
            ]
        result = launch(
            repo,
            task_id=task_id,
            command=[
                sys.executable,
                "-m",
                "minillm_forge.experiments_gpu3c.worker",
                *worker_args,
            ],
            max_wall_time_seconds=max_wall,
            estimated_duration_seconds=estimate,
            expected_artifacts=expected,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
