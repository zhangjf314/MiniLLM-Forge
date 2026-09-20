from __future__ import annotations

import argparse
import json
from pathlib import Path

from minillm_forge.experiments_gpu4a.audit import run_audit
from minillm_forge.experiments_gpu4a.runtime import (
    evaluate_generations,
    load_native_model,
    release,
    write_json,
)
from minillm_forge.experiments_gpu4a.train import (
    run_formal_sft,
    run_prewarm,
    run_tiny_overfit,
)


def run_baseline(repo: Path) -> dict:
    model, tokenizer, identity = load_native_model(repo)
    result = evaluate_generations(
        repo,
        model,
        tokenizer,
        model_id=f"native-pretrained:{identity['sha256']}",
        output_path="runs/gpu4a/baseline/generations.jsonl",
    )
    result["checkpoint"] = identity
    write_json(repo / "artifacts/gpu4a/baseline_evaluation.json", result)
    release(model)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-4A native 37M SFT experiment")
    parser.add_argument(
        "stage",
        choices=("audit", "tiny-overfit", "prewarm", "baseline", "formal", "sft-eval", "finalize"),
    )
    parser.add_argument("--repo", default=".")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    if args.stage == "audit":
        result = run_audit(repo)
    elif args.stage == "tiny-overfit":
        result = run_tiny_overfit(repo)
    elif args.stage == "prewarm":
        result = run_prewarm(repo, args.steps)
    elif args.stage == "baseline":
        result = run_baseline(repo)
    elif args.stage == "formal":
        result = run_formal_sft(repo)
    elif args.stage == "sft-eval":
        formal = json.loads(
            (repo / "artifacts/gpu4a/formal_sft_result.json").read_text(encoding="utf-8")
        )
        checkpoint = formal["selected_checkpoint"]["path"]
        model, tokenizer, identity = load_native_model(repo, checkpoint)
        result = evaluate_generations(
            repo,
            model,
            tokenizer,
            model_id=f"native-full-sft:{identity['sha256']}",
            output_path="runs/gpu4a/sft-evaluation/generations.jsonl",
        )
        result["checkpoint"] = identity
        write_json(repo / "artifacts/gpu4a/sft_evaluation.json", result)
        release(model)
    else:
        from minillm_forge.experiments_gpu4a.reporting import write_final_reports

        result = write_final_reports(repo)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
