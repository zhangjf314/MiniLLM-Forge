from __future__ import annotations

import argparse
import json

from minillm_forge.cli.common import read_jsonl
from minillm_forge.data.contamination import audit_contamination


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit train/benchmark text contamination")
    parser.add_argument("--train", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--train-field", default="problem")
    parser.add_argument("--benchmark-field", default="problem")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--ngram-size", type=int, default=8)
    parser.add_argument("--output", default="artifacts/data_manifests/contamination_report.json")
    args = parser.parse_args()
    training = [str(row[args.train_field]) for row in read_jsonl(args.train)]
    benchmark = [str(row[args.benchmark_field]) for row in read_jsonl(args.benchmark)]
    report = audit_contamination(
        training,
        benchmark,
        threshold=args.threshold,
        ngram_size=args.ngram_size,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
