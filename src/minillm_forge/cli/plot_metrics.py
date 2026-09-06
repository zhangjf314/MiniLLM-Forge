from __future__ import annotations

import argparse

from minillm_forge.evaluation.plots import plot_training_curves


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render loss and gradient curves from JSONL metrics"
    )
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(plot_training_curves(args.metrics, args.output))


if __name__ == "__main__":
    main()
