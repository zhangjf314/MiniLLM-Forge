from __future__ import annotations

import argparse
import json
from pathlib import Path

from minillm_forge.tokenization import iter_text_files, train_bpe_tokenizer, validate_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate the Track A BPE tokenizer")
    parser.add_argument("inputs", nargs="+", help="UTF-8 text or JSONL-free line files")
    parser.add_argument("--output", default="artifacts/tokenizers/minillm-tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=24_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--validation-samples", type=int, default=1_000)
    args = parser.parse_args()

    validation_texts = []

    def training_iterator():
        for text in iter_text_files(args.inputs):
            if len(validation_texts) < args.validation_samples:
                validation_texts.append(text)
            yield text

    tokenizer = train_bpe_tokenizer(
        training_iterator(),
        args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    stats = validate_tokenizer(tokenizer, validation_texts)
    stats_path = Path(args.output).with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(stats.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(stats.to_dict(), indent=2))


if __name__ == "__main__":
    main()
