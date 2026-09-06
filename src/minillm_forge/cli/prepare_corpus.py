from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minillm_forge.data.cpt import stream_huggingface_texts
from minillm_forge.data.dedup import text_hash
from minillm_forge.tokenization import normalize_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream, normalize, deduplicate, and pin a text corpus"
    )
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-documents", type=int, default=100_000)
    parser.add_argument("--target-tokens", type=int)
    parser.add_argument("--tokenizer", help="Track A tokenizer JSON for exact token counting")
    parser.add_argument("--output", default="data/processed/fineweb_edu_train.txt")
    parser.add_argument("--manifest", default="artifacts/data_manifests/fineweb_edu.json")
    args = parser.parse_args()
    tokenizer: Any = None
    if args.tokenizer:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(args.tokenizer)
    if args.target_tokens and tokenizer is None:
        raise ValueError("--target-tokens requires --tokenizer for exact counting")

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    content_digest = hashlib.sha256()
    document_count = 0
    token_count = 0
    duplicate_count = 0
    texts = stream_huggingface_texts(
        args.dataset,
        subset=args.subset,
        split=args.split,
        text_field=args.text_field,
        revision=args.revision,
        max_samples=args.max_documents,
    )
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for raw_text in texts:
            text = normalize_text(raw_text).replace("\n", " ")
            if not text:
                continue
            digest = text_hash(text)
            if digest in seen:
                duplicate_count += 1
                continue
            seen.add(digest)
            document_tokens = (
                len(tokenizer.encode(text, add_special_tokens=False).ids) if tokenizer else 0
            )
            encoded_line = (text + "\n").encode("utf-8")
            handle.write(text + "\n")
            content_digest.update(encoded_line)
            document_count += 1
            token_count += document_tokens
            if args.target_tokens and token_count >= args.target_tokens:
                break

    manifest = {
        "dataset": args.dataset,
        "subset": args.subset,
        "revision": args.revision,
        "sample_count": document_count,
        "token_count": token_count if tokenizer else None,
        "sha256": content_digest.hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filter_config": {"normalization": "NFKC", "exact_dedup": True},
        "duplicates_removed": duplicate_count,
        "benchmark_overlap_removed": 0,
        "benchmark_audit_required_before_training": True,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
