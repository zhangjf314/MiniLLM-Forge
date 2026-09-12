"""Frozen document-level partitions and memory-mapped tokens for the formal run."""

from __future__ import annotations

import hashlib
import json
from array import array
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from minillm_forge.data.dedup import text_hash
from minillm_forge.data.manifest import file_digest
from minillm_forge.tokenization import normalize_text, train_bpe_tokenizer, validate_tokenizer

SOURCE = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SHARD = "sample/10BT/000_00000.parquet"
MANIFEST = Path("artifacts/data_manifests/minillm_pretrain_formal.json")
TOKENIZER = Path("artifacts/tokenizers/minillm-tokenizer.json")
TOKENIZER_MANIFEST = TOKENIZER.with_name("minillm-tokenizer-manifest.json")


def json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def partition(text: str) -> str:
    # 1% hash partition. Chosen BEFORE tokenizer training, and never changed by batch size.
    return "validation" if int(text_hash(text)[:8], 16) % 100 == 0 else "train"


class FrozenTokenDataset(Dataset):
    def __init__(self, path: str | Path, sequence_length: int = 1024):
        self.path = Path(path)
        self.sequence_length = sequence_length
        self.tokens = np.memmap(self.path, dtype="<u2", mode="r")
        if len(self.tokens) < sequence_length:
            raise ValueError("frozen corpus has no full sequence")

    def __len__(self):
        return len(self.tokens) // self.sequence_length

    def __getitem__(self, index):
        start = index * self.sequence_length
        ids = torch.from_numpy(self.tokens[start : start + self.sequence_length].astype(np.int64))
        return {"input_ids": ids, "attention_mask": ids.ne(0), "labels": ids.clone()}


def verify_frozen() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for record in manifest["files"].values():
        if file_digest(record["path"]) != record["sha256"]:
            raise ValueError(f"frozen data identity mismatch: {record['path']}")
    if file_digest(TOKENIZER) != manifest["tokenizer_hash"]:
        raise ValueError("frozen tokenizer identity mismatch")
    return manifest


def prepare(documents: int = 30000) -> dict:
    if MANIFEST.exists():
        return verify_frozen()
    root = Path("data/processed/minillm-formal")
    root.mkdir(parents=True, exist_ok=True)
    raw = Path("data/raw/minillm-fineweb-edu.jsonl")
    receipt = raw.with_suffix(".receipt.json")
    if not receipt.exists():
        if raw.exists():
            raise ValueError(
                "incomplete acquisition exists; retain it and investigate before retry"
            )
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw_bytes = 0
        count = 0
        with raw.open("w", encoding="utf-8", newline="\n") as handle:
            from datasets import load_dataset

            stream = load_dataset(
                "parquet",
                data_files=f"https://huggingface.co/datasets/{SOURCE}/resolve/{REVISION}/{SHARD}",
                split="train",
                streaming=True,
            )
            for row in stream.take(documents):
                text = row["text"]
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                raw_bytes += len(text.encode("utf-8"))
                count += 1
                if count % 1000 == 0:
                    print(f"downloaded documents={count} raw_text_bytes={raw_bytes}", flush=True)
        json_write(
            receipt,
            {
                "documents": count,
                "raw_text_bytes": raw_bytes,
                "sha256": file_digest(raw),
                "revision": REVISION,
            },
        )
    acquisition = json.loads(receipt.read_text(encoding="utf-8"))
    if file_digest(raw) != acquisition["sha256"] or acquisition["revision"] != REVISION:
        raise ValueError("raw receipt mismatch")
    seen: set[str] = set()
    splits: dict[str, list[str]] = {"train": [], "validation": []}
    duplicates, empty = 0, 0
    with raw.open(encoding="utf-8") as handle:
        for line in handle:
            text = normalize_text(json.loads(line)["text"]).replace("\n", " ")
            if not text:
                empty += 1
                continue
            key = text_hash(text)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            splits[partition(text)].append(text)
    for split, texts in splits.items():
        with (root / f"{split}.txt").open("w", encoding="utf-8", newline="\n") as handle:
            for text in texts:
                handle.write(text + "\n")
    tokenizer_texts = splits["train"][:5000]
    tokenizer_input_hash = hashlib.sha256("\n".join(tokenizer_texts).encode()).hexdigest()
    print(f"training 24K BPE on {len(tokenizer_texts)} train-only documents", flush=True)
    tokenizer = train_bpe_tokenizer(tokenizer_texts, TOKENIZER, full_byte_alphabet=True)
    stats = validate_tokenizer(tokenizer, splits["validation"])
    if stats.vocab_size != 24000 or stats.unk_ratio != 0 or stats.roundtrip_success_rate != 1:
        raise ValueError(f"tokenizer qualification failed: {stats}")
    json_write(
        TOKENIZER_MANIFEST,
        {
            "dataset": SOURCE,
            "revision": REVISION,
            "algorithm": "byte-level BPE",
            "tokenizer_input_corpus_hash": tokenizer_input_hash,
            "training_documents": len(tokenizer_texts),
            "training_partition": "train only",
            "tokenizer_artifact_hash": file_digest(TOKENIZER),
            "special_tokens": {
                token: tokenizer.token_to_id(token)
                for token in ("<pad>", "<bos>", "<eos>", "<unk>")
            },
            "statistics": stats.to_dict(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    files, counts = {}, {}
    for split, texts in splits.items():
        token_path = root / f"{split}.bin"
        token_count = 0
        with token_path.open("wb") as handle:
            for start in range(0, len(texts), 128):
                encoded = tokenizer.encode_batch(
                    texts[start : start + 128], add_special_tokens=False
                )
                for sequence in encoded:
                    values = array("H", [*sequence.ids, tokenizer.token_to_id("<eos>")])
                    values.tofile(handle)
                    token_count += len(values)
        files[split] = {"path": str(token_path), "sha256": file_digest(token_path)}
        counts[split] = {
            "documents": len(texts),
            "raw_tokens": token_count,
            "packed_tokens": token_count // 1024 * 1024,
            "sequences": token_count // 1024,
            "tail_tokens_dropped": token_count % 1024,
            "text_sha256": file_digest(root / f"{split}.txt"),
        }
    manifest = {
        "dataset": SOURCE,
        "subset": "sample-10BT",
        "revision": REVISION,
        "shard": SHARD,
        "acquisition": acquisition,
        "normalized_documents": sum(map(len, splits.values())),
        "duplicate_removals": duplicates,
        "empty_removals": empty,
        "files": files,
        "partitions": counts,
        "tokenizer_hash": file_digest(TOKENIZER),
        "corpus_hash": hashlib.sha256("".join(sorted(seen)).encode()).hexdigest(),
        "creation_config": {
            "documents": documents,
            "sequence_length": 1024,
            "dtype": "little-endian uint16",
            "split_rule": "normalized SHA256 first 8 hex modulo 100 == 0 => val",
            "dedup": "normalized exact",
            "packing": "independent streams; EOS between docs",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    json_write(MANIFEST, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest
