from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def dataset_digest(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_digest(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file_manifest(manifest_path: str | Path, data_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    actual = file_digest(data_path)
    if manifest.get("sha256") != actual:
        raise ValueError(
            f"data hash mismatch for {data_path}: expected {manifest.get('sha256')}, got {actual}"
        )
    return manifest


def write_data_manifest(
    output_path: str | Path,
    *,
    dataset: str,
    revision: str,
    records: list[dict[str, Any]],
    token_count: int,
    filter_config: dict[str, Any],
    benchmark_overlap_removed: int,
) -> dict[str, Any]:
    manifest = {
        "dataset": dataset,
        "revision": revision,
        "sample_count": len(records),
        "token_count": token_count,
        "sha256": dataset_digest(records),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filter_config": filter_config,
        "benchmark_overlap_removed": benchmark_overlap_removed,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
