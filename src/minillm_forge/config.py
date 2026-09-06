from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(config_path: Path, seen: set[Path]) -> dict[str, Any]:
    config_path = config_path.resolve()
    if config_path in seen:
        raise ValueError(f"cyclic config inheritance detected at {config_path}")
    seen.add(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    parent = config.pop("extends", None)
    if parent is None:
        seen.remove(config_path)
        return config
    parent_paths = [parent] if isinstance(parent, str) else parent
    if not isinstance(parent_paths, list) or not all(
        isinstance(item, str) for item in parent_paths
    ):
        raise ValueError("extends must be a path or a list of paths")
    merged: dict[str, Any] = {}
    for parent_path in parent_paths:
        merged = _deep_merge(merged, _load_config(config_path.parent / parent_path, seen))
    seen.remove(config_path)
    return _deep_merge(merged, config)


def load_config(path: str | Path) -> dict[str, Any]:
    return _load_config(Path(path), set())


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
