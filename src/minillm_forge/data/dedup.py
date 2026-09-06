from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable


def normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^\w]+", "", normalized, flags=re.UNICODE)


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_match(text).encode("utf-8")).hexdigest()


def exact_deduplicate(texts: Iterable[str]) -> tuple[list[str], int]:
    unique: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    for text in texts:
        digest = text_hash(text)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        unique.append(text)
    return unique, duplicates
