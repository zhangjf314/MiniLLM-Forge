from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence


def pack_token_sequences(
    documents: Iterable[Sequence[int]],
    sequence_length: int,
    *,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    drop_remainder: bool = True,
) -> Iterator[list[int]]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    buffer: list[int] = []
    for document in documents:
        buffer.extend(int(token) for token in document)
        if eos_token_id is not None and (not buffer or buffer[-1] != eos_token_id):
            buffer.append(eos_token_id)
        while len(buffer) >= sequence_length:
            yield buffer[:sequence_length]
            del buffer[:sequence_length]
    if buffer and not drop_remainder:
        if pad_token_id is None:
            raise ValueError("pad_token_id is required when retaining a partial sequence")
        yield buffer + [pad_token_id] * (sequence_length - len(buffer))
