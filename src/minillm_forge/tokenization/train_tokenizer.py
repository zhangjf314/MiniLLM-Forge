from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_text_files(paths: Iterable[str | Path]) -> Iterator[str]:
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                normalized = normalize_text(line)
                if normalized:
                    yield normalized


def train_bpe_tokenizer(
    texts: Iterable[str],
    output_path: str | Path,
    *,
    vocab_size: int = 24_000,
    min_frequency: int = 2,
    full_byte_alphabet: bool = False,
):
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    except ImportError as exc:
        raise RuntimeError("install the 'tokenizers' dependency to train a tokenizer") from exc

    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError("vocab_size must exceed the number of special tokens")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if full_byte_alphabet else [],
    )
    tokenizer.train_from_iterator((normalize_text(text) for text in texts), trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    return tokenizer
