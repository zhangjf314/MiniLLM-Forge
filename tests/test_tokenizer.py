from minillm_forge.tokenization import train_bpe_tokenizer, validate_tokenizer


def test_tokenizer_trains_roundtrips_and_defines_special_tokens(tmp_path):
    texts = ["the quick brown fox", "the quick blue bird", "math: 2 + 2 = 4"] * 10
    tokenizer = train_bpe_tokenizer(
        texts, tmp_path / "tokenizer.json", vocab_size=80, min_frequency=1
    )
    stats = validate_tokenizer(tokenizer, texts[:3])
    assert stats.vocab_size <= 80
    assert stats.unk_ratio == 0.0
    assert stats.roundtrip_success_rate == 1.0
