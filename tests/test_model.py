import pytest
import torch

from minillm_forge.model import MiniLLM, MiniLLMConfig


def test_model_output_and_weight_tying():
    config = MiniLLMConfig(
        vocab_size=80,
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_seq_len=12,
    )
    model = MiniLLM(config)
    output = model(torch.randint(0, 80, (3, 9)))
    assert output.logits.shape == (3, 9, 80)
    assert model.lm_head.weight is model.token_embedding.weight


def test_invalid_gqa_ratio_fails_fast():
    with pytest.raises(ValueError, match="divisible"):
        MiniLLMConfig(
            vocab_size=80,
            hidden_size=32,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=3,
            intermediate_size=64,
        )
