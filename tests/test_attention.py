import torch

from minillm_forge.model import GroupedQueryAttention, MiniLLMConfig
from minillm_forge.model.attention import repeat_key_value


def tiny_config(**updates):
    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "num_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "max_seq_len": 16,
    }
    values.update(updates)
    return MiniLLMConfig(**values)


def test_attention_shape_and_finite_values():
    attention = GroupedQueryAttention(tiny_config())
    output = attention(torch.randn(2, 7, 32))
    assert output.shape == (2, 7, 32)
    assert torch.isfinite(output).all()


def test_gqa_repeats_each_kv_head_as_one_group():
    values = torch.tensor([[[[1.0]], [[2.0]]]])
    repeated = repeat_key_value(values, 2)
    assert repeated[:, :, 0, 0].tolist() == [[1.0, 1.0, 2.0, 2.0]]


def test_attention_mask_shape_is_validated():
    attention = GroupedQueryAttention(tiny_config())
    try:
        attention(torch.randn(2, 7, 32), torch.ones(2, 6))
    except ValueError as error:
        assert "attention_mask" in str(error)
    else:
        raise AssertionError("invalid mask shape was accepted")
