import torch

from minillm_forge.model import MiniLLM, MiniLLMConfig


def test_future_token_cannot_change_earlier_logits():
    torch.manual_seed(7)
    config = MiniLLMConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_seq_len=16,
    )
    model = MiniLLM(config).eval()
    first = torch.tensor([[1, 8, 9, 10, 11]])
    changed_future = torch.tensor([[1, 8, 9, 10, 27]])
    with torch.no_grad():
        first_logits = model(first).logits
        changed_logits = model(changed_future).logits
    torch.testing.assert_close(first_logits[:, :4], changed_logits[:, :4], atol=1e-6, rtol=1e-6)
