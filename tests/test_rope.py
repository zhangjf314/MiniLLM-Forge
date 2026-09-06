import torch

from minillm_forge.model.rope import RotaryEmbedding


def test_rope_preserves_vector_norm_and_changes_nonzero_positions():
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    query = torch.randn(2, 4, 6, 8)
    key = torch.randn(2, 2, 6, 8)
    positions = torch.arange(6).expand(2, -1)
    rotated_query, rotated_key = rope(query, key, positions)
    torch.testing.assert_close(query.norm(dim=-1), rotated_query.norm(dim=-1))
    torch.testing.assert_close(key.norm(dim=-1), rotated_key.norm(dim=-1))
    torch.testing.assert_close(query[:, :, 0], rotated_query[:, :, 0])
    assert not torch.allclose(query[:, :, 1], rotated_query[:, :, 1])
