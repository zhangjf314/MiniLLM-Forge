import torch
import torch.nn.functional as F

from minillm_forge.model import causal_lm_loss


def test_causal_loss_shifts_labels_once():
    logits = torch.tensor([[[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0], [9.0, 0.0, 0.0]]])
    labels = torch.tensor([[2, 0, 1, 2]])
    actual = causal_lm_loss(logits, labels)
    expected = F.cross_entropy(logits[:, :-1].reshape(-1, 3), labels[:, 1:].reshape(-1))
    torch.testing.assert_close(actual, expected)


def test_causal_loss_ignores_masked_assistant_labels():
    logits = torch.randn(1, 5, 8)
    labels = torch.tensor([[-100, -100, -100, 4, 5]])
    actual = causal_lm_loss(logits, labels)
    expected = F.cross_entropy(logits[:, 2:4].reshape(-1, 8), torch.tensor([4, 5]))
    torch.testing.assert_close(actual, expected)
