import pytest

from minillm_forge.training import trainer


def test_explicit_cuda_fails_closed_when_unavailable(monkeypatch):
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        trainer.resolve_device("cuda")
