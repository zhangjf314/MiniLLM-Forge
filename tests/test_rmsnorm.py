import torch

from minillm_forge.model import RMSNorm


def test_rmsnorm_produces_unit_root_mean_square():
    values = torch.randn(4, 5, 16) * 3 + 2
    output = RMSNorm(16, eps=1e-8)(values)
    rms = output.float().pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-5, rtol=1e-5)
