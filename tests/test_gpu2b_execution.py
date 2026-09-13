import hashlib

import torch

from minillm_forge.execution.gpu2b import FrozenSFTDataset, permutation_digest


def test_permutation_digest_uses_frozen_uint32_little_endian_contract():
    order = torch.randperm(7, generator=torch.Generator().manual_seed(42)).tolist()
    payload = b"".join(value.to_bytes(4, "little") for value in order)
    assert permutation_digest(7, 42) == hashlib.sha256(payload).hexdigest()


def test_frozen_sft_dataset_reconstructs_assistant_only_labels(tmp_path):
    path = tmp_path / "tokens.pt"
    torch.save(
        {
            "input_ids": torch.tensor([10, 11, 12, 20, 21], dtype=torch.int32),
            "offsets": torch.tensor([0, 3, 5]),
            "assistant_starts": torch.tensor([2, 1], dtype=torch.int32),
        },
        path,
    )
    dataset = FrozenSFTDataset(path)
    assert len(dataset) == 2
    assert dataset[0]["labels"].tolist() == [-100, -100, 12]
    assert dataset[1]["labels"].tolist() == [-100, 21]
