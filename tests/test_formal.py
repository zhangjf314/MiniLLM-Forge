import copy
from array import array
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from minillm_forge.cli.common import StatefulRandomSampler
from minillm_forge.cli.formal_pretrain import digest_state
from minillm_forge.data.formal import FrozenTokenDataset, partition
from minillm_forge.model import MiniLLM, MiniLLMConfig
from minillm_forge.training import TrainingConfig, build_adamw, build_cosine_scheduler
from minillm_forge.training.formal import FormalTrainer


def test_frozen_blocks_count_and_shift_targets(tmp_path):
    path = tmp_path / "tokens.bin"
    with path.open("wb") as handle:
        array("H", [4, 5, 6, 7, 8, 9, 10, 11, 12]).tofile(handle)
    dataset = FrozenTokenDataset(path, sequence_length=4)
    assert len(dataset) == 2
    assert dataset[1]["labels"].tolist() == [8, 9, 10, 11]
    assert partition(" Example document!") == partition("example document")


def test_formal_resume_preserves_schedule_dropout_and_cursor(tmp_path):
    model_config = MiniLLMConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        max_seq_len=8,
        residual_dropout=0.1,
    )
    data = [
        {
            "input_ids": torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]),
            "labels": torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]),
            "attention_mask": torch.ones(8, dtype=torch.bool),
        }
        for _ in range(5)
    ]

    def build(name):
        torch.manual_seed(31)
        model = MiniLLM(model_config)
        optimizer = build_adamw(model, lr=0.001)
        scheduler = build_cosine_scheduler(optimizer, total_steps=6, warmup_steps=1)
        config = TrainingConfig(
            max_steps=6,
            gradient_accumulation_steps=2,
            precision="fp32",
            device="cpu",
            output_dir=str(tmp_path / name),
            tensorboard=False,
            eval_every=0,
            log_every=1,
        )
        loader = DataLoader(
            data, batch_size=2, sampler=StatefulRandomSampler(data, 42), drop_last=True
        )
        return FormalTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=loader,
            config=config,
            run_config={
                "model": asdict(model_config),
                "training": asdict(config),
                "data_identity": {"sha256": "frozen"},
            },
        )

    continuous = build("continuous")
    continuous.train()
    split = build("split")
    split.train(stop_after_step=3)
    restored = build("restored")
    restored.resume(tmp_path / "split" / "last.pt")
    assert restored.state.global_step == 3
    restored.train()
    for key, value in continuous.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[key], atol=0, rtol=0)
    assert digest_state(copy.deepcopy(continuous.optimizer.state_dict())) == digest_state(
        restored.optimizer.state_dict()
    )
    assert continuous.state.tokens_seen == restored.state.tokens_seen == 6 * 2 * 2 * 8
    assert continuous.state.supervised_tokens_seen == 6 * 2 * 2 * 7
