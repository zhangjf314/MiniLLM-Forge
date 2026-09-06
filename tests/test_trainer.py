from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from minillm_forge.cli.common import StatefulRandomSampler
from minillm_forge.training import ForgeTrainer, TrainingConfig


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 1)

    def forward(self, input_ids, labels=None, attention_mask=None):
        prediction = self.projection(input_ids.float().unsqueeze(-1)).squeeze(-1)
        return SimpleNamespace(loss=((prediction - labels.float()) ** 2).mean())


def test_gradient_accumulation_controls_optimizer_steps(tmp_path):
    dataset = [
        {
            "input_ids": torch.tensor([value]),
            "labels": torch.tensor([value * 2]),
            "attention_mask": torch.tensor([1]),
        }
        for value in range(1, 5)
    ]
    model = ToyLM()
    optimizer = CountingSGD(model.parameters(), lr=0.01)
    trainer = ForgeTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=DataLoader(dataset, batch_size=1, shuffle=False),
        config=TrainingConfig(
            max_steps=2,
            gradient_accumulation_steps=2,
            precision="fp32",
            log_every=1,
            eval_every=0,
            save_every=0,
            output_dir=str(tmp_path),
            device="cpu",
        ),
    )
    state = trainer.train()
    assert state.global_step == 2
    assert state.batches_seen == 4
    assert optimizer.step_count == 2


def test_interrupted_resume_matches_continuous_training(tmp_path):
    dataset = [
        {
            "input_ids": torch.tensor([value]),
            "labels": torch.tensor([value * 2]),
            "attention_mask": torch.tensor([1]),
        }
        for value in range(1, 9)
    ]

    def build_trainer(name, max_steps, initial_state):
        model = ToyLM()
        model.load_state_dict(initial_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=StatefulRandomSampler(dataset, seed=17),
        )
        trainer = ForgeTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            config=TrainingConfig(
                max_steps=max_steps,
                gradient_accumulation_steps=1,
                precision="fp32",
                log_every=1,
                eval_every=0,
                save_every=0,
                output_dir=str(tmp_path / name),
                seed=23,
                device="cpu",
            ),
        )
        return model, trainer

    torch.manual_seed(5)
    initial = ToyLM().state_dict()
    continuous_model, continuous = build_trainer("continuous", 4, initial)
    continuous.train()

    _, first_leg = build_trainer("first-leg", 2, initial)
    first_leg.train()
    resumed_model, resumed = build_trainer("resumed", 4, initial)
    resumed.resume(tmp_path / "first-leg" / "last.pt")
    resumed.train()

    for continuous_parameter, resumed_parameter in zip(
        continuous_model.parameters(), resumed_model.parameters(), strict=True
    ):
        torch.testing.assert_close(continuous_parameter, resumed_parameter, atol=0, rtol=0)
