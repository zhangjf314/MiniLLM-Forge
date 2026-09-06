import random

import numpy as np
import torch

from minillm_forge.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_restores_model_optimizer_scheduler_and_rng(tmp_path):
    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_weights = model.weight.detach().clone()
    path = save_checkpoint(
        tmp_path / "state.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        trainer_state={"global_step": 1},
        config={"seed": 11},
    )
    expected_random = (random.random(), float(np.random.rand()), float(torch.rand(())))
    with torch.no_grad():
        model.weight.zero_()
    checkpoint = load_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=True
    )
    actual_random = (random.random(), float(np.random.rand()), float(torch.rand(())))
    torch.testing.assert_close(model.weight, expected_weights)
    assert checkpoint["trainer_state"]["global_step"] == 1
    assert actual_random == expected_random


def test_rng_restore_keeps_default_generator_state_on_cpu():
    from minillm_forge.training.checkpoint import restore_rng_state

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    restore_rng_state(state)
    assert torch.get_rng_state().device.type == "cpu"
