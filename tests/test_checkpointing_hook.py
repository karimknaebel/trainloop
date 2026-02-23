import itertools

import torch

from trainloop import BaseTrainer, CheckpointingHook


class _CheckpointingTrainer(BaseTrainer):
    def __init__(
        self,
        max_steps: int,
        hook: CheckpointingHook,
        workspace,
    ):
        self.hook = hook
        super().__init__(
            max_steps=max_steps,
            device="cpu",
            workspace=workspace,
        )

    def build_data_loader(self):
        return itertools.repeat(torch.ones(1, 1))

    def build_model(self):
        return torch.nn.Linear(1, 1, bias=False)

    def build_optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

    def build_hooks(self):
        return [self.hook]

    def forward(self, input):
        loss = self.model(input).sum()
        return loss, {}


def test_keep_steps_forces_save_and_keeps_checkpoint(tmp_path):
    hook = CheckpointingHook(
        interval=5,
        keep_previous=0,
        keep_steps=[3],
        path="checkpoints",
        load=None,
    )
    trainer = _CheckpointingTrainer(max_steps=6, hook=hook, workspace=tmp_path)

    trainer.train()

    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "3_keep").is_dir()
    assert not (ckpt_dir / "5").exists()
    assert (ckpt_dir / "6").is_dir()


def test_keep_steps_also_keeps_when_step_is_on_interval(tmp_path):
    hook = CheckpointingHook(
        interval=5,
        keep_previous=0,
        keep_steps=[5],
        path="checkpoints",
        load=None,
    )
    trainer = _CheckpointingTrainer(max_steps=6, hook=hook, workspace=tmp_path)

    trainer.train()

    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "5_keep").is_dir()
    assert (ckpt_dir / "6").is_dir()


def test_keep_interval_forces_save_and_keeps_checkpoint(tmp_path):
    hook = CheckpointingHook(
        interval=10,
        keep_previous=0,
        keep_interval=3,
        path="checkpoints",
        load=None,
    )
    trainer = _CheckpointingTrainer(max_steps=4, hook=hook, workspace=tmp_path)

    trainer.train()

    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "3_keep").is_dir()
    assert not (ckpt_dir / "3").exists()
    assert (ckpt_dir / "4").is_dir()
