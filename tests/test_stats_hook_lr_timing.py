import itertools

import torch

from trainloop import BaseTrainer, StatsHook


class _LRScheduleTrainer(BaseTrainer):
    def __init__(self, param_group_keys=("name", "lr")):
        self.stats = []
        self.param_group_keys = param_group_keys
        super().__init__(max_steps=2, device="cpu")

    def build_data_loader(self):
        return itertools.repeat(torch.ones(1, 1))

    def build_model(self):
        return torch.nn.Linear(1, 1, bias=False)

    def build_optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

    def build_lr_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)

    def build_hooks(self):
        return [
            StatsHook(
                lambda trainer, stats: self.stats.append(stats),
                interval=1,
                sync=False,
                param_group_keys=self.param_group_keys,
            )
        ]

    def forward(self, input):
        loss = self.model(input).sum()
        return loss, {}


def test_stats_hook_reports_pre_step_lr():
    trainer = _LRScheduleTrainer()

    trainer.train()

    lrs = [stats.param_groups[0]["lr"] for stats in trainer.stats]
    assert lrs == [0.1, 0.05]
    assert "name" not in trainer.stats[0].param_groups[0]


def test_stats_hook_reports_configured_param_group_keys():
    trainer = _LRScheduleTrainer(param_group_keys=("lr", "momentum"))

    trainer.train()

    assert trainer.stats[0].param_groups[0] == {"lr": 0.1, "momentum": 0}


def test_stats_hook_reports_param_group_name_when_present():
    class _NamedGroupTrainer(_LRScheduleTrainer):
        def build_optimizer(self):
            return torch.optim.SGD(
                [{"params": self.model.parameters(), "lr": 0.1, "name": "model"}]
            )

    trainer = _NamedGroupTrainer()

    trainer.train()

    assert trainer.stats[0].param_groups[0]["name"] == "model"


def test_stats_hook_reports_aggregated_stats():
    trainer = _LRScheduleTrainer()

    trainer.train()

    assert trainer.stats[0].records == {}
    assert isinstance(trainer.stats[0].loss, float)
    assert trainer.stats[0].grad_norm is None
