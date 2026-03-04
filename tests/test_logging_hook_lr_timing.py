import itertools

import torch

from trainloop import BaseHook, BaseTrainer, LoggingHook


class _CaptureLogHook(BaseHook):
    def __init__(self):
        self.records = []

    def on_log(self, trainer: BaseTrainer, records: dict, dry_run: bool = False):
        self.records.append(records)


class _LRScheduleTrainer(BaseTrainer):
    def __init__(self, capture_hook: _CaptureLogHook):
        self.capture_hook = capture_hook
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
        return [LoggingHook(interval=1, sync=False), self.capture_hook]

    def forward(self, input):
        loss = self.model(input).sum()
        return loss, {}


def test_logging_hook_logs_pre_step_lr():
    capture_hook = _CaptureLogHook()
    trainer = _LRScheduleTrainer(capture_hook)

    trainer.train()

    lrs = [record["train"]["lr"]["group_0"] for record in capture_hook.records]
    assert lrs == [0.1, 0.05]
