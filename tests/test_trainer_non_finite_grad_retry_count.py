import itertools

import torch

from trainloop import BaseHook, BaseTrainer, LoggingHook


class _CaptureLogHook(BaseHook):
    def __init__(self):
        self.records = []

    def on_log(self, trainer: BaseTrainer, records: dict, dry_run: bool = False):
        self.records.append(records)


class _RetryTrainer(BaseTrainer):
    def __init__(self, capture_hook: _CaptureLogHook):
        self.capture_hook = capture_hook
        self.forward_calls = 0
        super().__init__(
            max_steps=1,
            max_non_finite_grad_retries=1,
            device="cpu",
        )

    def build_data_loader(self):
        return itertools.repeat(torch.ones(1, 1))

    def build_model(self):
        return torch.nn.Linear(1, 1, bias=False)

    def build_optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

    def build_hooks(self):
        return [LoggingHook(interval=1, sync=False), self.capture_hook]

    def forward(self, input):
        self.forward_calls += 1
        loss = self.model(input).sum()
        if self.forward_calls == 1:
            loss = loss * torch.tensor(float("inf"))
        return loss, {"metric": 1.0}


def test_non_finite_grad_retry_count_saved_and_logged():
    capture_hook = _CaptureLogHook()
    trainer = _RetryTrainer(capture_hook)

    trainer.train()

    assert trainer.step_info["non_finite_grad_retry_count"] == 1
    assert capture_hook.records[0]["train"]["non_finite_grad_retry_count"] == 1.0
