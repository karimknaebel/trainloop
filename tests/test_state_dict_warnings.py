import logging
from types import SimpleNamespace

import torch

import trainloop.hooks
import trainloop.trainer
from trainloop import BaseTrainer, EMAHook


class _Trainer(BaseTrainer):
    def __init__(self, logger):
        super().__init__(max_steps=1, device="cpu", logger=logger)

    def build_data_loader(self):
        return []

    def build_model(self):
        return torch.nn.Linear(1, 1)

    def build_optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

    def forward(self, input):
        return self.model(input).sum(), {}


def test_trainer_logs_incompatible_model_state_dict_keys(monkeypatch, caplog):
    trainer = _Trainer(logging.getLogger("state-dict-test"))
    trainer._build()
    state_dict = trainer.state_dict()
    monkeypatch.setattr(
        trainloop.trainer,
        "set_state_dict",
        lambda *args, **kwargs: SimpleNamespace(
            missing_keys=["model.missing"], unexpected_keys=["model.unexpected"]
        ),
    )

    with caplog.at_level(logging.WARNING, logger="state-dict-test"):
        trainer.load_state_dict(state_dict)

    assert 'Missing keys in state_dict: "model.missing".' in caplog.messages
    assert 'Unexpected keys in state_dict: "model.unexpected".' in caplog.messages


def test_ema_hook_logs_incompatible_model_state_dict_keys(monkeypatch, caplog):
    logger = logging.getLogger("ema-state-dict-test")
    trainer = SimpleNamespace(logger=logger)
    hook = EMAHook(decay=0.9, name="ema_slow")
    hook.ema_model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        trainloop.hooks,
        "set_model_state_dict",
        lambda *args, **kwargs: SimpleNamespace(
            missing_keys=["ema.missing"], unexpected_keys=["ema.unexpected"]
        ),
    )

    with caplog.at_level(logging.WARNING, logger="ema-state-dict-test"):
        hook.on_load_state_dict(trainer, {"ema_slow": {}})

    assert 'Missing keys in state_dict: "ema.missing".' in caplog.messages
    assert 'Unexpected keys in state_dict: "ema.unexpected".' in caplog.messages
