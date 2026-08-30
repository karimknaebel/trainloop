import logging
from types import SimpleNamespace

import pytest
import torch

from trainloop import EMAHook


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.register_buffer("running", torch.tensor([1.0]))


def test_ema_hook_defaults():
    hook = EMAHook()

    assert hook.decay == 0.999
    assert not hook.use_buffers
    assert hook.name == "ema"


def test_ema_hook_uses_multi_avg_fn_and_buffers():
    model = _Model()
    trainer = SimpleNamespace(logger=logging.getLogger("ema-test"), model=model)
    hook = EMAHook(decay=0.5, use_buffers=True)
    hook.on_before_train(trainer)

    assert hook.ema_model.avg_fn is None
    assert hook.ema_model.multi_avg_fn is not None
    assert hook.ema_model.use_buffers

    with torch.no_grad():
        model.weight.fill_(3)
        model.running.fill_(3)
    hook.on_after_step(trainer)
    with torch.no_grad():
        model.weight.fill_(5)
        model.running.fill_(5)
    hook.on_after_step(trainer)

    torch.testing.assert_close(hook.ema_model.module.weight, torch.tensor([4.0]))
    torch.testing.assert_close(hook.ema_model.module.running, torch.tensor([4.0]))


def test_ema_hooks_use_names_for_state_dict():
    trainer = SimpleNamespace(logger=logging.getLogger("ema-test"), model=_Model())
    hooks = [EMAHook(name="ema_fast"), EMAHook(name="ema_slow")]
    state_dict = {}

    for hook in hooks:
        hook.on_before_train(trainer)
        hook.on_state_dict(trainer, state_dict)

    assert state_dict.keys() == {"ema_fast", "ema_slow"}


def test_ema_hook_rejects_duplicate_state_dict_name():
    trainer = SimpleNamespace(logger=logging.getLogger("ema-test"), model=_Model())
    hook = EMAHook()
    hook.on_before_train(trainer)

    with pytest.raises(ValueError, match="State dict key 'ema' already exists"):
        hook.on_state_dict(trainer, {"ema": {}})
