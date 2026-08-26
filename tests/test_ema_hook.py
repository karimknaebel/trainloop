import logging
from types import SimpleNamespace

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
