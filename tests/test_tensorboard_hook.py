from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from trainloop import TensorBoardHook


def test_tensorboard_uses_dot_namespaces_for_images():
    hook = TensorBoardHook(namespace_separator=".")
    hook.writer = Mock()
    trainer = SimpleNamespace(step=10)

    hook.on_log_images(
        trainer,
        {
            "vis": {
                "mesh": {
                    "Hypersim.scene": Image.new("RGB", (16, 9)),
                    "ETH3D.scene": Image.new("RGBA", (16, 9)),
                },
                "depth": {"Hypersim.scene": Image.new("L", (16, 9))},
            }
        },
    )

    calls = hook.writer.add_image.call_args_list
    assert [call.args[0] for call in calls] == [
        "vis.mesh.Hypersim.scene/image",
        "vis.mesh.ETH3D.scene/image",
        "vis.depth.Hypersim.scene/image",
    ]
    assert all(call.args[1].shape == (9, 16, 3) for call in calls)
    hook.writer.flush.assert_called_once_with()


def test_tensorboard_purges_events_after_loaded_checkpoint(tmp_path, monkeypatch):
    writers = [Mock(), Mock()]
    summary_writer = Mock(side_effect=writers)
    monkeypatch.setattr("trainloop.hooks.SummaryWriter", summary_writer)
    trainer = SimpleNamespace(workspace=tmp_path, step=0)
    hook = TensorBoardHook()

    hook.on_before_train(trainer)

    assert summary_writer.call_args.kwargs["purge_step"] is None
    assert summary_writer.call_args.kwargs["max_queue"] == 1_000

    trainer.step = 5000
    hook.on_load_state_dict(trainer, {})

    writers[0].close.assert_called_once_with()
    assert summary_writer.call_args.kwargs["purge_step"] == 5001


def test_tensorboard_remembers_checkpoint_loaded_before_writer(tmp_path, monkeypatch):
    summary_writer = Mock()
    monkeypatch.setattr("trainloop.hooks.SummaryWriter", summary_writer)
    trainer = SimpleNamespace(workspace=tmp_path, step=5000)
    hook = TensorBoardHook()

    hook.on_load_state_dict(trainer, {})
    hook.on_before_train(trainer)

    assert summary_writer.call_args.kwargs["purge_step"] == 5001


def test_tensorboard_uses_metric_first_scalar_sections():
    hook = TensorBoardHook(namespace_separator=".")
    hook.writer = Mock()
    trainer = SimpleNamespace(step=10)

    hook.on_log_scalars(
        trainer,
        {
            "loss": {"train": 0.5, "val.Hypersim": 0.4},
            "criterion": {
                "global": {
                    "loss": {"train": 0.3, "val.Hypersim": 0.2},
                }
            },
        },
    )

    assert [call.args[0] for call in hook.writer.add_scalar.call_args_list] == [
        "loss/train",
        "loss/val.Hypersim",
        "criterion.global.loss/train",
        "criterion.global.loss/val.Hypersim",
    ]
    hook.writer.flush.assert_called_once_with()
