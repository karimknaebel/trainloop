from types import SimpleNamespace

from PIL import Image

from trainloop import ImageFileLoggerHook, WandbHook, hooks


def test_image_file_logger_image_format_receives_full_key(tmp_path):
    trainer = SimpleNamespace(workspace=tmp_path, step=7, logger=None)
    hook = ImageFileLoggerHook(
        image_format=lambda key: "jpg" if key[0] == "mesh" else "png"
    )

    hook.on_log_images(
        trainer,
        {
            "mesh": {"dataset.filename": Image.new("RGB", (1, 1))},
            "depth": {"dataset.filename": Image.new("RGB", (1, 1))},
        },
    )

    assert (
        tmp_path / "visualizations" / "7" / "mesh" / "dataset.filename.jpg"
    ).is_file()
    assert (
        tmp_path / "visualizations" / "7" / "depth" / "dataset.filename.png"
    ).is_file()


def test_wandb_hook_image_format_receives_full_key(monkeypatch):
    logged = {}
    trainer = SimpleNamespace(step=3, logger=None)
    hook = WandbHook(
        project="test",
        image_format=lambda key: "jpeg" if key[-2] == "mesh" else "png",
    )
    hook.wandb = SimpleNamespace(log=lambda data, step: logged.update(data))

    class FakeImage:
        def __init__(self, img, caption, file_type):
            self.img = img
            self.caption = caption
            self.file_type = file_type

    monkeypatch.setattr(hooks.wandb, "Image", FakeImage)

    hook.on_log_images(
        trainer,
        {
            "vis": {
                "mesh": {"dataset.filename": Image.new("RGBA", (1, 1))},
                "depth": {"dataset.filename": Image.new("RGB", (1, 1))},
            }
        },
    )

    assert logged["vis/mesh"][0].caption == "dataset.filename"
    assert logged["vis/mesh"][0].file_type == "jpeg"
    assert logged["vis/mesh"][0].img.mode == "RGB"
    assert logged["vis/depth"][0].file_type == "png"
