# trainloop

[![PyPI version](https://img.shields.io/pypi/v/trainloop.svg)](https://pypi.org/project/trainloop/)

Minimal PyTorch training loop with hooks for logging, checkpointing, and customization.

Docs: https://karimknaebel.github.io/trainloop/

## Install

```bash
pip install trainloop
```

## Basic example

```python
import logging

import torch
import torch.nn as nn

from trainloop import BaseTrainer, CheckpointingHook, ProgressHook

logging.basicConfig(level=logging.INFO)


class MyTrainer(BaseTrainer):
    def build_data_loader(self):
        class ToyDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                while True:
                    data = torch.randn(784)
                    target = torch.randint(0, 10, (1,)).item()
                    yield data, target

        return torch.utils.data.DataLoader(ToyDataset(), batch_size=32)

    def build_model(self):
        return nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ).to(self.device)

    def build_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=3e-4)

    def build_hooks(self):
        return [
            ProgressHook(interval=50, with_records=True),
            CheckpointingHook(interval=500, keep_previous=2),
        ]

    def forward(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        accuracy = (logits.argmax(1) == y).float().mean().item()
        return loss, {"accuracy": accuracy}


trainer = MyTrainer(max_steps=2000, device="cpu", workspace="runs/demo")
trainer.train()
```
