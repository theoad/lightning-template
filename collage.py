from typing import Any, Optional

import pytorch_lightning as pl
import wandb
import torch
from torchvision.utils import make_grid
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning import Callback


class Collage(Callback):
    """
    Taken from https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/callbacks/vision/sr_image_logger.py
    """

    def __init__(self, methods: str, log_interval: int = 100, num_samples: int = 8) -> None:
        """
        Args:
            log_interval: Number of steps between logging. Default: ``100``.
            num_samples: Number of images of displayed in the grid. Default: ``8``.
        """
        super().__init__()
        self.methods = methods
        self.log_interval = log_interval
        self.num_samples = num_samples

    def log_images(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any):
        for method in self.methods:
            inputs = pl_module.batch_preprocess(batch)
            img_list = getattr(pl_module, method)(inputs)
            if len(img_list) == 0:
                continue
            collage = torch.cat(img_list, dim=-1).clamp(0, 1)  # concatenate on width dimension
            wb_collage = wandb.Image(make_grid(collage[:min(collage.size(0), self.num_samples)], nrow=1), caption=method)
            trainer.logger.experiment.log({method: wb_collage})

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if batch_idx == 0:
            self.log_images(trainer, pl_module, batch)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        pl_module.eval()
        with torch.no_grad():
            if trainer.global_step % trainer.log_every_n_steps == 0:
                self.log_images(trainer, pl_module, batch)
        pl_module.train()
