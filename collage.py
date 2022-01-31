import wandb
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
import numpy as np
from pytorch_lightning.utilities import rank_zero_only


def create_collage(tensor_list):
    t = torch.cat(tensor_list, dim=3)  # One long horizontal image (batchwize)
    t = t.permute(0, 2, 3, 1).reshape(1, t.shape[0] * t.shape[2], t.shape[3], t.shape[1]).permute(0, 3, 1, 2)  # One big rectangle (concat batch)
    return t


class Collage(Callback):
    def __init__(self, mode='val', num_samples=8):
        super().__init__()
        self.num_samples = num_samples
        if mode not in ['val', 'test']:
            raise ValueError("Expecting the dataset type to be either 'val' or 'test'.")
        self.mode = mode
        self.batch = None

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module: LightningModule):
        if self.mode == 'val':
            datamodule = pl_module.val_dataloader()
        else:
            datamodule = pl_module.test_dataloader()
        iterator = iter(datamodule)
        self.batch = next(iterator)

        def get_batch_size(batch):
            if type(batch) is list:
                return batch[0].shape[0]
            return batch.shape[0]

        def append_to_batch(batch, new_elems):
            if type(batch) is list:
                batch = torch.cat((batch[0], new_elems), dim=0)
                return batch
            return torch.cat((batch, new_elems), dim=0)

        def shrink_batch(batch, new_batch_size):
            if type(batch) is list:
                return batch[0][:new_batch_size]
            return batch[:new_batch_size]

        while get_batch_size(self.batch) < self.num_samples:
            self.batch = append_to_batch(self.batch, next(iterator))
        self.batch = shrink_batch(self.batch, self.num_samples)

    def to_np(self, image):
        image = image.clamp_(0, 1).detach().cpu().numpy()
        image = np.transpose(image, (0, 2, 3, 1))
        return image

    def log_batch(self, images, log, caption='collage'):
        if type(images) is list:
            for i in range(len(images)):
                images[i] = self.to_np(images[i])
            log[caption] = [wandb.Image(np.concatenate(image, axis=1), caption=caption) for image in zip(*images)]
        elif images.shape[0] == 1:
            images = self.to_np(images)
            log[caption] = wandb.Image(images[0], caption=caption)
        else:
            images = self.to_np(images)
            log[caption] = [wandb.Image(image, caption=caption) for image in images]

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        pl_module.eval()
        log = {}
        batch = self.batch.to(pl_module.device)
        if hasattr(pl_module, 'collage'):
            pl_log = pl_module.collage(batch)
            for caption, im in pl_log.items():
                self.log_batch(create_collage(im), log, caption)
        else:
            raise NotImplementedError
        pl_module.train()
        trainer.logger.experiment.log(log)
