#!/usr/bin/env python3
import os
from itertools import chain
from collections import OrderedDict
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
import torchvision
import torchmetrics

from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from pytorch_lightning import loggers as pl_loggers
from collage import Collage
from fid import FID


class LitModule(pl.LightningModule):
    def __init__(self,
                 metrics,
                 img_size,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters("img_size", *kwargs.keys())
        self.channels, self.height, self.width = img_size

        # nn
        self.encoder, self.decoder = None, None
        self.losses = []

        # Checkpoint. Let you load partial attributes.
        if self.hparams.ckpt is not None:
            for attr in ['encoder', 'decoder']:
                self.load_attr_state_dict(attr)

        # dataset & evaluation
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.val = metrics.clone(prefix='val_')
        self.test = metrics.clone(prefix='test_')

    def load_attr_state_dict(self, attr):
        if getattr(self, attr) is None:
            return
        assert os.path.exists(self.hparams.ckpt), f'Error: Path {self.hparams.ckpt} not found.'
        checkpoint = torch.load(self.hparams.ckpt)
        state_dict = OrderedDict()
        found = False
        for key, val in checkpoint['state_dict'].items():
            if attr == key.split('.')[0]:
                found = True
                state_dict['.'.join(key.split('.')[1:])] = val
        if found:
            getattr(self, attr).load_state_dict(state_dict, strict=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument("--dataset", type=str, default='CIFAR10',
                            choices=['MNIST', 'CIFAR10', 'CIFAR100', 'FFHQ', 'ImageNet'], help="training data")
        parser.add_argument("--dataset_root", type=str, default=os.path.expanduser("~/.cache"))
        parser.add_argument("--img_size", type=int, nargs="*", default=[3, 32, 32])
        parser = parent_parser.add_argument_group("Optimization")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        parser.add_argument("--batch_size", type=int, default=64, help="training batch size")
        parser.add_argument("--betas", type=float, nargs="*", default=[0.5, 0.9], help="parameters of Adam")  # ADAM
        parser.add_argument("--num_workers", type=int, default=10, help="number of CPUs available")
        parser.add_argument("--seed", type=int, default=42, help="Fixed seed for reproducibility")
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--ckpt", type=str, default=None, help="path to pretrained model")
        return parent_parser

    def prepare_data(self):
        transforms = T.Compose([T.RandomHorizontalFlip(), T.Resize(self.hparams.img_size[1]), T.ToTensor()])
        if self.hparams.dataset in ['MNIST', 'CIFAR10', 'CIFAR100']:
            if self.hparams.dataset == 'MNIST':
                transforms = T.Compose([T.Pad(2), T.ToTensor()])
            root = os.path.expanduser("~/.cache") if self.hparams.dataset_root is None else self.hparams.dataset_root
            ds_cls = getattr(torchvision.datasets, self.hparams.dataset)
            ds = ds_cls(root, download=True, train=True, transform=transforms)
            train_size = int(len(ds) * 0.8)
            self.train_ds, self.val_ds = random_split(ds, [train_size, len(ds) - train_size], torch.Generator().manual_seed(self.hparams.seed))
            self.test_ds = ds_cls(root, download=True, train=False, transform=transforms)
        elif self.hparams.dataset == 'FFHQ':
            root = '~/data/images1024x1024/' if self.hparams.dataset_root is None else self.hparams.dataset_root
            ds = torchvision.datasets.ImageFolder(root, transform=transforms)
            self.train_ds, self.val_ds, self.test_ds = random_split(ds, [66000, 2000, 2000], torch.Generator().manual_seed(self.hparams.seed))
        elif self.hparams.dataset == 'ImageNet':
            root = '~/data/ImageNet' if self.hparams.dataset_root is None else self.hparams.dataset_root
            transforms = T.Compose([T.RandomHorizontalFlip(), T.Resize(256), T.CenterCrop(224), T.ToTensor()])
            ds = torchvision.datasets.ImageNet(root, 'train', transform=transforms)
            train_size = int(len(ds) * 0.8)
            self.train_ds, self.val_ds = random_split(ds, [train_size, len(ds) - train_size], torch.Generator().manual_seed(self.hparams.seed))
            self.test_ds = torchvision.datasets.ImageNet(root, 'val', transform=transforms)
        else:
            raise NotImplementedError

    def on_fit_start(self) -> None:
        for val in self.val.values():
            if hasattr(val, 'prepare_metric'):
                val.prepare_metric(self)

    def on_test_start(self) -> None:
        for test in self.test.values():
            if hasattr(test, 'prepare_metric'):
                test.prepare_metric(self)

    def configure_optimizers(self):
        optim_list = []
        adam_kwargs = dict(lr=self.hparams.lr, betas=self.hparams.betas, eps=1e-8)
        optim_list.append(torch.optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), **adam_kwargs))
        assert len(optim_list) == len(self.losses)
        return optim_list

    def loss(self, img):
        raise NotImplementedError()

    def forward(self, img):
        raise NotImplementedError()

    def collage(self, img):
        log = {}
        log['training_images'] = [img for _ in range(5)]  # logs a batch_size x 5 collage
        return log

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        img, _ = batch
        loss, logs = self.losses[optimizer_idx](img)
        self.log_dict(logs, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        self.val['PSNR'].update(x_hat, x)
        self.val['FID'].update(x_hat)

    def validation_epoch_end(self, outputs):
        res = self.val.compute()
        self.val.reset()
        return self.log_dict(res)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        self.test['PSNR'].update(x_hat, x)
        self.test['FID'].update(x_hat)

    def test_epoch_end(self, outputs):
        res = self.test.compute()
        self.test.reset()
        return self.log_dict(res)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True
        )


def main(args):
    # ensures that weight initializations are all the same
    pl.seed_everything(args.seed)
    gpu_num = (args.gpus if torch.cuda.is_available() else 0)
    logger = None
    callbacks = []
    metrics = {'PSNR': torchmetrics.PSNR()}
    if not args.fast_dev_run:
        logger = pl_loggers.WandbLogger(name=f'ivae_{args.dataset}', project=f'ivae_{args.dataset}', entity='gip')
        logger.log_hyperparams(args)
        if args.dataset == 'MNIST':  # Inception v3 is not trained for MNIST
            metrics['FID'] = FID()
        callbacks += [Collage()]
    metrics = torchmetrics.MetricCollection(metrics)
    dict_args = vars(args)
    model = LitModule(metrics, **dict_args)
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir='logs', logger=logger,
                                            callbacks=callbacks, strategy='ddp' if gpu_num > 1 else None)
    trainer.fit(model)
    trainer.test(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = LitModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
