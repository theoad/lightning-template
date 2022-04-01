import os
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, random_split

import torchvision
from torchvision import transforms as T
import pytorch_lightning as pl


class LitBase(pl.LightningModule):
    def __init__(self,
                 metrics,
                 img_size,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters("img_size", *kwargs.keys())
        self.channels, self.height, self.width = img_size

        # nn
        self.losses = []

        # Checkpoint. Let you load partial attributes.
        if self.hparams.checkpoint is not None:
            for attr in self.children():
                self.load_attr_state_dict(attr)

        # dataset & evaluation
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.val = metrics.clone(prefix='val_')
        self.test = metrics.clone(prefix='test_')

    def load_attr_state_dict(self, attr):
        if getattr(self, attr) is None or not isinstance(getattr(self, attr), torch.nn.Module):
            return
        assert os.path.exists(self.hparams.checkpoint), f'Error: Path {self.hparams.checkpoint} not found.'
        checkpoint = torch.load(self.hparams.checkpoint)
        state_dict = OrderedDict()
        found = False
        for key, val in checkpoint['state_dict'].items():
            if attr == key.split('.')[0]:
                found = True
                state_dict['.'.join(key.split('.')[1:])] = val
        if found:
            getattr(self, attr).load_state_dict(state_dict, strict=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument("-d", "--dataset", type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'FFHQ', 'ImageNet'])
        parser.add_argument("-r", "--dataset_root", type=str, default=os.path.expanduser("~/.cache"))
        parser.add_argument("-s", "--img_size", type=int, nargs="*", default=[1, 32, 32])
        parser = parent_parser.add_argument_group("Optimization")
        parser.add_argument("-j", "--num_workers", type=int, default=10, help="number of CPUs available")
        parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="learning rate")
        parser.add_argument("-bs", "--batch_size", type=int, default=256, help="training batch size")
        parser.add_argument("--betas", type=str, default="0.9, 0.999", help="parameters of Adam")  # ADAM
        parser.add_argument("--seed", type=int, default=42, help="Fixed seed for reproducibility")
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("-ckpt", "--checkpoint", type=str, default=None, help="path to pretrained model")
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
        adam_kwargs = dict(lr=self.hparams.learning_rate, betas=self.hparams.betas, eps=1e-8)
        optim_list.append(torch.optim.Adam(self.model.parameters(), **adam_kwargs))
        assert len(optim_list) == len(self.losses)
        return optim_list

    def loss(self, img):
        raise NotImplementedError()

    def forward(self, img):
        raise NotImplementedError()

    @staticmethod
    def collage_methods():
        raise NotImplementedError()

    @staticmethod
    def batch_preprocess(batch):
        return batch[0]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss, logs = self.losses[optimizer_idx](self.batch_preprocess(batch))
        self.log_dict(logs, sync_dist=True, prog_bar=True)
        return loss

    def update_metric(self, x, mode='val'):
        x_hat = self(x)
        getattr(self, mode)['PSNR'].update(x_hat, x)
        if 'FID' in getattr(self, mode):
            getattr(self, mode)['FID'].update(x_hat)

    def log_metric(self, mode='val'):
        res = getattr(self, mode).compute()
        getattr(self, mode).reset()
        return self.log_dict(res, sync_dist=True, logger=True)

    def validation_step(self, batch, batch_idx):
        return self.update_metric(batch[0], 'val')

    def test_step(self, batch, batch_idx):
        return self.update_metric(batch[0], 'test')

    def validation_epoch_end(self, outputs):
        return self.log_metric('val')

    def test_epoch_end(self, outputs):
        return self.log_metric('test')

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
