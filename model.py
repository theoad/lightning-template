from functools import partial
from torch.nn import functional as F

import utils
from base_model import LitBase
from architecture import UNET


class Model(LitBase):
    def __init__(self,
                 metrics,
                 img_size,
                 **kwargs
                 ):
        super().__init__(metrics, img_size, **kwargs)
        self.expand = partial(utils.expand_4d_batch, n=self.hparams.expansion)
        self.reduce = partial(utils.restore_expanded_4d_batch, n=self.hparams.expansion)
        self.reduce_mean = partial(utils.mean_expanded_batch, n=self.hparams.expansion)
        self.reduce_std = partial(utils.std_expanded_batch, n=self.hparams.expansion)

        # nn
        self.losses = [self.loss]

        # TODO: define model here
        self.model = UNET(
            self.channels, self.height, self.hparams.bottleneck_size[1],
            self.hparams.bottleneck_size[0], self.hparams.channel_base * 4,
            self.hparams.scaling_factor, self.hparams.n_residual_blocks,
            self.hparams.channel_base
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        LitBase.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Optimization")
        parser.add_argument("-e", "--expansion", type=int, default=1)

        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--bottleneck_size", type=str, default="128, 2, 2")
        parser.add_argument("--scaling_factor", type=int, default=4)
        parser.add_argument("--channel_base", type=int, default=32)
        parser.add_argument("--n_residual_blocks", type=int, default=2)
        return parent_parser

    def loss(self, img):
        # TODO: implement loss (ex: auto-encoder loss)
        x_hat = self(self.expand(img))
        recon_loss = F.mse_loss(self.reduce_mean(x_hat), img)
        return recon_loss, {'reconstruction': recon_loss.item()}

    def forward(self, img):
        return self.model(img)

    @staticmethod
    def collage_methods():
        # TODO: add collage panel
        return ['recon', 'sampling']

    def recon(self, img):
        x = self(self.expand(img))
        x_mean, x_std = self.reduce_mean(x), self.reduce_std(x)
        return [img, x_mean, x_std] + [x[img.size(0) * i:img.size(0) * (i + 1)] for i in range(self.hparams.expansion)]

    def sampling(self, img):
        return [self.model.sample(img.shape[0], self.device, img.dtype) for _ in range(self.hparams.expansion)] if hasattr(self.model, 'sample') else []
