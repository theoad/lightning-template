import torch
import torch.nn as nn
from math import log2
import utils

class ResBlock(nn.Sequential):
    def __init__(self, in_channel, channel):
        super().__init__(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, x):
        out = super().forward(x)
        out += x
        return out


class Upsample(nn.Sequential):
    def __init__(self, channel, out_channel, scaling_factor):
        super().__init__(*([nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True)] * (int(log2(scaling_factor))-1)
                             + [nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)]))


class Encoder(nn.Sequential):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]
        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]
        else:
            raise ValueError(f'stride parameter must be either 2 or 4. given stride={stride}')

        blocks.extend([ResBlock(channel, n_res_channel) for _ in range(n_res_block)])
        blocks.append(nn.ReLU(inplace=True))

        super().__init__(*blocks)


class Decoder(nn.Sequential):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        blocks.extend([ResBlock(channel, n_res_channel) for _ in range(n_res_block)])
        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend([
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1)
            ])
        elif stride == 2:
            blocks.append(nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))
        else:
            raise ValueError(f'stride parameter must be either 2 or 4. given stride={stride}')

        super().__init__(*blocks)


class UNET(nn.Module):
    def __init__(
            self,
            in_channel,
            img_res,
            lowest_res=4,
            embed_dim=64,
            channel=128,
            scaling_factor=2,
            n_res_block=2,
            n_res_channel=32,
    ):
        super().__init__()
        block_scaling = utils.get_block_scaling(img_res, lowest_res, scaling_factor)

        encoders = [Encoder(in_channel, channel, n_res_block, n_res_channel, block_scaling[0])]
        for scaling in block_scaling[1:]:
            encoders.append(Encoder(channel, channel, n_res_block, n_res_channel, scaling))
        self.enc = nn.Sequential(*encoders)

        block_scaling.reverse()

        decoders = [Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, block_scaling[0])]
        for scaling in block_scaling[1:-1]:
            decoders.append(Decoder(2 * embed_dim, embed_dim, channel, n_res_block, n_res_channel, scaling))
        decoders.append(Decoder(2 * embed_dim, in_channel, channel, n_res_block, n_res_channel, block_scaling[-1]))
        self.dec = nn.Sequential(*decoders)

        crossing_conv = [nn.Conv2d(channel, embed_dim, 1)]
        for _ in block_scaling[1:]:
            crossing_conv.append(nn.Conv2d(channel + embed_dim, embed_dim, 1))
        self.crossing_conv = nn.ModuleList(crossing_conv)

        self.upsample = nn.ModuleList([Upsample(embed_dim, embed_dim, block_scaling[0])] + [Upsample(2 * embed_dim, embed_dim, scaling) for scaling in block_scaling[1:-1]])

    def forward(self, inputs):
        x = inputs

        # Encode
        encodings = []
        for enc in self.enc:
            x = enc(x)
            encodings.append(x)
        encodings.reverse()

        # bottleneck
        x = self.crossing_conv[0](x)

        # Decode
        for cross_conv, up, dec, enc in zip(self.crossing_conv[1:], self.upsample, self.dec[:-1], encodings[1:]):
            prev = up(x)
            x = cross_conv(torch.cat((enc, dec(x)), dim=1))
            x = torch.cat((prev, x), dim=1)
        x = self.dec[-1](x)

        return x

