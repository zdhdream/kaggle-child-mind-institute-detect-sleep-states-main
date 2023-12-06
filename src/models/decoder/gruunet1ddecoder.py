# ref: https://github.com/bamps53/kaggle-dfl-3rd-place-solution/blob/master/models/cnn_3d.py
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        (
            b,
            c,
            _,
        ) = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


def block(kernel_size, input_size, hidden_size):
    return nn.Sequential(
        nn.Conv1d(input_size, hidden_size, kernel_size, padding="same"),
        nn.LeakyReLU(),
    )


class ResidualBiGRU(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 n_layers=1,
                 bidir=True):
        super(ResidualBiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        dir_factor = 2 if bidir else 1
        self.fc1 = nn.Linear(
            hidden_size * dir_factor, hidden_size * dir_factor * 2
        )
        self.ln1 = nn.LayerNorm(hidden_size * dir_factor * 2)
        self.fc2 = nn.Linear(hidden_size * dir_factor * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, h=None):
        res, new_h = self.gru(x)

        res = self.fc1(res)
        res = self.ln1(res)
        res = nn.functional.relu(res)

        res = self.fc2(res)
        res = self.ln2(res)
        res = nn.functional.relu(res)

        # skip connection
        res = res + x

        return res, new_h


class MultiResidualBiGRU(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 out_size: int,
                 n_layers: int,
                 bidir: bool):
        super(MultiResidualBiGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_layers = n_layers

        self.fc_in = nn.Linear(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.res_bigrus = nn.ModuleList(
            [
                ResidualBiGRU(hidden_size, n_layers=1, bidir=bidir)
                for _ in range(n_layers)
            ]
        )
        self.fc_out = nn.Linear(hidden_size, out_size)

    def forward(self, x, h=None):
        if h is None:
            h = [None for _ in range(self.n_layers)]

        x = self.fc_in(x)
        x = self.ln(x)
        x = nn.functional.relu(x)

        new_h = []
        for i, res_bigru in enumerate(self.res_bigrus):
            x, new_hi = res_bigru(x, h[i])
            new_h.append(new_hi)
        x = self.fc_out(x)
        return x, new_h


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
            self,
            in_channels,
            out_channels,
            mid_channels=None,
            norm=nn.BatchNorm1d,
            se=False,
            res=False,
    ):
        super().__init__()
        self.res = res
        if not mid_channels:
            mid_channels = out_channels
        if se:
            non_linearity = SEModule(out_channels)
        else:
            non_linearity = nn.ReLU(inplace=True)
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm(out_channels),
            non_linearity,
        )

    def forward(self, x):
        if self.res:  # x:[bs, emb_dim, num_timestamps]
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
            self, in_channels, out_channels, scale_factor, norm=nn.BatchNorm1d, se=False, res=False
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(scale_factor),
            DoubleConv(in_channels, out_channels, norm=norm, se=se, res=res),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
            self, in_channels, out_channels, bilinear=True, scale_factor=2, norm=nn.BatchNorm1d
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels // 2, kernel_size=scale_factor, stride=scale_factor
            )
            self.conv = DoubleConv(in_channels, out_channels, norm=norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def create_layer_norm(channel, length):
    return nn.LayerNorm([channel, length])


class GRUUNet1DDecoder(nn.Module):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            duration: int,
            bilinear: bool = True,
            se: bool = False,
            res: bool = False,
            scale_factor: int = 2,
            dropout: float = 0.2,
    ):
        super().__init__()
        self.n_channels = n_channels  # 64
        self.n_classes = n_classes  # 3
        self.duration = duration  # 2880
        self.bilinear = bilinear
        self.se = se
        self.res = res
        self.scale_factor = scale_factor  # 2

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(
            self.n_channels, 64, norm=partial(create_layer_norm, length=self.duration)
        )
        self.down1 = Down(
            64, 128, scale_factor, norm=partial(create_layer_norm, length=self.duration // 2)
        )
        self.down2 = Down(
            128, 256, scale_factor, norm=partial(create_layer_norm, length=self.duration // 4)
        )
        self.down3 = Down(
            256, 512, scale_factor, norm=partial(create_layer_norm, length=self.duration // 8)
        )
        self.down4 = Down(
            512,
            1024 // factor,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 16),
        )
        self.conv_nets = nn.ModuleList([
            block(kernel_size, 1024, 1024) for kernel_size in [1, 3, 5, 7]
        ])
        self.res_grus = MultiResidualBiGRU(
            input_size= 1024,
            hidden_size=2048,
            out_size=1024,
            n_layers=3,
            bidir=True
        )
        self.up1 = Up(
            1024,
            512 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 8),
        )
        self.up2 = Up(
            512,
            256 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 4),
        )
        self.up3 = Up(
            256,
            128 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 2),
        )
        self.up4 = Up(
            128, 64, bilinear, scale_factor, norm=partial(create_layer_norm, length=self.duration)
        )

        self.cls = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, self.n_classes, kernel_size=1, padding=0),
            nn.Dropout(dropout),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
            self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ):
        """Forward

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """

        # 1D U-Net
        x1 = self.inc(x)  # [64,64,2880]
        x2 = self.down1(x1)  # [64,128,1440]
        x3 = self.down2(x2)  # [64,256,720]
        x4 = self.down3(x3)  # [64,512, 360]
        x = self.down4(x4)  # [64,1024,180]
        x = x.transpose(2, 1)  # [64, 180, 1024]
        x,_ = self.res_grus(x)  # [64, 180, 1024]
        x = x.transpose(1, 2)  # [64, 1024, 180]
        x = self.up1(x, x4)  # [64, 512, 360]
        x = self.up2(x, x3)  # [64, 256, 720]
        x = self.up3(x, x2)  # [64, 128, 1440]
        x = self.up4(x, x1)  # [64, 64, 2880]

        # classifier
        logits = self.cls(x)  # (batch_size, n_classes, n_timesteps) [64,3,2880]
        return logits.transpose(1, 2)  # (batch_size, n_timesteps, n_classes)
