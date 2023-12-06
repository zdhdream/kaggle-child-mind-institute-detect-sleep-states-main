from typing import Optional

import torch
import torch.nn as nn


def block(kernel_size, input_size, hidden_size):
    return nn.Sequential(
        nn.Conv1d(input_size, hidden_size, kernel_size, padding="same"),
        nn.LeakyReLU(),
    )

class ResidualBiGRU(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            n_layers: int,
            bidir: bool=True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidir
        )
        dir_factor = 2 if bidir else 1
        self.fc1 = nn.Linear(
            hidden_size * dir_factor, hidden_size * dir_factor * 2
        )
        self.ln1 = nn.LayerNorm(hidden_size * dir_factor * 2)
        self.fc2 = nn.Linear(hidden_size * dir_factor * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size*dir_factor)
        # self.ln3 = nn.LayerNorm(hidden_size * dir_factor)

    def forward(self, x, h=None):
        res, new_h = self.gru(x, h) # x[32,2880, 64]
        # res.shape = (batch_size, sequence_size, 2*hidden_size)

        res = self.fc1(res) # [32, 2880, 256]
        res = self.ln1(res)
        res = nn.functional.relu(res)

        res = self.fc2(res) # [32, 2880, 128]
        res = self.ln2(res)
        res = nn.functional.relu(res)

        # x = self.fc3(x)
        # x = self.ln3(x)
        # x = nn.functional.relu(x)

        # skip connection
        res = res + x

        return res, new_h


class MultiResidualBiGRU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        n_layers: int,
        kernels: list = [1,3,5,7],
        out_size: Optional[int] = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.out_chans = 1
        self.height = hidden_size * len(kernels) * ( 2 if bidirectional else 1 )
        self.out_size = out_size
        self.fc1 = nn.Linear(in_channels, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.conv_nets = nn.ModuleList(
            block(kernel_size, hidden_size, hidden_size)
            for kernel_size in kernels
        )
        self.res_bigrus = nn.ModuleList([
            ResidualBiGRU(hidden_size=4*hidden_size, n_layers=num_layers, bidir=bidirectional)
            for _ in range(n_layers)
        ])
        self.fc2 = nn.Linear(4*hidden_size, 8*hidden_size)
        self.ln2 = nn.LayerNorm(8*hidden_size)
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (32, 4, 5760)"""
        h = [None for _ in range(self.n_layers)]
        x = x.transpose(2,1) # [32,5760,4]
        x = self.fc1(x) # [32, 5760, 16]
        x = self.ln1(x)
        x = nn.functional.relu(x)
        x = x.transpose(2, 1) # [32,16, 5760]
        conv_res = []
        for i, net in enumerate(self.conv_nets):
            conv_res.append(net(x))
        x = torch.cat(conv_res, dim=1) # [32, 4 * 16, 5760]
        if self.out_size is not None:
            x = x.unsqueeze(1)  # [32,1,64,5760]
            x = self.pool(x) # [32,1,64, 2880]
            x = x.squeeze(1) # [32,64,2880]
        x = x.transpose(2,1) # [32, 2880, 64]
        for i, res_bigru in enumerate(self.res_bigrus):
            x, _  =  res_bigru(x, h[i]) # [32, 2880, 64]
        x = self.fc2(x)
        x = self.ln2(x)
        x = nn.functional.relu(x) # [32, 2880, 128]
        x = x.transpose(2,1) # [32,128,2880]
        x = x.unsqueeze(1) # [32,1,128,2880]
        return x



