from typing import Optional
from conformer import Conformer
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_layers: int,
        out_size: Optional[int] = None,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_size)
        self.height = hidden_size
        self.conformer = Conformer(
            input_dim=hidden_size,
            encoder_dim=2*hidden_size,
            num_encoder_layers=num_layers,
        )
        self.fc2 = nn.Linear(2*hidden_size, hidden_size)
        self.input_lengths = torch.LongTensor(out_size)
        self.out_chans = 1
        self.out_size = out_size
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): (batch_size, in_channels, time_steps)

        Returns:
            torch.Tensor: (batch_size, out_chans, height, time_steps)
        """
        # x: (batch_size, in_channels, time_steps)
        if self.out_size is not None:
            x = x.unsqueeze(1)  # x: (batch_size, 1, in_channels, time_steps) (32,1,4,5760)
            x = self.pool(x)  # x: (batch_size, 1, in_channels, output_size) (32,1,4,2880)
            x = x.squeeze(1)  # x: (batch_size, in_channels, output_size)  (32,4,2880)
        x = x.transpose(1, 2)  # x: (batch_size, output_size, in_channels) (32,2880,4)
        x = self.fc1(x)  # x: (batch_size, output_size, hidden_size) (32,2880,64)
        x = self.conformer(x, self.input_lengths)  # x: (batch_size, output_size, hidden_size * num_directions) (32,2880,128)
        x = x.transpose(2,1) # (32, 256, num_timestamps)
        x = F.interpolate(x, size=self.out_size, mode='linear', align_corners=False) # (32,2880,256)
        x = x.transpose(2,1) # (32, 2880, 256)
        x = self.fc2(x) # (32, 2880, 128)
        x = x.transpose(1, 2)  # x: (batch_size, hidden_size * num_directions, output_size) (32,128, 2880)
        x = x.unsqueeze(1)  # x: (batch_size, out_chans, hidden_size * num_directions, time_steps) (32,1,128,2880)
        return x