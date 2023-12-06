from typing import Optional

import torch
import torch.nn as nn


class LSTMFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        out_size: Optional[int] = None,
    ):
        super().__init__()
        self.fc = nn.Linear(in_channels, hidden_size)
        self.height = hidden_size * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.out_dim = hidden_size * 2 if bidirectional else hidden_size * 1
        self.out_chans = 1
        self.out_size = out_size
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

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
        x = self.fc(x)  # x: (batch_size, output_size, hidden_size) (32,2880,64)
        x, _ = self.lstm(x)  # x: (batch_size, output_size, hidden_size * num_directions) (32,2880,128)
        x = x.transpose(1, 2)  # x: (batch_size, hidden_size * num_directions, output_size) (32,128, 2880)
        x = x.unsqueeze(1)  # x: (batch_size, out_chans, hidden_size * num_directions, time_steps) (32,1,128,2880)
        return x
