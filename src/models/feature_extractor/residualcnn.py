from typing import Callable, Optional

import torch
import torch.nn as nn


# ref: https://github.com/analokmaus/kaggle-g2net-public/tree/main/models1d_pytorch
class ResidualCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_filters=128,
        kernel_sizes: tuple = (32, 16, 4, 2),
        stride: int = 4,
        sigmoid: bool = False,
        output_size: Optional[int] = None,
        conv: Callable = nn.Conv1d,
        reinit: bool = True,
    ):
        super().__init__()
        self.out_chans = len(kernel_sizes) # 输出通道数
        self.out_size = output_size # 2880 = duration // downsample_rate
        self.sigmoid = sigmoid
        if isinstance(base_filters, int):
            base_filters = tuple([base_filters])
        self.height = base_filters[-1]
        self.spec_conv = nn.ModuleList()
        for i in range(self.out_chans):
            if i == 0:
                tmp_block = [
                    conv(
                        in_channels, # input_channels
                        base_filters[0], # output_channels
                        kernel_size=kernel_sizes[i],
                        stride=stride,
                        padding=(kernel_sizes[i] - 1) // 2,
                    )
                ]
            else:
                tmp_block = [
                    conv(
                        base_filters[0],
                        base_filters[0],
                        kernel_size=kernel_sizes[i],
                        stride=stride,
                        padding = (kernel_sizes[i] - 1) // 2
                    )
                ]
            if len(base_filters) > 1:
                for j in range(len(base_filters) - 1):
                    tmp_block = tmp_block + [
                        nn.BatchNorm1d(base_filters[j]),
                        nn.ReLU(inplace=True),
                        conv(
                            base_filters[j],
                            base_filters[j + 1],
                            kernel_size=kernel_sizes[i],
                            stride=stride,
                            padding=(kernel_sizes[i] - 1) // 2,
                        ),
                    ]
                self.spec_conv.append(nn.Sequential(*tmp_block))
            else:
                self.spec_conv.append(tmp_block[0])

        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size)) # 高度维度不变，宽度维度调整为out_size

        if reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (_type_): (batch_size, in_channels, time_steps)

        Returns:
            _type_: (batch_size, out_chans, height, time_steps)
        """
        # x: (batch_size, in_channels, time_steps) [64,4,5760]
        out: list[torch.Tensor] = []
        for i in range(self.out_chans):
            x = self.spec_conv[i](x) # [32, 64,2880]
            out.append(x)
        img = torch.stack(out, dim=1)  # (batch_size, out_chans, height, time_steps) [64,3,64,2880]
        if self.out_size is not None:
            img = self.pool(img)  # (batch_size, out_chans, height, out_size)
        if self.sigmoid:
            img = img.sigmoid()
        return img