from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModel


class US_BLK(nn.Module):
    def __init__(self, in_c, out_c, in_len, ks=7, dilation=1, us_factor=2):
        super().__init__()
        padding = ((ks - 1) * dilation) // 2
        # 通过线性插值将输入信号的长度放大,增加时间维度的分辨率
        self.us = nn.Upsample(scale_factor=us_factor, mode='linear')
        self.conv1 = nn.Conv1d(in_c, out_c, ks, padding=padding, dilation=dilation)
        self.ln1 = nn.LayerNorm([out_c, in_len])
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(out_c, out_c, ks, padding=padding, dilation=dilation)
        self.ln2 = nn.LayerNorm([out_c, in_len])
        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.us(x)
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.ln2(x)
        x = self.act2(x)
        return x


class BertFeatureExtractor(nn.Module):
    def __init__(self,
                 in_channels: int,
                 model_name: str,
                 emb_dim: int = 8,
                 base = 96,
                 pretrained: bool = True,
                 out_size: Optional[int] = None,
                 ):
        super(BertFeatureExtractor, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, add_pooling_layer=False)
        if pretrained:
            self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.bert = AutoModel.from_config(self.config)
        self.bert = self.bert.encoder
        self.fc_in = nn.Linear(in_channels, self.config.hidden_size - emb_dim)
        self.hr_emb = nn.Embedding(24, emb_dim)
        self.upsample_blocks = nn.ModuleList([])
        self.upsample_blocks.append(US_BLK(self.config.hidden_size, base*4, out_size))
        self.out_chans = 1
        self.out_size = out_size
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))


    def forward(self, x):
        if self.out_size is not None:
            x = x.unsqueeze(1)
            x = self.pool(x)
            x = x.squeeze(1)
        x = x.transpose(1,2) # (bs,seq_len, emb_dim)
        x = self.fc_in(x)
        t = self.hr_emb(t)

