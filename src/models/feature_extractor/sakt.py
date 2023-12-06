from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualLSTM(nn.Module):
    def __init__(self, d_model):
        super(ResidualLSTM, self).__init__()
        self.LSTM = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc1 = nn.Linear(d_model * 2, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        res = x
        x, _ = self.LSTM(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = res + x
        return x




class SAKTModel(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 nlayers: int,
                 rnnlayers: int,
                 dropout: float,
                 nheads: int,
                 out_size: Optional[int] = None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.out_size = out_size
        self.height = embed_dim
        self.out_chans = 1
        self.pos_encoder = nn.ModuleList(
            [
                ResidualLSTM(embed_dim)
                for _ in range(rnnlayers)
            ]
        )
        self.pos_encoder_dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(in_channels, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        encoder_layers = [
            nn.TransformerEncoderLayer(embed_dim, nheads, embed_dim*4, dropout)
            for i in range(nlayers)
        ]
        conv_layers = [
            nn.Conv1d(embed_dim,embed_dim,(nlayers-i)*2-1,stride=1,padding=0)
            for i in range(nlayers)
        ]
        deconv_layers = [
            nn.ConvTranspose1d(embed_dim,embed_dim,(nlayers-i)*2-1,stride=1,padding=0)
            for i in range(nlayers)
        ]
        layer_norm_layers = [
            nn.LayerNorm(embed_dim) for i in range(nlayers)
        ]
        layer_norm_layers2 = [
            nn.LayerNorm(embed_dim) for i in range(nlayers)
        ]
        self.transformer_encoder = nn.ModuleList(encoder_layers) # transformer层
        self.conv_layers = nn.ModuleList(conv_layers) # 卷积层
        self.layer_norm_layers = nn.ModuleList(layer_norm_layers)
        self.layer_norm_layers2 = nn.ModuleList(layer_norm_layers2)
        self.deconv_layers = nn.ModuleList(deconv_layers) # 反卷积层
        self.nheads = nheads
        self.downsample = nn.Linear(embed_dim*2,embed_dim)
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

    def forward(self, x):
        """x: (32,4,5760)"""
        if self.out_size is not None:
            x = x.unsqueeze(1) # [32,1,4,5760]
            x = self.pool(x) # [32,1,4,2880]
            x = x = x.squeeze(1) # [32,4,2880]
        x = x.transpose(2,1) # [32, 2880, 4]
        x = self.embedding(x) # 先进行线性映射 (32, 2880, emb_dim)
        for lstm in self.pos_encoder:
            lstm.LSTM.flatten_parameters()
            x=lstm(x)

        x = self.pos_encoder_dropout(x)
        x = self.layer_norm(x)



        for conv, transformer_layer, layer_norm1, layer_norm2, deconv in zip(self.conv_layers,
                                                               self.transformer_encoder,
                                                               self.layer_norm_layers,
                                                               self.layer_norm_layers2,
                                                               self.deconv_layers):
            #LXBXC to BXCXL
            res=x
            # x: [bs, seq_len, emb_size] -> [bs, emb_size, seq_len]
            # x=F.relu(conv(x.permute(1,2,0)).permute(2,0,1))
            x=F.relu(conv(x.permute(0,2,1)).permute(0,2,1))
            x=layer_norm1(x) # [bs, seq_len, num_features]
            x=transformer_layer(x)
            x=F.relu(deconv(x.permute(0,2,1)).permute(0,2,1))
            x=layer_norm2(x)
            x=res+x

        x = x.transpose(2,1).unsqueeze(1) # [32,1,128,2880]


        return x

