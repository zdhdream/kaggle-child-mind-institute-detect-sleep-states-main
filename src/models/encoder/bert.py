from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModel

def block(kernel_size, in_c, out_c, duration=2880):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, padding="same"),
        nn.LayerNorm([out_c, duration]),
        nn.LeakyReLU(0.2),
        nn.Conv1d(in_channels=out_c, out_channels=out_c, kernel_size=kernel_size, padding="same"),
        nn.LayerNorm([out_c, duration]),
        nn.LeakyReLU(0.2)
    )

class BertEncoder(nn.Module):
    def __init__(self,
                 model_name: str,
                 in_channels: int,
                 pretrained: bool,
                 duration: int=2880,
                 ):
        super(BertEncoder, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, add_pooling_layer=False)
        if pretrained:
            self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.bert = AutoModel.from_config(self.config)
        self.bert = self.bert.encoder
        self.fc_in = nn.Linear(in_channels, self.config.hidden_size)
        self.conv1 = block(kernel_size=1, in_c=self.config.hidden_size, out_c=self.config.hidden_size, duration=duration)
        self.conv2 = block(kernel_size=3, in_c=self.config.hidden_size, out_c=self.config.hidden_size, duration=duration)
        self.conv3 = block(kernel_size=5, in_c=self.config.hidden_size, out_c=self.config.hidden_size, duration=duration)
        self.conv4 = block(kernel_size=7, in_c=self.config.hidden_size, out_c=self.config.hidden_size, duration=duration)
        self.fc_out = nn.Linear(self.config.hidden_size, in_channels)

    def forward(self, x, att_mask=None):
        """
        x: (bs, emb_dim, num_timestamps)
        """
        x = x.transpose(1,2) # (bs, num_timestamps, emb_dim)
        x = self.fc_in(x) # (bs, num_timestamps, hidden_size)

        if att_mask is None:
            att_mask = torch.ones(x.size()[:2])
        bert_output = self.bert(x, attention_mask=att_mask)
        x = bert_output.last_hidden_state # (bs, num_timestamps, hidden_size)
        x = x.transpose(1,2) # (bs, hidden_size, num_timestamps)
        old_x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + old_x
        x = x.transpose(1,2) # (bs, num_timestamps, hidden_size)
        x = self.fc_out(x) # (bs, num_timestamps, in_channels)
        x = x.transpose(1,2) # (bs,in_channels, num_timestamps)
        x = x.unsqueeze(1) # (bs, 1, emb_dim, num_timestamps)
        return x

if __name__ == "__main__":
    input = torch.randn((32, 128, 2880))
    model = BertEncoder(model_name="../../../weights/deberta-v3-small",
                        in_channels=128,
                        pretrained=True,
                        duration=2880)
    print(model(input).shape)
