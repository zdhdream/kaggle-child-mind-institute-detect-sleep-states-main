from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torchvision.transforms.functional import resize
from transformers import AutoConfig, AutoModel

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup
from src.models.base import BaseModel


def block(kernel_size, in_c, out_c, duration=2880):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, padding="same"),
        nn.LayerNorm([out_c, duration]),
        nn.LeakyReLU(0.2),
        nn.Conv1d(in_channels=out_c, out_channels=out_c, kernel_size=kernel_size, padding="same"),
        nn.LayerNorm([out_c, duration]),
        nn.LeakyReLU(0.2)
    )


class TransformerAutoModel(BaseModel):
    def __init__(
            self,
            feature_extractor: nn.Module,
            model_name: str,
            n_classes: int,
            mixup_alpha: float = 0.5,
            cutmix_alpha: float = 0.5,
            duration: int = 2880,
            pretrained: bool = True,
    ):
        super(TransformerAutoModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, add_pooling_layer=False)
        if pretrained:
            self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.bert = AutoModel.from_config(self.config)
        self.feature_extractor = feature_extractor
        self.height = feature_extractor.height
        self.bert = self.bert.encoder
        self.conv1 = block(kernel_size=1, in_c=self.height, out_c=self.config.hidden_size, duration=duration)
        self.conv2 = block(kernel_size=3, in_c=self.config.hidden_size, out_c=self.config.hidden_size,
                           duration=duration)
        self.conv3 = block(kernel_size=5, in_c=self.config.hidden_size, out_c=self.config.hidden_size,
                           duration=duration)
        self.conv4 = block(kernel_size=7, in_c=self.config.hidden_size, out_c=self.config.hidden_size,
                           duration=duration)
        # self.pool = nn.AdaptiveAvgPool1d(out_size)
        self.head = nn.Linear(self.config.hidden_size, n_classes)
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _forward(
            self,
            x: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            do_mixup: bool = False,
            do_cutmix: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.feature_extractor(x).squeeze(1)  # (bs, height, n_timesteps)
        old_x = self.conv1(x)  # (bs, hidden_size,  n_timesteps)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + old_x  # (bs, hidden_size, n_timesteps)
        x = x.transpose(1, 2)  # (bs, duration, n_channels)

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        x = self.backbone(
            inputs_embeds=x
        ).last_hidden_state  # (batch_size, n_timesteps, hidden_size)
        logits = self.head(x)  # (batch_size, n_timesteps, n_classes)

        if labels is not None:
            return logits, labels
        else:
            return logits

    def _logits_to_proba_per_step(self, logits: torch.Tensor, org_duration: int) -> torch.Tensor:
        preds = logits.sigmoid()
        return resize(preds, size=[org_duration, preds.shape[-1]], antialias=False)[:, :, [1, 2]]

    def _correct_labels(self, labels: torch.Tensor, org_duration: int) -> torch.Tensor:
        return resize(labels, size=[org_duration, labels.shape[-1]], antialias=False)[:, :, [1, 2]]
