import timm
import torch.nn as nn


class DogsVsCatsModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=True,
            num_classes=1,
        )

    def forward(self, x):
        return self.model(x)