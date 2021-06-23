"""Resnet50 model"""

# internal imports
from configs import config

# external imports
from torchvision import models
from torch import nn


class Resnet50(nn.Module):
    """Resnet50 model"""

    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=config.PRETRAINED)
        self.model_name = 'resnet50'

        if not config.REQUIRES_GRAD:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model._fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, config.model_fc['layer_1']),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(config.model_fc['layer_1'], config.model_fc['layer_2'])
        )

    def forward(self, x):
        x = self.model(x)

        return x
