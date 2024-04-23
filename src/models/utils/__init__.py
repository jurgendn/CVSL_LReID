import torch
from torch import nn


def get_num_channels(backbone: nn.Module, in_channels: int, output_name: str) -> int:
    dummies_input = torch.randn(1, in_channels, 224, 224)
    with torch.no_grad():
        features = backbone(dummies_input)[output_name]
    return features.shape[1]
