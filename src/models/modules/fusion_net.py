import torch
from torch import nn


class FusionNet(nn.Module):

    def __init__(self, out_features: int = 1024) -> None:
        super(FusionNet, self).__init__()
        self.out_features = out_features

        self.appearance_net = nn.Sequential(nn.LazyLinear(out_features=1024),
                                            nn.LeakyReLU())
        self.shape_net = nn.Sequential(nn.LazyLinear(out_features=1024),
                                       nn.LeakyReLU())

        self.agg_net = nn.parameter.Parameter(
            data=torch.FloatTensor([0.5, 0.5]))

    def forward(self, appearance_features: torch.Tensor,
                shape_features: torch.Tensor) -> torch.Tensor:
        appearance_features = self.appearance_net(appearance_features)
        shape_features = self.shape_net(shape_features)

        agg_features = self.agg_net[0] * appearance_features + self.agg_net[
            1] * shape_features
        return agg_features
