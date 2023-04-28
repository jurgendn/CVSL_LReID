import torch
from torch import nn


class AggregationNet(nn.Module):

    def __init__(self) -> None:
        super(AggregationNet, self).__init__()
        self.theta = nn.parameter.Parameter(torch.randn(1, 2))
        nn.init.kaiming_uniform_(self.theta)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        theta = torch.exp(self.theta) / torch.exp(self.theta).sum()
        out = theta[0][0] * x + theta[0][1] * y
        return out


class FusionNet(nn.Module):

    def __init__(self, out_features: int = 1024) -> None:
        super(FusionNet, self).__init__()
        self.out_features = out_features

        self.appearance_net = nn.Sequential(nn.Linear(in_features=512, out_features=1024),
                                            nn.LeakyReLU())
        self.shape_net = nn.Sequential(nn.Linear(in_features=512, out_features=1024),
                                       nn.LeakyReLU())

        self.theta = AggregationNet()

    def forward(self, appearance_features: torch.Tensor,
                shape_features: torch.Tensor) -> torch.Tensor:
        appearance = self.appearance_net(appearance_features)
        shape = self.shape_net(shape_features)
        agg_features = self.theta(x=appearance, y=shape)
        return agg_features
