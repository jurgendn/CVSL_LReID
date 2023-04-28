"""
These modules are built based on this paper
[1]X. Qian et al., Long-Term Cloth-Changing Person Re-identification

In this module, the Relation Network module use
Graph Convolutional Network instead of Convolutional Layer
"""

from typing import List, Tuple

import torch_geometric.nn as gnn
from torch import Tensor, nn
from torch.nn import functional as F


class PositionEmbedding(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super(PositionEmbedding, self).__init__()
        self.position_embedding = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            # nn.Linear(in_features=in_features, out_features=128),
            # nn.Linear(in_features=128, out_features=512),
            # nn.Linear(in_features=512, out_features=out_features),
            # nn.Linear(in_features=1024, out_features=out_features)
        )
        
        

    def forward(self, pose: Tensor) -> Tensor:
        x = self.position_embedding(pose)
        return x


# class SemanticEmbedding(nn.Module):

#     def __init__(self, in_features: int, out_features: int) -> None:
#         super(SemanticEmbedding, self).__init__()
#         self.semantic_embedding = nn.Linear(in_features=in_features,
#                                             out_features=out_features)

#     def forward(self, sem: Tensor) -> Tensor:
#         y = self.semantic_embedding(sem)
#         return y


class RefineNetwork(nn.Module):

    def __init__(self, pose_n_features: int, n_hidden: int,
                 out_features: int) -> None:
        super(RefineNetwork, self).__init__()
        self.position_embedding = PositionEmbedding(
            in_features=pose_n_features, out_features=n_hidden)
        self.fc = nn.Linear(in_features=n_hidden, out_features=out_features)

    def forward(self, p: Tensor):
        x = self.position_embedding(p)
        x = F.leaky_relu(x)
        y = self.fc(x)
        return y


class RelationNetwork(nn.Module):

    def __init__(self, layers: List[int]) -> None:
        super(RelationNetwork, self).__init__()
        self.__num_modules = len(layers)
        for i, (in_channel, out_channel) in enumerate(layers):
            setattr(
                self, f"gcn_{i+1}",
                gnn.GCNConv(in_channels=in_channel, out_channels=out_channel)
            )

    def forward(self, x: Tensor, a: Tensor):
        """
        forward

        Parameters
        ----------
        x : Tensor
            Node features, 2D Tensor
        a : Tensor
            Edge index, in format
            [[0, 1, 1, 2, 1, 9], [1, 0, 2, 1, 8, 1]]

        Returns
        -------
        Tensor
            Node embedding, (n, out_channels)
        """
        for i in range(self.__num_modules):
            module = getattr(self, f"gcn_{i+1}")
            x = module(x, a)
            x = F.leaky_relu(x)
        return x

class ShapeEmbedding(nn.Module):

    def __init__(self, pose_n_features: int, n_hidden: int, out_features: int,
                 relation_layers: List[Tuple[int]]) -> None:
        super(ShapeEmbedding, self).__init__()
        assert out_features == relation_layers[0][0]
        self.n_dim = relation_layers[-1][-1]
        self.refine_net = RefineNetwork(pose_n_features=pose_n_features,
                                        n_hidden=n_hidden,
                                        out_features=out_features)
        self.relation_net = RelationNetwork(layers=relation_layers)
        self.graph_pooling = gnn.MeanAggregation()

    def forward(self, pose: Tensor, edge_index: Tensor) -> Tensor:
        pose_features = self.refine_net(p=pose)
        relation_features = self.relation_net(x=pose_features, a=edge_index)
        graph_representation = self.graph_pooling(x=relation_features)
        return graph_representation.reshape(-1, self.n_dim)


if __name__ == "__main__":
    net = ShapeEmbedding(pose_n_features=4,
                         n_hidden=1024,
                         out_features=128,
                         relation_layers=[[128, 256], [256, 512]])
