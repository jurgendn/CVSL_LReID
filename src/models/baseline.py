from typing import Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F

from src.models.modules.fusion_net import FusionNet
from src.models.modules.r50 import FTNet
from src.models.modules.shape_embedding import ShapeEmbedding


class LitModule(LightningModule):

    def __init__(
        self,
        shape_semantic: torch.FloatTensor,
        shape_edge_index: torch.LongTensor,
        r50_class_num: int = 751,
        r50_droprate: float = 0.5,
        r50_stride: int = 2,
        r50_circle: bool = False,
        r50_ibn: bool = False,
        r50_linear_num: int = 512,
        shape_pose_n_features: int = 4,
        shape_sem_n_features: int = 33,
        shape_n_hidden: int = 1024,
        shape_out_features: int = 512,
        shape_relation_layers: List[Tuple[int]] = [(512, 256), (256, 128)]
    ) -> None:
        super(LitModule, self).__init__()
        self.shape_semantic = shape_semantic
        self.shape_edge_index = shape_edge_index
        self.ft_net = FTNet(class_num=r50_class_num,
                            droprate=r50_droprate,
                            stride=r50_stride,
                            circle=r50_circle,
                            ibn=r50_ibn,
                            linear_num=r50_linear_num)
        self.shape_embedding = ShapeEmbedding(
            pose_n_features=shape_pose_n_features,
            sem_n_features=shape_sem_n_features,
            n_hidden=shape_n_hidden,
            out_features=shape_out_features,
            relation_layers=shape_relation_layers)
        self.fusion = FusionNet(out_features=1024)
        self.id_classification = nn.Linear(in_features=1024, out_features=751)

    def forward(self, x_image: torch.Tensor,
                x_pose_features: torch.FloatTensor,
                x_semantic_features: torch.FloatTensor,
                edge_index: torch.LongTensor) -> torch.Tensor:
        appearance_feature = self.ft_net(x=x_image)
        pose_feature = self.shape_embedding(pose=x_pose_features,
                                            semantic=x_semantic_features,
                                            edge_index=edge_index)

        fusion_feature = self.fusion(appearance_features=appearance_feature,
                                     shape_features=pose_feature)
        return fusion_feature

    def configure_optimizers(self, lr: float):
        optimizer = optim.Adam(params=self.parameters(), lr=lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        (a_img, p_img, n_img), (a_pose, p_pose, n_pose), a_id = batch
        a_features = self.forward(x_image=a_img,
                                  x_pose_features=a_pose,
                                  x_semantic_features=self.shape_semantic,
                                  edge_index=self.shape_edge_index)
        p_features = self.forward(x_image=p_img,
                                  x_pose_features=p_pose,
                                  x_semantic_features=self.shape_semantic,
                                  edge_index=self.shape_edge_index)
        n_features = self.forward(x_image=n_img,
                                  x_pose_features=n_pose,
                                  x_semantic_features=self.shape_semantic,
                                  edge_index=self.shape_edge_index)

        triplet_loss = F.triplet_margin_loss(anchor=a_features,
                                             positive=p_features,
                                             negative=n_features)
        logits = self.id_classification(a_features)
        id_loss = F.cross_entropy(logits, a_id)
        return {'loss': id_loss + triplet_loss, 'extras': []}

    def on_train_batch_end(self, outputs: Dict):
        return outputs

    def on_train_epoch_end(self) -> None:
        self.log(name="train/f1", value=1)
