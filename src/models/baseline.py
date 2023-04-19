from typing import Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F

from src.models.modules.fusion_net import FusionNet
from src.models.modules.r50 import FTNet
from src.models.modules.shape_embedding import ShapeEmbedding


class LitModule(LightningModule):

    def __init__(self,
                 shape_edge_index: torch.LongTensor,
                 shape_pose_n_features: int = 4,
                 shape_n_hidden: int = 1024,
                 shape_out_features: int = 512,
                 shape_relation_layers: List[Tuple[int]] = [(512, 256),
                                                            (256, 128)],
                 class_num: int = 751,
                 r50_stride: int = 2,
                 r50_pretrained_weight: str = None,
                 lr: float = 1e-4) -> None:
        super(LitModule, self).__init__()
        shape_edge_index = torch.LongTensor(shape_edge_index)
        self.register_buffer("shape_edge_index", shape_edge_index)
        self.ft_net = FTNet(stride=r50_stride)
        self.shape_embedding = ShapeEmbedding(
            pose_n_features=shape_pose_n_features,
            n_hidden=shape_n_hidden,
            out_features=shape_out_features,
            relation_layers=shape_relation_layers)
        self.fusion = FusionNet(out_features=1024)
        self.id_classification = nn.Linear(in_features=1024,
                                           out_features=class_num)

        self.load_pretrained_r50(r50_weight_path=r50_pretrained_weight)

        self.train_batch_outputs: List = []
        self.validation_batch_outputs: List = []

    def load_pretrained_r50(self, r50_weight_path: str):
        self.ft_net.load_state_dict(torch.load(f=r50_weight_path))

    def forward(self, x_image: torch.Tensor,
                x_pose_features: torch.FloatTensor,
                edge_index: torch.LongTensor) -> torch.Tensor:
        appearance_feature = self.ft_net(x=x_image)
        pose_feature = self.shape_embedding(pose=x_pose_features,
                                            edge_index=edge_index)

        fusion_feature = self.fusion(appearance_features=appearance_feature,
                                     shape_features=pose_feature)
        return fusion_feature

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                 step_size=30,
                                                 gamma=0.1)
        return optimizer, lr_scheduler

    def training_step(self, batch, batch_idx) -> Dict:
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
        loss = (id_loss + triplet_loss) / 2
        self.train_batch_outputs.append(loss)
        return dict(loss=loss, logits=logits, targets=a_id)

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        outputs = self.train_batch_outputs
        self.log(name="some metrics", value=outputs.mean())
        self.train_batch_outputs = []
