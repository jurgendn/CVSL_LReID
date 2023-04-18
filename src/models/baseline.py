from typing import Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F

from src.models.modules.fusion_net import FusionNet
from src.models.modules.r50 import FTNet
from src.models.modules.shape_embedding import ShapeEmbedding
from config import BASIC_CONFIG

device = BASIC_CONFIG.DEVICE
class LitModule(LightningModule):

    def __init__(self,
                 shape_edge_index: torch.LongTensor,
                 shape_pose_n_features: int = 4,
                 shape_n_hidden: int = 1024,
                 shape_out_features: int = 512,
                 shape_relation_layers: List[Tuple[int]] = [(512, 256),
                                                            (256, 128)],
                 class_num: int = BASIC_CONFIG.NUM_CLASSES,
                 r50_stride: int = 2,
                 r50_pretrained_weight: str = None) -> None:
        super(LitModule, self).__init__()
        self.shape_edge_index = shape_edge_index.to(device)
        self.ft_net = FTNet(stride=r50_stride).to(device)
        self.shape_embedding = ShapeEmbedding(
            pose_n_features=shape_pose_n_features,
            n_hidden=shape_n_hidden,
            out_features=shape_out_features,
            relation_layers=shape_relation_layers).to(device)
        self.fusion = FusionNet(out_features=1024).to(device)
        self.id_classification = nn.Linear(in_features=1024,
                                           out_features=class_num).to(device)

        self.load_pretrained_r50(r50_weight_path=r50_pretrained_weight)
        for param in self.ft_net.parameters():
            param.requires_grad = False

    def load_pretrained_r50(self, r50_weight_path: str):
        self.ft_net.load_state_dict(torch.load(f=r50_weight_path), strict=False)

    def forward(self, x_image: torch.Tensor,
                x_pose_features: torch.FloatTensor,
                edge_index: torch.LongTensor) -> torch.Tensor:
        # with torch.inference_mode():
        appearance_feature = self.ft_net(x=x_image)
        pose_feature = self.shape_embedding(pose=x_pose_features,
                                            edge_index=edge_index)

        fusion_feature = self.fusion(appearance_features=appearance_feature,
                                     shape_features=pose_feature)
        return fusion_feature

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params' : self.shape_embedding.parameters(),
             'params': self.fusion.parameters()} 
        ], lr=BASIC_CONFIG.LR)
        # optimizer = optim.Adam(self.parameters(), lr=BASIC_CONFIG.LR)
        return optimizer

    def training_step(self, batch, batch_idx) -> Dict:
        (a_img, p_img, n_img), (a_pose, p_pose, n_pose), a_target = batch
        a_img, p_img, n_img, a_pose, p_pose, n_pose, a_target = a_img.to(device), p_img.to(device), n_img.to(device), a_pose.to(device), p_pose.to(device), n_pose.to(device), a_target.to(device)  
        
        a_features = self.forward(x_image=a_img,
                                  x_pose_features=a_pose,
                                  edge_index=self.shape_edge_index)
        p_features = self.forward(x_image=p_img,
                                  x_pose_features=p_pose,
                                  edge_index=self.shape_edge_index)
        n_features = self.forward(x_image=n_img,
                                  x_pose_features=n_pose,
                                  edge_index=self.shape_edge_index)

        triplet_loss = F.triplet_margin_loss(anchor=a_features,
                                             positive=p_features,
                                             negative=n_features)
        logits = self.id_classification(a_features)
        # id_loss = F.cross_entropy(logits, torch.LongTensor([a_target]))
        id_loss = F.cross_entropy(logits, a_target.long())
        loss = (id_loss + triplet_loss) / 2
        return dict(loss=loss, logits=logits, targets=a_target)

    # def on_train_batch_end(self, outputs: Dict):
    #     with torch.inference_mode():
    #         loss = outputs['loss']

    # def on_train_epoch_end(self) -> None:
    #     self.log(name="train/f1", value=1)
