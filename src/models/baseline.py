from typing import Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from src.models.modules.fusion_net import FusionNet
from src.models.modules.r50 import FTNet
from src.models.modules.shape_embedding import ShapeEmbedding
from config import BASIC_CONFIG
from src.losses.circle_loss import CircleLoss, convert_label_to_similarity
from pytorch_metric_learning import losses
# from apex.fp16_utils import *
# from apex import amp
# from apex.optimizers import FusedSGD



class Baseline(LightningModule):

    def __init__(self,
                 class_num: int,
                 r50_stride: int,
                 r50_pretrained_weight: str,
                 lr: float,
                 dataset_size: int,
                 shape_edge_index: torch.LongTensor,
                 shape_pose_n_features: int = 4,
                 shape_n_hidden: int = 1024,
                 shape_out_features: int = 512,
                 shape_relation_layers: List[Tuple[int]] = [(512, 256),
                                                            (256, 128)]) -> None:
        
        super(Baseline, self).__init__()

        shape_edge_index = torch.LongTensor(shape_edge_index)
        self.register_buffer("shape_edge_index", shape_edge_index)

        self.class_num = class_num
        self.dataset_size = dataset_size
        self.fp16 = BASIC_CONFIG.FP16

        self.ft_net = FTNet(stride=r50_stride)
        self.shape_embedding = ShapeEmbedding(
            pose_n_features=shape_pose_n_features,
            n_hidden=shape_n_hidden,
            out_features=shape_out_features,
            relation_layers=shape_relation_layers)
        self.fusion = FusionNet(out_features=1024)
        self.id_classification = nn.Linear(in_features=1024,
                                           out_features=self.class_num)

        self.load_pretrained_r50(r50_weight_path=r50_pretrained_weight)

        # for param in self.ft_net.parameters
        # self.ft_net.requires_grad_(False)
        self.warm_epoch = BASIC_CONFIG.WARM_EPOCH
        self.warm_up = BASIC_CONFIG.WARM_UP

        self.training_step_outputs = []
        # self.validation_batch_outputs: List = []
        self.save_hyperparameters()

    def load_pretrained_r50(self, r50_weight_path: str):
        self.ft_net.load_state_dict(torch.load(f=r50_weight_path),
                                    strict=False)

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
        # optimizer = FusedSGD(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                 step_size=10,
                                                 gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx) -> Dict:
        (a_img, p_img, n_img), (a_pose, p_pose, n_pose), a_id = batch
        now_batch_size, _, _, _ = a_img.shape
        # self = self.half()

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
                                             negative=n_features, margin=0.3)

        


        logits = self.id_classification(a_features)
        loss = F.cross_entropy(logits, a_id)

        loss += triplet_loss
        # normalize feature
        a_fnorm = torch.norm(a_features, p=2, dim=1, keepdim=True)
        a_features = a_features.div(a_fnorm.expand_as(a_features))

        # Circle Loss
        circle_loss = CircleLoss(m=0.25, gamma=64)
        loss += circle_loss(*convert_label_to_similarity(a_features, a_id))/now_batch_size

        # ArcFace Loss 
        # arcface = losses.ArcFaceLoss(num_classes=self.class_num, embedding_size=1024)
        # arcface_loss = arcface(a_features, a_id)/now_batch_size
        # loss += arcface_loss
        

        # Warm Up
        warm_iteration = round(self.dataset_size/BASIC_CONFIG.BATCH_SIZE)*self.warm_epoch # first 5 epoch
        if self.current_epoch < self.warm_epoch:
            warm_up = min(1.0, self.warm_up + 0.9 / warm_iteration)
            loss = loss * warm_up

        self.training_step_outputs.append(loss)
        self.log('train_loss', loss)
        # return loss 
        return dict(loss=loss, logits=logits, targets=a_id)

    def on_train_epoch_end(self):
        # compute and log average training loss
        avg_loss = torch.stack(self.training_step_outputs).mean()
        self.log('avg_train_loss', avg_loss)

class InferenceBaseline(LightningModule):

    def __init__(self,
                 shape_edge_index: torch.LongTensor,
                 shape_pose_n_features: int,
                 shape_n_hidden: int,
                 shape_out_features: int ,
                 shape_relation_layers: List,
                 r50_stride: int) -> None:
        super(InferenceBaseline, self).__init__()
        shape_edge_index = torch.LongTensor(shape_edge_index)
        self.register_buffer("shape_edge_index", shape_edge_index)
        self.ft_net = FTNet(stride=r50_stride)
        self.shape_embedding = ShapeEmbedding(
            pose_n_features=shape_pose_n_features,
            n_hidden=shape_n_hidden,
            out_features=shape_out_features,
            relation_layers=shape_relation_layers)
        self.fusion = FusionNet(out_features=1024)

    def forward(self, x_image: torch.Tensor,
                x_pose_features: torch.FloatTensor,
                edge_index: torch.LongTensor) -> torch.Tensor:
        appearance_feature = self.ft_net(x=x_image)
        pose_feature = self.shape_embedding(pose=x_pose_features,
                                            edge_index=edge_index)

        fusion_feature = self.fusion(appearance_features=appearance_feature,
                                     shape_features=pose_feature)
        return fusion_feature


# def contrastive_orientation_guided_loss(anchor, positive, negative):
    