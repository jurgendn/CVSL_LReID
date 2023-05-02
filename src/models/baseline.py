from typing import Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from src.models.modules.fusion_net import FusionNet
from src.models.modules.r50 import FTNet, FTNet_HR, FTNet_Swin, weights_init_classifier
from src.models.modules.shape_embedding import ShapeEmbedding
from config import BASIC_CONFIG
from src.losses.circle_loss import CircleLoss, CircleLossTriplet, convert_label_to_similarity
from pytorch_metric_learning import losses
from utils.misc import normalize_feature

class Baseline(LightningModule):

    def __init__(self,
                 class_num: int,
                 r50_stride: int,
                 r50_pretrained_weight: str,
                 lr: float,
                 train_shape: bool, 
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
        self.train_shape = train_shape

        if self.train_shape: 
            self.return_f = True 
        else: self.return_f = False 

        if BASIC_CONFIG.USE_RESTNET:
            self.ft_net = FTNet(stride=r50_stride, class_num=self.class_num, return_f=self.return_f)
        elif BASIC_CONFIG.USE_HRNET:
            self.ft_net = FTNet_HR(class_num=self.class_num, return_f=self.return_f)
        elif BASIC_CONFIG.USE_SWIN:
            self.ft_net = FTNet_Swin(class_num=self.class_num, return_f=self.return_f)

        # can try ClassBlock for id_classification
        if self.train_shape:
            self.shape_embedding = ShapeEmbedding(
                pose_n_features=shape_pose_n_features,
                n_hidden=shape_n_hidden,
                out_features=shape_out_features,
                relation_layers=shape_relation_layers)
            
            """
            need to change the input shape of appearance net and shape net 
            if change the relation layers
            """
            self.fusion = FusionNet(out_features=512)

            self.id_classification = nn.Linear(in_features=512, out_features=self.class_num)
            self.id_classification.apply(weights_init_classifier)
        
        if not BASIC_CONFIG.TRAIN_FROM_SCRATCH:
            self.load_pretrained_r50(r50_weight_path=r50_pretrained_weight)

        # self.ft_net.requires_grad_(False)
        self.use_warm_epoch = BASIC_CONFIG.USE_WARM_EPOCH
        self.warm_epoch = BASIC_CONFIG.WARM_EPOCH
        self.warm_up = BASIC_CONFIG.WARM_UP

        self.use_circle = BASIC_CONFIG.USE_CIRCLE_LOSS
        self.use_circle_appearance = BASIC_CONFIG.USE_CIRCLE_LOSS_APP

        self.training_step_outputs = []
        # self.validation_batch_outputs: List = []
        self.save_hyperparameters()

    def load_pretrained_r50(self, r50_weight_path: str):
        self.ft_net.load_state_dict(torch.load(f=r50_weight_path), strict=True)

    def forward(self, x_image: torch.Tensor,
                x_pose_features: torch.FloatTensor,
                edge_index: torch.LongTensor) -> torch.Tensor:
        
        appearance_feature = self.ft_net(x=x_image)

        if self.train_shape:
            pose_feature = self.shape_embedding(pose=x_pose_features,
                                                edge_index=edge_index)

            fusion_feature = self.fusion(appearance_features=appearance_feature,
                                        shape_features=pose_feature)
            return fusion_feature
        else:            
            return appearance_feature
    def configure_optimizers(self):
        
        optim_name = optim.SGD
        
        ignored_params = list(map(id, self.ft_net.classifier.parameters()))
        classifier_ft_net_params = self.ft_net.classifier.parameters()
        if self.train_shape:
            ignored_params += list(map(id, self.id_classification.parameters()))
            classifier_params = self.id_classification.parameters()

        base_params = filter(lambda p: id(p) not in ignored_params, self.parameters())
        
        if self.train_shape:
            optimizer = optim_name([
                        {'params': base_params, 'lr': 0.1*self.hparams.lr},
                        {'params': classifier_ft_net_params, 'lr': self.hparams.lr},
                        {'params': classifier_params, 'lr': self.hparams.lr}
                    ], weight_decay=BASIC_CONFIG.WEIGHT_DECAY, momentum=0.9, nesterov=True)
        else:
            optimizer = optim_name([
                        {'params': base_params, 'lr': 0.1*self.hparams.lr},
                        {'params': classifier_ft_net_params, 'lr': self.hparams.lr},
                    ], weight_decay=BASIC_CONFIG.WEIGHT_DECAY, momentum=0.9, nesterov=True)
        # optimizer = FusedSGD(self.parameters(), lr=self.hparams.lr)
        if BASIC_CONFIG.USE_REDUCE_LR:
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode='min', patience=10,  
            )
            return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "epoch_loss"}}
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                 #step_size=BASIC_CONFIG.EPOCHS * 2 // 3,
                                                 step_size=23,
                                                 gamma=0.1,)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        

    def training_step(self, batch, batch_idx) -> Dict:
        (a_img, p_img, n_img), (a_pose, p_pose, n_pose), a_id = batch
        now_batch_size, _, _, _ = a_img.shape

        a_feature = self.forward(x_image=a_img,
                                  x_pose_features=a_pose,
                                  edge_index=self.shape_edge_index)
        p_feature = self.forward(x_image=p_img,
                                  x_pose_features=p_pose,
                                  edge_index=self.shape_edge_index)
        n_feature = self.forward(x_image=n_img,
                                  x_pose_features=n_pose,
                                  edge_index=self.shape_edge_index)
        
        if self.train_shape:
            logits = self.id_classification(a_feature)
            id_loss = F.cross_entropy(logits, a_id)
        else:
            id_loss = F.cross_entropy(a_feature, a_id)  

        # Normalize features
        a_features = normalize_feature(a_feature)
        p_features = normalize_feature(p_feature)
        n_features = normalize_feature(n_feature)
        
        if BASIC_CONFIG.USE_TRIPLET_LOSS:
            triplet_margin_loss = nn.TripletMarginLoss(margin=0.5)
            triplet_loss = triplet_margin_loss(a_features, p_features, n_features)

        if BASIC_CONFIG.USE_CIRCLE_TRIPLET_LOSS:
            circle_loss_triplet = CircleLossTriplet(scale=64, margin=0.25)
            triplet_loss = circle_loss_triplet(p_features, n_features, a_features)

        loss = id_loss + triplet_loss

        if self.use_circle:
            # normalize feature
            if self.use_circle_appearance: 
                feature = self.ft_net(a_img)
            else:
                feature = a_feature
            circle_loss = CircleLoss(m=0.25, gamma=64)
            a_fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
            feature = feature.div(a_fnorm.expand_as(feature))
            loss += circle_loss(*convert_label_to_similarity(feature, a_id))/now_batch_size
        
        if self.use_warm_epoch:
            #Warm Up
            warm_iteration = round(self.dataset_size/BASIC_CONFIG.BATCH_SIZE)*self.warm_epoch # first 5 epoch
            if self.current_epoch < self.warm_epoch:
                warm_up = min(1.0, self.warm_up + 0.9 / warm_iteration)
                loss = loss * warm_up

        self.training_step_outputs.append(loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # return loss 
        return loss 

    # def on_fit_start(self) -> None:
    #     self.has_warmed_up = False
    
    def on_train_epoch_end(self):
        epoch_loss = sum(self.training_step_outputs) / len(self.training_step_outputs)
        self.log('epoch_loss', epoch_loss)
        self.training_step_outputs.clear()

class InferenceBaseline(LightningModule):

    def __init__(self,
                 shape_edge_index: torch.LongTensor,
                 shape_pose_n_features: int,
                 shape_n_hidden: int,
                 shape_out_features: int ,
                 shape_relation_layers: List,
                 r50_stride: int, 
                 with_pose: bool) -> None:
        
        super(InferenceBaseline, self).__init__()
        shape_edge_index = torch.LongTensor(shape_edge_index)
        self.register_buffer("shape_edge_index", shape_edge_index)
        if BASIC_CONFIG.USE_RESTNET:
            self.ft_net = FTNet(stride=r50_stride, return_f=True)
        elif BASIC_CONFIG.USE_HRNET:
            self.ft_net = FTNet_HR(return_f=True)
        elif BASIC_CONFIG.USE_SWIN:
            self.ft_net = FTNet_Swin(return_f=True)

        self.test_with_pose = with_pose

        self.shape_embedding = ShapeEmbedding(
            pose_n_features=shape_pose_n_features,
            n_hidden=shape_n_hidden,
            out_features=shape_out_features,
            relation_layers=shape_relation_layers)
        self.fusion = FusionNet(out_features=512)

    def forward(self, x_image: torch.Tensor,
                x_pose_features: torch.FloatTensor,
                edge_index: torch.LongTensor) -> torch.Tensor:
        
        appearance_feature = self.ft_net(x=x_image)
        if self.test_with_pose:
            pose_feature = self.shape_embedding(pose=x_pose_features,
                                                edge_index=edge_index)

            fusion_feature = self.fusion(appearance_features=appearance_feature,
                                        shape_features=pose_feature)
        
            return fusion_feature
        else:
            return appearance_feature


# def contrastive_orientation_guided_loss(anchor, positive, negative):
    