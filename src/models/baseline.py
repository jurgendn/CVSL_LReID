from typing import Dict, List, Tuple
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from src.models.modules.fusion_net import FusionNet
from src.models.modules.r50 import FTNet, FTNet_HR, FTNet_Swin, weights_init_classifier
from src.models.modules.shape_embedding import ShapeEmbedding
from src.losses.triplet_loss import TripletLoss
from src.losses.cross_entropy_loss_with_label_smooth import CrossEntropyWithLabelSmooth
from src.losses.circle_loss import CircleLoss, PairwiseCircleLoss
from pytorch_metric_learning import losses
from utils.misc import normalize_feature
from torch.nn import init
from config import BASIC_CONFIG
from src.datasets.base_dataset import TestDataset, TrainDataset, TrainDatasetOrientation
from torch.utils.data import DataLoader
from src.datasets.samplers import RandomIdentitySampler


conf = BASIC_CONFIG

class Baseline(LightningModule):

    def __init__(self,
                 orientation_guided: bool,
                 r50_stride: int,
                 r50_pretrained_weight: str,
                 lr: float,
                 train_shape: bool, 
                 shape_edge_index: torch.LongTensor,
                 shape_pose_n_features: int,
                 shape_n_hidden: int,
                 shape_out_features: int,
                 shape_relation_layers: List[Tuple[int]],
                 out_features: int) -> None:
        
        super(Baseline, self).__init__()

        shape_edge_index = torch.LongTensor(shape_edge_index)
        self.register_buffer("shape_edge_index", shape_edge_index)

        self.train_shape = train_shape

        self.orientation_guided = orientation_guided

        if self.orientation_guided:
            self.train_data = TrainDatasetOrientation(conf.TRAIN_JSON_PATH, conf.TRAIN_TRANSFORM)
        else:
            self.train_data = TrainDataset(conf.TRAIN_JSON_PATH, conf.TRAIN_TRANSFORM)
        self.class_num = self.train_data.num_classes
        self.dataset_size = len(self.train_data)
        self.sampler = RandomIdentitySampler(self.train_data, num_instances=8)

        if self.train_shape: 
            self.return_f = True 
        else: self.return_f = False 

        if conf.USE_RESTNET:
            self.ft_net = FTNet(stride=r50_stride, class_num=self.class_num, return_f=self.return_f)
        elif conf.USE_HRNET:
            self.ft_net = FTNet_HR(class_num=self.class_num, return_f=self.return_f)
        elif conf.USE_SWIN:
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
            self.fusion = FusionNet(out_features=out_features)

            self.bn = nn.BatchNorm1d(out_features)
            init.normal_(self.bn.weight.data, 1.0, 0.02)
            init.constant_(self.bn.bias.data, 0.0),

            self.id_classification = nn.Linear(in_features=out_features, out_features=self.class_num)
        
            self.id_classification.apply(weights_init_classifier)
        
        if not conf.TRAIN_FROM_SCRATCH:
            self.load_pretrained_r50(r50_weight_path=r50_pretrained_weight)

        # self.ft_net.requires_grad_(False)
        self.use_warm_epoch = conf.USE_WARM_EPOCH
        self.warm_epoch = conf.WARM_EPOCH
        self.warm_up = conf.WARM_UP

        self.training_step_outputs = []
        # self.validation_batch_outputs: List = []
        self.save_hyperparameters()

    def load_pretrained_r50(self, r50_weight_path: str):
        self.ft_net.load_state_dict(torch.load(f=r50_weight_path), strict=True)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if conf.SAMPLER:
            train_loader = DataLoader(self.train_data, batch_size=conf.BATCH_SIZE, shuffle=False, num_workers=conf.NUM_WORKER, pin_memory=conf.PIN_MEMORY, sampler=self.sampler)
        else:
            train_loader = DataLoader(self.train_data, batch_size=conf.BATCH_SIZE, shuffle=True, num_workers=conf.NUM_WORKER, pin_memory=conf.PIN_MEMORY)
        return train_loader
    
    def on_epoch_start(self):
        self.sampler.set_epoch(self.current_epoch)

    def forward(self, x_image: torch.Tensor,
                x_pose_features: torch.FloatTensor,
                edge_index: torch.LongTensor) -> torch.Tensor:
        
        appearance_feature = self.ft_net(x=x_image)

        if self.train_shape:
            pose_feature = self.shape_embedding(pose=x_pose_features,
                                                edge_index=edge_index)

            fusion_feature = self.fusion(appearance_features=appearance_feature,
                                        shape_features=pose_feature)
            fusion_feature = self.bn(fusion_feature)
            return fusion_feature
        else:            
            return appearance_feature
    def configure_optimizers(self):
        if conf.OPTIMIZER == 'adam':
            optim_name = optim.Adam
            optimizer = optim_name(params=self.parameters(), lr=self.hparams.lr, weight_decay=conf.WEIGHT_DECAY)
        elif conf.OPTIMIZER == 'sgd':
            optim_name = optim.SGD
            optimizer = optim_name(params=self.parameters(), lr=self.hparams.lr, weight_decay=conf.WEIGHT_DECAY, momentum=0.9, nesterov=True)
        
        # ignored_params = list(map(id, self.ft_net.classifier.parameters()))
        # classifier_ft_net_params = self.ft_net.classifier.parameters()
        # if self.train_shape:
        #     ignored_params += list(map(id, self.id_classification.parameters()))
        #     classifier_params = self.id_classification.parameters()

        # base_params = filter(lambda p: id(p) not in ignored_params, self.parameters())
        
        # if self.train_shape:
        #     optimizer = optim_name([
        #                 {'params': base_params, 'lr': 0.1*self.hparams.lr},
        #                 {'params': classifier_ft_net_params, 'lr': self.hparams.lr},
        #                 {'params': classifier_params, 'lr': self.hparams.lr}
        #             ], weight_decay=conf.WEIGHT_DECAY, momentum=0.9, nesterov=True)
        # else:
        #     optimizer = optim_name([
        #                 {'params': base_params, 'lr': 0.1*self.hparams.lr},
        #                 {'params': classifier_ft_net_params, 'lr': self.hparams.lr},
        #             ], weight_decay=conf.WEIGHT_DECAY, momentum=0.9, nesterov=True)
        # optimizer = FusedSGD(self.parameters(), lr=self.hparams.lr)
        if conf.USE_REDUCE_LR:
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode='min', patience=10,  
            )
            return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "epoch_loss"}}
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=20,
                                                    gamma=0.1)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        

    def training_step(self, batch, batch_idx) -> Dict:
        if self.orientation_guided:
            (a_img, p_img, n_img), (a_pose, p_pose, n_pose), a_id = batch
            p_feature = self.forward(x_image=p_img,
                                    x_pose_features=p_pose,
                                    edge_index=self.shape_edge_index)
            n_feature = self.forward(x_image=n_img,
                                    x_pose_features=n_pose,
                                    edge_index=self.shape_edge_index)
            if conf.NORM_FEATURE:
                p_feature = normalize_feature(p_feature)
                n_feature = normalize_feature(n_feature)
        
        else:
            a_img, a_pose, a_cloth_id, a_id = batch

        now_batch_size, _, _, _ = a_img.shape

        a_feature = self.forward(x_image=a_img,
                                 x_pose_features=a_pose,
                                 edge_index=self.shape_edge_index)
            
            
        
        if conf.USE_CE_LOSS:
            id_loss_func = nn.CrossEntropyLoss()
        if conf.USE_CELABELSMOOTH_LOSS:
            id_loss_func = CrossEntropyWithLabelSmooth()

        if self.train_shape:
            logits = self.id_classification(a_feature)
            id_loss = id_loss_func(logits, a_id)
        else:
            id_loss = id_loss_func(a_feature, a_id)  

        # Normalize features
        if conf.NORM_FEATURE:
            a_feature = normalize_feature(a_feature)

        if conf.USE_TRIPLET_LOSS:
            triplet_margin_loss = nn.TripletMarginLoss(margin=0.3,)
            triplet_loss = triplet_margin_loss(a_feature, p_feature, n_feature)
        
        if conf.USE_TRIPLETPAIRWISE_LOSS:
            triplet_margin_loss = TripletLoss(margin=0.3, distance='cosine')
            triplet_loss = triplet_margin_loss(a_feature, a_id)

        # if conf.USE_CIRCLE_TRIPLET_LOSS:
        #     circle_loss_triplet = PairwiseCircleLoss(scale=16, margin=0.3)
        #     triplet_loss = circle_loss_triplet(p_feature, n_feature, a_feature)


        loss = id_loss + triplet_loss

        if conf.USE_CIRCLE_LOSS:
            feature = a_feature
            circle_loss = CircleLoss(scale=16, margin=0.3)
            a_fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
            feature = feature.div(a_fnorm.expand_as(feature))
            # loss += circle_loss(*convert_label_to_similarity(feature, a_id))/now_batch_size
            loss += circle_loss(feature, a_id)
        
        if self.use_warm_epoch:
            #Warm Up
            warm_iteration = round(self.dataset_size/conf.BATCH_SIZE)*self.warm_epoch # first 5 epoch
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
        if conf.USE_RESTNET:
            self.ft_net = FTNet(stride=r50_stride, return_f=True)
        elif conf.USE_HRNET:
            self.ft_net = FTNet_HR(return_f=True)
        elif conf.USE_SWIN:
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
    