from typing import Dict, List, Tuple
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from src.models.modules.fusion_net import FusionNet
from src.models.modules.r50 import FTNet, FTNet_HR, FTNet_Swin
from src.models.modules.shape_embedding import ShapeEmbedding
from losses import build_losses
from models import build_models
from utils.misc import normalize_feature
from config import BASIC_CONFIG
from torch.utils.data import DataLoader
from src.datasets.get_loader import get_train_data

conf = BASIC_CONFIG

class Baseline(LightningModule):

    def __init__(self,
                 orientation_guided: bool,
                 lr: float,
                 train_shape: bool, 
                 out_features: int) -> None:
        
        super(Baseline, self).__init__()

        shape_edge_index = torch.LongTensor(shape_edge_index)
        self.register_buffer("shape_edge_index", shape_edge_index)

        self.train_shape = train_shape

        self.orientation_guided = orientation_guided

        self.train_data, self.class_num, self.dataset_size, self.sampler = get_train_data(self.orientation_guided)
        self.losses = build_losses(conf)
        self.models = build_models(self.train_shape, conf, self.class_num, out_features)

        self.use_warm_epoch = conf.USE_WARM_EPOCH
        self.warm_epoch = conf.WARM_EPOCH
        self.warm_up = conf.WARM_UP

        self.training_step_outputs = []
        # self.validation_batch_outputs: List = []
        self.save_hyperparameters()
    
    def train_dataloader(self):
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
        #     ignored_params += list(map(id, self.id_classifier.parameters()))
        #     classifier_params = self.id_classifier.parameters()

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
            
            
        
        id_loss = self.losses['cla']
        triplet_loss = self.losses['triplet']
        pair_loss = self.losses['pair']
        clothes_loss_func = self.losses['clothes']
        cal_func = self.losses['cal']

        loss = 0.
        if self.train_shape:
            logits = self.id_classifier(a_feature)
            loss += id_loss(logits, a_id)
        else:
            loss += id_loss(a_feature, a_id)  

        # Normalize features
        if conf.NORM_FEATURE:
            a_feature = normalize_feature(a_feature)

        if conf.USE_TRIPLET_LOSS:
            loss += triplet_loss(a_feature, p_feature, n_feature)
        
        if conf.USE_PAIRWISE_LOSS:
            loss += pair_loss(a_feature, a_id)
        
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
    