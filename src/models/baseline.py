from typing import Dict, List, Tuple
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from src.models.modules.fusion_net import FusionNet
from src.models.modules.r50 import FTNet, FTNet_HR, FTNet_Swin
from src.models.modules.shape_embedding import ShapeEmbedding
from src.losses import build_losses
from src.models import build_models
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
                 out_features: int,
                 shape_edge_index) -> None:
        
        super(Baseline, self).__init__()

        self.shape_edge_index = torch.LongTensor(shape_edge_index)
        # self.register_buffer("shape_edge_index", self.shape_edge_index)

        self.train_shape = train_shape

        self.orientation_guided = orientation_guided

        self.train_data, self.sampler = get_train_data(self.orientation_guided)
        self.class_num = self.train_data.num_classes
        self.num_clothes = self.train_data.num_clothes
        self.pid2clothes = torch.from_numpy(self.train_data.pid2clothes)
        self.dataset_size = len(self.train_data)

        self.losses = build_losses()
        self.id_loss = self.losses['cla']
        if conf.USE_TRIPLET_LOSS:
            self.triplet_loss = self.losses['triplet']
        if conf.USE_PAIRWISE_LOSS:
            self.pair_loss = self.losses['pair']

        self.models = build_models(self.train_shape, self.class_num, self.num_clothes, out_features)
        self.ft_net = self.models['cnn']
        self.shape_embedding = self.models['shape']
        self.fusion = self.models['fusion']
        self.id_classifier = self.models['id_clf']

        if conf.USE_CLOTHES_LOSS:
            self.clothes_loss = self.losses['clothes']
            self.cal = self.losses['cal']
            self.clothes_classifier = self.models['clothes_clf']

        self.use_warm_epoch = conf.USE_WARM_EPOCH
        self.warm_epoch = conf.WARM_EPOCH
        self.warm_up = conf.WARM_UP

        self.automatic_optimization = False
        
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
        parameters_cc = list(map(id, self.clothes_classifier.parameters()))
        parameters = filter(lambda p: id(p) not in parameters_cc, self.parameters())
        
        if conf.OPTIMIZER == 'adam':
            optim_name = optim.Adam
            optimizer = optim_name(params=parameters, lr=self.hparams.lr, weight_decay=conf.WEIGHT_DECAY)
            optimizer_cc = optim_name(params=self.clothes_classifier.parameters(),\
                        lr=self.hparams.lr, weight_decay=conf.WEIGHT_DECAY)
        elif conf.OPTIMIZER == 'sgd':
            optim_name = optim.SGD
            optimizer = optim_name(params=parameters, lr=self.hparams.lr, \
                        weight_decay=conf.WEIGHT_DECAY, momentum=0.9, nesterov=True)
            optimizer_cc = optim_name(params=self.clothes_classifier.parameters(), lr=self.hparams.lr, \
                        weight_decay=conf.WEIGHT_DECAY, momentum=0.9, nesterov=True)
            
        # if conf.USE_REDUCE_LR:
        #     lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer=self.optimizer, mode='min', patience=10,  
        #     )
        #     return {"optimizer": [self.optimizer, self.optimizer_cc], 
        #         "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "epoch_loss"}}
        # else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=20,
                                                gamma=0.1)
        # return {"optimizer": [self.optimizer, self.optimizer_cc], "lr_scheduler": lr_scheduler}
        return [optimizer, optimizer_cc], [lr_scheduler]
    

    def training_step(self, batch, batch_idx) -> Dict:
        optimizer, optimizer_cc = self.optimizers()

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
            a_img, a_pose, a_id, a_cloth_id = batch

        a_feature = self.forward(x_image=a_img,
                                 x_pose_features=a_pose,
                                 edge_index=self.shape_edge_index)
            
        pos_mask = self.pid2clothes[a_id].float()

        # Normalize features
        if conf.NORM_FEATURE:
            a_feature = normalize_feature(a_feature)

        loss = 0.
        if conf.USE_CLOTHES_LOSS:
            pred_clothes = self.clothes_classifier(a_feature)
            clothes_loss = self.clothes_loss(pred_clothes, a_cloth_id)
            if self.current_epoch >= conf.START_EPOCH_CC:
                optimizer_cc.zero_grad()
                self.manual_backward(clothes_loss)
                optimizer_cc.step()

        new_pred_clothes = self.clothes_classifier(a_feature)

        if self.train_shape:
            logits = self.id_classifier(a_feature)
            id_loss = self.id_loss(logits, a_id)
        else:
            id_loss = self.id_loss(a_feature, a_id)  

        loss += id_loss
        if conf.USE_TRIPLET_LOSS:
            triplet_loss = self.triplet_loss(a_feature, p_feature, n_feature)
            loss += triplet_loss
        
        if conf.USE_PAIRWISE_LOSS:
            pair_loss = self.pair_loss(a_feature, a_id)
            loss += pair_loss * conf.WEIGHT_PAIR
        
        if conf.USE_CLOTHES_LOSS:
            clothes_adv_loss = self.cal(new_pred_clothes, a_cloth_id, pos_mask)
            if self.current_epoch >= conf.START_EPOCH_ADV:
                loss += clothes_adv_loss

        if self.use_warm_epoch:
            #Warm Up
            warm_iteration = round(self.dataset_size/conf.BATCH_SIZE)*self.warm_epoch # first 5 epoch
            if self.current_epoch < self.warm_epoch:
                warm_up = min(1.0, self.warm_up + 0.9 / warm_iteration)
                loss = loss * warm_up
        
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        self.training_step_outputs.append(loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
    