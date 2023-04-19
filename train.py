import json
from typing import List

import torch
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from torch import nn
from torchvision import transforms as T
from tqdm.auto import tqdm
import os.path as osp
from config import BASIC_CONFIG, FT_NET_CFG, SHAPE_EMBEDDING_CFG
from src.datasets.get_loader import get_train_loader
from src.models.baseline import LitModule



train_loader = get_train_loader()
net = LitModule(shape_edge_index=SHAPE_EMBEDDING_CFG.EDGE_INDEX,
                shape_pose_n_features=SHAPE_EMBEDDING_CFG.POSE_N_FEATURES,
                shape_n_hidden=SHAPE_EMBEDDING_CFG.N_HIDDEN,
                shape_out_features=SHAPE_EMBEDDING_CFG.OUT_FEATURES,
                shape_relation_layers=SHAPE_EMBEDDING_CFG.RELATION_LAYERS,
                class_num=BASIC_CONFIG.NUM_CLASSES,
                r50_stride=FT_NET_CFG.R50_STRIDE,
                r50_pretrained_weight=FT_NET_CFG.PRETRAINED, lr=BASIC_CONFIG.LR)


# model_checkpoint = ModelCheckpoint(every_n_epochs=1)
# early_stopping = EarlyStopping(mode='min', patience=20, monitor=)
epochs = 60

trainer = Trainer(accelerator='gpu', max_epochs=epochs)# ,callbacks=[model_checkpoint, early_stopping])

trainer.fit(model=net, train_dataloaders=train_loader)

name = f"net_last_shape_{BASIC_CONFIG.DATASET_NAME}_numepochs_{epochs}.pth"

torch.save(net.state_dict(), osp.join(BASIC_CONFIG.SAVE_PATH, name))