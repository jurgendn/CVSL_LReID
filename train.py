from typing import List

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
# from pytorch_lightning.loggers import NeptuneLogger
from tqdm.auto import tqdm
import os.path as osp
from config import BASIC_CONFIG, FT_NET_CFG, SHAPE_EMBEDDING_CFG
from src.datasets.get_loader import get_train_loader
from src.models.baseline import Baseline
import matplotlib.pyplot as plt 


train_loader, num_classes, dataset_size = get_train_loader()

net = Baseline(shape_edge_index=SHAPE_EMBEDDING_CFG.EDGE_INDEX,
                shape_pose_n_features=SHAPE_EMBEDDING_CFG.POSE_N_FEATURES,
                shape_n_hidden=SHAPE_EMBEDDING_CFG.N_HIDDEN,
                shape_out_features=SHAPE_EMBEDDING_CFG.OUT_FEATURES,
                shape_relation_layers=SHAPE_EMBEDDING_CFG.RELATION_LAYERS,
                class_num=num_classes,
                train_shape=BASIC_CONFIG.TRAIN_SHAPE,
                dataset_size=dataset_size,
                r50_stride=FT_NET_CFG.R50_STRIDE,
                r50_pretrained_weight=FT_NET_CFG.PRETRAINED, lr=BASIC_CONFIG.LR)

model_name = BASIC_CONFIG.MODEL_NAME
print(model_name)

model_checkpoint = ModelCheckpoint(every_n_epochs=10)
# early_stopping = EarlyStopping(monitor='avg_train_loss', patience=20, mode='min')
epochs = BASIC_CONFIG.EPOCHS

trainer = Trainer(accelerator='gpu', max_epochs=epochs, callbacks=[model_checkpoint])

trainer.fit(model=net, train_dataloaders=train_loader)

torch.save(net.state_dict(), osp.join(BASIC_CONFIG.SAVE_PATH, model_name))