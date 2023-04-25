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
from src.models.baseline import Baseline
import matplotlib.pyplot as plt 


train_loader, num_classes, dataset_size = get_train_loader()

net = Baseline(shape_edge_index=SHAPE_EMBEDDING_CFG.EDGE_INDEX,
                shape_pose_n_features=SHAPE_EMBEDDING_CFG.POSE_N_FEATURES,
                shape_n_hidden=SHAPE_EMBEDDING_CFG.N_HIDDEN,
                shape_out_features=SHAPE_EMBEDDING_CFG.OUT_FEATURES,
                shape_relation_layers=SHAPE_EMBEDDING_CFG.RELATION_LAYERS,
                class_num=num_classes,
                dataset_size=dataset_size,
                r50_stride=FT_NET_CFG.R50_STRIDE,
                r50_pretrained_weight=FT_NET_CFG.PRETRAINED, lr=BASIC_CONFIG.LR)


model_checkpoint = ModelCheckpoint(every_n_epochs=10)
early_stopping = EarlyStopping(monitor='avg_train_loss', patience=20, mode='min')
# early_stopping = EarlyStopping(mode='min', patience=20, monitor=)
epochs = BASIC_CONFIG.EPOCHS

trainer = Trainer(accelerator='gpu', max_epochs=epochs, callbacks=[model_checkpoint, early_stopping], precision=16)

trainer.fit(model=net, train_dataloaders=train_loader)

name = f"net_last_shape_{BASIC_CONFIG.DATASET_NAME}_{epochs}epochs_{BASIC_CONFIG.WARM_EPOCH}warmepoch_{BASIC_CONFIG.LR}.pth"

torch.save(net.state_dict(), osp.join(BASIC_CONFIG.SAVE_PATH, name))

# # extract logged loss values
# train_loss = trainer.callback_metrics['train_loss']
# avg_train_loss = trainer.callback_metrics['avg_train_loss']

# # plot loss curve
# epochs = range(1, len(avg_train_loss) + 1)
# plt.plot(epochs, train_loss, 'b', label='Training loss')
# plt.plot(epochs, avg_train_loss, 'r', label='Average training loss')
# plt.title('Training loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# loss_name = f"loss_curve_{BASIC_CONFIG.DATASET_NAME}"
# plt.savefig(osp.join("output", loss_name))