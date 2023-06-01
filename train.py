import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os.path as osp
import glob
from config import BASIC_CONFIG, FT_NET_CFG, SHAPE_EMBEDDING_CFG
# from src.datasets.get_loader import get_train_loader
from src.models.baseline import Baseline
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(save_dir=BASIC_CONFIG.LOG_PATH)

# train_loader, num_classes, dataset_size = get_train_loader()

net = Baseline(orientation_guided=BASIC_CONFIG.ORIENTATION_GUIDED,
                lr=BASIC_CONFIG.LR, train_shape=BASIC_CONFIG.TRAIN_SHAPE,
                out_features=BASIC_CONFIG.OUT_FEATURES,
                shape_edge_index=SHAPE_EMBEDDING_CFG.EDGE_INDEX)

model_name = BASIC_CONFIG.MODEL_NAME
print(model_name)

model_checkpoint = ModelCheckpoint(every_n_epochs=5)
early_stopping = EarlyStopping(monitor='epoch_loss', patience=20, mode='min')

epochs = BASIC_CONFIG.EPOCHS

trainer = Trainer(accelerator='gpu', 
                  max_epochs=epochs, 
                  callbacks=[model_checkpoint, early_stopping], 
                  logger=logger)

if BASIC_CONFIG.TRAIN_FROM_CKPT:
    ckpt_path = BASIC_CONFIG.CKPT_PATH
    trainer.fit(model=net, ckpt_path=ckpt_path)
else:
    trainer.fit(model=net)

torch.save(net.state_dict(), osp.join(BASIC_CONFIG.SAVE_PATH, model_name))