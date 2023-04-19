from dataclasses import dataclass
import os.path as osp
import torch
from dynaconf import Dynaconf
from torchvision import transforms as T

CFG = Dynaconf(envvar_prefix="DYNACONF",
               settings_files=["config/main_cfg.yaml"])

HRNET_CFG = Dynaconf(envvar_prefix="DYNACONF",
                     settings_files=["config/pose_hrnet_w32_256_192.yaml"])

DATASET_CFG = Dynaconf(envar_prefix="DYNACONF",
                       settings_file=["config/datasets.yaml"])

FT_NET_CFG = Dynaconf(envar_prefix="DYNACONF",
                      settings_file=["config/ft_net.yaml"])
SHAPE_EMBEDDING_CFG = Dynaconf(envar_prefix="DYNACONF",
                               settings_file=["config/shape_embedding.yaml"])
# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.


@dataclass
class BASIC_CONFIG:
    INPUT_SIZE = (256, 128)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TRAIN_TRANSFORM = T.Compose([
        #T.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        T.Resize(INPUT_SIZE, interpolation=3),
        T.Pad(10),
        T.RandomCrop(INPUT_SIZE),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    TEST_TRANSFORM = T.Compose([
        T.Resize(INPUT_SIZE, interpolation=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    DATA_PATH = "./data"

    DATASET_NAME = "market1501"
    CLOTH_CHANGING_MODE = False
    TRAIN_JSON_PATH = osp.join(DATA_PATH, DATASET_NAME, "jsons/train.json")
    QUERY_JSON_PATH = osp.join(DATA_PATH, DATASET_NAME, "jsons/query.json")
    GALLERY_JSON_PATH = osp.join(DATA_PATH, DATASET_NAME, "jsons/gallery.json")
    
    NUM_CLASSES = 751
    LR = 1e-4
    BATCH_SIZE = 32
    PIN_MEMORY = True
    NUM_WORKER = 4
    SAVE_PATH = "./work_space/save"