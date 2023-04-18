from dataclasses import dataclass

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
        T.Resize(INPUT_SIZE, interpolation=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    TEST_TRANSFORM = T.Compose([
        T.Resize(INPUT_SIZE, interpolation=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    TRAIN_JSON_PATH = "./data/market1501/jsons/train.json"
    QUERY_JSON_PATH = None
    GALLERY_JSON_PATH = None
    NUM_CLASSES = 751
    LR = 0.0001
    BATCH_SIZE = 32
    PIN_MEMORY = True
    NUM_WORKER = 16
    SAVE_PATH = None

