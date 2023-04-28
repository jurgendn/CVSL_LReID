from dataclasses import dataclass
import os.path as osp
import torch
from dynaconf import Dynaconf
from torchvision import transforms as T
from utils.random_erasing import RandomErasing

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

    COLOR_JITTER = False
    RANDOM_ERASING = False

    train_transform_list = [
        #T.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        T.Resize(INPUT_SIZE, interpolation=3),
        T.Pad(10),
        T.RandomCrop(INPUT_SIZE),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    if COLOR_JITTER:
        train_transform_list = [T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),] + train_transform_list
    if RANDOM_ERASING:
        train_transform_list += [RandomErasing(probability = 0.5, mean=[0.0, 0.0, 0.0]),]

    TRAIN_TRANSFORM = T.Compose(train_transform_list)

    TEST_TRANSFORM = T.Compose([
        T.Resize(INPUT_SIZE, interpolation=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    DATA_PATH = "./data"
    DATASET_NAME = "ltcc"

    if DATASET_NAME == "market1501" or DATASET_NAME == "cuhk03":
        CLOTH_CHANGING_MODE = False
    else:
        CLOTH_CHANGING_MODE = True
        
    TRAIN_JSON_PATH = osp.join(DATA_PATH, DATASET_NAME, "jsons/train.json")
    QUERY_JSON_PATH = osp.join(DATA_PATH, DATASET_NAME, "jsons/query.json")
    GALLERY_JSON_PATH = osp.join(DATA_PATH, DATASET_NAME, "jsons/gallery.json")

    LR = 0.05
    WEIGHT_DECAY =  5e-4

    USE_WARM_EPOCH = False
    WARM_EPOCH = 5
    WARM_UP = 0.1

    USE_CIRCLE_LOSS = False
    USE_ARCFACE_LOSS = False

    TRAIN_FROM_SCRATCH = True
    TRAIN_SHAPE = True

    EPOCHS = 60
    BATCH_SIZE = 16
    PIN_MEMORY = True
    NUM_WORKER = 4
    
    TEST_WITH_POSE = False

    SAVE_PATH = "./work_space/save"

    NAME = f"model_{DATASET_NAME}_{EPOCHS}epochs_{LR}lr"

    if TRAIN_FROM_SCRATCH:
        NAME += "_fromscratch"
    else:
        NAME += "_transfered"

    if TRAIN_SHAPE:
        NAME += "_withshape"
    
    if USE_WARM_EPOCH:
        NAME += f"_{WARM_EPOCH}warmepoch"
    
    if USE_CIRCLE_LOSS:
        NAME += "_circle"
    
    if USE_ARCFACE_LOSS:
        NAME += "_arcface"
    
    if COLOR_JITTER:
        NAME += "_colorjitter"
    if RANDOM_ERASING:
        NAME += "_randomerasing"

    MODEL_NAME = NAME + ".pth"

