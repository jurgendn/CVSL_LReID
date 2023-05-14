from dataclasses import dataclass
import os.path as osp
import torch
from dynaconf import Dynaconf
from torchvision import transforms as T
from utils.img_transforms import RandomErasing, RandomCroping

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
    USE_RESTNET = True
    USE_HRNET = False
    USE_SWIN = False

    OUT_FEATURES = 512

    if USE_SWIN:
        INPUT_SIZE = (224, 224)
    else:
        INPUT_SIZE = (384, 192)

    if USE_SWIN or USE_HRNET:
        LR = 0.02
    else:
        LR = 0.00035

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    COLOR_JITTER = False
    RANDOM_ERASING = True  

    train_transform_list = [
        #T.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        T.Resize(INPUT_SIZE),
        # T.Pad(10),
        RandomCroping(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    if COLOR_JITTER:
        train_transform_list = [T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),] + train_transform_list
    if RANDOM_ERASING:
        train_transform_list += [RandomErasing(probability=0.5)]

    TRAIN_TRANSFORM = T.Compose(train_transform_list)

    TEST_TRANSFORM = T.Compose([
        T.Resize(INPUT_SIZE),
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

    ORIENTATION_GUIDED = False
    SAMPLER = True

    OPTIMIZER = 'adam' # or 'sgd'
    USE_REDUCE_LR = False
    WEIGHT_DECAY =  5e-4

    USE_WARM_EPOCH = False
    WARM_EPOCH = 5
    WARM_UP = 0.1


    """
    Loss functions
    """

    CLA_LOSS = 'crossentropylabelsmooth' # crossentropy, arcface, cosface, circle
    CLA_S = 16.
    CLA_M = 0.

    USE_TRIPLET_LOSS = False
    if USE_TRIPLET_LOSS:
        TRIPLET_LOSS = 'triplet' # circle
        TRIP_M = 0.3

    USE_PAIRWISE_LOSS = True
    if USE_PAIRWISE_LOSS:
        PAIR_LOSS = 'triplet' # contrastive, cosface, circle
        PAIR_M = 0.3
        PAIR_S = 16.
    
    # use clothes loss
    USE_CLOTHES_LOSS = True
    if USE_CLOTHES_LOSS:
        CLOTHES_CLA_LOSS = 'cosface'
        CAL = 'cal'
        EPSILON = 0.1
        START_EPOCH_CC = 25
        START_EPOCH_ADV = 25

    TRAIN_FROM_SCRATCH = True
    TRAIN_FROM_CKPT = False
    CKPT_PATH = "work_space/lightning_logs/version_3/checkpoints/epoch=49-step=28200.ckpt"

    TRAIN_SHAPE = True 
    NUM_REFINE_LAYERS = 3 # or 2 or 1
    GCN_LAYER_TYPE = "GCNConv" # ResGCN or GCNConv
    NUM_GCN_LAYERS = 3
    AGGREGATION_TYPE = 'max' # max

    EPOCHS = 80
    BATCH_SIZE = 64
    PIN_MEMORY = True
    NUM_WORKER = 4
    
    NORM_FEATURE = False

    TEST_WITH_POSE = False

    SAVE_PATH = "./work_space/save"
    LOG_PATH = "./work_space/"

    NAME = f"model_{DATASET_NAME}_{EPOCHS}epochs_{LR}lr_{BATCH_SIZE}bs"

    if ORIENTATION_GUIDED:
        NAME += "_ori"

    if USE_RESTNET:
        NAME += "_resnet"
    if USE_HRNET:
        NAME += "_hrnet"
    if USE_SWIN:
        NAME += "_swin"

    if TRAIN_FROM_SCRATCH:
        NAME += "_fromscratch"
    else:
        NAME += "_transfered"

    if TRAIN_SHAPE:
        NAME += "_withshape"

    if SAMPLER:
        NAME += "_sampler"

    if USE_WARM_EPOCH:
        NAME += f"_{WARM_EPOCH}warmepoch"
    
    if NORM_FEATURE:
        NAME += "_norm"

    
    NAME += f"_{CLA_LOSS}"

    if USE_TRIPLET_LOSS:
        NAME += f"_{TRIPLET_LOSS}"

    if USE_PAIRWISE_LOSS:
        NAME += f"_{PAIR_LOSS}"
    
    if USE_CLOTHES_LOSS:
        NAME += "_clothesLoss"

    # if RANDOM_ERASING:
    #     NAME += "_randomErasing"

    NAME += f"_{NUM_REFINE_LAYERS}Refine"
    NAME += f"_{GCN_LAYER_TYPE}"
    NAME += f"_{NUM_GCN_LAYERS}GCN"
    NAME += f"_{AGGREGATION_TYPE}Agg"

    # MODEL_NAME = NAME + "_modified.pth"
