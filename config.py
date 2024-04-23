import os.path as osp
from dataclasses import dataclass

import torch
from torchvision import transforms as T

from utils.img_transforms import RandomCroping, RandomErasing


@dataclass
class BASIC_CONFIG:
    OUT_FEATURES = 512
    AGG = "concat"  #'sum

    INPUT_SIZE = (384, 192)
    LR = 0.0035
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COLOR_JITTER = False
    RANDOM_ERASING = True

    train_transform_list = [
        T.Resize(INPUT_SIZE),
        RandomCroping(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if COLOR_JITTER:
        train_transform_list = [
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        ] + train_transform_list
    if RANDOM_ERASING:
        train_transform_list += [
            RandomErasing(
                probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)
            )
        ]

    TRAIN_TRANSFORM = T.Compose(train_transform_list)

    TEST_TRANSFORM = T.Compose(
        [
            T.Resize(INPUT_SIZE),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    DATA_PATH = "./data"
    DATASET_PATH = "/home/dustin/Documents/Research/P003 - 2D ReID/Datasets/"
    DATASET_NAME = "ltcc"
    TRAIN_PATH = osp.join(DATASET_PATH, DATASET_NAME, "train")

    if DATASET_NAME == "market1501" or DATASET_NAME == "cuhk03":
        CLOTH_CHANGING_MODE = False
    else:
        CLOTH_CHANGING_MODE = True

    TRAIN_JSON_PATH = osp.join(DATA_PATH, DATASET_NAME, "jsons/train.json")
    QUERY_JSON_PATH = osp.join(DATA_PATH, DATASET_NAME, "jsons/query.json")
    GALLERY_JSON_PATH = osp.join(DATA_PATH, DATASET_NAME, "jsons/gallery.json")

    ORIENTATION_GUIDED = False
    SAMPLER = True

    OPTIMIZER = "adam"  # or 'sgd'
    WEIGHT_DECAY = 5e-4

    USE_WARM_EPOCH = False
    WARM_EPOCH = 5
    WARM_UP = 0.1

    """
    Loss functions
    """

    CLA_LOSS = "crossentropylabelsmooth"  # crossentropy, arcface, cosface, circle
    CLA_S = 16.0
    CLA_M = 0.0

    USE_TRIPLET_LOSS = False
    if USE_TRIPLET_LOSS:
        TRIPLET_LOSS = "triplet"  # circle
        TRIP_M = 0.3

    USE_PAIRWISE_LOSS = True
    if USE_PAIRWISE_LOSS:
        PAIR_LOSS = "triplet"  # contrastive, cosface, circle
        PAIR_M = 0.3
        PAIR_S = 16.0
        WEIGHT_PAIR = 0.2

    # use clothes loss
    USE_CLOTHES_LOSS = True
    if USE_CLOTHES_LOSS:
        CLOTHES_CLA_LOSS = "cosface"
        CAL = "cal"
        EPSILON = 0.1
        START_EPOCH_CC = 25
        START_EPOCH_ADV = 25

    TRAIN_FROM_SCRATCH = True
    TRAIN_FROM_CKPT = False
    CKPT_PATH = (
        "work_space/lightning_logs/version_7/checkpoints/epoch=14-step=17955.ckpt"
    )

    TRAIN_SHAPE = True
    NUM_REFINE_LAYERS = 3  # or 2 or 1
    GCN_LAYER_TYPE = "GCNConv"  # ResGCN or GCNConv

    NUM_GCN_LAYERS = 3
    AGGREGATION_TYPE = "max"  # max

    EPOCHS = 60
    BATCH_SIZE = 64
    PIN_MEMORY = True
    NUM_WORKER = 4

    OUT_FEATURES = 512
    NORM_FEATURE = False

    TEST_WITH_POSE = False

    SAVE_PATH = "./work_space/save"
    LOG_PATH = "./work_space/"

    NAME = f"model_{DATASET_NAME}_{EPOCHS}epochs_{LR}lr_{BATCH_SIZE}bs"

    if ORIENTATION_GUIDED:
        NAME += "_ori"

    if TRAIN_FROM_SCRATCH:
        NAME += "_fromscratch"
    else:
        NAME += "_transfered"

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

    if TRAIN_SHAPE:
        NAME += f"_{NUM_REFINE_LAYERS}Refine"
        NAME += f"_{GCN_LAYER_TYPE}"
        NAME += f"_{NUM_GCN_LAYERS}GCN"
        NAME += f"_{AGGREGATION_TYPE}Agg"

    MODEL_NAME = NAME + ".pth"


@dataclass
class DATASET_CFG:
    DATAPATH = "/media/jurgen/personal/personal_research/person-reid/reid/datasets/Market-1501-v15.09.15/bounding_box_test/"
    BATCH_SIZE = 32

    @dataclass
    class MARKET1501:
        IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


@dataclass
class FT_NET_CFG:
    R50_STRIDE = 1
    DROP_RATE = 0.5
    LINEAR_NUM = 512
    PRETRAINED = "pretrained/net_pretrained_market.pth"


@dataclass
class CFG:
    PRETRAINED = "resources/model.pth"
    DATAPATH = "/media/jurgen/personal/personal_research/person-reid/reid/datasets/Market-1501-v15.09.15/bounding_box_test/"
    BATCH_SIZE = 32


@dataclass
class HRNET_CFG:
    AUTO_RESUME = False

    @dataclass
    class CUDNN:
        BENCHMARK = True
        DETERMINISTIC = False
        ENABLED = True
    
    DATA_DIR = ""
    GPUS = (0, 1, 2, 3)
    OUTPUT_DIR = "output"
    LOG_DIR = "log"
    WORKERS = "8x"
    PRINT_FREQ = 30

    @dataclass
    class DATASET:
        COLOR_RGB = True
        DATASET = "COCO_HOE_Dataset"
        DATA_FORMAT = "jpg"
        FLIP = True
        NUM_JOINTS_HALF_BODY = 8
        PROB_HALF_BODY = 0.3
        TRAIN_ROOT = "data/coco"
        VAL_ROOT = """da    cfg.defrost()
            cfg.merge_from_list("")
            cfg.DATA_DIR = ""
            cfg.OUTPUT_DIR = ""
            cfg.LOG_DIR = ""ta/coco
            """
        ROT_FACTOR = 45
        SCALE_FACTOR = 0.35
        HOE_SIGMA = 4.0

    @dataclass
    class MODEL:
        INIT_WEIGHTS = True
        USE_FEATUREMAP = True
        NAME = "pose_hrnet"
        NUM_JOINTS = 17
        PRETRAINED = "models/pose_hrnet_w32_256x192.pth"
        TARGET_TYPE = "gaussian"
        IMAGE_SIZE = [192, 256]
        HEATMAP_SIZE = [48, 64]
        SIGMA = 2

        @dataclass
        class EXTRA:
            PRETRAINED_LAYERS = [
                "conv1",
                "bn1",
                "conv2",
                "bn2",
                "layer1",
                "transition1",
                "stage2",
                "transition2",
                "stage3",
                "transition3",
                "stage4",
            ]
            FINAL_CONV_KERNEL = 1

            @dataclass
            class STAGE2:
                NUM_MODULES = 1
                NUM_BRANCHES = 2
                BLOCK = "BASIC"
                NUM_BLOCKS = [4, 4]
                NUM_CHANNELS = [32, 64]
                FUSE_METHOD = "SUM"

            @dataclass
            class STAGE3:
                NUM_MODULES = 4
                NUM_BRANCHES = 3
                BLOCK = "BASIC"
                NUM_BLOCKS = [4, 4, 4]
                NUM_CHANNELS = [32, 64, 128]
                FUSE_METHOD = "SUM"

            @dataclass
            class STAGE4:
                NUM_MODULES = 3
                NUM_BRANCHES = 4
                BLOCK = "BASIC"
                NUM_BLOCKS = [4, 4, 4, 4]
                NUM_CHANNELS = [32, 64, 128, 256]
                FUSE_METHOD = "SUM"

    @dataclass
    class LOSS:
        USE_DIFFERENT_JOINTS_WEIGHT = False
        USE_TARGET_WEIGHT = True

    @dataclass
    class TRAIN:
        BATCH_SIZE_PER_GPU = 32
        SHUFFLE = True
        BEGIN_EPOCH = 0
        END_EPOCH = 80
        OPTIMIZER = "adam"
        LR = 0.001
        LR_FACTOR = 0.1
        LR_STEP = [170, 200]
        WD = 0.0001
        GAMMA1 = 0.99
        GAMMA2 = 0.0
        MOMENTUM = 0.9
        NESTEROV = False

    @dataclass
    class TEST:
        BATCH_SIZE_PER_GPU = 32
        COCO_BBOX_FILE = "data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json"
        BBOX_THRE = 1.0
        IMAGE_THRE = 0.0
        IN_VIS_THRE = 0.2
        MODEL_FILE = "output/tud_dataset/pose_hrnet/lrle-3/model_best.pth"
        NMS_THRE = 1.0
        OKS_THRE = 0.9
        USE_GT_BBOX = True
        FLIP_TEST = True
        POST_PROCESS = True
        SHIFT_HEATMAP = True

    @dataclass
    class DEBUG:
        DEBUG = True
        SAVE_BATCH_IMAGES_GT = True
        SAVE_BATCH_IMAGES_PRED = True
        SAVE_HEATMAPS_GT = True
        SAVE_HEATMAPS_PRED = True


@dataclass
class SHAPE_EMBEDDING_CFG:
    # NUM_REFINE_LAYERS = 3
    # GCN_LAYER_TYPE = "GCNConv" # "ResGCN"
    # AGGREGATION_TYPE = 'mean'
    # RELATION_LAYERS = [[512, 512], [512, 256], [256, 256], [256, 128]]
    POSE_N_FEATURES = 3
    N_HIDDEN = 1024
    OUT_FEATURES = 2048
    RELATION_LAYERS = [[2048, 1024], [1024, 1024], [1024, 512]]

    EDGE_INDEX = [
        [1, 1, 2, 3, 5, 6, 1, 8, 9, 1, 11, 12, 1, 0, 14, 0, 15, 2, 5],
        [2, 5, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 0, 14, 16, 15, 17, 16, 17],
    ]
