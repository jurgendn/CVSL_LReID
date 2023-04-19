import argparse
import torch
import time, os, scipy.io, yaml, math
import numpy as np


from src.models.modules.r50 import FTNet
from src.models.modules.shape_embedding import ShapeEmbedding
from src.models.modules.fusion_net import FusionNet
from src.models.baseline import LitModule

from src.datasets.get_loader import get_query_gallery_loader

from utils.extract_features import extract_feature_cc, extract_feature_standard
from utils.evaluate import evaluate, evaluate2

from config import BASIC_CONFIG, FT_NET_CFG, SHAPE_EMBEDDING_CFG


# parser = argparse.ArgumentParser(description='test')

# parser.add_argument("--gpu_ids", default=0, type=str, help="gpu ids")
# parser.add_argument("--which_epoch", default='last', type=str, help="load trained model from which epoch")
# parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
# parser.add_argument('--batch_size', default=32, type=int, help='batchsize')
# parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
# parser.add_argument('--PCB', action='store_true', help='use PCB' )
# parser.add_argument('--multi', action='store_true', help='use multiple query' )
# parser.add_argument('--ibn', action='store_true', help='use ibn.' )
# parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

# opt = parser.parse_args()

### Load config ### 

### Load data

query_loader, gallery_loader = get_query_gallery_loader()

"""
need to fix
# """
# def load_model(model):
    
#     model.load_state_dict(torch.load(save_path))
#     return model 

"""
need to fix
"""
model = LitModule(shape_edge_index=SHAPE_EMBEDDING_CFG.EDGE_INDEX,
                shape_pose_n_features=SHAPE_EMBEDDING_CFG.POSE_N_FEATURES,
                shape_n_hidden=SHAPE_EMBEDDING_CFG.N_HIDDEN,
                shape_out_features=SHAPE_EMBEDDING_CFG.OUT_FEATURES,
                shape_relation_layers=SHAPE_EMBEDDING_CFG.RELATION_LAYERS,
                class_num=BASIC_CONFIG.NUM_CLASSES,
                r50_stride=FT_NET_CFG.R50_STRIDE,
                r50_pretrained_weight=FT_NET_CFG.PRETRAINED, lr=BASIC_CONFIG.LR).to(BASIC_CONFIG.DEVICE)

save_path = os.path.join(BASIC_CONFIG.SAVE_PATH, "net_last_with_shape.pth")

model.load_state_dict(torch.load(save_path))

model.eval()

with torch.inference_mode():
    query = extract_feature_standard(model, query_loader, type='query')
    gallery = extract_feature_standard(model, gallery_loader, type='gallery')
    

standard_CMC, standard_mAP = evaluate(gallery, query)
print(f"Standard Protocols | CMC: {standard_CMC:.2f} | mAP: {standard_mAP:.2f}")

# cc_CMC, cc_mAP = evaluate2(gallery, query)
# print(f"Cloth-Changing Protocols | CMC: {cc_CMC:.2f} | mAP: {cc_mAP:.2f}")