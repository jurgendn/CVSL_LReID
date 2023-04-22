import argparse
import torch
import time, os, scipy.io, yaml, math
import numpy as np
import matplotlib.pyplot as plt

from src.models.modules.r50 import FTNet
from src.models.modules.shape_embedding import ShapeEmbedding
from src.models.modules.fusion_net import FusionNet
from src.models.baseline import InferenceBaseline

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
model = InferenceBaseline(shape_edge_index=SHAPE_EMBEDDING_CFG.EDGE_INDEX,
                shape_pose_n_features=SHAPE_EMBEDDING_CFG.POSE_N_FEATURES,
                shape_n_hidden=SHAPE_EMBEDDING_CFG.N_HIDDEN,
                shape_out_features=SHAPE_EMBEDDING_CFG.OUT_FEATURES,
                shape_relation_layers=SHAPE_EMBEDDING_CFG.RELATION_LAYERS,
                r50_stride=FT_NET_CFG.R50_STRIDE).to(BASIC_CONFIG.DEVICE)

model_name = f"net_last_shape_{BASIC_CONFIG.DATASET_NAME}_numepochs_{BASIC_CONFIG.EPOCHS}.pth"

save_path = os.path.join(BASIC_CONFIG.SAVE_PATH, model_name)

model.load_state_dict(torch.load(save_path), strict=False)

model.eval()

with torch.inference_mode():
    query_standard = extract_feature_standard(model, query_loader, type='query')
    gallery_standard = extract_feature_standard(model, gallery_loader, type='gallery')

    query_cc = extract_feature_cc(model, query_loader, type='query')
    gallery_cc = extract_feature_cc(model, gallery_loader, type='gallery')
    

standard_CMC, standard_mAP = evaluate(gallery_standard, query_standard)
standard_CMC = standard_CMC.numpy()
print("==============================")
print()
print(f"Standard Protocols | Rank-1 Accuracy: {standard_CMC[0]:.2f} | mAP: {standard_mAP:.2f}")

cc_CMC, cc_mAP = evaluate2(gallery_cc, query_cc)
cc_CMC = cc_CMC.numpy()
print(f"Cloth-Changing Protocols | Rank-1 Accuracy: {cc_CMC[0]:.2f} | mAP: {cc_mAP:.2f}")\

print()
print("==============================")

# Calculate the rank values for the x-axis
# ranks = np.arange(1, len(standard_CMC)+1)
ranks = np.arange(1, 21)

# Plot the CMC curve 
plt.plot(ranks, standard_CMC[:20], '-o', label='Standard Evaluation')
plt.plot(ranks, cc_CMC[:20], '-x', label='Cloth-Changing Evaluation')

plt.xlabel('Rank')
plt.ylabel('Identification Rate')
plt.title('CMC Curve on LTCC Dataset')
plt.grid(False)
# Save the plot to an output folder
path = f"output/cmc_curve_{BASIC_CONFIG.DATASET_NAME}"
plt.legend()
plt.savefig(path)
