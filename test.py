import argparse
import torch
import time, os, scipy.io, yaml, math
import numpy as np
import matplotlib.pyplot as plt

from src.models.baseline import InferenceBaseline

from src.datasets.get_loader import get_query_gallery_loader

from utils.extract_features import extract_feature_cc, extract_feature_standard
from utils.evaluate import evaluate, evaluate2

from config import BASIC_CONFIG, FT_NET_CFG


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

model = InferenceBaseline(r50_stride=FT_NET_CFG.R50_STRIDE).to(BASIC_CONFIG.DEVICE)

save_path = "work_space/save/model_ltcc_60epochs_0.0035lr_64bs_fromscratch_sampler_crossentropylabelsmooth_triplet_clothesLoss_3Refine_GCNConv_3GCN_maxAgg.pth"
save_path = os.path.join(BASIC_CONFIG.SAVE_PATH, BASIC_CONFIG.MODEL_NAME)
model.load_state_dict(torch.load(save_path), strict=False)

cloth_changing = BASIC_CONFIG.CLOTH_CHANGING_MODE

model.eval()

with torch.inference_mode():
    query_standard = extract_feature_standard(model, query_loader, type='query')
    gallery_standard = extract_feature_standard(model, gallery_loader, type='gallery')
    if cloth_changing:
        query_cc = extract_feature_cc(model, query_loader, type='query')
        gallery_cc = extract_feature_cc(model, gallery_loader, type='gallery')
    

standard_CMC, standard_mAP = evaluate(gallery_standard, query_standard)
standard_CMC = standard_CMC.numpy()
print("==============================")
print()
print(f"Results on {BASIC_CONFIG.DATASET_NAME}")

standard_results = f"Standard | R-1: {standard_CMC[0]:.2f} R-4: {standard_CMC[4]:.2f} R-10: {standard_CMC[9]:.2f} | mAP: {standard_mAP:.2f}"
print(standard_results)

if cloth_changing:
    cc_CMC, cc_mAP = evaluate2(gallery_cc, query_cc)
    cc_CMC = cc_CMC.numpy()
    cc_results = f"Cloth-Changing | R-1: {cc_CMC[0]:.2f} R-5: {cc_CMC[4]:.2f} R-10: {cc_CMC[9]:.2f} | mAP: {cc_mAP:.2f}"
    print(cc_results)

print()
print("==============================")

# Calculate the rank values for the x-axis
ranks = np.arange(1, len(standard_CMC)+1)
ranks = np.arange(1, 41)

# # Plot the CMC curve 
plt.plot(ranks, standard_CMC[:40], '-o', label=standard_results)
plt.plot(ranks, cc_CMC[:40], '-x', label=cc_results)

plt.xlabel('Rank')
plt.ylabel('Identification Rate')
plt.title(BASIC_CONFIG.MODEL_NAME)
plt.grid(False)
# Save the plot to an output folder
path = f"output/{BASIC_CONFIG.MODEL_NAME[:-4]}.png"
plt.legend()
plt.savefig(path)
