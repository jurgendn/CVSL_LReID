import argparse
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import time, os, scipy.io, yaml, math
from src.models.modules.r50 import FTNet
from src.models.modules.shape_embedding import ShapeEmbedding
from src.datasets.base_dataset import TestDataset

from utils.evaluate import evaluate, evaluate2


parser = argparse.ArgumentParser(description='test')

parser.add_argument("--gpu_ids", default=0, type=str, help="gpu ids")
parser.add_argument("--which_epoch", default='last', type=str, help="load trained model from which epoch")
parser.add_argument('--query_json_path',default='data/ltcc/jsons/query.json',type=str, help='path to query json file')
parser.add_argument('--gallery_json_path',default='data/ltcc/jsons/gallery.json',type=str, help='path to gallery json file')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_efficient', action='store_true', help='use efficient-b4' )
parser.add_argument('--use_hr', action='store_true', help='use hr18 net' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ibn', action='store_true', help='use ibn.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')

opt = parser.parse_args()

### Load config ### 

batch_size = opt.batchsize
num_workers = opt.num_workers

### Load data
h, w = 256, 128

gallery_json_path = opt.gallery_json_path
query_json_path = opt.query_json_path

data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

query_data = TestDataset(json_path=query_json_path, transforms=data_transforms)
gallery_data = TestDataset(json_path=gallery_json_path, transforms=data_transforms)

query_loader = DataLoader(dataset=query_data, batch_size=batch_size, shuffle=False)
gallery_loader = DataLoader(dataset=gallery_data, batch_size=batch_size, shuffle=False)

def load_model(model):
    save_path = os.path.join('./model', opt.name, 'net_%s.pth'%opt.which_epoch)
    model.load_state_dict(torch.load(save_path))
    return model 

