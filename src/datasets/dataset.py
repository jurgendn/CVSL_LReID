import os.path as osp
from pathlib import Path
import torchvision.transforms as T
from torch import nn, Tensor
from torch.utils.data import Dataset
from PIL import Image
from config import dataset
import json

class DatasetReID(Dataset):
    def __init__(self, dataset_name, transforms) -> None:
        # mode = 0: training, 1: query, 2: gallery
        super(DatasetReID, self).__init__()
        # self.dataset_name = dataset_name
        # if mode==0:
        self.path = osp.join('data', dataset_name, 'jsons', 'train.json')
        # elif mode==1:
        #     self.path = osp.join('data', dataset_name, 'jsons', 'query.json')
        # else:
        #     self.path = osp.join('data', dataset_name, 'jsons', 'gallery.json')
        self.transforms = transforms
        self.imgs_list = self.load_file()

    def load_file(self):
        with open(self.path, 'rb') as f:
            img_list = json.load(f)
        return img_list
    
    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        img_dict = self.imgs_list[index]
        img_path = img_dict['img_path']
        p_id = img_dict['p_id']

        pil = Image.open(img_path)
        if self.transforms is not None:
            pil = self.transforms(pil)
        
        if isinstance(pil, Tensor) is False:
            pil = T.ToTensor(pil)

        return pil, p_id

