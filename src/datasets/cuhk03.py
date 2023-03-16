import os
from typing import List
import pathlib
from PIL import Image
from torch import Tensor, nn 
from torch.utils.data import Dataset
from torchvision import transforms as T
import json

from config import cuhk03_cfg

class CUHK03(Dataset):
    def __init__(self, path, transforms) -> None:
        super(CUHK03, self).__init__()
        self.path = path
        self.transforms = transforms
        self.imgs, self.ids = self.load_file()

    def load_file(self):
        with open(self.path, 'rb') as f:
            img_list = json.load(f)
        imgs, ids = [], []
        for img in img_list:
            img_path, id = img[0], img[1]
            imgs.append(img_path)
            ids.append(id)
        return imgs, ids
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        id = self.ids[index]

        img = Image.open(img_path)
        if self.transforms is not None:
            pil = self.transforms(pil)
        
        if isinstance(pil, Tensor) is False:
            pil = T.ToTensor(pil)

        return pil, id