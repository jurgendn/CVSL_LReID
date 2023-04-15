import os
import random
from typing import List
import json
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from torch import tensor
"""
a_orientation buckets:
    - back: 63-71, 0-9
    - front: 27 - 45
    - other: side
"""

class TrainDataset(Dataset):

    def __init__(self,
                 json_path: str,
                 transforms: nn.Module = None) -> None:
        super(TrainDataset, self).__init__()
        self.json_path = json_path
        self.img_list = self.get_img_list()
        self.transforms = transforms

    def get_img_list(self):
        with open(self.json_path, 'rb') as f:
            img_list = json.load(f)
        return img_list

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        a_img = self.img_list[index]
        a_img_path =  a_img['img_path']
        a_id =  a_img['p_id']
        a_orientation = a_img['a_orientation']
        a_pose =  a_img['pose_landmarks']
        

        same_id, diff_id = [], []
        for item in self.img_list:
            if item['p_id'] == a_id:
                same_id.append(item)
            else:
                diff_id.append(item)

        same_id_diff_ori, diff_id_same_ori = [], []

        if 0 <= a_orientation <= 9 or 63 <= a_orientation <= 71 or 27 <= a_orientation <= 45:
            # anchor has back or front orientation
            for item in same_id:
                if 45 <= item["orientation"] < 63 or 9 <= item["orientation"] < 27:
                        # found positive sample of sideway orientation
                    same_id_diff_ori.append(item)
            for item in diff_id:
                if 0 <= item["orientation"] <= 9 or 63 <= item["orientation"] <= 71 or 27 <= item["orientation"] <= 45:
                    diff_id_same_ori.append(item)
        else:
            # anchor has sideway orientation
            for item in same_id:
                if 0 <= item["orientation"] <= 9 or 63 <= item["orientation"] <= 71 or 27 <= item["orientation"] <= 45:
                        # found positive sample of back orientation
                    same_id_diff_ori.append(item)
            for item in diff_id:
                if 45 <= item["orientation"] < 63 or 9 <= item["orientation"] < 27:
                    diff_id_same_ori.append(item)

        p_img = random.choice(same_id_diff_ori)
        p_img_path =  p_img['img_path']
        p_pose =  p_img['pose_landmarks']
        
        n_img = random.choice(diff_id_same_ori)
        n_img_path =  n_img['img_path']
        n_pose =  n_img['pose_landmarks']

        a_img_tensor = self.get_img_tensor(a_img_path)
        p_img_tensor = self.get_img_tensor(p_img_path)
        n_img_tensor = self.get_img_tensor(n_img_path)
        
        a_pose_tensor = self.get_pose_tensor(a_pose)
        p_pose_tensor = self.get_pose_tensor(p_pose)
        n_pose_tensor = self.get_pose_tensor(n_pose)

        return (a_img_tensor, p_img_tensor, n_img_tensor), (a_pose_tensor, p_pose_tensor, n_pose_tensor), a_id
        
    def get_img_tensor(self, img_path):
        img = Image.open(img_path)
        img_tensor = self.transforms(img)
        return img_tensor
    
    def get_pose_tensor(self, pose):
        return tensor(pose)
    
class TestDataset(Dataset):

    def __init__(self,
                 json_path: str,
                 transforms: nn.Module = None) -> None:
        super(TrainDataset, self).__init__()
        self.json_path = json_path
        self.img_list = self.get_img_list()
        self.transforms = transforms

    def get_img_list(self):
        with open(self.json_path, 'rb') as f:
            img_list = json.load(f)
        return img_list

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        sample = self.img_list[index]
        img_path =  sample['img_path']
        p_id =  sample['p_id']
        cam_id = sample['cam_id']
        cloth_id = sample['cloth_id']
        
        img_tensor = self.get_img_tensor(img_path)
        return img_tensor, p_id, cam_id, cloth_id        
        
    def get_img_tensor(self, img_path):
        img = Image.open(img_path)
        img_tensor = self.transforms(img)
        return img_tensor
   