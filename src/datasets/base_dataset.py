import json
import random
from typing import List
import torch 
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
import numpy as np

class TrainDataset(Dataset):

    def __init__(self, json_path: str, transforms: nn.Module) -> None:
        super(TrainDataset, self).__init__()
        self.json_path = json_path
        self.img_list = self.get_img_list()
        self.transforms = transforms
        self.id_to_idx, self.num_classes = self.classes_to_idx()

    def get_img_list(self):
        with open(self.json_path, 'rb') as f:
            img_list = json.load(f)
        return img_list
    
    def classes_to_idx(self):
        id_to_index = {}
        index = 0
        for item in self.img_list:
            a_id = item['p_id']
            if a_id not in id_to_index:
                id_to_index[a_id] = index
                index += 1
        return id_to_index, len(id_to_index)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        a_img = self.img_list[index]
        a_img_path = a_img['img_path']
        a_id = a_img['p_id']
        a_orientation = a_img['orientation']
        a_pose = a_img['pose_landmarks']

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

        try:        
            p_img = random.choice(same_id_diff_ori)
        except:
            p_img = random.choice(same_id)
        p_img_path = p_img['img_path']
        p_pose = p_img['pose_landmarks']
        
        try:
            n_img = random.choice(diff_id_same_ori)
        except:
            n_img = random.choice(diff_id)
        n_img_path = n_img['img_path']
        n_pose = n_img['pose_landmarks']

        a_img_tensor = self.get_img_tensor(a_img_path)
        p_img_tensor = self.get_img_tensor(p_img_path)
        n_img_tensor = self.get_img_tensor(n_img_path)

        a_pose_tensor = self.get_pose_tensor(a_pose)
        p_pose_tensor = self.get_pose_tensor(p_pose)
        n_pose_tensor = self.get_pose_tensor(n_pose)

        a_id_index = self.id_to_idx[a_id]
        a_target = torch.zeros(self.num_classes)
        a_target[a_id_index] = 1
        return (a_img_tensor, p_img_tensor, n_img_tensor), (a_pose_tensor, p_pose_tensor, n_pose_tensor), a_target

    def get_img_tensor(self, img_path):
        img = Image.open(img_path)
        img_tensor = self.transforms(img)
        return img_tensor

    def get_pose_tensor(self, pose: List[List[float]]):
        return Tensor(pose)


class TestDataset(Dataset):

    def __init__(self, json_path: str, transforms: nn.Module = None, cloth_changing: bool = False) -> None:
        super(TestDataset, self).__init__()
        self.json_path = json_path
        self.img_list = self.get_img_list()
        self.transforms = transforms
        self.cloth_changing_mode = cloth_changing
        self.num_classes = self.get_num_classes()

    def get_img_list(self):
        with open(self.json_path, 'rb') as f:
            img_list = json.load(f)
        return img_list

    def __len__(self):
        return len(self.img_list)

    def get_num_classes(self):
        classes = []
        for item in self.img_list:
            p_id = item['p_id']
            classes.append(p_id)

        return len(np.unique(classes))

    def __getitem__(self, index):
        sample = self.img_list[index]
        img_path = sample['img_path']
        p_id = sample['p_id']
        cam_id = sample['cam_id']
        pose = sample['pose_landmarks']
        img_tensor = self.get_img_tensor(img_path)
        pose_tensor = Tensor(pose)

        if self.cloth_changing_mode:
            cloth_id = sample['cloth_id']
            return img_tensor, pose_tensor, p_id, cam_id, cloth_id, img_path
        else:
            return img_tensor, pose_tensor, p_id, cam_id, img_path

    def get_img_tensor(self, img_path):
        img = Image.open(img_path)
        img_tensor = self.transforms(img)
        return img_tensor
