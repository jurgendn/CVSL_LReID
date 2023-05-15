import json
import random
from typing import List
import torch 
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
import numpy as np

cloth_change_ids = [0, 88, 2, 4, 89, 146, 136, 123, 93, 124, 122, 45,
                    7, 10, 66, 81, 140, 13, 21, 30, 40, 53, 61, 71, 
                    76, 111, 116, 128, 131, 132, 149, 151, 144, 27, 48, 115, 
                    117, 86, 87, 36, 39, 43, 109, 75, 135, 130]


class TrainDatasetOrientation(Dataset):

    def __init__(self, json_path: str, transforms: nn.Module) -> None:
        super(TrainDatasetOrientation, self).__init__()
        self.json_path = json_path
        self.img_list = get_img_list(self.json_path)
        self.transforms = transforms
        self.id_to_idx, self.num_classes = classes_to_idx(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = self.img_list[index]
        a_img_path = img['img_path']
        a_id = img['p_id']
        a_orientation = img['orientation']
        a_pose = img['pose_landmarks']
        a_cloth_id = img['cloth_id']

        a_img_tensor, a_size = get_img_tensor(a_img_path, self.transforms)
        pose_tensor = get_pose_tensor(a_pose, a_size)
        a_id_index = self.id_to_idx[a_id]
        
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
                    if a_id in cloth_change_ids:
                        if item['cloth_id'] != a_cloth_id:
                    # found positive sample of sideway orientation and different cloth
                            same_id_diff_ori.append(item)
                    else:
                        same_id_diff_ori.append(item)
            for item in diff_id:
                if 0 <= item["orientation"] <= 9 or 63 <= item["orientation"] <= 71 or 27 <= item["orientation"] <= 45:
                    diff_id_same_ori.append(item)
        else:
            # anchor has sideway orientation
            for item in same_id:
                if 0 <= item["orientation"] <= 9 or 63 <= item["orientation"] <= 71 or 27 <= item["orientation"] <= 45:
                    if a_id in cloth_change_ids:
                        if item['cloth_id'] != a_cloth_id:
                    # found positive sample of sideway orientation and different cloth
                            same_id_diff_ori.append(item)
                    else:
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

        
        p_img_tensor, p_size = get_img_tensor(p_img_path, self.transforms)
        n_img_tensor, n_size = get_img_tensor(n_img_path, self.transforms)

        
        p_pose_tensor = get_pose_tensor(p_pose, p_size)
        n_pose_tensor = get_pose_tensor(n_pose, n_size)

        
        # a_target = torch.zeros(self.num_classes)
        # a_target[a_id_index] = 1
        return (a_img_tensor, p_img_tensor, n_img_tensor), \
                (pose_tensor, p_pose_tensor, n_pose_tensor), a_id_index
    
    
        
class TrainDataset(Dataset):
    def __init__(self, json_path: str, transforms: nn.Module) -> None:
        super(TrainDataset, self).__init__()
        self.json_path = json_path
        self.img_list = get_img_list(self.json_path)
        self.transforms = transforms
        self.num_classes, self.num_clothes, self.pid2clothes, self.pid2label, self.clothes2label = process_train_data(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = self.img_list[index]
        img_path = img['img_path']
        pid = img['p_id']
        pose = img['pose_landmarks']
        cloth_id = img['cloth_id']

        img_tensor, size = get_img_tensor(img_path, self.transforms)
        pose_tensor = get_pose_tensor(pose, size)
        pid_index = self.pid2label[pid]
        cloth_id_index = self.clothes2label[cloth_id]

        return img_tensor, pose_tensor, pid_index, cloth_id_index
    
    
def process_train_data(img_list):
    pid_container = set()
    clothes_container = set()
    for img in img_list:
        pid = img['p_id']
        cloth_id = img['cloth_id']
        pid_container.add(pid)
        clothes_container.add(cloth_id)

    pid_container = sorted(pid_container)
    clothes_container = sorted(clothes_container)
    pid2label = {pid:label for label, pid in enumerate(pid_container)}
    clothes2label = {cloth_id:label for label, cloth_id in enumerate(clothes_container)}

    num_pids = len(pid_container)
    num_clothes = len(clothes_container)

    pid2clothes = np.zeros((num_pids, num_clothes))
    for img in img_list:
        pid = img['p_id']
        clothes = img['cloth_id']
        pid = pid2label[pid]
        cloth_id = clothes2label[clothes]
        pid2clothes[pid, cloth_id] = 1

    return num_pids, num_clothes, pid2clothes, pid2label, clothes2label

class TestDataset(Dataset):

    def __init__(self, json_path: str, transforms: nn.Module = None, cloth_changing: bool = False) -> None:
        super(TestDataset, self).__init__()
        self.json_path = json_path
        self.img_list = get_img_list(self.json_path)
        self.transforms = transforms
        self.cloth_changing_mode = cloth_changing
        self.num_classes = self.get_num_classes()

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
        img_tensor,_ = get_img_tensor(img_path, self.transforms)
        pose_tensor = Tensor(pose)

        if self.cloth_changing_mode:
            cloth_id = sample['cloth_id']
            return img_tensor, pose_tensor, p_id, cam_id, cloth_id, img_path
        else:
            return img_tensor, pose_tensor, p_id, cam_id, img_path




def get_img_tensor(img_path, transform):
        img = Image.open(img_path)
        img_size = img.size
        img_tensor = transform(img)
        return img_tensor, img_size

def get_pose_tensor(pose: List[List[float]], size):
    width, height = size
    processed_pose = [[item[0], item[1]] for item in pose]
    for item in processed_pose:
        item.append(width / height)
    return Tensor(processed_pose)

def classes_to_idx(img_list):
    id_to_index = {}
    index = 0
    for item in img_list:
        a_id = item['p_id']
        if a_id not in id_to_index:
            id_to_index[a_id] = index
            index += 1
    return id_to_index, len(id_to_index)

def get_img_list(json_path):
    with open(json_path, 'rb') as f:
        img_list = json.load(f)
    return img_list