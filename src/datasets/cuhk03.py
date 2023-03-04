import os
from typing import List
import pathlib
from PIL import Image
from torch import Tensor, nn 
from torch.utils.data import Dataset
from torchvision import transforms as T

from config import cuhk03_cfg

class CUHK03(Dataset):
    def __init__(self, image_folder, transforms) -> None:
        super(CUHK03, self).__init__()
        self.image_folder = image_folder
        self.transforms = transforms
        self.image = self.get_file_list()
    
    def get_file_list(self):
        def __is_valid_extension(s):
            path = pathlib.Path(s)
            if path.suffix in cuhk03_cfg.CUHK03.IMAGE_EXTENSIONS:
                return True
            else:
                return False
        
        def __make_fullpath(s):
            return f"{self.image_folder}/{s}"

        list_file = os.listdir(self.image_folder)

        valid_files = filter(__is_valid_extension, list_file)
        fullpath = list(map(__make_fullpath, valid_files))

        return fullpath
    
    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image = self.image[index]
        pil = Image.open(image)
        if self.transforms is not None:
            pil = self.transforms(pil)
        
        if isinstance(pil, Tensor) is False:
            pil = T.ToTensor(pil)

        return pil, 0