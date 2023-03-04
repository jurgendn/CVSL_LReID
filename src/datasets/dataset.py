import os
from pathlib import Path
import torchvision.transforms as T
from torch import nn, Tensor
from torch.utils.data import Dataset
from PIL import Image
from config import dataset

class DatasetReID(Dataset):
    def __init__(self, dataset_name, image_folder, transforms) -> None:
        super(DatasetReID, self).__init__()
        self.dataset_name = dataset_name
        self.image_folder = image_folder
        self.transforms = transforms
        self.image = self.get_file_list()
        
    def get_file_list(self):
        dataset_name = self.dataset_name
        if dataset_name == 'market1501':
            extension = dataset.MARKET1501.IMAGE_EXTENSIONS
        elif dataset_name == 'cuhk03':
            extension = dataset.CUHK03.IMAGE_EXTENSIONS
        elif dataset_name == 'msmt17':
            extension = dataset.MSMT17.IMAGE_EXTENSIONS
        elif dataset_name == 'viper':
            extension = dataset.VIPER.IMAGE_EXTENSIONS
        elif dataset_name == 'ltcc':
            extension = dataset.LTCC.IMAGE_EXTENSIONS
        elif dataset_name == 'prcc':
            extension = dataset.PRCC.IMAGE_EXTENSIONS
        elif dataset_name == 'vc_clothes':
            extension = dataset.VC_CLOTHES.IMAGE_EXTENSIONS
        elif dataset_name == 'real28':
            extension = dataset.REAL28.IMAGE_EXTENSIONS
        
        def __is_valid_extension(s):
            path = Path(s)
            if path.suffix in extension:
                return True
            else:
                return False

        def __make_fullpath(s):
            return f'{self.image_folder}/{s}'

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

