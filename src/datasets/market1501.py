import os
import pathlib
from typing import List

from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import transforms as T

from config import market1501_cfg


class Market1501(Dataset):

    def __init__(self,
                 image_folder: str,
                 transforms: nn.Module = None) -> None:
        super(Market1501, self).__init__()
        self.image_folder = image_folder
        self.image = self.get_file_list()
        self.transforms = transforms

    def get_file_list(self) -> List[str]:

        def __is_valid_extension(s: str) -> bool:
            path = pathlib.Path(s)
            if path.suffix in market1501_cfg.MARKET1501.IMAGE_EXTENSIONS:
                return True
            return False

        def __make_fullpath(s: str) -> str:
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
            pil = T.ToTensor()(pil)
        return pil, 0