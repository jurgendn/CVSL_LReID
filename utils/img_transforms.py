import math
import random
from typing import Tuple

import torch
from PIL import Image


class ResizeWithEqualScale:
    """
    Resize an image with equal scale as the original image.

    Args:
        height (int): resized height.
        width (int): resized width.
        interpolation: interpolation manner.
        fill_color (tuple): color for padding.
    """

    def __init__(
        self,
        height: int,
        width: int,
        interpolation: Image.Resampling = Image.Resampling.BILINEAR,
        fill_color: Tuple[int, int, int] = (0, 0, 0),
    ):
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.fill_color = fill_color

    def __call__(self, img: Image.Image):
        width, height = img.size
        if self.height / self.width >= height / width:
            height = int(self.width * (height / width))
            width = self.width
        else:
            width = int(self.height * (width / height))
            height = self.height

        resized_img = img.resize(size=(width, height), resample=self.interpolation)
        new_img = Image.new(
            mode="RGB", size=(self.width, self.height), color=self.fill_color
        )
        new_img.paste(
            im=resized_img,
            box=(int((self.width - width) / 2), int((self.height - height) / 2)),
        )

        return new_img


class RandomCroping:
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(
        self,
        p: float = 0.5,
        interpolation: Image.Resampling = Image.Resampling.BILINEAR,
    ):
        self.p: float = p
        self.interpolation: Image.Resampling = interpolation

    def __call__(self, img: Image.Image):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        width, height = img.size
        if random.uniform(0, 1) >= self.p:
            return img

        new_width, new_height = int(round(width * 1.125)), int(round(height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - width
        y_maxrange = new_height - height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + width, y1 + height))

        return croped_img


class RandomErasing:
    """
    Randomly selects a rectangle region in an image and erases its pixels.

    Reference:
        Zhong et al. Random Erasing Data Augmentation. arxiv: 1708.04896, 2017.

    Args:
        probability: The probability that the Random Erasing operation will be performed.
        sl: Minimum proportion of erased area against input image.
        sh: Maximum proportion of erased area against input image.
        r1: Minimum aspect ratio of erased area.
        mean: Erasing value.
    """

    def __init__(
        self,
        probability: float = 0.5,
        sl: float = 0.02,
        sh: float = 0.4,
        r1: float = 0.3,
        mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
    ):
        self.probability: float = probability
        self.mean: Tuple[float, float, float] = mean
        self.sl: float = sl
        self.sh: float = sh
        self.r1: float = r1

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) >= self.probability:
            return img

        for _ in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1 : x1 + h, y1 : y1 + w] = self.mean[0]
                    img[1, x1 : x1 + h, y1 : y1 + w] = self.mean[1]
                    img[2, x1 : x1 + h, y1 : y1 + w] = self.mean[2]
                else:
                    img[0, x1 : x1 + h, y1 : y1 + w] = self.mean[0]
                return img

        return img
