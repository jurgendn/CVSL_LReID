from typing import Tuple

import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms as T

from config import HRNET_SEG_CFG
from src.models.segmentations.hrnet_seg_ocr import HighResolutionNet

IMAGE_PATH = "./example/pose_1.jpg"


def prepare_input(image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
    original_size = image.size
    image = image.resize(size=(512, 1024))
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

    image = T.ToTensor()(image).unsqueeze(0)
    flipped_image = T.ToTensor()(flipped_image).unsqueeze(0)

    return torch.cat(tensors=[image, flipped_image], dim=0), original_size


def main():
    hrnet_seg = HighResolutionNet(config=HRNET_SEG_CFG)
    hrnet_seg.init_weights(HRNET_SEG_CFG.MODEL.PRETRAINED)
    hrnet_seg.eval()

    image = Image.open(IMAGE_PATH)
    image = image.convert(mode="RGB")

    input_tensors, original_size = prepare_input(image=image)

    with torch.no_grad():
        preds, scores = hrnet_seg(input_tensors)

    segment_map = preds[0].unsqueeze(0)
    flipped_segment_map = preds[1].unsqueeze(0)
    pred = F.interpolate(input=segment_map,
                         size=original_size,
                         mode='bilinear',
                         align_corners=True)
    flipped_pred = F.interpolate(input=flipped_segment_map,
                                 size=original_size,
                                 mode='bilinear',
                                 align_corners=True)
    flipped_flipped_pred = flipped_pred.numpy()[:, :, :, ::-1]
    segment_map = (pred + flipped_flipped_pred) * 0.5
    human_segment_map = segment_map[0][11]

    target_sihouette = human_segment_map.numpy()
    print(target_sihouette.shape)
    plt.imsave(fname="sihouette.jpg", arr=target_sihouette)


if __name__ == "__main__":
    main()
