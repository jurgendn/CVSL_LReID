import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image

from configs.factory import MainConfig
from src.datasets.get_loader import get_query_gallery_loader
from utils.extract_features import extract_feature_cc, extract_feature_standard


@torch.inference_mode()
def visualize_ranklist(
    model: nn.Module,
    model_path: str,
    topk: int,
    config: MainConfig,
    is_clothes_change: bool = False,
):
    """
    Visualize the rank list for person re-identification problem.

    Args:
        model (nn.Module): The neural network model used for feature extraction.
        model_path (str): The path to the saved model weights.
        topk (int): The number of top-ranked images to visualize.
        is_clothes_change (bool, optional): Flag indicating whether the person in the query image may have changed clothes.
            Defaults to False.

    Returns:
        None

    Raises:
        FileNotFoundError: If the model weights file is not found.
    """
    query_loader, gallery_loader = get_query_gallery_loader(config=config)
    # model.load_state_dict(torch.load(model_path, map_location="cpu")["state_dict"])
    model.eval()
    extract_fn = (
        extract_feature_standard if not is_clothes_change else extract_feature_cc
    )
    gallery_info = extract_fn(
        model=model, dataloader=gallery_loader, extracted_type="gallery"
    )
    query_info = extract_fn(
        model=model, dataloader=query_loader, extracted_type="query"
    )

    query_feature = query_info["feature"]
    query_cam = np.array(query_info["camera"])
    query_label = np.array(query_info["label"])
    query_cloth = np.array(query_info["cloth"])
    query_path = query_info["path"]
    gallery_feature = gallery_info["feature"]
    gallery_cam = np.array(gallery_info["camera"])
    gallery_label = np.array(gallery_info["label"])
    gallery_cloth = np.array(gallery_info["cloth"])
    gallery_path = gallery_info["path"]

    for i, _ in enumerate(query_label):
        # -----   modify this part of codes for different metrci learning  ------
        # -----   This part also can be replaced by _evaluate_  ------
        qf = query_feature[i]
        ql = query_label[i]
        qc = query_cam[i]
        qcl = query_cloth[i]
        gf = gallery_feature
        gl = gallery_label
        gc = gallery_cam
        gcl = gallery_cloth

        qff = qf.view(-1, 1)
        score = torch.mm(gf, qff)
        score = score.squeeze(1).cpu()
        score = score.numpy()
        index = np.argsort(score)  # from small to large
        index = index[::-1]
        # good index
        query_index = np.argwhere(gl == ql)  # same id
        camera_index = np.argwhere(gc == qc)  # same cam
        cloth_index = np.argwhere(gcl == qcl)  # same id same cloth

        good_index = np.setdiff1d(
            query_index, camera_index, assume_unique=True
        )  # same id different cam
        good_index = np.setdiff1d(
            good_index, cloth_index, assume_unique=True
        )  # same id different cloth
        junk_index1 = np.argwhere(gl == -1)  # id == -1
        junk_index2 = np.intersect1d(query_index, camera_index)  # same id same cam
        junk_index2 = np.union1d(junk_index2, cloth_index)
        junk_index = np.append(junk_index2, junk_index1)  # .flatten())

        mask = np.in1d(index, junk_index, invert=True)
        index = index[mask]
        index = index[:topk]

        def read_image(path: str, good=False):
            image = Image.open(fp=path)
            image = image.resize(size=(192, 384))
            draw = ImageDraw.Draw(im=image)
            if good:
                draw.rectangle(xy=((2, 2), (192, 384)), fill=(0, 0, 255), width=5)
            return T.ToTensor()(image)

        samples = torch.Tensor(topk + 1, 3, 384, 192).fill_(255.0)
        samples[0] = read_image(query_path[i], good=False)
        for k, v in enumerate(index):
            samples[k + 1] = read_image(
                path=os.path.join(gallery_path[v]), good=v in good_index
            )
        # Uncomment to log the grids
        # grid = make_grid(samples, nrow=11, padding=30, normalize=True)

        save_image(
            samples,
            "/home/qxl/ranklist/%s.png" % (query_path[i].split("/")[-1]),
            nrow=9,
            padding=30,
            normalize=True,
        )
