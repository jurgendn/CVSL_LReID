import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import BASIC_CONFIG, SHAPE_EMBEDDING_CFG

cloth_changing_mode = BASIC_CONFIG.CLOTH_CHANGING_MODE
dataset_name = BASIC_CONFIG.DATASET_NAME
edge_index = torch.LongTensor(SHAPE_EMBEDDING_CFG.EDGE_INDEX).to(BASIC_CONFIG.DEVICE)


def extract_feature_standard(
    model: nn.Module, dataloader: DataLoader, extracted_type: str
):
    features = []
    cameras = []
    labels = []
    paths = []

    for data in tqdm(dataloader, desc="-- Extract %s features: " % (extracted_type)):
        if cloth_changing_mode:
            imgs, poses, p_ids, cam_ids, _, img_paths = data
        else:
            imgs, poses, p_ids, cam_ids, img_paths = data

        labels += p_ids
        cameras += cam_ids
        paths += img_paths

        input_imgs = imgs.to(BASIC_CONFIG.DEVICE)
        input_poses = poses.to(BASIC_CONFIG.DEVICE)
        print(input_poses)
        output = model(input_imgs, input_poses, edge_index=edge_index)

        feature = output.data.cpu()
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True)
        feature = feature.div(feature_norm.expand_as(feature))
        features.append(feature)

    features = torch.cat(features, dim=0)

    return {"feature": features, "camera": cameras, "label": labels, "path": paths}


def extract_feature_cc(model: nn.Module, dataloader: DataLoader, extracted_type: str):
    features = []
    cameras = []
    labels = []
    clothes = []
    paths = []

    for data in tqdm(
        dataloader, desc="-- Extract %s features | Cloth-changing: " % (extracted_type)
    ):
        imgs, _, p_ids, cam_ids, cloth_ids, img_paths = data

        labels += p_ids
        cameras += cam_ids
        clothes += cloth_ids
        paths += img_paths

        input_imgs = imgs.to(BASIC_CONFIG.DEVICE)

        output = model(input_imgs)

        feature = output.data.cpu()
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True)
        feature = feature.div(feature_norm.expand_as(feature))

        features.append(feature)

    features = torch.cat(features, dim=0)

    return {
        "feature": features,
        "camera": cameras,
        "label": labels,
        "cloth": clothes,
        "path": paths,
    }
