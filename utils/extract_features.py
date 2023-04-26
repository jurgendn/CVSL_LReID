import torch
from tqdm.auto import tqdm

from config import BASIC_CONFIG, SHAPE_EMBEDDING_CFG

cloth_changing_mode = BASIC_CONFIG.CLOTH_CHANGING_MODE

def extract_feature_standard(model, dataloader, type):
    features = []
    cameras = []
    labels = []
    paths = []

    for data in tqdm(dataloader, desc='-- Extract %s features: ' % (type)):
        
        if cloth_changing_mode:
            imgs, poses, p_ids, cam_ids, _, img_paths = data
        else: 
            imgs, poses, p_ids, cam_ids, img_paths = data

        labels += p_ids
        cameras += cam_ids
        paths += img_paths

        n, c, h, w = imgs.size()
        input_imgs = imgs.to(BASIC_CONFIG.DEVICE)
        input_poses = poses.to(BASIC_CONFIG.DEVICE)

        output = model(
            input_imgs, input_poses,
            torch.LongTensor(SHAPE_EMBEDDING_CFG.EDGE_INDEX).to(
                BASIC_CONFIG.DEVICE))

        feature = output.data.cpu()
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True)
        feature = feature.div(feature_norm.expand_as(feature))
        features.append(feature)

    features = torch.cat(features, dim=0)
    # print(features.shape)
    return {
            'feature': features,
            'camera': cameras,
            'label': labels,
            'path': paths
        }


def extract_feature_cc(model, dataloader, type):
    features = []
    cameras = []
    labels = []
    clothes = []
    paths = []

    for data in tqdm(dataloader, desc='-- Extract %s features: ' % (type)):
        imgs, poses, p_ids, cam_ids, cloth_ids, img_paths = data

        labels += p_ids
        cameras += cam_ids
        clothes += cloth_ids
        paths += img_paths

        n, c, h, w = imgs.size()
        input_imgs = imgs.to(BASIC_CONFIG.DEVICE)
        input_poses = poses.to(BASIC_CONFIG.DEVICE)

        output = model(
            input_imgs, input_poses,
            torch.LongTensor(SHAPE_EMBEDDING_CFG.EDGE_INDEX).to(
                BASIC_CONFIG.DEVICE))

        feature = output.data.cpu()
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True)
        feature = feature.div(feature_norm.expand_as(feature))

        features.append(feature)

    features = torch.cat(features, dim=0)

    return {
        'feature': features,
        'camera': cameras,
        'label': labels,
        'cloth': clothes,
        'path': paths
    }