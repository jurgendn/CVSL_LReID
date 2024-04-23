from typing import Tuple, Union

from torch.utils.data import DataLoader

from configs.factory import MainConfig
from src.datasets.base_dataset import TestDataset, TrainDataset, TrainDatasetOrientation
from src.datasets.samplers import RandomIdentitySampler


def get_query_gallery_loader(config: MainConfig):
    query_data = TestDataset(
        json_path=config.query_json_path,
        transforms=config.test_transform,
        cloth_changing=config.cloth_changing_mode,
    )
    gallery_data = TestDataset(
        json_path=config.gallery_json_path,
        transforms=config.test_transform,
        cloth_changing=config.cloth_changing_mode,
    )
    query_loader = DataLoader(
        dataset=query_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    gallery_loader = DataLoader(
        dataset=gallery_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return query_loader, gallery_loader


def get_train_data(
    config: MainConfig,
) -> Tuple[Union[TrainDatasetOrientation, TrainDataset], RandomIdentitySampler]:
    """
    Get train data and sampler
    Args:
        config: MainConfig

    Returns:
        Tuple[Union[TrainDatasetOrientation, TrainDataset], RandomIdentitySampler]: Train data and sampler
    """
    if config.orientation_guided:
        train_data = TrainDatasetOrientation(
            json_path=config.train_json_path, transforms=config.train_transform
        )
    else:
        train_data = TrainDataset(
            config.train_path,
            config.train_json_path,
            config.train_transform,
        )
    sampler = RandomIdentitySampler(train_data, num_instances=8)
    return train_data, sampler
