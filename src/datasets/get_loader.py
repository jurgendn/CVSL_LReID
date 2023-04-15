from src.datasets.base_dataset import TestDataset, TrainDataset
from torch.utils.data import DataLoader
from config import get_config

conf = get_config()

train_json_path = conf.train_json_path
query_json_path = conf.query_json_path
gallery_json_path = conf.gallery_json_path
batch_size = conf.batch_size
test_transforms = conf.test_transform
train_transforms = conf.train_transform

def get_query_gallery_loader():
    query_data = TestDataset(query_json_path, test_transforms)
    gallery_data = TestDataset(gallery_json_path, test_transforms)
    query_loader = DataLoader(query_data, batch_size=batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=conf.pin_memory)
    gallery_loader = DataLoader(gallery_data, batch_size=batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=conf.pin_memory)
    return query_loader, gallery_loader

def get_train_loader():
    train_data = TrainDataset(train_json_path, train_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=conf.num_workers, pin_memory=conf.pin_memory)
    return train_loader