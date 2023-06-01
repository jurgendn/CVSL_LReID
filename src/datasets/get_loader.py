from src.datasets.base_dataset import TestDataset, TrainDataset, TrainDatasetOrientation
from torch.utils.data import DataLoader
from src.datasets.samplers import RandomIdentitySampler
from config import BASIC_CONFIG

conf = BASIC_CONFIG



def get_query_gallery_loader():
    query_data = TestDataset(conf.QUERY_JSON_PATH, conf.TEST_TRANSFORM, conf.CLOTH_CHANGING_MODE)
    gallery_data = TestDataset(conf.GALLERY_JSON_PATH, conf.TEST_TRANSFORM, conf.CLOTH_CHANGING_MODE)
    query_loader = DataLoader(query_data, batch_size=conf.BATCH_SIZE, shuffle=False, num_workers=conf.NUM_WORKER, pin_memory=conf.PIN_MEMORY)
    gallery_loader = DataLoader(gallery_data, batch_size=conf.BATCH_SIZE, shuffle=False, num_workers=conf.NUM_WORKER, pin_memory=conf.PIN_MEMORY)
    return query_loader, gallery_loader

def get_train_data(orientation_guided):
    if orientation_guided:
        train_data = TrainDatasetOrientation(conf.TRAIN_JSON_PATH, conf.TRAIN_TRANSFORM)
    else:
        train_data = TrainDataset(conf.TRAIN_PATH, conf.TRAIN_JSON_PATH, conf.TRAIN_TRANSFORM)
    sampler = RandomIdentitySampler(train_data, num_instances=8)
    return train_data, sampler