from src.datasets.base_dataset import TestDataset, TrainDataset
from torch.utils.data import DataLoader
from config import BASIC_CONFIG

conf = BASIC_CONFIG

def get_query_gallery_loader():
    query_data = TestDataset(conf.QUERY_JSON_PATH, conf.TEST_TRANSFORM)
    gallery_data = TestDataset(conf.GALLERY_JSON_PATH, conf.TEST_TRANSFORM)
    query_loader = DataLoader(query_data, batch_size=conf.BATCH_SIZE, shuffle=False, num_workers=conf.NUM_WORKER, pin_memory=conf.PIN_MEMORY)
    gallery_loader = DataLoader(gallery_data, batch_size=conf.BATCH_SIZE, shuffle=False, num_workers=conf.NUM_WORKER, pin_memory=conf.PIN_MEMORY)
    return query_loader, gallery_loader

def get_train_loader():
    train_data = TrainDataset(conf.TRAIN_JSON_PATH, conf.TRAIN_TRANSFORM)
    train_loader = DataLoader(train_data, batch_size=conf.BATCH_SIZE, shuffle=True, num_workers=conf.NUM_WORKER, pin_memory=conf.PIN_MEMORY)
    return train_loader