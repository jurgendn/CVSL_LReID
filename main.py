import json
from itertools import chain


import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from config import cfg, hrnet_cfg
from src.datasets.dataset import DatasetReID
from src.models.orientations.pose_hrnet import PoseHighResolutionNet
from utils.misc import get_filename, make_objects

transform = T.Compose([
    T.Resize(size=(256, 192)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = PoseHighResolutionNet(hrnet_cfg)
with open(cfg.PRETRAINED, "rb") as f:
    weights = torch.load(f, map_location='cpu')
model.load_state_dict(weights)
model.eval()
model.cuda()

dataset_names = ['market1501', 'cuhk03', 'msmt17', 'viper', 'ltcc', 'prcc', 'vc_clothes', 'real28']
for dataset_name in dataset_names:
    dataset = DatasetReID(dataset_name ,image_folder=cfg.DATAPATH, transforms=transform)
    loader = DataLoader(dataset=dataset, batch_size=cfg.BATCH_SIZE)

    out = []

    with torch.no_grad():
        for x, _ in tqdm(loader):
            x = x.cuda()
            _, hoe = model(x)
            orient = torch.argmax(hoe, dim=1)
            out.append(orient.cpu().view(1, -1)[0])

    out = list(map(lambda s: s.numpy(), out))
    out = list(chain.from_iterable(out))
    label = list(map(lambda s: int(s), out))

    file_name = list(map(get_filename, dataset.image))

    data = list(map(lambda s: make_objects(s[0], s[1]), zip(file_name, label)))

    with open("bbox_train_orientation.json", "w") as f:
        json.dump(data, f, ensure_ascii=False)