import json
from argparse import ArgumentParser
from itertools import chain

from PIL import Image
import torch
import yaml
from torchvision import transforms as T
from tqdm.auto import tqdm

from configs.factory import HRNetConfig
from src.models.orientations.pose_hrnet import PoseHighResolutionNet
from utils.misc import get_filename, make_objects
from src.datasets.ltcc import LTCC
from torchvision.datasets import ImageFolder


def parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str, default="configs/hrnet.yaml"
    )
    parser.add_argument(
        "--pretrained", dest="pretrained", type=str, default="pretrained/model_hboe.pt"
    )
    parser.add_argument("--dataset", dest="dataset", type=str, default="ltcc")
    parser.add_argument("--device", dest="device", type=str, default="cpu")
    args = parser.parse_args()
    return args


def get_orientation():
    args = parse()
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    with open(file=args.config, mode="r") as f:
        payload = yaml.load(stream=f, Loader=yaml.FullLoader)
    with open(file=args.pretrained, mode="rb") as f:
        weights = torch.load(f, map_location=device)
    config = HRNetConfig(**payload)

    model = PoseHighResolutionNet(cfg=config)
    model.load_state_dict(state_dict=weights, strict=True)
    model.eval().to(device)

    transform = T.Compose(
        [
            T.Resize(size=(256, 192)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    loader = LTCC(root=args.dataset)
    for data in loader.train[:100]:
        img_path = data[0]
        image = Image.open(img_path)
        x = transform(image).unsqueeze(0).to(device)
        _, hoe_output = model(x)
        ori = torch.argmax(hoe_output[0]) * 5
        print(ori)


if __name__ == "__main__":
    get_orientation()
