import json
from argparse import ArgumentParser

import torch
import yaml
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm

from configs.factory import HRNetConfig
from src.datasets.ltcc import LTCC
from src.models.orientations.pose_hrnet import PoseHighResolutionNet


def parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str, default="configs/hrnet.yaml"
    )
    parser.add_argument(
        "--pretrained", dest="pretrained", type=str, default="pretrained/model_hboe.pt"
    )
    parser.add_argument("--dataset", dest="dataset", type=str, default="ltcc")
    parser.add_argument("--target-set", dest="target_set", type=str, default="train")
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
    target_set = args.target_set
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

    if target_set == "train":
        image_set = loader.train
    elif target_set == "query":
        image_set = loader.query
    elif target_set == "gallery":
        image_set = loader.gallery
    else:
        raise ValueError(f"Invalid target set: {target_set}")
    output = []
    for data in tqdm(image_set):
        img_path, pid, camid, clothes_id = data
        image = Image.open(img_path)
        x = transform(image).unsqueeze(0).to(device)
        _, hoe_output = model(x)
        ori = torch.argmax(hoe_output[0]) * 5
        payload = {
            "img_path": img_path,
            "p_id": pid,
            "cam_id": camid,
            "clothes_id": clothes_id,
            "orientation": ori.numel(),
        }
        output.append(payload)
    with open(f"{target_set}.json", "w+") as f:
        json.dump(output, f)


if __name__ == "__main__":
    get_orientation()
