from pathlib import Path
from typing import Dict
import torch 

def get_filename(path: str) -> str:
    name = Path(path).name
    return name

def make_objects(filename: str, orient: int) -> Dict[str, int]:
    return {"name": filename, "orient": orient}

def normalize_feature(feature):
    fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
    normed_feature = feature.div(fnorm.expand_as(feature))
    return normed_feature