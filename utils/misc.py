from pathlib import Path
from typing import Dict


def get_filename(path: str) -> str:
    name = Path(path).name
    return name


def make_objects(filename: str, orient: int) -> Dict[str, int]:
    return {"name": filename, "orient": orient}
