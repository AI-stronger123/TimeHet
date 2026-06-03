from .base import BaseDataset, BaseLinkDataset
from .mag import MAGLinkDataset
from .wiki import WikiLinkDataset

DATASET_REGISTRY = {
    "mag_link": MAGLinkDataset,
    "wiki_link": WikiLinkDataset,
}


def build_dataset(config: dict) -> BaseDataset:
    return DATASET_REGISTRY[config["name"]](config)
