from dataclasses import dataclass

from .base import DatasetManager, DatasetName
from .accord import AccordConfig, AccordDatasetManager


@dataclass
class DatasetConfigs:
    accord: AccordConfig = AccordConfig()


def load_dataset_manager(
    name: DatasetName, cfg: DatasetConfigs, *args, **kwargs
) -> DatasetManager:
    cfg = getattr(cfg, name.value.lower(), None)
    if name == DatasetName.MISSING:
        raise ValueError(f"Missing runtime dataset: {name}")
    elif name == DatasetName.ACCORD:
        return AccordDatasetManager(cfg=cfg, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset type: {name}")
