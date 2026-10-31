from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from ..core import PromptKey

SliceName = str
InstanceID = str


@dataclass
class Instance:
    instance_id: InstanceID
    core_text: str
    flow_through_data: dict[str, Any] = field(default_factory=dict)


class DatasetName(Enum):
    MISSING = "MISSING"
    ACCORD = "ACCORD"


class DataSlice:
    def __init__(
        self, name: SliceName, dataset_name: DatasetName, instances: list[Instance]
    ):
        self.slice_name = name
        self.dataset_name = dataset_name
        self.instances = instances

    def get_prompt_key(self) -> PromptKey:
        raise NotImplementedError


@dataclass
class DataSliceManager:
    name: SliceName

    def load(self, *arg, **kwargs) -> DataSlice:
        raise NotImplementedError


@dataclass
class DatasetConfig:
    trim_indicator: str = "<missing-indicator>"


class DatasetManager:
    def __init__(self, cfg: DatasetConfig, data_slices: list[DataSliceManager]):
        self.cfg = cfg
        self.slices = {s.name: s for s in data_slices}

    def get_trim_indicator(self) -> str:
        if self.cfg.trim_indicator == "<missing-indicator>":
            raise ValueError("Missing trim indicator")
        return self.cfg.trim_indicator

    def get_slice_manager(self, slice_name: SliceName) -> DataSliceManager:
        return self.slices[slice_name]

    def get_slice_managers(self) -> list[DataSliceManager]:
        return list(self.slices.values())

    def get_slice_names(self) -> list[SliceName]:
        return list(self.slices.keys())
