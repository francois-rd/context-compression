from dataclasses import dataclass
from enum import Enum

from ..core import PromptKey, SystemPrompt, InstructionPrompt
from ..datasets import DataSlice, Instance
from ..llms import PromptData
from ..manipulations import Manipulation


class TransformName(Enum):
    MISSING = "MISSING"
    PLAIN = "PLAIN"
    ZERO_COT = "ZERO_COT"
    FORCED_COT = "FORCED_COT"


class Transform:
    def __init__(self, name: TransformName, manipulation: Manipulation | None = None):
        self.name = name
        self.manipulation = manipulation

    def get_manipulation(self) -> Manipulation | None:
        return self.manipulation

    def get_base_transform(self) -> TransformName:
        return self.name

    def get_prompt_key(self) -> PromptKey:
        return self.name.value.lower()

    def get_path_dir(self) -> str:
        return self.name.value

    def make_prompts(
        self,
        data: DataSlice,
        system: SystemPrompt,
        instruction: InstructionPrompt,
        *args,
        **kwargs,
    ) -> list[PromptData]:
        prompts = []
        for instance in data.instances:
            prompt = self._make_prompt(instance, system, instruction, *args, **kwargs)
            if prompt is not None:
                prompts.append(prompt)
        return prompts

    def _make_prompt(
        self,
        instance: Instance,
        system: SystemPrompt,
        instruction: InstructionPrompt,
        *args,
        **kwargs,
    ) -> PromptData | None:
        raise NotImplementedError


@dataclass
class SliceTransformProduct:
    data_slice: DataSlice
    transform: Transform

    def get_prompt_key(self) -> PromptKey:
        slice_key = self.data_slice.get_prompt_key()
        return f"{slice_key}_{self.transform.get_prompt_key()}"
