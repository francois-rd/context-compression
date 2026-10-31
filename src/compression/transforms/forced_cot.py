from dataclasses import dataclass

from ..core import PromptKey, SystemPrompt, InstructionPrompt
from ..datasets import Instance
from ..llms import Message, MessageType, PromptData
from ..manipulations import ManipulationName, ManipulationConfigs, load_manipulation

from .base import Transform, TransformName


@dataclass
class ForcedCoTConfig:
    suffix: str = "Let's think step by step."
    manipulation: ManipulationName = ManipulationName.MISSING


class ForcedCoTTransform(Transform):
    def __init__(
        self,
        name: TransformName,
        cfg: ForcedCoTConfig,
        *args,
        manipulations: ManipulationConfigs,
        **kwargs,
    ):
        super().__init__(name)
        self.cfg = cfg
        self.manipulation = load_manipulation(
            cfg.manipulation, manipulations, *args, **kwargs
        )

    def get_base_transform(self) -> TransformName:
        return TransformName.ZERO_COT

    def get_prompt_key(self) -> PromptKey:
        return self.get_base_transform().value.lower()

    def get_path_dir(self) -> str:
        return f"{super().get_path_dir()}/{self.manipulation.get_path_dir()}"

    def _make_prompt(
        self,
        instance: Instance,
        system: SystemPrompt,
        instruction: InstructionPrompt,
        *_,
        **kwargs,
    ) -> PromptData | None:
        try:
            self.manipulation.lazy_load_base_file(kwargs["base_cot_output_file"])
        except KeyError:
            raise KeyError(
                "To use a Manipulation, provide this Transform with a kwarg called "
                "'base_cot_output_file' that points to the base CoT output from a "
                "specific LLM that ought to be manipulated."
            )

        new_cot = self.manipulation.apply_cot(instance)
        if new_cot is None:
            return None

        messages = [
            Message(MessageType.SYSTEM, "{system}"),
            Message(
                MessageType.USER, "Instructions:\n{instruction}\n\n{core}\n\n{suffix}"
            ),
            Message(MessageType.ASSISTANT, "{cot}"),
        ]
        for message in messages:
            message.fill_placeholders(
                system=system,
                instruction=instruction,
                core=instance.core_text,
                suffix=self.cfg.suffix,
                cot=new_cot,
            )
        return PromptData(
            messages=messages,
            instance_id=instance.instance_id,
        )
