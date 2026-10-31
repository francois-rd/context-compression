from dataclasses import dataclass

from ..core import SystemPrompt, InstructionPrompt
from ..datasets import Instance
from ..llms import Message, MessageType, PromptData

from .base import Transform, TransformName


@dataclass
class ZeroCoTConfig:
    suffix: str = "Let's think step by step."


class ZeroCoTTransform(Transform):
    def __init__(self, name: TransformName, cfg: ZeroCoTConfig, *_, **__):
        super().__init__(name)
        self.cfg = cfg

    def _make_prompt(
        self,
        instance: Instance,
        system: SystemPrompt,
        instruction: InstructionPrompt,
        *__,
        **___,
    ) -> PromptData:
        messages = [
            Message(MessageType.SYSTEM, "{system}"),
            Message(
                MessageType.USER, "Instructions:\n{instruction}\n\n{text}\n\n{suffix}"
            ),
        ]
        for message in messages:
            message.fill_placeholders(
                system=system,
                instruction=instruction,
                text=instance.core_text,
                suffix=self.cfg.suffix,
            )
        return PromptData(
            messages=messages,
            instance_id=instance.instance_id,
        )
