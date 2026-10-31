from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from ..datasets import InstanceID


class MessageType(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "ASSISTANT"


@dataclass
class Message:
    message_type: MessageType
    text: str

    def fill_placeholders(self, **kwargs) -> None:
        self.text = self.text.format(**kwargs)


@dataclass
class PromptData:
    # Chat-like list of messages making up the current prompt (if any).
    messages: list[Message] | None

    # The identifier of the data instance from which this prompt derives.
    instance_id: InstanceID

    # Any additional relevant data.
    flow_through_data: dict[str, Any] = field(default_factory=dict)
