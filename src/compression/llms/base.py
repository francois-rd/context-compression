from dataclasses import dataclass
from typing import Any
from enum import Enum

from .prompting import PromptData

# A name to uniquely differentiate different LLMs.
Nickname = str
MISSING_NICKNAME: Nickname = "<missing-llm>"


def flatten(llm: Nickname) -> Nickname:
    return llm.replace("/", "-")


class LLMImplementation(Enum):
    MISSING = "MISSING"
    DUMMY = "DUMMY"
    OPENAI = "OPENAI"


DerivedDataKey = str


class DerivedData(Enum):
    LOGPROBS = "LOGPROBS"
    PROMPT_LOGPROBS = "PROMPT_LOGPROBS"
    CORRECTNESS = "CORRECTNESS"
    LENGTH = "LENGTH"

    def to_key(self) -> DerivedDataKey:
        return self.value.lower()


@dataclass
class LLMOutput:
    # The LLM's generated output text, or None if an error occurred.
    generated_text: str | None

    # Non-None only if an error occurred, in which case the error message is given here.
    error_message: str | None

    # Any other derived output data, or None if an error occurred or no data exists.
    derived_data: dict[DerivedDataKey, Any] | None = None


class LLM:
    """Wrapper interface for all LLM implementations."""

    def __init__(self, nickname: Nickname, *args, **kwargs):
        self.nickname = nickname

    def invoke(self, prompt_data: PromptData, *args, **kwargs) -> LLMOutput:
        """Invokes the underlying LLM, returning its output."""
        raise NotImplementedError
