from dataclasses import dataclass, field
from random import Random
import re

from ..core import Logprob, Logprobs, RankedLogprob

from .base import DerivedData, LLM, LLMOutput, Nickname
from .prompting import MessageType, PromptData


@dataclass
class OutputOption:
    text: str  # The text to output.
    weight: int  # The weighting of this text relative to others.


@dataclass
class DummyConfig:
    # Text to add as prefill before the prefill_coda.
    prefill_body: str = ""

    # If True, replace prefill_body with "<think>\n{prefill_body}\n</think>\n\n".
    add_reasoning_tags: bool = False

    # Add after prefill_body and before output distribution. Added after reasoning
    # tokens, if any. If no reasoning tokens, a "\n" is inserted between body and coda.
    prefill_coda: str = ""

    # Distribution over output options. Added after prefill_coda.
    output_distribution: list[OutputOption] = field(default_factory=list)

    # Random seed for distribution sampling.
    seed: int = 42


class DummyLLM(LLM):
    pattern = re.compile(r"(\S+\s*)")

    def __init__(
        self,
        nickname: Nickname,
        llm_cfg: DummyConfig,
        *args,
        trim_indicator: str | None = None,
        **kwargs,
    ):
        super().__init__(nickname, *args, **kwargs)
        self.cfg = llm_cfg
        self.indicator = trim_indicator
        self.rng = Random(self.cfg.seed)
        self.pop = [o.text for o in self.cfg.output_distribution]
        self.weights = [o.weight for o in self.cfg.output_distribution]

    def invoke(
        self,
        prompt_data: PromptData,
        *args,
        add_logprobs: bool = False,
        add_prompt_logprobs: bool = False,
        **kwargs,
    ) -> LLMOutput:
        text = self._fake_generated_text(prompt_data)
        err_msg = prompt_data.flow_through_data.get("error_message", None)
        data = prompt_data.flow_through_data.get("derived_data", {})
        logprobs = DerivedData.LOGPROBS.to_key()
        prompt_logprobs = DerivedData.PROMPT_LOGPROBS.to_key()
        if add_logprobs and add_prompt_logprobs:
            data[logprobs] = self._fake_logprobs(text, no_trim=True)
            data[prompt_logprobs] = self._fake_logprobs(prompt_data.messages[1].text)
        elif add_logprobs and not add_prompt_logprobs:
            data[logprobs] = self._fake_logprobs(text, no_trim=True)
        elif not add_logprobs and add_prompt_logprobs:
            full_text = prompt_data.messages[1].text + "\n" + text
            data[prompt_logprobs] = self._fake_logprobs(full_text)
        else:
            pass  # Add nothing.
        return LLMOutput(generated_text=text, error_message=err_msg, derived_data=data)

    def _fake_generated_text(self, prompt_data: PromptData) -> str:
        if prompt_data.messages[-1].message_type == MessageType.ASSISTANT:
            return prompt_data.messages[-1].text
        label = self.rng.choices(self.pop, self.weights)[0]
        return "{begin_think}{prefill_body}{end_think}{prefill_coda}{label}".format(
            begin_think="<think>\n" if self.cfg.add_reasoning_tags else "",
            prefill_body=self.cfg.prefill_body,
            end_think="\n</think>\n\n" if self.cfg.add_reasoning_tags else "\n",
            prefill_coda=self.cfg.prefill_coda,
            label=label,
        )

    def _fake_logprobs(self, text: str, no_trim: bool = False) -> Logprobs:
        logprobs = []
        for token in self.pattern.findall(text):
            logprob = RankedLogprob(
                chosen=Logprob(token=token, rank=2, logprob=-self.rng.random()),
                others={},
                ranking="absolute",
            )
            logprobs.append(logprob)
        logprobs = Logprobs(sequence=logprobs)
        if not no_trim:
            logprobs.maybe_trim(self.indicator, raise_if_not_unique=True, fuzzy=True)
        return logprobs
