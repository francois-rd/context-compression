from dataclasses import asdict, dataclass
from enum import Enum
import warnings
import re

import numpy as np
import spacy

from ..datasets import Instance, InstanceID
from ..io import load_dataclass_jsonl
from ..llms import Inference

_REASONING_PATTERN = re.compile("(<think>\n)(.*)(\n</think>\n\n)(.*)", flags=re.DOTALL)
_NEWLINE_PATTERN = re.compile(r"(.*)(\s*)", flags=re.MULTILINE)


@dataclass
class Step:
    text: str
    start_idx: int
    end_idx: int
    coda: str = ""
    surprisal: float = np.nan


class StepSplitterUnit(Enum):
    NEWLINE = "NEWLINE"
    SENTENCE = "SENTENCE"


class StepSplitter:
    def __init__(self, how: StepSplitterUnit):
        self.how = how
        if self.how == StepSplitterUnit.SENTENCE:
            warnings.simplefilter("always")
            warnings.warn(
                f"Splitting by {self.how} is not recommended for several reasons.\n"
                f"1. spaCy does not correctly parse bullet point chain-of-thought "
                f"reasoning steps. In particular, steps using Markdown, such as "
                f"'**Step 1:**' get incorrectly parsed as a full sentences as such: "
                f"'**Step 1:*' and '*'. However, '**' is often a complete token in "
                f"the LLM output, which causes token-to-text misalignment. This occurs "
                f"approximately 5% of the time.\n"
                f"2. Unlike {StepSplitterUnit.NEWLINE}, {self.how} does not retain "
                f"trailing whitespace. When pruning, this can cause the last token of "
                f"the previous sentence to be transposed to the first token of the "
                f"next sentence, leading to a fused word. This often still results in "
                f"two tokens for the LLM, but can possibly cause semantic shift from "
                f"compounding. Regardless, it can also change the token ID.\n"
                f"3. Empirically, {StepSplitterUnit.NEWLINE} more accurately reflects "
                f"the human intuition behind what a reasoning step is, whereas "
                f"{self.how} often splits each conceptual step into several sentences.",
                category=DeprecationWarning,
            )
            self.nlp = spacy.load("en_core_web_md")

    def split(self, cot_body: str, start_offset: int = 0) -> list[Step]:
        # NOTE: Returned Steps have start_idx & end_idx relative to cot_body,
        # regardless of the value of start_offset, which just affects the start
        # position from which the search for steps begins.
        steps = []
        if self.how == StepSplitterUnit.NEWLINE:
            for match in _NEWLINE_PATTERN.finditer(cot_body, pos=start_offset):
                if match.group():
                    steps.append(Step(match.group(1), *match.span(1), match.group(2)))
        elif self.how == StepSplitterUnit.SENTENCE:
            for s in self.nlp(cot_body[start_offset:]).sents:
                s = Step(s.text, s.start_char + start_offset, s.end_char + start_offset)
                steps.append(s)
            for i, step in enumerate(steps):
                try:
                    next_step = steps[i + 1]
                    step.coda = cot_body[step.end_idx : next_step.start_idx]
                except IndexError:
                    pass
        else:
            raise ValueError(f"Unsupported: {self.how}")
        return steps


@dataclass
class CoTComponents:
    begin_think: str
    body: str
    end_think: str
    coda: str

    @staticmethod
    def from_text(cot_text: str, output_span: tuple[int, int]) -> "CoTComponents":
        match = _REASONING_PATTERN.search(cot_text)
        if match is None:
            return CoTComponents(
                begin_think="",
                body=cot_text[: output_span[0]],
                end_think="",
                coda=cot_text[slice(*output_span)],
            )
        return CoTComponents(*match.groups())

    def to_text(self, new_coda: str | None = None) -> str:
        return "{begin_think}{body}{end_think}{coda}".format(
            begin_think=self.begin_think,
            body=self.body,
            end_think=self.end_think,
            coda=self.coda if new_coda is None else new_coda,
        )


class ManipulationName(Enum):
    MISSING = "MISSING"
    NOOP = "NOOP"
    TOKEN_SKIP = "TOKEN_SKIP"
    TAIL_PRUNING = "TAIL_PRUNING"


@dataclass
class ManipulationConfig:
    def get_path_dir(self) -> str:
        return "__".join([f"{k}={v}" for k, v in asdict(self).items()])


class Manipulation:
    def __init__(
        self,
        name: ManipulationName,
        cfg: ManipulationConfig,
        *_,
        new_coda: str | None = None,
        **__,
    ):
        self.name = name
        self.cfg = cfg
        self.new_coda = new_coda
        self.inferences: dict[InstanceID, Inference] | None = None

    def get_path_dir(self) -> str:
        return "/".join([str(self.name.value), self.cfg.get_path_dir()])

    def lazy_load_base_file(self, base_cot_output_file: str) -> None:
        if self.inferences is None:
            inferences = load_dataclass_jsonl(base_cot_output_file, t=Inference)
            self.inferences = {
                i.prompt_data.instance_id: i
                for i in inferences
                if i.parser_result is not None
            }

    def apply_cot(self, instance: Instance) -> str | None:
        inference = self.inferences.get(instance.instance_id, None)
        if inference is None:
            return None
        text = inference.raw_llm_output.generated_text
        comps = CoTComponents.from_text(text, inference.parser_result.span)
        new_body = self._do_apply_cot(comps.body, instance)
        if new_body is None:
            return None
        comps.body = new_body
        return comps.to_text(new_coda=self.new_coda)

    def _do_apply_cot(self, cot_body: str, reference_instance: Instance) -> str | None:
        raise NotImplementedError
