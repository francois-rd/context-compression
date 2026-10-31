from dataclasses import asdict, dataclass
from enum import Enum
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

    def to_text(self, keep_coda: bool = False) -> str:
        return "{begin_think}{body}{end_think}{coda}".format(
            begin_think=self.begin_think,
            body=self.body,
            end_think=self.end_think,
            coda=self.coda if keep_coda else "",
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
    def __init__(self, name: ManipulationName, cfg: ManipulationConfig, *_, **__):
        self.name = name
        self.cfg = cfg
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
        return comps.to_text()

    def _do_apply_cot(self, cot_body: str, reference_instance: Instance) -> str | None:
        raise NotImplementedError
