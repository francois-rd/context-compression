from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..core import AggregatorOption, Logprobs
from ..datasets import Instance
from ..llms import DerivedData

from .base import (
    Manipulation,
    ManipulationConfig,
    ManipulationName,
    Step,
    StepSplitter,
    StepSplitterUnit,
)


@dataclass
class NoopManipConfig(ManipulationConfig):
    pass


class NoopManipulation(Manipulation):
    def _do_apply_cot(self, cot_body: str, _: Instance) -> str:
        return cot_body


class TailPruningLexicalUnit(Enum):
    NEWLINE = "NEWLINE"
    SENTENCE = "SENTENCE"
    PEAK = "PEAK"


@dataclass
class TailPruningConfig(ManipulationConfig):
    keep_p: float = 50  # Percentile to keep expressed as percentage between 0 and 100.
    # NOTE: Implementation keeps 'up to' this percentile. For example, asking to keep
    # 75% of tokens if there are only 10 tokens ends up keeping 7/10 = 70% of tokens.

    unit: TailPruningLexicalUnit = TailPruningLexicalUnit.PEAK
    # hampel_window_size: int = 10
    # hampel_n_sigma: float = 3.0
    exclusivity_window: int = 10
    frequency_scaler: float = 1.0


class TailPruning(Manipulation):
    def __init__(self, name: ManipulationName, cfg: TailPruningConfig, *_, **__):
        super().__init__(name, cfg, *_, **__)
        self.cfg = cfg
        if cfg.unit == TailPruningLexicalUnit.NEWLINE:
            self.splitter = StepSplitter(StepSplitterUnit.NEWLINE)
        elif cfg.unit == TailPruningLexicalUnit.SENTENCE:
            self.splitter = StepSplitter(StepSplitterUnit.SENTENCE)
        elif cfg.unit == TailPruningLexicalUnit.PEAK:
            self.splitter = StepSplitter(StepSplitterUnit.NEWLINE)
        else:
            raise ValueError(f"Unsupported: {cfg.unit}")

    def _do_apply_cot(self, cot_body: str, reference_instance: Instance) -> str | None:
        inference = self.inferences.get(reference_instance.instance_id)
        logprobs_data: dict | None = inference.raw_llm_output.derived_data.get(
            DerivedData.LOGPROBS.to_key(), None
        )
        if logprobs_data is None:
            return None
        logprobs = Logprobs.from_dict(logprobs_data)
        body_seqs = list(logprobs.indices_of(cot_body, fuzzy=True))
        if len(body_seqs) != 1:
            return None
        body_start, body_end = body_seqs[0].start_idx(), body_seqs[0].end_idx()
        steps = self._get_steps(cot_body, body_start, body_end, logprobs)
        kept_steps = []
        for s in steps:
            seqs = list(logprobs.indices_of(s.text, s.start_idx, s.end_idx, fuzzy=True))
            if len(seqs) != 1:
                return None
            new_end_list_index = int(len(seqs[0].indices) * (self.cfg.keep_p / 100)) - 1
            if new_end_list_index < 0:
                # Keeping not a single token because "keep_p" is set too low. Discard.
                continue
            new_end = seqs[0].indices[new_end_list_index]
            s.text = logprobs.to_text(seqs[0].start_idx(), new_end)
            kept_steps.append(s)
        return "".join(step.text + step.coda for step in kept_steps)

    def _get_steps(
        self, cot_body: str, body_start: int, body_end: int, logprobs: Logprobs
    ) -> list[Step]:
        if self.cfg.unit in [
            TailPruningLexicalUnit.NEWLINE,
            TailPruningLexicalUnit.SENTENCE,
        ]:
            steps = self.splitter.split(cot_body)
            for step in steps:
                step.start_idx = logprobs.containing_token_of(
                    step.start_idx, start_idx=body_start
                )
                step.end_idx = logprobs.containing_token_of(
                    step.end_idx, start_idx=body_start
                )
                step.end_idx += 1
            return steps
        elif self.cfg.unit == TailPruningLexicalUnit.PEAK:
            return self._get_peak_steps(cot_body, body_start, body_end, logprobs)
        else:
            raise ValueError(f"Unsupported: {self.cfg.unit}")

    def _get_peak_steps(
        self, cot_body: str, body_start: int, body_end: int, logprobs: Logprobs
    ) -> list[Step]:
        steps = []
        peak_indices = self._get_top_k_peaks(cot_body, body_start, body_end, logprobs)
        peak_indices = sorted(peak_indices)
        for i, j in zip(peak_indices, peak_indices[1:] + [None]):
            start, end = body_start + i, body_start + j - 1 if j else body_end
            text = logprobs.to_text(start, end)
            end += 1
            steps.append(Step(text=text, start_idx=start, end_idx=end))
        return steps

    def _get_top_k_peaks(
        self, cot_body: str, body_start: int, body_end: int, logprobs: Logprobs
    ) -> list[int]:
        k = len(self.splitter.split(cot_body))
        surprisals = [abs(x) for x in logprobs.to_chosen_logprobs(body_start, body_end)]
        peak_indices, length = [], len(surprisals)
        for _ in range(int(k * self.cfg.frequency_scaler)):
            try:
                max_idx = np.nanargmax(surprisals)
            except ValueError:
                break
            peak_indices.append(max_idx)
            window_before = max(0, max_idx - self.cfg.exclusivity_window)
            window_after = min(length, max_idx + self.cfg.exclusivity_window + 1)
            for i in range(window_before, window_after):
                surprisals[i] = np.nan
        return np.array(peak_indices).tolist()

    # def _get_hampel_peaks(self, body_start: int, body_end: int, logprobs: Logprobs):
    #    from hampel import hampel
    #    return hampel(
    #        np.array(logprobs.to_chosen_logprobs(body_start, body_end)),
    #        window_size=self.cfg.hampel_window_size,
    #        n_sigma=self.cfg.hampel_n_sigma,
    #    ).outlier_indices.tolist()


class TokenSkipLexicalUnit(Enum):
    NEWLINE = "NEWLINE"
    SENTENCE = "SENTENCE"
    TOKEN = "TOKEN"


@dataclass
class TokenSkipConfig(ManipulationConfig):
    keep_p: float = 50  # Percentile to keep expressed as percentage between 0 and 100.
    # NOTE: Implementation keeps 'up to' this percentile. For example, asking to keep
    # 75% of tokens if there are only 10 tokens ends up keeping 7/10 = 70% of tokens.

    unit: TokenSkipLexicalUnit = TokenSkipLexicalUnit.TOKEN
    agg: AggregatorOption = AggregatorOption.SUM
    random: bool = False  # If true, 'keep_p' is interpreted as keeping random 'unit'
    # items up to this percentile, rather than the top percentile.


class TokenSkip(Manipulation):
    def __init__(
        self, name: ManipulationName, cfg: TokenSkipConfig, *_, seed: int, **__
    ):
        super().__init__(name, cfg, *_, **__)
        self.cfg = cfg
        if cfg.unit == TokenSkipLexicalUnit.NEWLINE:
            self.splitter = StepSplitter(StepSplitterUnit.NEWLINE)
        elif cfg.unit == TokenSkipLexicalUnit.SENTENCE:
            self.splitter = StepSplitter(StepSplitterUnit.SENTENCE)
        elif cfg.unit == TokenSkipLexicalUnit.TOKEN:
            pass
        else:
            raise ValueError(f"Unsupported: {cfg.unit}")
        self.rng = np.random.default_rng(seed)

    def _do_apply_cot(self, cot_body: str, reference_instance: Instance) -> str | None:
        inference = self.inferences.get(reference_instance.instance_id)
        logprobs_data: dict | None = inference.raw_llm_output.derived_data.get(
            DerivedData.LOGPROBS.to_key(), None
        )
        if logprobs_data is None:
            return None
        logprobs = Logprobs.from_dict(logprobs_data)
        body_seqs = list(logprobs.indices_of(cot_body, fuzzy=True))
        if len(body_seqs) != 1:
            return None
        body_start = body_seqs[0].start_idx()
        if self.cfg.unit in [
            TokenSkipLexicalUnit.NEWLINE,
            TokenSkipLexicalUnit.SENTENCE,
        ]:
            return self._apply_splitter(cot_body, body_start, logprobs)
        elif self.cfg.unit == TokenSkipLexicalUnit.TOKEN:
            return self._apply_token(body_start, body_seqs[0].end_idx(), logprobs)
        else:
            raise ValueError(f"Unsupported: {self.cfg.unit}")

    def _apply_splitter(
        self, cot_body: str, body_start: int, logprobs: Logprobs
    ) -> str | None:
        steps = []
        for s in self.splitter.split(cot_body):
            start_idx = logprobs.containing_token_of(s.start_idx, start_idx=body_start)
            end_idx = logprobs.containing_token_of(s.end_idx, start_idx=body_start) + 1
            seqs = list(logprobs.indices_of(s.text, start_idx, end_idx, fuzzy=True))
            if len(seqs) != 1:
                return None
            s.surprisal = abs(self.cfg.agg.aggregate(seqs[0].to_chosen_logprobs()))
            steps.append(s)

        keep_n = int(len(steps) * self.cfg.keep_p / 100)
        if self.cfg.random:
            indices = self.rng.choice(len(steps), size=keep_n, replace=False)
        else:
            if keep_n == 0:
                indices = []
            else:
                surprisals = [step.surprisal for step in steps]
                indices = np.argpartition(surprisals, -keep_n)[-keep_n:].tolist()
        top_steps = [steps[i] for i in sorted(indices)]
        return "".join(step.text + step.coda for step in top_steps)

    def _apply_token(self, body_start: int, body_end: int, logprobs: Logprobs) -> str:
        surprisals = [abs(x) for x in logprobs.to_chosen_logprobs(body_start, body_end)]
        keep_n = int(len(surprisals) * self.cfg.keep_p / 100)
        if self.cfg.random:
            indices = self.rng.choice(len(surprisals), size=keep_n, replace=False)
        else:
            if keep_n == 0:
                indices = []
            else:
                indices = np.argpartition(surprisals, -keep_n)[-keep_n:].tolist()
        drop = [i + body_start for i in range(len(surprisals)) if i not in indices]
        for i in sorted(drop, reverse=True):
            del logprobs.sequence[i]
        return logprobs.to_text(body_start, body_end - len(drop))
