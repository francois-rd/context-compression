from typing import Any, Callable, Iterable, Literal
from dataclasses import dataclass
from enum import Enum
import heapq
import re

Rank = int
Token = str
PrefixWhitespace = str


@dataclass
class Logprob:
    token: Token
    rank: Rank
    logprob: float

    def __post_init__(self):
        self.logprob = abs(self.logprob)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Logprob":
        return Logprob(token=data["token"], rank=data["rank"], logprob=data["logprob"])


@dataclass
class RankedLogprob:
    chosen: Logprob
    others: dict[Rank, Logprob]
    ranking: Literal["relative", "absolute"]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RankedLogprob":
        return RankedLogprob(
            chosen=Logprob.from_dict(data["chosen"]),
            others={rank: Logprob.from_dict(d) for rank, d in data["others"].items()},
            ranking=data["ranking"],
        )

    def clean_up_token(self) -> str:
        special = ".^$*+?{}[]()\\"
        return "".join(["\\" * (c in special) + c for c in self.chosen.token.strip()])


@dataclass
class SpacedSubsequence:
    indices: list[int]
    reference: "Logprobs"

    def start_idx(self) -> int:
        return self.indices[0]

    def end_idx(self) -> int:
        return self.indices[-1]

    def to_text(self) -> str:
        return self.reference.to_text(self.start_idx(), self.end_idx())

    def to_tokens(self) -> list[str]:
        return self.reference.to_tokens(self.start_idx(), self.end_idx())

    def to_chosen_logprobs(self) -> list[float]:
        return self.reference.to_chosen_logprobs(self.start_idx(), self.end_idx())

    def verify(
        self,
        prefix_text: str | None,
        suffix_text: str | None,
        fuzzy: bool = False,
    ) -> bool:
        if not self.indices:
            # If empty, cannot really verify accurately.
            return False
        has_prefix = self._do_verify(prefix_text, self.start_idx() - 1, max, fuzzy)
        has_suffix = self._do_verify(suffix_text, self.end_idx() + 1, min, fuzzy)
        return has_prefix and has_suffix

    def _do_verify(
        self, text: str | None, target_idx: int, agg: Callable, fuzzy: bool = False
    ) -> bool:
        if text is None:
            return False
        for ss in self.reference.indices_of(text, fuzzy=fuzzy):
            if ss.indices and agg(ss.indices) == target_idx:
                return True
        return False


@dataclass
class Logprobs:
    sequence: list[RankedLogprob]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Logprobs":
        return Logprobs([RankedLogprob.from_dict(d) for d in data["sequence"]])

    def maybe_trim(
        self,
        trim_indicator: str | None,
        include_indicator: bool = False,
        raise_if_indicator_not_found: bool = False,
        raise_if_not_unique: bool = False,
        fuzzy: bool = False,
    ) -> None:
        if trim_indicator is None:
            return
        sequences = list(self.indices_of(trim_indicator, fuzzy=fuzzy))
        if len(sequences) == 0:
            if raise_if_indicator_not_found:
                raise ValueError(f"{trim_indicator=}: not found in: {self.to_text()}")
            return  # Fail silently otherwise.
        if len(sequences) > 1:
            if raise_if_not_unique:
                raise ValueError(
                    f"{trim_indicator=}: found many times in: {self.to_text()}"
                )
            return  # Fail silently otherwise.
        indices = sequences[0].indices
        start_idx = min(indices) if include_indicator else max(indices) + 1
        if start_idx >= len(self.sequence):
            raise ValueError(f"{trim_indicator=}: trims all of: {self.to_text()}")
        self.sequence = self.sequence[start_idx:]

    def to_tokens(self, start_idx: int = 0, end_idx: int | None = None) -> list[str]:
        end_idx = len(self.sequence) if end_idx is None else end_idx + 1
        return [self.sequence[i].chosen.token for i in range(start_idx, end_idx)]

    def to_text(self, start_idx: int = 0, end_idx: int | None = None) -> str:
        return "".join(self.to_tokens(start_idx=start_idx, end_idx=end_idx))

    def to_chosen_logprobs(
        self, start_idx: int = 0, end_idx: int | None = None
    ) -> list[float]:
        end_idx = len(self.sequence) if end_idx is None else end_idx + 1
        return [self.sequence[i].chosen.logprob for i in range(start_idx, end_idx)]

    def indices_of(
        self,
        text: str,  # Text to search for.
        start_idx: int = 0,
        end_idx: int | None = None,
        fuzzy: bool = False,  # Whether BOS or EOS can mismatch in *whitespace*.
    ) -> Iterable[SpacedSubsequence]:
        working_sequence, working_prefix = [], ""
        end_idx = len(self.sequence) if end_idx is None else end_idx
        end_idx = min(len(self.sequence), end_idx)
        for i, logprob in enumerate(self.sequence):
            if i < start_idx or i >= end_idx:
                continue
            whitespace = self._determine_viability(logprob, working_prefix, text)
            if whitespace is None:
                # Not viable and working sequence is incomplete, so reset.
                working_sequence, working_prefix = [], ""
            else:
                working_sequence.append(i)
                working_prefix += whitespace + logprob.clean_up_token()
            if working_sequence:
                working_text = self.to_text(working_sequence[0], working_sequence[-1])
            else:
                working_text = ""
            if fuzzy:
                match = working_text.strip() == text.strip()
            else:
                match = working_text == text
            if match:
                # Working sequence is complete. Yield and then reset to look for more.
                yield SpacedSubsequence(indices=working_sequence, reference=self)
                working_sequence, working_prefix = [], ""

    def containing_token_of(self, char_offset: int, start_idx: int = 0) -> int | None:
        # Translates between the token space and the character space of the string
        # those tokens collectively represent.
        # Given a specific idx of a start token, what token contains the text that is
        # char_offset away? For example: given the token sequence:
        # ["Hello", ", ", "World"], if we set start_idx=1, so we start on ", ", and we
        # offset that by 3 characters (char_offset=3), then ',' is the 0th character,
        # ' ' is the 1st, 'W' is the 2nd, and 'o' is the third. That puts us in the next
        # token ("World"), which is token 2. So, we return 2.
        count = 0
        for i, logprob in enumerate(self.sequence):
            if i < start_idx:
                continue
            count += len(logprob.chosen.token)
            if count >= char_offset:
                return i
        return None

    @staticmethod
    def _determine_viability(
        logprob: RankedLogprob, working_prefix: str, text: str
    ) -> str | None:
        if working_prefix.strip() == logprob.chosen.token.strip() == "":
            # If the very first possible token is just whitespace, we can't
            # reasonably tell if we are correctly matching against it.
            return None
        pattern = working_prefix + r"(\s*)" + logprob.clean_up_token()
        match = re.match(pattern, text)  # match() matches against BOS. Unlike search().
        if match is None:
            return None
        return match.group(1)


AggregatorStr = str


class AggregatorOption(Enum):
    SUM = "SUM"
    MIN = "MIN"
    MAX = "MAX"

    def aggregate(self, logprobs: list[float], top: int | None = None) -> float:
        if top is not None:
            if all(lp >= 0 for lp in logprobs):
                logprobs = heapq.nlargest(top, logprobs)
            elif all(lp <= 0 for lp in logprobs):
                logprobs = heapq.nsmallest(top, logprobs)
            else:
                raise ValueError(
                    f"Logprobs should be all negative or all positive. Got: {logprobs}"
                )
        if self == AggregatorOption.SUM:
            return sum(logprobs)
        elif self == AggregatorOption.MIN:
            return min(logprobs)
        elif self == AggregatorOption.MAX:
            return max(logprobs)
        else:
            raise ValueError(f"Unsupported aggregator: {self}")
