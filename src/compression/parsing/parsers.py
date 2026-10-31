from typing import Callable, Iterable
import re

from ..core import AnswerLabel, AnswerTerm

from .base import JSONParser, OutputParser, ParserResult, RegexExtractionParser


def _none_safe_exact_match(x: str, y: str) -> bool:
    if x is None or y is None:
        return False
    return x.lower() == y.lower()


def _get_answer_choices(**kwargs) -> Iterable[tuple[AnswerLabel, AnswerTerm]]:
    try:
        return kwargs["prompt_data"].flow_through_data["answer_choices"].items()
    except (KeyError, AttributeError):
        raise ValueError(
            "'kwargs' must contain a 'prompt_data' key and the associated value must "
            "have a 'flow_through_data' field. 'flow_through_data' must contain an"
            "'answer_choices' key with a dict-like value."
        )


class MCQPatternParser(OutputParser):
    def __init__(
        self,
        pattern_format: str,
        match_label_group: int | None = 1,
        match_term_group: int | None = 1,
        comparator: Callable[[str, str], bool] = _none_safe_exact_match,
        flags=re.IGNORECASE,
    ):
        """
        Uses a regex pattern to parse LLM output that corresponds to some option from
        the answer choices of a multiple-choice question. Returns a Label or None on
        failure.

        :param pattern_format: A regex pattern with optional LABEL and TERM placeholders
            that will be replaced inline with the values from the answer choices.
        :param match_label_group: If a group number (not None), attempts to regex
            match against the label value, extracting the corresponding match group.
            Takes precedence over 'match_term_group', but both can be given.
        :param match_term_group: If a group number (not None), attempts to regex
            match against the term value, extracting the corresponding match group.
            'match_label_group' takes precedence over this, but both can be given.
        :param comparator: Comparator between extracted label/term and associated
            label/term as written in the answer choices.
        :param flags: Flags to pass to re.search(), if any.
        """
        super().__init__()
        self.pattern_format = pattern_format
        self.match_label_group = match_label_group
        self.match_term_group = match_term_group
        self.comparator = comparator
        self.flags = flags
        self.label_parsers = {}
        self.term_parsers = {}

    def __call__(self, generated_text: str, *args, **kwargs) -> ParserResult | None:
        for label, term in _get_answer_choices(**kwargs):
            label_result = self._label(label, term, generated_text, *args, **kwargs)
            if label_result is not None and self.comparator(label_result.output, label):
                return label_result
            term_result = self._term(label, term, generated_text, *args, **kwargs)
            if term_result is not None and self.comparator(term_result.output, term):
                return term_result
        return None

    def _label(
        self, label: AnswerLabel, term: AnswerTerm, generated_text: str, *args, **kwargs
    ) -> ParserResult | None:
        return self._do_extract(
            parsers=self.label_parsers,
            group=self.match_label_group,
            label=label,
            term=term,
            generated_text=generated_text,
            *args,
            **kwargs,
        )

    def _term(
        self, label: AnswerLabel, term: AnswerTerm, generated_text: str, *args, **kwargs
    ) -> ParserResult | None:
        return self._do_extract(
            parsers=self.label_parsers,
            group=self.match_term_group,
            label=label,
            term=term,
            generated_text=generated_text,
            *args,
            **kwargs,
        )

    def _do_extract(
        self,
        parsers: dict[tuple[AnswerLabel, AnswerTerm], RegexExtractionParser],
        group: int | None,
        label: AnswerLabel,
        term: AnswerTerm,
        generated_text: str,
        *args,
        **kwargs,
    ) -> ParserResult | None:
        if group is None:
            return None
        if (label, term) not in parsers:
            pattern = self.pattern_format.replace("LABEL", label).replace("TERM", term)
            parser = RegexExtractionParser(pattern, int(group), flags=self.flags)
            parsers[(label, term)] = parser
        return parsers[(label, term)](generated_text, *args, **kwargs)


class MCQJSONParser(OutputParser):
    def __init__(
        self,
        schema_key: str,
        pattern: str = r"({.*?})",  # NOTE: Doesn't catch JSON objects w/ nested dicts.
        match_label: bool = True,
        match_term: bool = True,
        comparator: Callable[[str, str], bool] = _none_safe_exact_match,
        flags=re.IGNORECASE,
    ):
        """
        Extracts JSON objects from generated_text, checking whether the value at the
        schema_key in each object corresponds to a label/term. Returns None on failure.

        :param schema_key: The key into the JSON object containing the answer Label.
        :param pattern: A regex pattern to extract JSON objects from generated_text
            that may also include other text.
        :param match_label: Whether the value corresponding to the schema key in the
            JSON object should match against the label value. Takes precedence over
            match_term, but both can be True.
        :param match_term: Whether the value corresponding to the schema key in the
            JSON object should match against the term value. match_label takes
            precedence over this, but both can be True.
        :param comparator: Comparator between extracted label/term and associated
            label/term as written in the QAData answer choices.
        :param flags: Flags to pass to re.compile(), if any.
        """
        super().__init__()
        self.parser = JSONParser(schema_key=schema_key, pattern=pattern, flags=flags)
        self.match_label = match_label
        self.match_term = match_term
        self.comparator = comparator

    def __call__(self, generated_text: str, *args, **kwargs) -> ParserResult | None:
        result = self.parser(generated_text=generated_text, *args, **kwargs)
        if result is None:
            return None
        for label, term in _get_answer_choices(**kwargs):
            if self.match_label and self.comparator(result.output, label):
                return result
            if self.match_term and self.comparator(result.output, term):
                return result
        return None


class SimpleMCQParser(OutputParser):
    # NOTE: These are in approximate descending order of confidence in the pattern.
    default_sub_parsers: list[OutputParser] = [
        MCQPatternParser(r'Answer:\s*"?(LABEL|TERM)"?'),
        MCQPatternParser(r'[Aa]nswer is:?\s*"?(LABEL|TERM)"?'),
        MCQPatternParser(r'^\s*"?(LABEL)"?$'),
        MCQPatternParser(r'"?(LABEL)"?\s*:\s*"?TERM"?', match_term_group=None),
        MCQPatternParser(r'{\s*"?answer"?\s*:\s*"?(LABEL|TERM)"?\s*}'),
        MCQPatternParser(r'{\s*"?answer"?\s*:\s*"(LABEL|TERM)"\s*}?'),
        MCQPatternParser(
            r'{\s*"?answer"?\s*:\s*"?(LABEL)\s*:\s*TERM"?\s*}?', match_term_group=None
        ),
        MCQPatternParser(r'^\s*"?(LABEL|TERM)"?\n'),
        MCQJSONParser("answer"),
    ]

    def __init__(self, sub_parsers: list[OutputParser] = None):
        """
        Parses LLM output using sub-parsers. These are checked in order, and so should
        be given in descending order of confidence in their ability to extract a valid
        label from the generated text. Uses SimpleLLMOutputParser.default_sub_parsers
        if sub_parsers is None.

        :param sub_parsers: Optional list of sub-parsers to call, in order.
        """
        super().__init__()
        self.parsers = self.default_sub_parsers if sub_parsers is None else sub_parsers

    def __call__(self, generated_text: str, *args, **kwargs) -> ParserResult | None:
        """Returns the Label of the first successful sub-parser, or None on failure."""
        for parser in self.parsers:
            result = parser(generated_text, *args, **kwargs)
            if result is not None:
                return result
        return None
