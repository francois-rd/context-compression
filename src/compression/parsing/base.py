from dataclasses import dataclass
from typing import Any
import json
import re


@dataclass
class ParserResult:
    output: Any  # The parser's output/answer itself.
    span: tuple[int, int]  # The span within 'generated_text' that derives 'output'.


class OutputParser:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, generated_text: str, *args, **kwargs) -> ParserResult | None:
        """
        Parses the generated_text of an LLM to extract some meaningful result.
        Returns None on parsing failure.
        """
        raise NotImplementedError


class NoOutputParser(OutputParser):
    def __init__(self, check_string_type: bool = True):
        """
        Returns 'generated_text' directly without performing any parsing.

        :param check_string_type: If True, checks that LLM's generated text is of
            type 'str'. This can be useful for catching errors that cause LLM to
            instead return None, for example.
        """
        super().__init__()
        self.check = check_string_type

    def __call__(self, generated_text: str, *args, **kwargs) -> ParserResult:
        if self.check and not isinstance(generated_text, str):
            raise ValueError(
                f"'{type(generated_text)}' instead of type 'str': {generated_text}"
            )
        return ParserResult(generated_text, (0, len(generated_text)))


def re_compile(pattern: str, flags=None) -> re.Pattern:
    return re.compile(pattern) if flags is None else re.compile(pattern, flags=flags)


class RegexMatchParser(OutputParser):
    def __init__(self, pattern: str, flags=re.IGNORECASE):
        """
        Uses a regex pattern to parse LLM output. Returns a re.Match or None on failure.

        :param pattern: A regex pattern from which to extract a match.
        :param flags: Flags to pass to re.search(), if any.
        """
        super().__init__()
        self.pattern = re_compile(pattern, flags=flags)

    def __call__(self, generated_text: str, *args, **kwargs) -> ParserResult | None:
        match = self.pattern.search(generated_text)
        if match is None:
            return None
        return ParserResult(match, match.span())


class RegexExtractionParser(OutputParser):
    def __init__(self, pattern: str, match_group: int = 1, flags=re.IGNORECASE):
        """
        Uses a regex pattern to parse LLM output. Returns a string result from
        the match group or None on failure.

        :param pattern: A regex pattern from which to extract a match.
        :param match_group: The group index of result within the pattern Match object.
        :param flags: Flags to pass to re.search(), if any.
        """
        super().__init__()
        self.parser = RegexMatchParser(pattern, flags=flags)
        self.match_group = match_group

    def __call__(self, generated_text: str, *args, **kwargs) -> ParserResult | None:
        try:
            match = self.parser(generated_text, *args, **kwargs).output
            return ParserResult(match.group(self.match_group), match.span())
        except (AttributeError, ValueError):
            return None


class JSONParser(OutputParser):
    def __init__(
        self,
        schema_key: str,
        pattern: str = r"({.*?})",  # NOTE: Doesn't catch JSON objects w/ nested dicts.
        flags=re.IGNORECASE,
    ):
        """
        Extracts JSON objects from generated_text, checking whether the value at the
        schema_key in each object corresponds to a score. Returns None on failure.

        :param schema_key: The key into the JSON object containing the score.
        :param pattern: A regex pattern to extract JSON objects from generated_text
            that may also include other text.
        :param flags: Flags to pass to re.findall(), if any.
        """
        super().__init__()
        self.schema_key = schema_key
        self.pattern = re_compile(pattern, flags=flags)

    def __call__(self, generated_text: str, *args, **kwargs) -> ParserResult | None:
        for string in [generated_text, *self.pattern.findall(generated_text)]:
            try:
                output = json.loads(string, **kwargs)[self.schema_key]
                span = re.search(string, generated_text).span()
                return ParserResult(output, span)
            except (
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                json.decoder.JSONDecodeError,
            ):
                continue
        return None
