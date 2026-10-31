from .base import LLM, LLMImplementation, MISSING_NICKNAME, Nickname
from .dummy import DummyLLM
from .openai import OpenAILLM


def load_llm(llm: Nickname, implementation: LLMImplementation, *args, **kwargs) -> LLM:
    if llm == MISSING_NICKNAME:
        raise ValueError(f"Missing runtime LLM: {llm}")
    if implementation == LLMImplementation.MISSING:
        raise ValueError(f"Missing implementation type for LLM: {llm}")
    elif implementation == LLMImplementation.DUMMY:
        return DummyLLM(llm, *args, **kwargs)
    elif implementation == LLMImplementation.OPENAI:
        return OpenAILLM(llm, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM implementation type: {implementation}")
