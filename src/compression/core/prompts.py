from dataclasses import dataclass

PromptKey = str
SystemPrompt = str
InstructionPrompt = str


@dataclass
class SystemPromptsConfig:
    prompts: dict[PromptKey, SystemPrompt]


@dataclass
class InstructionPromptsConfig:
    prompts: dict[PromptKey, InstructionPrompt]
