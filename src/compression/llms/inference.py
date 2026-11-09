from dataclasses import dataclass, replace
from typing import Any
from time import time

from langchain_core.runnables import Runnable, chain
from langchain_core.runnables.base import RunnableEach
from tqdm import tqdm

from ..datasets import InstanceID
from ..io import ConditionalPrinter, load_dataclass_jsonl, save_dataclass_jsonl
from ..parsing import OutputParser, ParserResult

from .base import LLMImplementation, LLMOutput, Nickname
from .load import load_llm
from .prompting import PromptData


@dataclass
class Inference:
    prompt_data: PromptData
    raw_llm_output: LLMOutput
    parser_result: ParserResult | None


@dataclass
class PromptWrapper:
    """
    Surely, there is a better way to pass args and kwargs to each Runnable in a
    Chain, but I cannot figure it out from the documentation.
    """

    prompt_data: PromptData
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class ParallelInference:
    def __init__(
        self,
        llm: Nickname,
        llm_implementation: LLMImplementation,
        parser: OutputParser,
        *args,
        **kwargs,
    ):
        self.llm = load_llm(llm, llm_implementation, *args, **kwargs)
        self.parser = parser
        self.chain = self._generate_chain()

    def __call__(self, prompts: list[PromptData], *args, **kwargs) -> list[Inference]:
        return self.chain.invoke([PromptWrapper(p, args, kwargs) for p in prompts])

    def _generate_chain(self) -> Runnable:
        @chain
        def runnable(w: PromptWrapper) -> Inference:
            llm_output = self.llm.invoke(w.prompt_data, *w.args, **w.kwargs)
            if llm_output.error_message is not None:
                llm_output.derived_data = None
                return Inference(
                    prompt_data=replace(w.prompt_data, messages=None),
                    raw_llm_output=llm_output,
                    parser_result=None,
                )
            parser_result = self.parser(
                generated_text=llm_output.generated_text,
                prompt_data=w.prompt_data,
                *w.args,
                **w.kwargs,
            )
            return Inference(
                prompt_data=replace(w.prompt_data, messages=None),
                raw_llm_output=llm_output,
                parser_result=parser_result,
            )

        return RunnableEach(bound=runnable)


class CheckpointIndicator:
    def __init__(self, frequency: float):
        """
        frequency == 0 disables checkpointing altogether.
        0 < frequency < 1 is interpreted as a percentile. For example, frequency==0.1
            means checkpoint every 10th percentile of the data (or 10 times in total).
        frequency >= 1 is interpreted as an interval of seconds. For example,
            frequency == 60 means checkpoint every 60 seconds.
        """
        self.disabled = frequency == 0
        if 0 < frequency < 1:
            self.percentile = frequency
            self.interval = None
        else:
            self.percentile = None
            self.interval = frequency
        self.start_time = None
        self.modulus = None

    def start(self, total: int) -> None:
        self.start_time = None if self.interval is None else time()
        self.modulus = None
        if self.percentile is not None:
            # If percent * total < 1, then int floors to 0, which leads to DivZeroErr.
            self.modulus = max(int(self.percentile * total), 1)

    def do_checkpoint(self, iteration: int) -> bool:
        if self.disabled:
            return False
        if self.percentile is not None:
            if iteration == 0:
                return False
            return iteration % self.modulus == 0
        current_time = time()
        if self.start_time - current_time >= self.interval:
            self.start_time = current_time
            return True
        return False


class CheckpointedParallelInference:
    def __init__(
        self,
        infer: ParallelInference,
        complete_id_list: list[InstanceID],  # List of all Instance IDs (in order) that
        # might need to be processed through inference (including ones that will be
        # skipped due to potential checkpointing.
        out_file: str,
        batch_size: int,
        verbose: bool,
        frequency: int | float,
    ):
        self.infer = infer
        self.out_file = out_file
        self.batch_size = batch_size
        self.print = ConditionalPrinter(verbose)
        self.indicator = CheckpointIndicator(frequency)
        self.print("Loading checkpoint...")
        self.retry = {instance_id: True for instance_id in complete_id_list}
        self.done_inferences = {}
        if not self.indicator.disabled:
            try:
                # Fails occur when error is *not* None or when ParserResult is None.
                for i in load_dataclass_jsonl(out_file, t=Inference):
                    needs_retry = (
                        i.raw_llm_output.error_message is not None
                        or i.parser_result is None
                    )
                    self.retry[i.prompt_data.instance_id] = needs_retry
                    if not needs_retry:
                        self.done_inferences[i.prompt_data.instance_id] = i
                with open(out_file, "w"):
                    pass  # This should empty the file.
            except FileNotFoundError:
                pass  # Keep retry as all True if no checkpoints.
        self.print("Done.")

    def get_needs_retry(self) -> list[InstanceID]:
        return [i for i, needs_retry in self.retry.items() if needs_retry]

    def __call__(self, prompts: list[PromptData], *args, **kwargs) -> None:
        """
        :param prompts: Any prompts here are processed through the inference engine,
        even if they've previously succeeded. Use 'get_needs_retry()' to figure out
        the subset of prompts that haven't previously succeeded in order to not retry
        prompts unnecessarily.
        :param args: Passed to inference engine.
        :param kwargs: Passed to inference engine.
        """
        # Apply checkpoint skips. Summarize if verbose.
        prompts_map = {p.instance_id: p for p in prompts}
        total, missing, skipped, prompts = len(self.retry), 0, 0, []
        for i, retry in self.retry.items():
            if i in prompts_map:
                prompts.append(prompts_map[i])
            elif retry:  # If not in prompts_map, but we should have retried it...
                missing += 1  # ... mark it as missing.
            else:  # If not given in prompts_maps and was previously done, skip it.
                prompts.append(self.done_inferences[i])
                skipped += 1
        self.print(f"Skipping {skipped}/{total} prompts due to checkpointing.")
        self.print(f"Missing {missing} prompts that should be processed.")

        # Make batches.
        if self.batch_size > 0:
            all_batches, current_batch, prompt_count = [], [], 0
            for i, prompt_or_inference in enumerate(prompts):
                current_batch.append(prompt_or_inference)
                if isinstance(prompt_or_inference, PromptData):
                    prompt_count += 1
                if prompt_count == self.batch_size or i == len(prompts) - 1:
                    all_batches.append(current_batch)
                    current_batch, prompt_count = [], 0
            prompts = all_batches
        else:
            prompts = [prompts]

        self.print("Running inference...")
        inferences = []
        self.indicator.start(len(prompts))
        for i, batch in enumerate(tqdm(prompts) if self.print.condition else prompts):
            infer = [p_or_i for p_or_i in batch if isinstance(p_or_i, PromptData)]
            results = {
                r.prompt_data.instance_id: r for r in self.infer(infer, *args, **kwargs)
            }
            for p_or_i in batch:
                if isinstance(p_or_i, PromptData):
                    inferences.append(results[p_or_i.instance_id])
                else:
                    inferences.append(p_or_i)
            if self.indicator.do_checkpoint(i):
                self._save(inferences)
                inferences = []
        self._save(inferences)
        self.print("Done.")

    def _save(self, inferences: list[Inference]) -> None:
        mode = "w" if self.indicator.disabled else "a"
        save_dataclass_jsonl(self.out_file, *inferences, mode=mode, ensure_ascii=False)
