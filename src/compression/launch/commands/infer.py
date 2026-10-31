from coma import InvocationData, command, preload

from ...core import SystemPromptsConfig, InstructionPromptsConfig
from ...datasets import DatasetConfigs, load_dataset_manager
from ...io import ConditionalPrinter, PathConfig
from ...manipulations import ManipulationConfigs
from ...llms import (
    CheckpointedParallelInference,
    DummyConfig,
    LLMImplementation,
    OpenAIConfig,
    ParallelInference,
)
from ...parsing import ParserManager, ParsersConfig
from ...transforms import SliceTransformProduct, TransformConfigs, load_transform

from .base import GeneralConfig


def pre_config_hook(data: InvocationData) -> None:
    preload(data, "general")
    general: GeneralConfig = data.parameters.get_config("general").get_latest()
    if general.active_llm_implementation == LLMImplementation.MISSING:
        raise ValueError("Missing LLM implementation.")
    elif general.active_llm_implementation == LLMImplementation.DUMMY:
        cfg_name, drop_cfg_names = "dummy", ["openai"]
    elif general.active_llm_implementation == LLMImplementation.OPENAI:
        cfg_name, drop_cfg_names = "openai", ["dummy"]
    else:
        raise ValueError(
            f"Unsupported implementation: {general.active_llm_implementation}"
        )
    preload(data, cfg_name)
    cfg = data.parameters.get_config(cfg_name)
    data.parameters.replace("llm_cfg", cfg.as_primitive(cfg.get_latest_key()))
    data.parameters.delete(*drop_cfg_names)


@command(
    name="infer",
    pre_config_hook=pre_config_hook,
    dummy=DummyConfig,
    openai=OpenAIConfig,
)
class Infer:
    def __init__(
        self,
        general: GeneralConfig,
        path: PathConfig,
        datasets: DatasetConfigs,
        transforms: TransformConfigs,
        manipulations: ManipulationConfigs,
        parsers: ParsersConfig,
        system_prompts: SystemPromptsConfig,
        instruction_prompts: InstructionPromptsConfig,
        llm_cfg,
    ):
        self.general = general
        self.path = path
        self.transforms = transforms
        self.manipulations = manipulations
        self.print = ConditionalPrinter(general.verbose)

        # Load data and transform.
        self.print("Loading data...")
        data_manager = load_dataset_manager(
            name=general.active_dataset, cfg=datasets, path=path
        )
        data = data_manager.get_slice_manager(general.active_data_slice).load()
        transform = load_transform(
            name=general.active_transform,
            cfg=transforms,
            manipulations=manipulations,
            seed=general.seed,
        )
        self.stp = SliceTransformProduct(data, transform)
        self.print("Done.")

        # Select prompt components.
        self.system = system_prompts.prompts[self.stp.get_prompt_key()]
        self.instruction = instruction_prompts.prompts[self.stp.get_prompt_key()]

        self.out_file = general.llm_output_file(path, self.stp)
        self.infer = CheckpointedParallelInference(
            infer=ParallelInference(
                llm=general.active_llm,
                llm_implementation=general.active_llm_implementation,
                parser=ParserManager(parsers).get(general.active_parser),
                llm_cfg=llm_cfg,  # Passed to all LLMs' init.
                log_dir=path.log_dir,  # Passed to OpenaiLLM's init.
                trim_indicator=data_manager.get_trim_indicator(),  # OpenaiLLM.
            ),
            out_file=self.out_file,
            batch_size=self.general.prompt_batch_size,
            verbose=self.general.verbose,
            frequency=self.general.checkpoint_frequency,
        )

    def run(self):
        base_transform_name = self.stp.transform.get_base_transform()
        base_transform = load_transform(
            name=base_transform_name,
            cfg=self.transforms,
            manipulations=self.manipulations,
            seed=self.general.seed,
        )
        base_stp = SliceTransformProduct(self.stp.data_slice, base_transform)
        base_cot_output_file = self.general.llm_output_file(self.path, base_stp)

        self.print("Generating prompts...")
        prompts = self.stp.transform.make_prompts(
            data=self.stp.data_slice,
            system=self.system,
            instruction=self.instruction,
            base_cot_output_file=base_cot_output_file,
        )

        self.print("Copying flow-through data...")
        instances = {i.instance_id: i for i in self.stp.data_slice.instances}
        for prompt in prompts:
            instance = instances[prompt.instance_id]
            prompt.flow_through_data.update(instance.flow_through_data)
        self.print("Done.")

        self.print("Running inference...")
        self.infer(prompts, add_logprobs=True, add_prompt_logprobs=True)
