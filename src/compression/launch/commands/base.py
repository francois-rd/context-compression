from dataclasses import dataclass, field
import os

from ...datasets import DatasetName, SliceName
from ...io import PathConfig
from ...llms import LLMImplementation, MISSING_NICKNAME, Nickname, flatten
from ...manipulations import (
    ManipulationConfig,
    ManipulationName,
    TailPruningConfig,
    TokenSkipConfig,
)
from ...parsing import ParserID
from ...transforms import SliceTransformProduct, TransformName


@dataclass
class AnalysisDatasetNameToSliceMap:
    accord: list[SliceName] = field(default_factory=list)

    def get_slice_names(self, dataset_name: DatasetName) -> list[SliceName]:
        return getattr(self, dataset_name.value.lower())


@dataclass
class TransformNameToManipulationMap:
    forced_cot: list[ManipulationName] = field(default_factory=list)

    def get_manipulation_names(
        self, transform_name: TransformName
    ) -> list[ManipulationName]:
        try:
            return getattr(self, transform_name.value.lower())
        except AttributeError:
            return []


@dataclass
class ManipulationNameToHPMap:
    token_skip: list[TokenSkipConfig] = field(default_factory=list)
    tail_pruning: list[TailPruningConfig] = field(default_factory=list)

    def get_manipulation_configs(
        self, name: ManipulationName
    ) -> list[ManipulationConfig]:
        return getattr(self, name.value.lower())


@dataclass
class GeneralConfig:
    # Global fields.
    verbose: bool = True  # Whether to be verbose during processing steps.
    seed: int = 42  # Random seed for reproducibility.
    active_dataset: DatasetName = DatasetName.MISSING  # The dataset to actively use.
    active_transform: TransformName = TransformName.MISSING  # The transform to use.
    new_forced_cot_coda: str = "Answer:\n"  # New coda to add to FORCED_COT transforms.

    # Inference fields.
    active_data_slice: SliceName = "<missing>"  # The slice within the dataset to use.
    active_llm: Nickname = MISSING_NICKNAME  # The LLM to actively use.
    active_llm_implementation: LLMImplementation = (
        LLMImplementation.MISSING
    )  # The implementation type of the active LLM.
    prompt_batch_size: int = 10  # How many samples to run at once.
    checkpoint_frequency: float = 5 * 60  # E.g., save every 5 minutes.
    active_parser: ParserID = "simple_mcq"  # The ID of the parser to actively use.

    # Analysis fields.
    # Which data slices from the active_dataset to include in the analysis.
    data_slices: AnalysisDatasetNameToSliceMap = field(
        default_factory=AnalysisDatasetNameToSliceMap
    )
    # Which transforms to include in the analysis.
    transforms: list[TransformName] = field(default_factory=list)
    # Which manipulations from the above transforms to include in the analysis.
    manipulations: TransformNameToManipulationMap = field(
        default_factory=TransformNameToManipulationMap
    )
    # Which hyperparameter variants from the above manipulations to include in analysis.
    manipulation_hps: ManipulationNameToHPMap = field(
        default_factory=ManipulationNameToHPMap
    )
    # Which LLMs to include in the analysis.
    llms: list[Nickname] = field(default_factory=list)

    def llm_output_file(
        self, path: PathConfig, stp: SliceTransformProduct, llm: Nickname | None = None
    ) -> str:
        llm = llm or self.active_llm
        return self._get_file_or_dir_base(path, "output", stp, flatten(llm)) + ".jsonl"

    def postprocess_file(self, path: PathConfig, llm: Nickname) -> str:
        return str(
            os.path.join(
                path.postprocess_dir,
                self.active_dataset.value,
                flatten(llm) + ".jsonl",
            )
        )

    @staticmethod
    def _get_file_or_dir_base(
        path: PathConfig,
        command: str,
        stp: SliceTransformProduct,
        *args,
    ) -> str:
        return str(
            os.path.join(
                path.transforms_dir,
                stp.transform.get_path_dir(),
                command,
                stp.data_slice.dataset_name.value,
                stp.data_slice.slice_name,
                *args,
            )
        )
