from dataclasses import dataclass, field
from collections import Counter
import os

from coma import command
from scipy.stats import sem, t as t_dist
from statsmodels.stats.proportion import proportion_confint
import numpy as np

from ...datasets import DatasetConfigs, DataSlice, SliceName, load_dataset_manager
from ...io import (
    ConditionalPrinter,
    PathConfig,
    load_dataclass_jsonl,
    save_dataclass_jsonl,
)
from ...llms import DerivedData, Inference, Nickname
from ...manipulations import (
    ManipulationConfigs,
    ManipulationName,
    TailPruningLexicalUnit,
    TokenSkipLexicalUnit,
)
from ...transforms import (
    SliceTransformProduct,
    Transform,
    TransformConfigs,
    TransformName,
    load_transform,
)

from .base import GeneralConfig

KEEP_P = float
WINDOW = int
SCALER = float


class SliceTransformDatabase:
    def __init__(
        self,
        general: GeneralConfig,
        path: PathConfig,
        datasets: DatasetConfigs,
        transforms: TransformConfigs,
        manipulations: ManipulationConfigs,
    ):
        self.general = general
        self.path = path
        self.datasets = datasets
        self.transforms = transforms
        self.manipulations = manipulations
        self.print = ConditionalPrinter(general.verbose)
        self.data_slices = self._load_data_slices()
        self.loaded_transforms = self._load_transforms()
        self.print("Summary:")
        self.print("    Data slices:", [d.slice_name for d in self.data_slices])
        transform_names = [t.name.value for t in self.loaded_transforms]
        self.print("    Transforms:", list(Counter(transform_names).items()))
        self.print("Done.")

    def _load_data_slices(self) -> list[DataSlice]:
        data_slices, name = [], self.general.active_dataset
        data_manager = load_dataset_manager(name, cfg=self.datasets, path=self.path)
        self.print(f"Loading {name} data slices...")
        for slice_name in self.general.data_slices.get_slice_names(name):
            data_slices.append(data_manager.get_slice_manager(slice_name).load())
        self.print("Done.")
        return data_slices

    def _load_transforms(self) -> list[Transform]:
        self.print("Loading transforms...")
        transforms = []
        for transform_name in self.general.transforms:
            names = self.general.manipulations.get_manipulation_names(transform_name)
            if names:
                loaded = self._load_transform_with_manipulations(transform_name, names)
                transforms.extend(loaded)
            else:
                transform = load_transform(
                    name=transform_name, cfg=self.transforms, seed=self.general.seed
                )
                transforms.append(transform)
        self.print("Done.")
        return transforms

    def _load_transform_with_manipulations(
        self, transform_name: TransformName, manipulation_names: list[ManipulationName]
    ) -> list[Transform]:
        transforms = []
        for name in manipulation_names:
            self.print(f"    Loading {transform_name} with manipulation: {name}...")
            transform_cfg = getattr(self.transforms, transform_name.value.lower())
            setattr(transform_cfg, "manipulation", name)
            cfgs_list = self.general.manipulation_hps.get_manipulation_configs(name)
            for cfg in cfgs_list:
                setattr(self.manipulations, name.value.lower(), cfg)
                transform = load_transform(
                    name=transform_name,
                    cfg=self.transforms,
                    manipulations=self.manipulations,
                    seed=self.general.seed,
                )
                transforms.append(transform)
        return transforms

    def get_all_slice_transform_products(self) -> list[SliceTransformProduct]:
        stp = []
        for data in self.data_slices:
            for transform in self.loaded_transforms:
                stp.append(SliceTransformProduct(data, transform))
        return stp


@dataclass
class Outcome:
    accuracy: float | None
    acc_ci: tuple[float, float] | None
    length_mean: float | None
    length_ci: tuple[float, float] | None
    n: int | None

    @staticmethod
    def from_data(binary_scores: list[int], text_lengths: list[int]) -> "Outcome":
        count, nobs = np.count_nonzero(binary_scores), len(binary_scores)
        if len(text_lengths) != nobs:
            raise ValueError(f"Got {nobs} scores, but {len(text_lengths)} lengths.")
        if nobs == 0:
            return Outcome(
                accuracy=None, acc_ci=None, length_mean=None, length_ci=None, n=None
            )
        length_mean = np.mean(text_lengths).item()
        return Outcome(
            accuracy=count / nobs,
            acc_ci=proportion_confint(count=count, nobs=nobs, alpha=0.05),
            length_mean=length_mean,
            length_ci=t_dist.interval(
                1 - 0.05, df=nobs - 1, loc=length_mean, scale=sem(text_lengths)
            ),
            n=nobs,
        )


@dataclass
class StructuredOutcomes:
    # Name of slice within larger dataset to which these outcomes belong.
    slice_name: SliceName

    # Single-value baselines.
    plain: Outcome | None = None
    zero_cot: Outcome | None = None

    # Random baseline by keep percentage.
    random_newline: list[tuple[KEEP_P, Outcome]] = field(default_factory=list)
    random_sentence: list[tuple[KEEP_P, Outcome]] = field(default_factory=list)
    random_token: list[tuple[KEEP_P, Outcome]] = field(default_factory=list)

    # Token skip by keep percentage.
    token_skip_newline: list[tuple[KEEP_P, Outcome]] = field(default_factory=list)
    token_skip_sentence: list[tuple[KEEP_P, Outcome]] = field(default_factory=list)
    token_skip_token: list[tuple[KEEP_P, Outcome]] = field(default_factory=list)

    # Basic tail pruning by keep percentage.
    tail_pruning_newline: list[tuple[KEEP_P, Outcome]] = field(default_factory=list)
    tail_pruning_sentence: list[tuple[KEEP_P, Outcome]] = field(default_factory=list)

    # Ablated tail pruning (keep percentage, exclusivity window, frequency scaler).
    tail_pruning_peak: list[tuple[KEEP_P, WINDOW, SCALER, Outcome]] = field(
        default_factory=list
    )

    def add(
        self,
        stp: SliceTransformProduct,
        binary_scores: list[int],
        text_lengths: list[int],
    ) -> None:
        outcome = Outcome.from_data(binary_scores, text_lengths)
        transform_name = stp.transform.name
        if transform_name == TransformName.PLAIN:
            self.plain = outcome
        elif transform_name == TransformName.ZERO_COT:
            self.zero_cot = outcome
        elif transform_name == TransformName.FORCED_COT:
            self._add_forced_cot(stp, outcome)
        else:
            raise ValueError(f"Unsupported: {transform_name}")

    def _add_forced_cot(self, stp: SliceTransformProduct, outcome: Outcome) -> None:
        manipulation = stp.transform.get_manipulation()
        if manipulation is None:
            raise ValueError(f"Missing Manipulation for: {TransformName.FORCED_COT}")
        elif manipulation.name == ManipulationName.TOKEN_SKIP:
            cfg = manipulation.cfg
            keep_p: KEEP_P = getattr(cfg, "keep_p")
            unit: TokenSkipLexicalUnit = getattr(cfg, "unit")
            random: bool = getattr(cfg, "random")
            attr = ("random_" if random else "token_skip_") + unit.value.lower()
            outcome_list: list = getattr(self, attr)
            outcome_list.append((keep_p, outcome))
        elif manipulation.name == ManipulationName.TAIL_PRUNING:
            cfg = manipulation.cfg
            keep_p: KEEP_P = getattr(cfg, "keep_p")
            unit: TailPruningLexicalUnit = getattr(cfg, "unit")
            window: WINDOW = getattr(cfg, "exclusivity_window")
            scaler: SCALER = getattr(cfg, "frequency_scaler")
            if unit in [
                TailPruningLexicalUnit.NEWLINE,
                TailPruningLexicalUnit.SENTENCE,
            ]:
                attr = "tail_pruning_" + unit.value.lower()
                outcome_list: list = getattr(self, attr)
                outcome_list.append((keep_p, outcome))
            elif unit == TailPruningLexicalUnit.PEAK:
                attr = "tail_pruning_" + unit.value.lower()
                outcome_list: list = getattr(self, attr)
                outcome_list.append((keep_p, window, scaler, outcome))
            else:
                raise ValueError(f"Unsupported: {unit}")
        else:
            raise ValueError(f"Unsupported: {manipulation.name}")


@command(name="postprocess")
class Postprocess:
    def __init__(
        self,
        general: GeneralConfig,
        path: PathConfig,
        datasets: DatasetConfigs,
        transforms: TransformConfigs,
        manipulations: ManipulationConfigs,
    ):
        self.general = general
        self.path = path
        self.stp_db = SliceTransformDatabase(
            general=general,
            path=path,
            datasets=datasets,
            transforms=transforms,
            manipulations=manipulations,
        )
        self.print = ConditionalPrinter(general.verbose)

    def run(self):
        for llm in self.general.llms:
            out_file = self.general.postprocess_file(self.path, llm)
            save_dataclass_jsonl(out_file, *self._process_llm_outputs(llm))

    def _process_llm_outputs(self, llm: Nickname) -> list[StructuredOutcomes]:
        outcomes, slice_names = {}, set()
        self.print(f"Processing: {llm}...")
        for stp in self.stp_db.get_all_slice_transform_products():
            name, path = stp.data_slice.slice_name, stp.transform.get_path_dir()
            scores, lens = [], []
            inferences = self._preprocess_inferences(llm, stp)
            if inferences is None:
                continue
            if name not in slice_names:
                slice_names.add(name)
                self.print(f"    Processing: {name}...")
            for inference in inferences:
                correct_key = DerivedData.CORRECTNESS.to_key()
                scores.append(inference.raw_llm_output.derived_data[correct_key])
                length_key = DerivedData.LENGTH.to_key()
                lens.append(inference.raw_llm_output.derived_data[length_key])
            outcomes.setdefault(name, StructuredOutcomes(name)).add(stp, scores, lens)
        self.print("Done.")
        return list(outcomes.values())

    def _preprocess_inferences(
        self, llm: Nickname, stp: SliceTransformProduct
    ) -> list[Inference] | None:
        in_file = self.general.llm_output_file(self.path, stp, llm)
        if not os.path.exists(in_file):
            return None
        instances = {i.instance_id: i for i in stp.data_slice.instances}
        kept_inferences = []
        for inference in load_dataclass_jsonl(in_file, t=Inference):
            instance = instances[inference.prompt_data.instance_id]
            inference.prompt_data.flow_through_data["instance"] = instance
            result = inference.parser_result
            if result is None:
                continue
            correct = instance.is_correct(result.output)
            correct_key = DerivedData.CORRECTNESS.to_key()
            inference.raw_llm_output.derived_data[correct_key] = correct

            text = inference.raw_llm_output.generated_text
            length = len(text) if text else 0
            length_key = DerivedData.LENGTH.to_key()
            inference.raw_llm_output.derived_data[length_key] = length
            kept_inferences.append(inference)
        return kept_inferences
