from dataclasses import dataclass
from enum import Enum

from ...core import PromptKey
from ...io import PathConfig, load_dataclass_jsonl

from ..base import (
    DatasetConfig,
    DatasetManager,
    DatasetName,
    DataSlice,
    DataSliceManager,
    Instance,
    SliceName,
)

from .dataclasses import AccordInstance, CaseLink, CsqaBase, MetaData, SurfaceForms
from .surfacer import (
    InstanceSurfacer,
    OrderingSurfacer,
    QADataSurfacer,
    StatementSurfacer,
    TermSurfacer,
)


@dataclass
class AccordConfig(DatasetConfig):
    pass


class AccordFactuality(Enum):
    F = "F"
    AF = "AF"

    def is_factual(self) -> bool:
        return self == AccordFactuality.F


class AccordSubset(Enum):
    BASELINE = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5

    def get_meta_data_file(self, path: PathConfig) -> str:
        if self == AccordSubset.BASELINE:
            return path.accord_baseline_file
        elif self == AccordSubset.ONE:
            return path.accord_tree_size_1_file
        elif self == AccordSubset.TWO:
            return path.accord_tree_size_2_file
        elif self == AccordSubset.THREE:
            return path.accord_tree_size_3_file
        elif self == AccordSubset.FOUR:
            return path.accord_tree_size_4_file
        elif self == AccordSubset.FIVE:
            return path.accord_tree_size_5_file
        else:
            raise ValueError(f"Unsupported ACCORD subset: {self}")

    def get_reductions_file(self, path: PathConfig) -> str | None:
        return None if self == AccordSubset.BASELINE else path.accord_reductions_file

    def get_factuality_options(self) -> list[AccordFactuality]:
        if self == AccordSubset.BASELINE:
            return [AccordFactuality.F]
        return [AccordFactuality.F, AccordFactuality.AF]


class AccordDataSlice(DataSlice):
    def __init__(
        self,
        name: SliceName,
        dataset_name: DatasetName,
        subset: AccordSubset,
        factuality: AccordFactuality,
        instances: list[AccordInstance],
    ):
        wrapped = [
            Instance(
                instance_id=i.meta_data.id,
                core_text=i.text,
                flow_through_data=dict(
                    answer_choices=i.meta_data.answer_choices.copy(),
                    csqa_label=i.csqa_label,
                ),
            )
            for i in instances
        ]
        super().__init__(name, dataset_name, wrapped)
        self.subset = subset
        self.factuality = factuality

    def get_prompt_key(self) -> PromptKey:
        subset_key = "baseline" if self.subset == AccordSubset.BASELINE else "general"
        return f"{self.dataset_name.value.lower()}_{subset_key}"


class AccordLoader:
    def __init__(
        self,
        slice_name: SliceName,
        subset: AccordSubset,
        factuality: AccordFactuality,
        path: PathConfig,
    ):
        self.slice_name = slice_name
        self.subset = subset
        self.factuality = factuality
        self.data_file = subset.get_meta_data_file(path)
        self.forms_file = subset.get_reductions_file(path)
        self.csqa_file = path.accord_csqa_file
        self.surfacer: InstanceSurfacer | None = None
        self.is_factual = factuality.is_factual()

    def load(self) -> AccordDataSlice:
        self.surfacer = self._create_surfacer()
        meta_datas = load_dataclass_jsonl(self.data_file, t=MetaData)
        csqa = load_dataclass_jsonl(self.csqa_file, t=CsqaBase)
        csqa = {instance.identifier: instance for instance in csqa}
        instances = [
            AccordInstance(self.surfacer(md), csqa[md.qa_id].correct_answer_label, md)
            for md in meta_datas
        ]
        instances = [i for i in instances if i.is_factual() == self.is_factual]
        return AccordDataSlice(
            name=self.slice_name,
            dataset_name=DatasetName.ACCORD,
            subset=self.subset,
            factuality=self.factuality,
            instances=instances,
        )

    def _load_surface_forms(self) -> SurfaceForms:
        """
        Loads and registers CaseLink subsumed surface forms from a CSV-formatted file.

        Empty lines are skipped, as are lines starting with the comment string.

        Relevant headers from the CSV file are:
            "relation1_or_pairing" -> CaseLink.r1_type
            "relation2" -> CaseLink.r2_type
            "case" -> CaseLink.case
            "relation2_surface_form" -> subsumed surface form for associated CaseLink
        """
        line_comment = "#"
        with open(self.forms_file, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            lines = [
                line
                for line in f.readlines()
                if line.strip() and not line.strip().startswith(line_comment)
            ]
        forms = SurfaceForms()
        for r in [dict(zip(header, line.strip().split(","))) for line in lines]:
            cl = CaseLink(r["relation1_or_pairing"], r["relation2"], int(r["case"]))
            forms.register(cl, r["relation2_surface_form"])
        return forms

    def _create_surfacer(self) -> InstanceSurfacer:
        """
        Creates an InstanceSurfacer with all config parameters matching those used
        to generate and evaluate ACCORD_CSQA from the ACCORD paper.
        """
        forms = None if self.forms_file is None else self._load_surface_forms()
        ordering_surfacer = None
        if forms is not None:
            ordering_surfacer = OrderingSurfacer(
                prefix="Statements:\n",
                statement_separator="\n",
                statement_surfacer=StatementSurfacer(
                    prefix="- ",
                    term_surfacer=TermSurfacer(prefix="[", suffix="]"),
                    forms=forms,
                ),
            )
        return InstanceSurfacer(
            prefix="",
            surfacer_separator="\n",
            prefix_surfacer=None,
            ordering_surfacer=ordering_surfacer,
            qa_data_surfacer=QADataSurfacer(
                prefix="Question:\n",
                question_answer_separator="\n",
                answer_choice_separator="    ",
                answer_choice_formatter="{}: {}",
            ),
            suffix_surfacer=None,
        )


@dataclass
class AccordSliceManager(DataSliceManager):
    subset: AccordSubset
    factuality: AccordFactuality
    path: PathConfig

    def load(self, *_, **__) -> DataSlice:
        return AccordLoader(self.name, self.subset, self.factuality, self.path).load()


class AccordDatasetManager(DatasetManager):
    def __init__(self, *_, path: PathConfig, cfg: AccordConfig, **__):
        slices = []
        for subset in AccordSubset:
            for factuality in subset.get_factuality_options():
                data_slice = AccordSliceManager(
                    name=f"{factuality.value}{subset.value}",
                    subset=subset,
                    factuality=factuality,
                    path=path,
                )
                slices.append(data_slice)
        super().__init__(cfg=cfg, data_slices=slices)
