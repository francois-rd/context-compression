from enum import Enum

from coma import command
from dacite import Config
import plotly.graph_objects as go

from ...datasets import DatasetName
from ...io import PathConfig, load_dataclass_jsonl
from ...manipulations import (
    ManipulationName,
    TailPruningLexicalUnit,
    TokenSkipLexicalUnit,
)

from .base import GeneralConfig
from .postprocess import KEEP_P, WINDOW, SCALER, Outcome, StructuredOutcomes


def maybe_init_fig(
    fig: go.Figure | None = None,
    title: str = "Title Goes Here",
    re_init: bool = False,
) -> go.Figure:
    if fig is None or re_init:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            xaxis_title="Compression Rate (%)",
            yaxis_title="Accuracy (%)",
            template="simple_white",
        )
        fig.update_yaxes(range=[0, 100])
    return fig


def add_horizontal_baselines(fig, outcome: Outcome, name: str, dash: str) -> None:
    if outcome is None:
        return None
    x, (low_ci, high_ci) = [0.0, 100.0], outcome.acc_ci
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],  # Forward, then reverse.
            y=[high_ci, high_ci] + [low_ci, low_ci],  # High, then low.
            fill="toself",  # Connect together.
            fillcolor="rgba(0, 0, 0, 0.10)",  # Low opacity fill colour.
            line=dict(color="rgba(255, 255, 255, 0)"),  # 0 opacity border.
            showlegend=False,  # Don't show the fill in the legend.
            name=name,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[outcome.accuracy, outcome.accuracy],
            mode="lines",
            line=dict(
                color="rgba(0, 0, 0, 0.25)",  # Same colour but more opacity.
                width=2,
                dash=dash,
            ),
            name=name,
        )
    )


def add_data_by_keep_p(
    fig,
    outcomes_by_keep_p: list[tuple[KEEP_P, dict]],
    name: str,
    color: str | None,
    dash: str | None,
    symbol: str | None,
) -> bool:
    if len(outcomes_by_keep_p) == 0:
        return False
    x_axis, y_axis, low_cis, high_cis = [], [], [], []
    for keep_p, outcome in outcomes_by_keep_p:
        outcome = Outcome(**outcome)
        x_axis.append(100 - keep_p)  # Convert keep into reject.
        y_axis.append(outcome.accuracy)
        low_ci, high_ci = outcome.acc_ci
        low_cis.append(outcome.accuracy - low_ci)
        high_cis.append(high_ci - outcome.accuracy)
    line = dict(width=2)
    if color is not None:
        line["color"] = color
    if dash is not None:
        line["dash"] = dash
    kwargs = dict(
        error_y=dict(
            type="data",
            symmetric=False,
            array=high_cis,
            arrayminus=low_cis,
        ),
        name=name,
        line=line,
    )
    if symbol is not None:
        kwargs.update(dict(mode="lines+markers", marker=dict(symbol=symbol)))
    fig.add_trace(go.Scatter(x=x_axis, y=y_axis, **kwargs))
    return True


def get_tail_pruning_outcome_by_keep_p(
    outcomes: StructuredOutcomes,
    unit: TailPruningLexicalUnit,
    select_window: WINDOW,
    select_scaler: SCALER,
) -> tuple[str, list[tuple[KEEP_P, dict]]]:
    tail_pruning_attr = ManipulationName.TAIL_PRUNING.value.lower()
    unit_attr = unit.value.lower()
    attr = f"{tail_pruning_attr}_{unit_attr}"
    outcome_list = getattr(outcomes, attr)
    if unit != TailPruningLexicalUnit.PEAK:
        return attr, outcome_list
    outcome_by_keep_p = []
    for keep_p, window, scaler, outcome in outcome_list:
        if window == select_window and scaler == select_scaler:
            outcome_by_keep_p.append((keep_p, outcome))
    return attr, outcome_by_keep_p


def compare_random_and_token_skip_and_tail_pruning(
    outcomes: StructuredOutcomes,
    dataset: DatasetName,
    include_random: bool = False,
    include_token_skip: bool = False,
    include_tail_pruning: bool = False,
    select_window: WINDOW = 10,
    select_scaler: SCALER = 1.0,
    split_units: bool = False,
) -> None:
    if not include_random and not include_token_skip and not include_tail_pruning:
        raise ValueError("Include at least one: [random, token skip, or tail pruning]")

    fig = None
    added_attrs = set()
    tail_pruning_name = ManipulationName.TAIL_PRUNING.value
    token_skip_name = ManipulationName.TOKEN_SKIP.value
    dash = {tail_pruning_name: None, token_skip_name: "dash", "random": "dot"}
    symbol = {
        tail_pruning_name: "circle",
        token_skip_name: "square",
        "random": "diamond",
    }
    color = {
        tail_pruning_name: None,  # "#1F77B4",
        token_skip_name: None,  # "#FF7F0E",
        "random": None,  # "#4D9900",
    }
    tail_pruning_to_random_and_token_unit_map = {
        TailPruningLexicalUnit.NEWLINE: TokenSkipLexicalUnit.TOKEN,
        TailPruningLexicalUnit.SENTENCE: TokenSkipLexicalUnit.TOKEN,
        TailPruningLexicalUnit.PEAK: TokenSkipLexicalUnit.TOKEN,
    }
    unit_type = TailPruningLexicalUnit if include_tail_pruning else TokenSkipLexicalUnit
    for unit in unit_type:
        title = f"{dataset.value.title()}/{outcomes.slice_name}"
        if split_units:
            title += f"/{unit.value.title()}"
        fig = maybe_init_fig(fig=fig, title=title, re_init=split_units)
        if include_tail_pruning:
            attr_name, outcome_by_keep_p = get_tail_pruning_outcome_by_keep_p(
                outcomes, unit, select_window, select_scaler
            )
            if split_units or attr_name not in added_attrs:
                added_tail_pruning = add_data_by_keep_p(
                    fig,
                    outcome_by_keep_p,
                    name=f"{tail_pruning_name.title()}/{unit.value.title()}",
                    color=color[tail_pruning_name],
                    dash=dash[tail_pruning_name],
                    symbol=symbol[tail_pruning_name],
                )
                added_attrs.add(attr_name)
            else:
                added_tail_pruning = True
            map_unit = tail_pruning_to_random_and_token_unit_map[unit]
        else:
            added_tail_pruning = True
            map_unit = unit
        if include_token_skip:
            attr_name = f"{token_skip_name.lower()}_{map_unit.value.lower()}"
            if split_units or attr_name not in added_attrs:
                added_token_skip = add_data_by_keep_p(
                    fig,
                    getattr(outcomes, attr_name),
                    name=f"{token_skip_name.title()}/{map_unit.value.title()}",
                    color=color[token_skip_name],
                    dash=dash[token_skip_name],
                    symbol=symbol[token_skip_name],
                )
                added_attrs.add(attr_name)
            else:
                added_token_skip = True
        else:
            added_token_skip = True
        if include_random:
            attr_name = f"random_{map_unit.value.lower()}"
            if split_units or attr_name not in added_attrs:
                added_random = add_data_by_keep_p(
                    fig,
                    getattr(outcomes, attr_name),
                    name=f"Random/{map_unit.value.title()}",
                    color=color["random"],
                    dash=dash["random"],
                    symbol=symbol["random"],
                )
                added_attrs.add(attr_name)
            else:
                added_random = True
        else:
            added_random = True
        if split_units:
            add_horizontal_baselines(
                fig, outcomes.zero_cot, name="Zero CoT", dash="dash"
            )
            add_horizontal_baselines(fig, outcomes.plain, name="No CoT", dash="dot")
            if added_tail_pruning and added_token_skip and added_random:
                fig.show()
    if not split_units:
        add_horizontal_baselines(fig, outcomes.zero_cot, name="Zero CoT", dash="dash")
        add_horizontal_baselines(fig, outcomes.plain, name="No CoT", dash="dot")
        fig.show()


@command(name="analyze")
class Analyze:
    def __init__(self, general: GeneralConfig, path: PathConfig):
        self.general = general
        self.path = path
        if general.active_dataset == DatasetName.MISSING:
            raise ValueError(f"Missing runtime dataset: {general.active_dataset}")

    def run(self):
        for llm in self.general.llms:
            in_file = self.general.postprocess_file(self.path, llm)
            outcomes_per_slice = load_dataclass_jsonl(
                in_file,
                t=StructuredOutcomes,
                dacite_config=Config(cast=[Enum, tuple], check_types=False),
            )
            for outcomes in outcomes_per_slice:
                self._do_process_outcomes(outcomes)

    def _do_process_outcomes(self, outcomes: StructuredOutcomes) -> None:
        compare_random_and_token_skip_and_tail_pruning(
            outcomes,
            self.general.active_dataset,
            include_tail_pruning=True,
            include_token_skip=True,
            include_random=True,
            select_window=10,
            select_scaler=1.0,
            split_units=True,
        )
