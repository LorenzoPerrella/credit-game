"""What a page can place besides a figure: a table, or a single number in running text.

A number on the site is read from a view or from a record a command wrote -- never typed into
a page -- so that regenerating the views cannot leave the prose saying something else. The
pages name them with placeholders, which :mod:`creditsurv.site.hooks` replaces:

    <!-- figure: km_vs_model -->
    <!-- table: acceptance -->
    The backtest's actual over expected is <!-- value: backtest.ae -->.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pandas as pd

from creditsurv import names
from creditsurv.backtest.runner import ACCEPTANCE
from creditsurv.site import figures
from creditsurv.site.figures import (
    SHAPE_PARAMETERS,
    group_key,
    group_label,
    registry,
    segment_title,
)
from creditsurv.views.tables import load_manifest, load_view

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


class Sources(Mapping[str, pd.DataFrame]):
    """The committed views, read once each, and the JSON records beside the reports."""

    def __init__(self, tables: Path, reports: Path) -> None:
        self.tables = tables
        self.reports = reports
        self.manifest = load_manifest(tables)
        self._frames: dict[str, pd.DataFrame] = {}

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self.manifest:
            message = f"No view {name!r} in {self.tables}. Run `uv run creditsurv views`."
            raise KeyError(message)
        if name not in self._frames:
            self._frames[name] = load_view(name, self.tables)
        return self._frames[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.manifest)

    def __len__(self) -> int:
        return len(self.manifest)

    def record(self, filename: str) -> dict[str, object]:
        loaded: dict[str, object] = json.loads((self.reports / filename).read_text())
        return loaded


@dataclass(frozen=True)
class Table:
    name: str
    views: tuple[str, ...]
    build: Callable[[Mapping[str, pd.DataFrame]], pd.DataFrame]


@dataclass(frozen=True)
class Value:
    name: str
    views: tuple[str, ...]
    read: Callable[[Sources], str]


def _mark(passed: object) -> str:
    return "pass" if bool(passed) else "**fail**"


def acceptance(views: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    table = views["acceptance_by_segment"].copy()
    table["order"] = [group_key(group) for group in table["group"]]
    table["segment_order"] = table["segment"].map(
        {name: position for position, name in enumerate(dict.fromkeys(table["segment"]))}
    )
    table = table.sort_values(["segment_order", "order"])
    return pd.DataFrame(
        {
            "Segment": table["segment"].map(segment_title),
            "Group": [
                group_label(segment, group)
                for segment, group in zip(table["segment"], table["group"], strict=True)
            ],
            "Loan-months": table["loan_months"].map("{:,.0f}".format),
            "Defaults": table["defaults"].map("{:,.0f}".format),
            "Actual / expected": table["actual_over_expected"].map("{:.3f}".format),
            "Gini": table["gini"].map("{:.3f}".format),
            "Deciles, lowest to highest": [
                f"{low:.3f} to {high:.3f}"
                for low, high in zip(table["decile_low"], table["decile_high"], strict=True)
            ],
            "Overall": table["overall_passed"].map(_mark),
            "Gini test": table["gini_passed"].map(_mark),
            "Every decile": table["deciles_passed"].map(_mark),
        }
    )


def scenario_summary(views: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    table = views["scenarios_by_segment"]
    table = table[table["segment"] != "all"].copy()
    table["order"] = [group_key(group) for group in table["group"]]
    table = table.sort_values(["segment", "order"])
    return pd.DataFrame(
        {
            "Segment": table["segment"].map(segment_title),
            "Group": [
                group_label(segment, group)
                for segment, group in zip(table["segment"], table["group"], strict=True)
            ],
            "12-month PD": table["pd_12m"].map("{:.2%}".format),
            "Lifetime PD, baseline": table["lifetime_pd_baseline"].map("{:.2%}".format),
            "Lifetime PD, adverse": table["lifetime_pd_adverse"].map("{:.2%}".format),
            "Adverse multiple": table["adverse_multiple"].map("{:.2f}x".format),
        }
    )


def coefficient_table(views: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    table = views["coefficients"]
    return pd.DataFrame(
        {
            "Parameter": table["parameter"].map(lambda value: names.PARAMETERS.get(value, value)),
            "Term": table["term"].map(names.term_label),
            "Coefficient": table["coef"].map("{:+.5f}".format),
            "Standard error": table["se"].map("{:.2e}".format),
            "95% interval": [
                f"{low:+.5f} to {high:+.5f}"
                for low, high in zip(table["lower"], table["upper"], strict=True)
            ],
            "One sd": table["one_sd"].map(lambda v: "" if pd.isna(v) else f"{v:.4f}"),
            "Effect of one sd": table["effect_1sd"].map(
                lambda v: "" if pd.isna(v) else f"{v:+.4f}"
            ),
        }
    )


def elimination(views: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    table = views["selection_elimination"]
    return pd.DataFrame(
        {
            "Step": table["step"],
            "Removed": table["removed"].map(names.label),
            "Rule that fired": table["reason"].map(names.in_words),
            "Covariates left": table["remaining"],
        }
    )


def inflation(views: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    table = views["selection_inflation"]
    return pd.DataFrame(
        {
            "Step": table["step"],
            "Removed": table["removed"].map(names.label),
            "Variance inflation": table["vif"].map("{:.1f}".format),
            "Covariates left": table["remaining"],
        }
    )


def adverse_legs(views: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Built from the scenario itself, so the table cannot describe a path the model never saw."""
    from creditsurv.config import TIME_VARYING_CONTINUOUS
    from creditsurv.features import MACRO_SOURCES
    from creditsurv.models.lifetime_pd import ADVERSE, scenario_legs

    legs = scenario_legs(ADVERSE, {name: MACRO_SOURCES[name] for name in TIME_VARYING_CONTINUOUS})
    return names.readable(legs)


def _glossary(kind: names.Kind) -> Callable[[Mapping[str, pd.DataFrame]], pd.DataFrame]:
    """Four columns, so a name is never broken across lines: values and source join the meaning."""

    def build(views: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        rows = []
        for row in names.glossary([kind]):
            meaning = row["What it is"]
            if row["Values"]:
                meaning += f" Values: {row['Values']}."
            if row["Source"]:
                meaning += f" Source: {row['Source']}."
            name = row["Name"] + (f"<br>formerly {row['Formerly']}" if row["Formerly"] else "")
            rows.append(
                {
                    "Label": row["Label"],
                    "Name": name,
                    "Unit": row["Unit"],
                    "Meaning": meaning.replace("|", "/"),
                }
            )
        return pd.DataFrame(rows)

    return build


TABLES: Final[dict[str, Table]] = registry(
    Table("acceptance", ("acceptance_by_segment",), acceptance),
    Table("scenario_summary", ("scenarios_by_segment",), scenario_summary),
    Table("coefficients", ("coefficients",), coefficient_table),
    Table("selection_elimination", ("selection_elimination",), elimination),
    Table("selection_inflation", ("selection_inflation",), inflation),
    Table("adverse_legs", (), adverse_legs),
    Table("variables_loan", (), _glossary(names.Kind.LOAN)),
    Table("variables_macro", (), _glossary(names.Kind.MACRO)),
    Table("variables_series", (), _glossary(names.Kind.SERIES)),
    Table("variables_structure", (), _glossary(names.Kind.STRUCTURE)),
)


def to_markdown(frame: pd.DataFrame) -> str:
    """A pipe table, so the theme styles it as every other table on the site."""
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.to_numpy()]
    return "\n".join([header, rule, *rows])


# ----- numbers in running text -------------------------------------------------------------


def _whole(sources: Sources, view: str) -> pd.Series:
    table = sources[view]
    row: pd.Series = table[table["segment"] == "all"].iloc[0]
    return row


def _in_sample_years(sources: Sources) -> pd.Series:
    table = sources["ae_by_year"]
    ratios: pd.Series = table[table["segment"] == "all"].set_index("year")["actual_over_expected"]
    return ratios


def _shape(sources: Sources) -> pd.Series:
    table = sources["coefficients"]
    shape = table["parameter"].isin(SHAPE_PARAMETERS) & (table["term"] == "Intercept")
    row: pd.Series = table[shape].iloc[0]
    return row


def _fit(sources: Sources) -> str:
    fits = {entry["fit"] for entry in sources.manifest.values() if entry.get("fit")}
    if len(fits) != 1:
        message = f"The views come from {len(fits)} fits, {sorted(map(str, fits))}; expected one."
        raise ValueError(message)
    return f"`{fits.pop()}`"


def _generated(sources: Sources) -> str:
    return max(str(entry["generated"]) for entry in sources.manifest.values())


def _summary(key: str, shown: Callable[[float], str]) -> Callable[[Sources], str]:
    def read(sources: Sources) -> str:
        value = sources.record("portfolio_summary.json")[key]
        return shown(float(str(value)))

    return read


def _count(value: float) -> str:
    return f"{value:,.0f}"


_CRITERIA: Final = ("overall_passed", "gini_passed", "deciles_passed")


def _verdict(sources: Sources) -> str:
    row = _whole(sources, "acceptance_by_segment")
    failed = sum(not bool(row[criterion]) for criterion in _CRITERIA)
    if failed == 0:
        return "passes all three acceptance criteria"
    return f"fails {failed} of the three acceptance criteria"


def _criterion(column: str) -> Callable[[Sources], str]:
    def read(sources: Sources) -> str:
        return "passes" if bool(_whole(sources, "acceptance_by_segment")[column]) else "fails"

    return read


VALUES: Final[dict[str, Value]] = registry(
    Value("views.fit", (), _fit),
    Value("views.generated", (), _generated),
    Value("figures.floor", (), lambda _: _count(figures.EXPOSURE_FLOOR)),
    Value("model.rho", ("coefficients",), lambda s: f"{math.exp(_shape(s)['coef']):.4f}"),
    Value("model.rho_z", ("coefficients",), lambda s: _count(_shape(s)["coef"] / _shape(s)["se"])),
    Value(
        "backtest.ae",
        ("acceptance_by_segment",),
        lambda s: f"{_whole(s, 'acceptance_by_segment')['actual_over_expected']:.3f}",
    ),
    Value(
        "backtest.gini",
        ("acceptance_by_segment",),
        lambda s: f"{_whole(s, 'acceptance_by_segment')['gini']:.3f}",
    ),
    Value(
        "backtest.deciles",
        ("acceptance_by_segment",),
        lambda s: "{:.3f} to {:.3f}".format(
            *_whole(s, "acceptance_by_segment")[["decile_low", "decile_high"]]
        ),
    ),
    Value(
        "backtest.defaults",
        ("acceptance_by_segment",),
        lambda s: _count(_whole(s, "acceptance_by_segment")["defaults"]),
    ),
    Value(
        "backtest.loan_months",
        ("acceptance_by_segment",),
        lambda s: _count(_whole(s, "acceptance_by_segment")["loan_months"]),
    ),
    Value("backtest.verdict", ("acceptance_by_segment",), _verdict),
    *(
        Value(
            f"backtest.{criterion.removesuffix('_passed')}_verdict",
            ("acceptance_by_segment",),
            _criterion(criterion),
        )
        for criterion in _CRITERIA
    ),
    Value(
        "acceptance.band",
        (),
        lambda _: f"{ACCEPTANCE.ae_low:.2f} to {ACCEPTANCE.ae_high:.2f}",
    ),
    Value("acceptance.gini", (), lambda _: f"{ACCEPTANCE.gini_min:.2f}"),
    Value(
        "in_sample.years",
        ("ae_by_year",),
        lambda s: f"{_in_sample_years(s).min():.2f} to {_in_sample_years(s).max():.2f}",
    ),
    Value(
        "scenario.pd_12m",
        ("scenarios_by_segment",),
        lambda s: f"{_whole(s, 'scenarios_by_segment')['pd_12m']:.2%}",
    ),
    Value(
        "scenario.baseline",
        ("scenarios_by_segment",),
        lambda s: f"{_whole(s, 'scenarios_by_segment')['lifetime_pd_baseline']:.2%}",
    ),
    Value(
        "scenario.adverse",
        ("scenarios_by_segment",),
        lambda s: f"{_whole(s, 'scenarios_by_segment')['lifetime_pd_adverse']:.2%}",
    ),
    Value(
        "scenario.multiple",
        ("scenarios_by_segment",),
        lambda s: f"{_whole(s, 'scenarios_by_segment')['adverse_multiple']:.2f}x",
    ),
    Value("book.loans", (), _summary("loans_originated", _count)),
    Value(
        "book.amount",
        (),
        _summary("amount_originated", lambda v: f"${v / 1e12:.2f} trillion"),
    ),
    Value("book.performance_rows", (), _summary("performance_rows", _count)),
    Value("book.loan_months", (), _summary("loan_months_outstanding", _count)),
    Value("book.peak_contracts", (), _summary("peak_contracts_outstanding", _count)),
    Value(
        "book.peak_balance",
        (),
        _summary("peak_balance_outstanding", lambda v: f"${v / 1e12:.2f} trillion"),
    ),
    Value("model.loan_months", (), _summary("loan_months_modelled", _count)),
    Value("model.defaults", (), _summary("defaults_modelled", _count)),
)
