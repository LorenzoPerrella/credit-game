"""One builder per figure on the site, each reading only the views it names.

A view opened by segment becomes a figure with a menu: one entry per segment, each showing a
trace per group, so the reader picks "Credit score band" and sees a curve per band. The
tables decide which segments and groups exist; a builder never computes a statistic the
table does not carry, and only rescales one -- a survival probability drawn as cumulative
default in percent, a monthly rate in basis points.

Thin tails are cut here, not in the tables: a point is drawn only when it rests on at least
:data:`EXPOSURE_FLOOR` loan-months, and the figure says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, TypeVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from creditsurv import names
from creditsurv.backtest.runner import ACCEPTANCE
from creditsurv.views.segments import SEGMENTS

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

#: Loan-months a point must rest on to be drawn. At 100,000 a monthly default rate of 10 basis
#: points has a standard error of 1 basis point; below it a curve draws the noise of a few
#: hundred loans as if it were a shape.
EXPOSURE_FLOOR: float = 100_000

#: The parameter block a covariate acts on, whatever the family calls it.
SCALE_PARAMETERS: Final = ("lambda_", "alpha_", "mu_")

#: The parameter block holding the shape of the hazard.
SHAPE_PARAMETERS: Final = ("rho_", "beta_", "sigma_")

#: Tableau 10: distinguishable for the common colour-vision deficiencies, and legible on the
#: light and the dark scheme alike.
PALETTE: Final[tuple[str, ...]] = (
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
)

#: Menu order: the whole book, the model's own segments, then the fields only the portfolio
#: is opened by.
_TITLES: Final[dict[str, str]] = {
    "all": "All loans",
    **{name: segment.title for name, segment in SEGMENTS.items()},
    "channel": "Origination channel",
    "region": "Region",
    "property_type": "Property type",
    "vintage_year": "Vintage year",
}

_LEADING_NUMBER: Final = re.compile(r"^-?\d+(\.\d+)?")


@dataclass(frozen=True)
class Figure:
    """A figure the pages can place, and the views it is built from."""

    name: str
    views: tuple[str, ...]
    build: Callable[[Mapping[str, pd.DataFrame]], go.Figure]


class _Named(Protocol):
    @property
    def name(self) -> str: ...


Named = TypeVar("Named", bound=_Named)


def registry(*items: Named) -> dict[str, Named]:
    """Items by name, refusing a second item under a name already taken."""
    found: dict[str, Named] = {}
    for item in items:
        if item.name in found:
            message = f"Two items are named {item.name!r}."
            raise ValueError(message)
        found[item.name] = item
    return found


# ----- shared construction -----------------------------------------------------------------


def segment_title(name: str) -> str:
    """A segment's menu entry: its title, or the label of the variable it was named after."""
    return _TITLES.get(name) or names.label(name)


def group_label(segment: str, group: object) -> str:
    """A group as a reader sees it: a level's label, or the band as it is written.

    Looked up through the variable the segment opens, under its current or former codes, so a
    table written before the rename still reads "Cash-out refinance".
    """
    column = SEGMENTS[segment].columns[0] if segment in SEGMENTS else segment
    return names.level_label(column, group)


def _segment_key(name: str) -> tuple[int, int, str]:
    order = list(_TITLES)
    return (0, order.index(name), name) if name in order else (1, 0, name)


def group_key(label: object) -> tuple[int, float, str]:
    """Bands in the order of their lower edge, names alphabetically, unmapped codes last."""
    text = str(label)
    if text == "not mapped":
        return (2, 0.0, text)
    if text.startswith("up to"):
        return (0, -np.inf, text)
    match = _LEADING_NUMBER.match(text)
    if match:
        return (0, float(match.group()), text)
    return (1, 0.0, text)


def _segments(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["segment"].unique(), key=_segment_key)


def _groups(rows: pd.DataFrame, column: str = "group") -> list[object]:
    return sorted(rows[column].unique(), key=group_key)


def _base(x_title: str, y_title: str, *, height: int = 460) -> go.Figure:
    figure = go.Figure()
    _style(figure, x_title, y_title, height=height)
    return figure


def _style(figure: go.Figure, x_title: str | None, y_title: str | None, *, height: int) -> None:
    grid = "rgba(127, 127, 127, 0.25)"
    figure.update_layout(
        template=go.layout.Template(),
        height=height,
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font={"family": "Roboto, Helvetica, Arial, sans-serif", "size": 13, "color": "#333333"},
        hovermode="closest",
        legend={"orientation": "v", "x": 1.02, "xanchor": "left", "y": 1, "yanchor": "top"},
        colorway=list(PALETTE),
    )
    figure.update_xaxes(gridcolor=grid, zerolinecolor=grid, title_text=x_title, automargin=True)
    # Covariate names are long, and a fixed margin cut them to their last few letters.
    figure.update_yaxes(gridcolor=grid, zerolinecolor=grid, title_text=y_title, automargin=True)


def _menu(figure: go.Figure, choices: Sequence[tuple[str, Sequence[int]]]) -> go.Figure:
    """Show the traces of the first choice, and offer the others in a menu."""
    if not choices:
        message = "Nothing to draw: no group rests on the exposure floor."
        raise ValueError(message)
    total = len(figure.data)

    def visible(members: Sequence[int]) -> list[bool]:
        shown = set(members)
        return [position in shown for position in range(total)]

    for trace, shown in zip(figure.data, visible(choices[0][1]), strict=True):
        trace.visible = shown
    if len(choices) > 1:
        figure.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "label": label,
                            "method": "update",
                            "args": [{"visible": visible(members)}],
                        }
                        for label, members in choices
                    ],
                    "direction": "down",
                    "showactive": True,
                    "x": 0,
                    "xanchor": "left",
                    "y": 1.02,
                    "yanchor": "bottom",
                    "bgcolor": "#ffffff",
                    "bordercolor": "#bbbbbb",
                    "font": {"color": "#333333"},
                }
            ],
        )
        figure.update_layout(margin={"t": 70})
    return figure


def _note(figure: go.Figure, text: str) -> None:
    figure.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=1,
        y=1.02,
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        font={"size": 11},
    )


def _acceptance_band(figure: go.Figure) -> None:
    figure.add_hrect(
        y0=ACCEPTANCE.ae_low,
        y1=ACCEPTANCE.ae_high,
        fillcolor="#59a14f",
        opacity=0.12,
        line_width=0,
    )
    figure.add_hline(y=1.0, line={"color": "#888888", "width": 1, "dash": "dot"})


def _rounded(values: pd.Series) -> pd.Series:
    """Six significant figures.

    A figure's JSON carries every digit it is given, and on a page of segmented monthly series
    the digits past the sixth are most of its weight.
    """
    rounded: pd.Series = values.map(lambda value: float(f"{value:.6g}"))
    return rounded


def _along(rows: pd.DataFrame, x: str) -> pd.DataFrame:
    """Rows in the order of ``x``: bands by their lower edge, not as text sorts them."""
    if rows[x].dtype != object:
        return rows.sort_values(x)
    order = sorted(range(len(rows)), key=lambda position: group_key(rows[x].iloc[position]))
    return rows.iloc[order]


def _categories(figure: go.Figure, frame: pd.DataFrame, x: str) -> None:
    """Fix a text axis to the band order, whichever group happens to be drawn first."""
    if frame[x].dtype == object:
        ordered = sorted(frame[x].unique(), key=group_key)
        figure.update_xaxes(categoryorder="array", categoryarray=ordered)


def _floored(rows: pd.DataFrame, column: str | None) -> pd.DataFrame:
    return rows if column is None else rows[rows[column] >= EXPOSURE_FLOOR]


def _lines_by_segment(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    x_title: str,
    y_title: str,
    scale: float = 1.0,
    exposure: str | None = None,
    mode: str = "lines",
    stack: bool = False,
) -> go.Figure:
    """A trace per group, a menu entry per segment."""
    figure = _base(x_title, y_title)
    choices = []
    for segment in _segments(frame):
        rows = _floored(frame[frame["segment"] == segment], exposure)
        members = []
        for position, group in enumerate(_groups(rows)):
            points = _along(rows[rows["group"] == group], x)
            members.append(len(figure.data))
            figure.add_trace(
                go.Scatter(
                    x=points[x],
                    y=_rounded(points[y] * scale),
                    mode=mode,
                    name=group_label(segment, group),
                    line={"color": PALETTE[position % len(PALETTE)], "width": 1.8},
                    stackgroup=segment if stack else None,
                    hovertemplate=f"{group_label(segment, group)}<br>%{{x}}: %{{y:,.3~f}}"
                    "<extra></extra>",
                )
            )
        if members:
            choices.append((segment_title(segment), members))
    _categories(figure, frame, x)
    return _menu(figure, choices)


def _pairs_by_segment(
    frame: pd.DataFrame,
    x: str,
    observed: str,
    predicted: str,
    *,
    x_title: str,
    y_title: str,
    names: tuple[str, str],
    transform: Callable[[pd.Series], pd.Series],
    exposure: str | None,
    mode: str = "lines",
) -> go.Figure:
    """Observed solid and predicted dashed, in one colour per group, a menu entry per segment."""
    figure = _base(x_title, y_title)
    choices = []
    for segment in _segments(frame):
        rows = _floored(frame[frame["segment"] == segment], exposure)
        members = []
        for position, group in enumerate(_groups(rows)):
            points = _along(rows[rows["group"] == group], x)
            colour = PALETTE[position % len(PALETTE)]
            for column, dash, shown in ((observed, "solid", True), (predicted, "dash", False)):
                label = names[0] if column == observed else names[1]
                members.append(len(figure.data))
                figure.add_trace(
                    go.Scatter(
                        x=points[x],
                        y=_rounded(transform(points[column])),
                        mode=mode,
                        name=group_label(segment, group),
                        legendgroup=f"{segment}:{group}",
                        showlegend=shown,
                        line={"color": colour, "dash": dash, "width": 1.8},
                        hovertemplate=f"{group_label(segment, group)}, {label}<br>"
                        "%{x}: %{y:,.3~f}<extra></extra>",
                    )
                )
        if members:
            choices.append((segment_title(segment), members))
    _note(figure, f"solid: {names[0]} · dashed: {names[1]}")
    _categories(figure, frame, x)
    return _menu(figure, choices)


def _percent_default(survival: pd.Series) -> pd.Series:
    return (1.0 - survival) * 100.0


def _basis_points(rate: pd.Series) -> pd.Series:
    return rate * 1e4


# ----- calibration and backtest ------------------------------------------------------------


def km_vs_model(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    figure = _pairs_by_segment(
        views["km_vs_model"],
        "age",
        "km_survival",
        "predicted_survival",
        x_title="Loan age, months",
        y_title="Cumulative default, %",
        names=("Kaplan-Meier", "model"),
        transform=_percent_default,
        exposure="at_risk",
    )
    return figure


def km_deviation(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    frame = views["km_vs_model"].copy()
    # Model minus Kaplan-Meier in cumulative default: positive where the model expects more.
    frame["gap_pp"] = -frame["deviation"] * 100.0
    figure = _lines_by_segment(
        frame,
        "age",
        "gap_pp",
        x_title="Loan age, months",
        y_title="Model minus Kaplan-Meier, percentage points",
        exposure="at_risk",
    )
    figure.add_hline(y=0.0, line={"color": "#888888", "width": 1, "dash": "dot"})
    return figure


def hazard_by_age(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    return _pairs_by_segment(
        views["km_vs_model"],
        "age",
        "observed_hazard",
        "predicted_hazard",
        x_title="Loan age, months",
        y_title="Monthly default rate, basis points",
        names=("observed", "model"),
        transform=_basis_points,
        exposure="at_risk",
    )


def _ae(
    view: str, x: str, x_title: str, *, mode: str = "lines+markers"
) -> Callable[[Mapping[str, pd.DataFrame]], go.Figure]:
    def build(views: Mapping[str, pd.DataFrame]) -> go.Figure:
        figure = _lines_by_segment(
            views[view],
            x,
            "actual_over_expected",
            x_title=x_title,
            y_title="Actual over expected",
            exposure="exposure",
            mode=mode,
        )
        _acceptance_band(figure)
        return figure

    return build


def _rates(
    view: str, x: str, x_title: str, *, mode: str
) -> Callable[[Mapping[str, pd.DataFrame]], go.Figure]:
    def build(views: Mapping[str, pd.DataFrame]) -> go.Figure:
        return _pairs_by_segment(
            views[view],
            x,
            "actual_rate",
            "expected_rate",
            x_title=x_title,
            y_title="Monthly default rate, basis points",
            names=("actual", "expected"),
            transform=_basis_points,
            exposure="exposure",
            mode=mode,
        )

    return build


def families_vs_km(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    frame = _floored(views["families_vs_km"], "at_risk")
    figure = _base("Loan age, months", "Cumulative default, %")
    reference = frame[frame["distribution"] == frame["distribution"].iloc[0]].sort_values("age")
    level, gap = [], []
    level.append(len(figure.data))
    figure.add_trace(
        go.Scatter(
            x=reference["age"],
            y=_percent_default(reference["km_survival"]),
            name="Kaplan-Meier",
            line={"color": "#333333", "width": 2.2},
        )
    )
    band = pd.concat(
        [_percent_default(reference["km_lower"]), _percent_default(reference["km_upper"])[::-1]]
    )
    level.append(len(figure.data))
    figure.add_trace(
        go.Scatter(
            x=pd.concat([reference["age"], reference["age"][::-1]]),
            y=band,
            fill="toself",
            fillcolor="rgba(127, 127, 127, 0.25)",
            line={"width": 0},
            name="Greenwood 95% band",
            hoverinfo="skip",
        )
    )
    for position, (name, rows) in enumerate(frame.groupby("distribution", sort=True)):
        rows = rows.sort_values("age")
        colour = PALETTE[position % len(PALETTE)]
        level.append(len(figure.data))
        figure.add_trace(
            go.Scatter(
                x=rows["age"],
                y=_percent_default(rows["predicted_survival"]),
                name=names.DISTRIBUTIONS.get(str(name), str(name)),
                line={"color": colour, "dash": "dash", "width": 1.8},
            )
        )
        gap.append(len(figure.data))
        figure.add_trace(
            go.Scatter(
                x=rows["age"],
                y=-rows["deviation"] * 100.0,
                name=names.DISTRIBUTIONS.get(str(name), str(name)),
                line={"color": colour, "width": 1.8},
            )
        )
    return _menu(
        figure,
        [
            ("Cumulative default, %", level),
            ("Model minus Kaplan-Meier, percentage points", gap),
        ],
    )


# ----- the model ---------------------------------------------------------------------------


def coefficients(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    table = views["coefficients"]
    # The block the covariates act on: lambda_ for the Weibull, alpha_ for the log-logistic.
    scale = table[table["parameter"].isin(SCALE_PARAMETERS) & (table["term"] != "Intercept")]
    continuous = scale[scale["one_sd"].notna()].copy()
    categorical = scale[scale["one_sd"].isna()].copy()
    continuous = continuous.reindex(continuous["effect_1sd"].abs().sort_values().index)
    categorical = categorical.reindex(categorical["coef"].abs().sort_values().index)

    figure = _base("Change in log survival time", "", height=520)
    choices = []
    for label, rows, value, lower, upper in (
        (
            "Continuous: one standard deviation",
            continuous,
            continuous["effect_1sd"],
            continuous["lower"] * continuous["one_sd"],
            continuous["upper"] * continuous["one_sd"],
        ),
        (
            "Categorical: each level against its reference",
            categorical,
            categorical["coef"],
            categorical["lower"],
            categorical["upper"],
        ),
    ):
        if rows.empty:
            continue
        colours = np.where(value > 0, PALETTE[0], PALETTE[2])
        choices.append((label, [len(figure.data)]))
        figure.add_trace(
            go.Bar(
                x=value,
                y=[names.term_label(term) for term in rows["term"]],
                orientation="h",
                marker={"color": colours},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": (upper - value).abs(),
                    "arrayminus": (value - lower).abs(),
                    "color": "#888888",
                },
                name=label,
                showlegend=False,
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            )
        )
    figure.update_xaxes(title_text="Change in log survival time: positive lengthens survival")
    return _menu(figure, choices)


def term_structure(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    return _lines_by_segment(
        views["term_structure_by_segment"],
        "month",
        "cumulative_pd",
        x_title="Months from today",
        y_title="Cumulative PD, %",
        scale=100.0,
    )


def scenarios(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    frame = views["scenarios_by_segment"]
    measures = [
        ("pd_12m", "12-month PD", PALETTE[9]),
        ("lifetime_pd_baseline", "Lifetime PD, baseline", PALETTE[0]),
        ("lifetime_pd_adverse", "Lifetime PD, adverse", PALETTE[2]),
    ]
    figure = _base("", "PD, %")
    choices = []
    for segment in _segments(frame):
        rows = frame[frame["segment"] == segment]
        rows = rows.set_index("group").loc[_groups(rows)].reset_index()
        members = []
        for column, name, colour in measures:
            if column not in rows.columns:
                continue
            members.append(len(figure.data))
            figure.add_trace(
                go.Bar(
                    x=[group_label(segment, group) for group in rows["group"]],
                    y=rows[column] * 100.0,
                    name=name,
                    marker={"color": colour},
                    hovertemplate=f"%{{x}}, {name}: %{{y:.2f}}%<extra></extra>",
                )
            )
        choices.append((segment_title(segment), members))
    figure.update_layout(barmode="group")
    return _menu(figure, choices)


def covariates_over_time(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    frame = views["covariates_over_time"]
    figure = _base("Month", "Exposure-weighted mean")
    choices = []
    for name, rows in frame.groupby("covariate", sort=True):
        rows = rows.sort_values("month")
        choices.append((names.label(str(name)), [len(figure.data)]))
        figure.add_trace(
            go.Scatter(
                x=rows["month"],
                y=rows["mean"],
                name=names.label(str(name)),
                line={"color": PALETTE[0], "width": 1.8},
                showlegend=False,
            )
        )
    return _menu(figure, choices)


# ----- the portfolio -----------------------------------------------------------------------


def default_rate(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    return _lines_by_segment(
        views["book_by_segment"],
        "month",
        "default_rate_bp",
        x_title="Month",
        y_title="Monthly default rate, basis points",
        exposure="loan_months",
    )


def prepayment_rate(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    return _lines_by_segment(
        views["book_by_segment"],
        "month",
        "cpr",
        x_title="Month",
        y_title="Conditional prepayment rate, % a year",
        scale=100.0,
        exposure="loan_months",
    )


def outstanding(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    return _lines_by_segment(
        views["book_by_segment"],
        "month",
        "loan_months",
        x_title="Month",
        y_title="Loans outstanding",
        stack=True,
    )


def lending_volume(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    frame = views["lending_by_segment"]
    whole = frame[frame["segment"] == "all"].sort_values("year")
    # Two panels rather than two y-axes: a count and an amount share no scale.
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    _style(figure, None, None, height=520)
    figure.add_trace(
        go.Bar(x=whole["year"], y=whole["loans"] / 1e6, name="Loans", marker={"color": PALETTE[0]}),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=whole["year"], y=whole["amount"] / 1e9, name="Amount", marker={"color": PALETTE[1]}
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="Loans, millions", row=1, col=1)
    figure.update_yaxes(title_text="Amount, $ billion", row=2, col=1)
    figure.update_xaxes(title_text="Vintage year", row=2, col=1)
    figure.update_layout(showlegend=False)
    return figure


def lending_mix(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    frame = views["lending_by_segment"]
    frame = frame[frame["segment"] != "all"]
    figure = _base("Vintage year", "Share of the year's loans, %")
    choices = []
    for segment in _segments(frame):
        rows = frame[frame["segment"] == segment]
        members = []
        for position, group in enumerate(_groups(rows)):
            points = rows[rows["group"] == group].sort_values("year")
            members.append(len(figure.data))
            figure.add_trace(
                go.Bar(
                    x=points["year"],
                    y=points["loan_share"] * 100.0,
                    name=group_label(segment, group),
                    marker={"color": PALETTE[position % len(PALETTE)]},
                    hovertemplate=f"{group_label(segment, group)}, %{{x}}: %{{y:.1f}}%"
                    "<extra></extra>",
                )
            )
        choices.append((segment_title(segment), members))
    figure.update_layout(barmode="stack")
    return _menu(figure, choices)


def underwriting(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    frame = views["underwriting_by_vintage"]
    names = {
        "score": "Credit score",
        "ltv": "Loan-to-value, %",
        "dti": "Debt-to-income, %",
    }
    figure = _base("Vintage year", "Median and interquartile range")
    choices = []
    for measure in (name for name in names if name in set(frame["measure"])):
        rows = frame[frame["measure"] == measure].sort_values("year")
        members = [len(figure.data), len(figure.data) + 1]
        figure.add_trace(
            go.Scatter(
                x=pd.concat([rows["year"], rows["year"][::-1]]),
                y=pd.concat([rows["q25"], rows["q75"][::-1]]),
                fill="toself",
                fillcolor="rgba(78, 121, 167, 0.25)",
                line={"width": 0},
                name="25th to 75th percentile",
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=rows["year"],
                y=rows["q50"],
                name="median",
                line={"color": PALETTE[0], "width": 2},
                hovertemplate="%{x}: %{y:.1f}<extra></extra>",
            )
        )
        choices.append((names[measure], members))
    return _menu(figure, choices)


def vintage_curves(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    frame = _floored(views["vintage_curves"], "at_risk")
    figure = _base("Loan age, months", "Cumulative default, %")
    years = sorted(frame["vintage_year"].unique())
    shades = _sequential(len(years))
    for year, colour in zip(years, shades, strict=True):
        rows = frame[frame["vintage_year"] == year].sort_values("age")
        figure.add_trace(
            go.Scatter(
                x=rows["age"],
                y=rows["cumulative_default_pct"],
                name=str(year),
                line={"color": colour, "width": 1.6},
                hovertemplate=f"{year}, age %{{x}}: %{{y:.2f}}%<extra></extra>",
            )
        )
    return figure


def _sequential(count: int) -> list[str]:
    """Evenly spaced colours from blue through yellow to red, old vintages to new."""
    from plotly.colors import sample_colorscale

    points = [position / max(count - 1, 1) for position in range(count)]
    return list(sample_colorscale("Turbo", points))


def macro_series(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    frame = views["macro_series"]
    figure = _base("Month", "")
    choices = []
    for name, rows in frame.groupby("series", sort=True):
        rows = rows.sort_values("month")
        shown = names.label(str(name), kind=names.Kind.SERIES)
        choices.append((shown, [len(figure.data)]))
        figure.add_trace(
            go.Scatter(
                x=rows["month"],
                y=rows["value"],
                name=shown,
                showlegend=False,
                line={"color": PALETTE[0], "width": 1.8},
            )
        )
    return _menu(figure, choices)


# ----- the selection -----------------------------------------------------------------------


def selection_correlation(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    long = views["selection_correlation"]
    order = list(dict.fromkeys(long["first"]))
    wide = long.pivot(index="first", columns="second", values="correlation").loc[order, order]
    figure = _base("", "", height=620)
    figure.add_trace(
        go.Heatmap(
            z=wide.to_numpy(),
            x=[names.label(name) for name in order],
            y=[names.label(name) for name in order],
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            hovertemplate="%{y} and %{x}: %{z:.2f}<extra></extra>",
        )
    )
    # Every covariate named: left to itself the axis shows every other one.
    figure.update_xaxes(dtick=1)
    figure.update_yaxes(autorange="reversed", dtick=1)
    return figure


def selection_screening(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    table = views["selection_screening"]
    table = table.reindex(table["effect_1sd"].abs().sort_values().index)
    agrees = table["sign_agrees"].astype(bool)
    figure = _base("Effect of one standard deviation on log survival time", "", height=560)
    for shown, name, colour in (
        (True, "sign as expected", PALETTE[0]),
        (False, "sign against", PALETTE[2]),
    ):
        rows = table[agrees == shown]
        figure.add_trace(
            go.Bar(
                x=rows["effect_1sd"],
                y=[names.label(str(name)) for name in rows["covariate"]],
                orientation="h",
                name=name,
                marker={"color": colour},
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            )
        )
    figure.update_layout(barmode="overlay")
    return figure


def selection_stability(views: Mapping[str, pd.DataFrame]) -> go.Figure:
    table = views["selection_stability"]
    figure = _base("Effect of one standard deviation on log survival time", "", height=560)
    choices = []
    for round_number, rows in table.groupby("round", sort=True):
        members = []
        for column, name, colour, symbol in (
            ("effect_all", "training half", "#333333", "circle"),
            ("effect_even", "even vintage years", PALETTE[0], "triangle-right"),
            ("effect_odd", "odd vintage years", PALETTE[1], "triangle-left"),
        ):
            members.append(len(figure.data))
            figure.add_trace(
                go.Scatter(
                    x=rows[column],
                    y=[names.label(str(name)) for name in rows["covariate"]],
                    mode="markers",
                    name=name,
                    marker={"color": colour, "symbol": symbol, "size": 10},
                    hovertemplate=f"%{{y}}, {name}: %{{x:.3f}}<extra></extra>",
                )
            )
        choices.append((f"Round {round_number}", members))
    figure.add_vline(x=0.0, line={"color": "#888888", "width": 1, "dash": "dot"})
    return _menu(figure, choices)


FIGURES: Final[dict[str, Figure]] = registry(
    Figure("km_vs_model", ("km_vs_model",), km_vs_model),
    Figure("km_deviation", ("km_vs_model",), km_deviation),
    Figure("hazard_by_age", ("km_vs_model",), hazard_by_age),
    Figure("ae_by_year", ("ae_by_year",), _ae("ae_by_year", "year", "Calendar year")),
    Figure(
        "ae_by_vintage",
        ("ae_by_vintage",),
        _ae("ae_by_vintage", "vintage_year", "Vintage year"),
    ),
    Figure(
        "ae_by_age_band",
        ("ae_by_age_band",),
        _ae("ae_by_age_band", "age_band", "Loan age band, months"),
    ),
    Figure(
        "ae_by_decile",
        ("ae_by_decile",),
        _rates("ae_by_decile", "decile", "Decile of predicted risk", mode="lines+markers"),
    ),
    Figure("families_vs_km", ("families_vs_km",), families_vs_km),
    Figure(
        "backtest_by_month",
        ("backtest_by_month",),
        _rates("backtest_by_month", "month", "Month", mode="lines"),
    ),
    Figure(
        "backtest_ae_by_month",
        ("backtest_by_month",),
        _ae("backtest_by_month", "month", "Month", mode="lines"),
    ),
    Figure(
        "backtest_by_decile",
        ("backtest_by_decile",),
        _rates("backtest_by_decile", "decile", "Decile of predicted risk", mode="lines+markers"),
    ),
    Figure("coefficients", ("coefficients",), coefficients),
    Figure("term_structure", ("term_structure_by_segment",), term_structure),
    Figure("scenarios", ("scenarios_by_segment",), scenarios),
    Figure("covariates_over_time", ("covariates_over_time",), covariates_over_time),
    Figure("default_rate", ("book_by_segment",), default_rate),
    Figure("prepayment_rate", ("book_by_segment",), prepayment_rate),
    Figure("outstanding", ("book_by_segment",), outstanding),
    Figure("lending_volume", ("lending_by_segment",), lending_volume),
    Figure("lending_mix", ("lending_by_segment",), lending_mix),
    Figure("underwriting", ("underwriting_by_vintage",), underwriting),
    Figure("vintage_curves", ("vintage_curves",), vintage_curves),
    Figure("macro_series", ("macro_series",), macro_series),
    Figure("selection_correlation", ("selection_correlation",), selection_correlation),
    Figure("selection_screening", ("selection_screening",), selection_screening),
    Figure("selection_stability", ("selection_stability",), selection_stability),
)


def render(figure: go.Figure, name: str) -> str:
    """The figure as an HTML fragment, with plotly.js left to the page."""
    html: str = figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=f"figure-{name}",
        default_height=f"{figure.layout.height or 460}px",
        config={"displaylogo": False, "responsive": True},
    )
    return f'<div class="creditsurv-figure">{html}</div>'
