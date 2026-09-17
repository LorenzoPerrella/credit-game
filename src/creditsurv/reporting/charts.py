"""Figures for the generated reports.

Every figure ships beside the table it draws, which is deliberate. Two of the
palette hues sit below a 3:1 contrast ratio on a light surface, so identity is
never left to colour alone: each chart carries a legend, and the numbers are
always available in the accompanying table.

Two rules do most of the work here. Quantities on different scales go in separate
panels rather than onto a second y-axis -- a dual axis lets the author choose the
apparent relationship between two series by picking the scales. And categorical
hues are assigned in fixed order rather than cycled, so a series keeps its colour
when the number of series changes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # No display in CI, and none needed to write a file.

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from lifelines import KaplanMeierFitter
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Categorical slots, in fixed assignment order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e3e2df"

_LINE_WIDTH = 2.0
_MARKER_SIZE = 6.0


def _style(axis: Axes, *, title: str, xlabel: str, ylabel: str) -> None:
    """Recessive frame: the data should be the only assertive thing on the page."""
    axis.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    axis.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    axis.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    axis.tick_params(colors=INK_MUTED, labelsize=8.5)
    axis.grid(visible=True, color=GRID, linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)


def _save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=144, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    return path


def survival_vs_kaplan_meier(curve: KaplanMeierFitter, predicted: pd.Series, path: Path) -> Path:
    """Fitted survival against the non-parametric estimate and its band."""
    figure, axis = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)

    band = curve.confidence_interval_survival_function_
    axis.fill_between(
        band.index,
        band.iloc[:, 0],
        band.iloc[:, 1],
        color=SERIES[0],
        alpha=0.16,
        linewidth=0,
        label="Kaplan-Meier 95% band",
    )
    axis.plot(
        curve.survival_function_.index,
        curve.survival_function_.iloc[:, 0],
        color=SERIES[0],
        linewidth=_LINE_WIDTH,
        label="Kaplan-Meier",
    )
    axis.plot(
        predicted.index,
        predicted.to_numpy(),
        color=SERIES[1],
        linewidth=_LINE_WIDTH,
        linestyle=(0, (5, 2)),
        label="Fitted model",
    )

    _style(
        axis,
        title="Survival: fitted model against Kaplan-Meier",
        xlabel="Loan age (months)",
        ylabel="Share still performing",
    )
    axis.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="lower left")
    return _save(figure, path)


def term_structure(table: pd.DataFrame, path: Path) -> Path:
    """Cumulative and marginal PD, in two panels.

    Two panels rather than two axes: the quantities differ by an order of
    magnitude, and putting them on a shared frame with separate scales would let
    the choice of scales decide how related they appear.
    """
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True, facecolor=SURFACE)
    for axis in (top, bottom):
        axis.set_facecolor(SURFACE)

    top.plot(table.index, table["cumulative_pd"], color=SERIES[0], linewidth=_LINE_WIDTH)
    _style(top, title="Cumulative probability of default", xlabel="", ylabel="Cumulative PD")

    bottom.plot(table.index, table["marginal_pd"], color=SERIES[1], linewidth=_LINE_WIDTH)
    _style(
        bottom,
        title="Marginal probability of default by month",
        xlabel="Months since reporting date",
        ylabel="Marginal PD",
    )
    figure.tight_layout()
    return _save(figure, path)


def calibration(table: pd.DataFrame, path: Path) -> Path:
    """Realised against predicted default rate, by bucket of predicted risk."""
    figure, axis = plt.subplots(figsize=(5.6, 5.2), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)

    limit = float(max(table["expected"].max(), table["actual"].max())) * 1.15
    axis.plot(
        [0, limit],
        [0, limit],
        color=INK_MUTED,
        linewidth=1.0,
        linestyle=(0, (3, 3)),
        label="Perfect calibration",
    )
    axis.scatter(
        table["expected"],
        table["actual"],
        s=_MARKER_SIZE**2,
        color=SERIES[0],
        zorder=3,
        label="Decile of predicted PD",
        edgecolor=SURFACE,
        linewidth=1.5,
    )

    _style(
        axis,
        title="Calibration: realised against predicted default rate",
        xlabel="Predicted PD",
        ylabel="Realised default rate",
    )
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="upper left")
    return _save(figure, path)


_TREATMENT = re.compile(r"C\((\w+),\s*Treatment\('[^']*'\)\)\[T\.([^\]]+)\]")


def _label(parameter: str, covariate: str) -> str:
    """Turn a formulaic term into something readable on an axis.

    ``C(purpose, Treatment('purchase'))[T.cash_out_refinance]`` carries the encoding
    scheme, the reference level and the level itself. Only the last two are news
    once the reference is stated in the caption.
    """
    match = _TREATMENT.match(covariate)
    readable = f"{match.group(1)}: {match.group(2)}" if match else covariate
    # Two parameter blocks each own an intercept, so the block has to be named.
    return f"{parameter.rstrip('_')}: {readable}" if readable == "Intercept" else readable


def coefficients(table: pd.DataFrame, path: Path) -> Path:
    """Covariate effects with confidence intervals, largest first.

    Intercepts are excluded. They are not covariate effects, and on this model the
    scale intercept sits near 6 while every covariate is inside +/-0.5 -- plotted
    together, the covariates compress into an unreadable sliver against the axis.
    They appear in the coefficient table instead, where the scale does not matter.
    """
    covariates = table[[str(name[-1]) != "Intercept" for name in table.index]]
    ordered = covariates.reindex(covariates["coef"].abs().sort_values().index)
    labels = [_label(str(name[0]), str(name[-1])) for name in ordered.index]
    positions = np.arange(len(ordered))

    figure, axis = plt.subplots(figsize=(7.4, 0.34 * len(ordered) + 1.6), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)

    lower = ordered["coef"] - ordered["coef lower 95%"]
    upper = ordered["coef upper 95%"] - ordered["coef"]
    axis.errorbar(
        ordered["coef"],
        positions,
        xerr=[lower, upper],
        fmt="o",
        color=SERIES[0],
        ecolor=SERIES[0],
        elinewidth=1.6,
        markersize=_MARKER_SIZE,
        capsize=3,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
    )
    axis.axvline(0.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)))

    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8.5, color=INK_MUTED)
    axis.set_ylim(-0.7, len(ordered) - 0.3)
    _style(
        axis,
        title="Covariate effects on log survival time, with 95% intervals",
        xlabel="Coefficient (positive lengthens survival, lowering risk)",
        ylabel="",
    )
    return _save(figure, path)


def scenario_comparison(pds: pd.DataFrame, path: Path) -> Path:
    """Distribution of lifetime PD under each macro scenario."""
    figure, axis = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)

    for slot, column in enumerate(pds.columns):
        values = np.sort(pds[column].to_numpy(dtype=float))
        share = np.linspace(0, 1, len(values))
        axis.plot(
            values,
            share,
            color=SERIES[slot % len(SERIES)],
            linewidth=_LINE_WIDTH,
            label=str(column).capitalize(),
        )

    _style(
        axis,
        title="Lifetime PD by macroeconomic scenario",
        xlabel="Lifetime probability of default",
        ylabel="Share of loans at or below",
    )
    axis.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="lower right")
    return _save(figure, path)


def backtest_over_time(table: pd.DataFrame, path: Path) -> Path:
    """Predicted against realised default rate, month by month of the test window.

    The one picture the whole backtest reduces to. Two series on one axis because
    both are monthly hazards in the same units -- which is the case a shared axis is
    actually for, and the gap between them is the result.
    """
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.4, 5.2), sharex=True, facecolor=SURFACE, height_ratios=[2, 1]
    )
    time = _timeline(top, table["group"])

    top.set_facecolor(SURFACE)
    top.plot(
        time,
        table["expected_rate"] * 1e4,
        color=SERIES[0],
        linewidth=_LINE_WIDTH,
        label="Predicted",
    )
    top.plot(
        time,
        table["actual_rate"] * 1e4,
        color=SERIES[1],
        linewidth=_LINE_WIDTH,
        label="Realised",
    )
    _style(top, title="Monthly default rate over the test window", xlabel="", ylabel="Basis points")
    top.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="upper left")

    bottom.set_facecolor(SURFACE)
    bottom.plot(time, table["actual_over_expected"], color=SERIES[2], linewidth=_LINE_WIDTH)
    bottom.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)))
    _style(bottom, title="", xlabel="Month", ylabel="Actual / expected")

    figure.tight_layout()
    return _save(figure, path)


# --------------------------------------------------------------------------------------
# Portfolio description
# --------------------------------------------------------------------------------------


def _timeline(axis: Axes, periods: pd.Series) -> np.ndarray:
    """Monthly periods as a numeric axis matplotlib can draw."""
    del axis
    return np.asarray([p.year + (p.month - 1) / 12 for p in periods], dtype=float)


def outstanding_book(table: pd.DataFrame, path: Path) -> Path:
    """Contracts and balance outstanding, in two panels.

    Two panels rather than two y-axes. A count and a currency amount share no scale,
    and a dual axis would let the choice of scales decide how related they appear.
    """
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(8.4, 5.4), sharex=True, facecolor=SURFACE)
    time = _timeline(top, table["period"])

    top.set_facecolor(SURFACE)
    top.plot(time, table["contracts"] / 1e6, color=SERIES[0], linewidth=_LINE_WIDTH)
    _style(top, title="Contracts outstanding", xlabel="", ylabel="Millions of loans")

    bottom.set_facecolor(SURFACE)
    bottom.plot(time, table["balance"] / 1e12, color=SERIES[1], linewidth=_LINE_WIDTH)
    _style(bottom, title="Unpaid balance", xlabel="Year", ylabel="USD trillions")

    figure.tight_layout()
    return _save(figure, path)


def new_lending(table: pd.DataFrame, path: Path) -> Path:
    """New loans written each month, by count and by amount."""
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(8.4, 5.4), sharex=True, facecolor=SURFACE)
    time = _timeline(top, table["period"])

    top.set_facecolor(SURFACE)
    top.fill_between(time, table["loans"] / 1e3, color=SERIES[0], alpha=0.75, linewidth=0)
    _style(top, title="Loans originated per month", xlabel="", ylabel="Thousands")

    bottom.set_facecolor(SURFACE)
    bottom.fill_between(time, table["amount"] / 1e9, color=SERIES[1], alpha=0.75, linewidth=0)
    _style(bottom, title="Amount originated per month", xlabel="Year", ylabel="USD billions")

    figure.tight_layout()
    return _save(figure, path)


def underwriting_over_time(table: pd.DataFrame, path: Path) -> Path:
    """Median and interquartile range of the underwriting covariates, by vintage.

    Distribution and drift in one figure: the band is who was being lent to, the line
    is the middle of the book. A median that moves is a change in what was written.
    """
    figure, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True, facecolor=SURFACE)
    years = table["year"].astype(int).to_numpy()

    for axis, (name, label, colour) in zip(
        axes,
        (
            ("score", "Credit score", SERIES[0]),
            ("ltv", "Loan-to-value (%)", SERIES[1]),
            ("dti", "Debt-to-income (%)", SERIES[2]),
        ),
        strict=True,
    ):
        axis.set_facecolor(SURFACE)
        axis.fill_between(
            years,
            table[f"{name}_q25"],
            table[f"{name}_q75"],
            color=colour,
            alpha=0.20,
            linewidth=0,
        )
        axis.plot(years, table[f"{name}_q50"], color=colour, linewidth=_LINE_WIDTH)
        _style(axis, title=label, xlabel="", ylabel="")

    axes[-1].set_xlabel("Vintage year", color=INK_MUTED, fontsize=9)
    figure.tight_layout()
    return _save(figure, path)


def origination_mix_over_time(table: pd.DataFrame, path: Path, *, title: str) -> Path:
    """Share of new lending by category and year, stacked.

    A mix that moves is the reason a model fitted on one decade can mislead about
    another, and it is worth seeing before the coefficients rather than after.
    """
    wide = table.pivot(index="year", columns="level", values="share").fillna(0.0)
    figure, axis = plt.subplots(figsize=(8.4, 4.0), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)

    colours = [SERIES[i % len(SERIES)] for i in range(len(wide.columns))]
    axis.stackplot(
        wide.index.astype(int),
        *[wide[column].to_numpy() for column in wide.columns],
        labels=[str(column) for column in wide.columns],
        colors=colours,
        edgecolor=SURFACE,
        linewidth=0.6,
    )

    _style(axis, title=title, xlabel="Vintage year", ylabel="Share of loans written")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED, loc="lower left", ncols=3)
    return _save(figure, path)


def default_rate_and_unemployment(defaults: pd.DataFrame, macro: pd.DataFrame, path: Path) -> Path:
    """Realised default hazard beside unemployment, in two panels.

    The whole argument for a time-varying model in one figure — provided the two are
    not forced onto a shared frame, where the scales would be doing the arguing.
    """
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(8.4, 5.4), sharex=True, facecolor=SURFACE)

    time = _timeline(top, defaults["period"])
    top.set_facecolor(SURFACE)
    top.plot(time, defaults["default_rate"] * 10000, color=SERIES[1], linewidth=_LINE_WIDTH)
    _style(top, title="Realised monthly default rate", xlabel="", ylabel="Basis points")

    macro_time = np.asarray([p.year + (p.month - 1) / 12 for p in macro.index], dtype=float)
    bottom.set_facecolor(SURFACE)
    bottom.plot(
        macro_time, macro["unemployment_rate"].to_numpy(), color=SERIES[0], linewidth=_LINE_WIDTH
    )
    _style(bottom, title="Unemployment rate", xlabel="Year", ylabel="Percent")

    figure.tight_layout()
    return _save(figure, path)


def macro_panel(macro: pd.DataFrame, path: Path) -> Path:
    """The macroeconomic series the model reads, one panel each."""
    columns = [
        c
        for c in (
            "unemployment_rate",
            "house_price_index",
            "mortgage_rate_30y",
            "financial_conditions_index",
        )
        if c in macro
    ]
    figure, axes = plt.subplots(
        len(columns), 1, figsize=(8.0, 1.7 * len(columns) + 1.0), sharex=True, facecolor=SURFACE
    )
    time = np.asarray([p.year + (p.month - 1) / 12 for p in macro.index], dtype=float)

    labels = {
        "unemployment_rate": "Unemployment rate (%)",
        "house_price_index": "House price index (Jan 2000 = 100)",
        "mortgage_rate_30y": "30-year mortgage rate (%)",
        "financial_conditions_index": "Financial conditions (0 = average)",
    }
    for axis, name in zip(np.atleast_1d(axes), columns, strict=True):
        axis.set_facecolor(SURFACE)
        axis.plot(time, macro[name].to_numpy(), color=SERIES[0], linewidth=_LINE_WIDTH)
        _style(axis, title=labels.get(name, name), xlabel="", ylabel="")

    np.atleast_1d(axes)[-1].set_xlabel("Year", color=INK_MUTED, fontsize=9)
    figure.tight_layout()
    return _save(figure, path)
