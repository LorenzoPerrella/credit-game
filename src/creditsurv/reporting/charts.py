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

    ``C(purpose, Treatment('purchase'))[T.refinance_cashout]`` carries the encoding
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


def backtest_discrimination(summary: pd.DataFrame, path: Path) -> Path:
    """Concordance by reporting date, one series per macro mode."""
    figure, axis = plt.subplots(figsize=(7.2, 4.0), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)

    for slot, (mode, rows) in enumerate(summary.groupby("macro_mode", observed=True)):
        axis.plot(
            rows["as_of"],
            rows["concordance"],
            marker="o",
            markersize=_MARKER_SIZE,
            linewidth=_LINE_WIDTH,
            color=SERIES[slot % len(SERIES)],
            label=str(mode).capitalize(),
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
        )

    axis.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)))
    _style(
        axis,
        title="Discrimination by reporting date",
        xlabel="Reporting date",
        ylabel="Concordance",
    )
    axis.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="lower right")
    return _save(figure, path)
