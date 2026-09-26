"""Views of the things the September 2026 model added.

Prepayment as a competing risk, the two levels the key gained, the master scale, the
anchoring multiplier, the backtest windows and the in-sample cycle. They are here rather
than in :mod:`creditsurv.views.model` because that module is about one model of one exit,
and every table below either compares two exits or reports a rule from ``docs/rules.md``
firing.

Each is computed the way everything else on the cells is: exposure-weighted, summed with
``bincount``, and never through a loan-level frame, which does not exist at this scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from creditsurv.backtest.metrics import grade_backtest, master_scale_passed
from creditsurv.data.panel import AGE, CAUSES, DEFAULT_CAUSE, WEIGHT, ended_in
from creditsurv.models.nonparametric import cumulative_incidence, predicted_incidence_curve
from creditsurv.views.tables import View

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from creditsurv.models.anchoring import Anchor


def competing_incidence(
    cells: pd.DataFrame,
    hazards: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Observed cumulative incidence by age, and the model's beside it where there is one.

    The observed side is Aalen-Johansen on the cells; the model's is its cause-specific
    hazards chained the same way. One minus Kaplan-Meier is deliberately absent: it answers
    what the default rate would be if loans could not be repaid, which is the figure this
    whole part of the model exists to stop publishing.
    """
    observed = cumulative_incidence(cells).rename(
        columns={cause: f"observed_{cause}" for cause in CAUSES} | {"survival": "observed_survival"}
    )
    if hazards is None:
        return observed

    ages = cells[AGE].to_numpy(dtype=int)
    renamed = {cause: f"model_{cause}" for cause in hazards} | {"survival": "model_survival"}
    predicted = predicted_incidence_curve(
        hazards, ages, weight=cells[WEIGHT].to_numpy(dtype=float)
    ).rename(columns=renamed)
    return observed.merge(predicted.drop(columns="at_risk"), on="age", how="left")


def exposure_by_level(
    cells: pd.DataFrame, column: str, *, causes: Sequence[str] = CAUSES
) -> pd.DataFrame:
    """Loan-months and each exit's rate, level by level of a key column.

    What the HARP level and the payment state are for, in one table: how much of the book
    each level holds, and how differently it behaves. Monthly rates, not annual, because
    that is the unit the cells are in and an annualisation here would be a second
    approximation on top of the first.
    """
    codes, levels = pd.factorize(cells[column], sort=True)
    weight = cells[WEIGHT].to_numpy(dtype=float)
    present = codes >= 0
    size = len(levels)

    def total(values: np.ndarray) -> np.ndarray:
        summed: np.ndarray = np.bincount(codes[present], weights=values[present], minlength=size)
        return summed

    exposure = total(weight)
    table = pd.DataFrame({column: [str(level) for level in levels], "loan_months": exposure})
    for cause in causes:
        events = total(weight * ended_in(cells, cause))
        table[cause] = events
        table[f"{cause}_rate"] = np.where(exposure > 0, events / exposure, np.nan)
    table["share"] = exposure / exposure.sum() if exposure.sum() else np.nan
    return table[table["loan_months"] > 0].reset_index(drop=True)


def cycle_in_band(by_year: pd.DataFrame, *, low: float = 0.80, high: float = 1.25) -> pd.DataFrame:
    """The in-sample cycle criterion: the share of calendar years inside the band.

    The criterion the previous model was found not to test. Its actual over expected ran
    from 0.47 to 1.60 across years, and a single out-of-time ratio near one says nothing
    beside that spread -- rule 5 asks for at least 70% of years in the band.
    """
    ratios = by_year["actual_over_expected"].dropna()
    inside = ratios.between(low, high)
    return pd.DataFrame(
        [
            {
                "years": len(ratios),
                "years_in_band": int(inside.sum()),
                "share_in_band": float(inside.mean()) if len(ratios) else np.nan,
                "lowest": float(ratios.min()) if len(ratios) else np.nan,
                "highest": float(ratios.max()) if len(ratios) else np.nan,
                "threshold": 0.70,
                "passed": bool(len(ratios) and inside.mean() >= 0.70),
            }
        ]
    )


def anchoring_view(
    anchor: Anchor,
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> View:
    """The multiplier, and what it did to the level on either side of it.

    ``before`` and ``after`` are actual-against-expected tables of the same rows, scored
    with the unanchored and the anchored hazard, so a reader can see that one number moved
    the level and nothing else -- the ranking is identical by construction.
    """
    frame = pd.concat(
        [before.assign(model="unanchored"), after.assign(model="anchored")], ignore_index=True
    )
    # Named here rather than spread from describe(): the view is a table people read, and
    # a column called "actual_defaults" beside a table of actual defaults is a trap.
    frame["anchor_window"] = f"{anchor.window[0]} to {anchor.window[1]}"
    frame["anchor_multiplier"] = anchor.multiplier
    frame["anchor_actual_defaults"] = anchor.actual_defaults
    frame["anchor_expected_defaults"] = anchor.expected_defaults
    return View(
        "anchoring",
        "The level, before and after anchoring",
        f"One multiplier of {anchor.multiplier:.4f} on the default hazard, estimated as "
        f"actual over expected on {anchor.window[0]} to {anchor.window[1]} and applied to "
        "every loan-month. Nothing else changes: not the coefficients, not the shape, not "
        "the ranking.",
        frame,
        source="fit",
    )


def window_view(results: Sequence[Mapping[str, object]]) -> View:
    """One row per backtest cut: the window, its exposure, and how the model did on it."""
    return View(
        "backtest_windows",
        "The backtest, cut three times",
        "Each cut estimated on everything up to it and judged on the 24 months after, so "
        "no window is scored by a model that saw it: a tightening cycle, a pandemic under "
        "a moratorium regime, and a rate shock.",
        pd.DataFrame(list(results)),
        source="fit",
    )


def grade_view(
    hazard: np.ndarray,
    cells: pd.DataFrame,
    *,
    cause: str = DEFAULT_CAUSE,
) -> View:
    """Twelve-month PD by grade, against what each grade's loans did."""
    weight = cells[WEIGHT]
    table = grade_backtest(
        pd.Series(hazard), pd.Series(ended_in(cells, cause).astype(float)), weight
    )
    passed = master_scale_passed(table)
    return View(
        "pd_by_grade",
        "Twelve-month PD by grade",
        "Eight grades on geometric thresholds of predicted twelve-month PD, each floor "
        "twice the one before. A grade passes when its predicted PD falls inside the 95% "
        f"Jeffreys interval around what its loans did. The scale "
        f"{'passes' if passed else 'fails'}: "
        f"{int(table['passed'].sum())} of {len(table)} populated grades hold.",
        table,
        source="fit",
    )
