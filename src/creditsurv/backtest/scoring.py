"""The book on the shelf, scored: the twelve-month PD the backtest validated, and nothing else.

Every number the model produced until November 2026 was produced on the way to a report -- a
view, a backtest table, a campaign -- and there was no operation that took a book and returned
its PD. This module is that operation, and it is deliberately narrow: it delivers **the
quantity rule 5's criteria were measured on**, computed the way they measured it, and refuses
to deliver what has not been validated.

The quantity is :func:`creditsurv.backtest.metrics.annualised` of the **anchored** monthly
hazard at each cell's own covariates: ``1 - (1 - k h)^12``. It is what the master scale graded
and found inside rule 5's tolerance in **8 of 8** grades on the test window, with the deciles
between 0.803 and 1.150 -- fit ``bc016657b2b40bfc``, multiplier 1.0111 on 2022-01 to 2024-12,
scored on the months after it. A different twelve-month PD -- chained along a projected macro
path, say, or net of prepayment -- would be a better-sounding number that nothing has tested.

What it does not deliver, and why:

- **No lifetime PD.** The shape along age holds 2 of 7 bands out of sample where rule 5 asks for
  five, and a lifetime PD is that shape integrated. `views` still publishes a term structure;
  it is a description of the model, not a validated forecast.
- **No individually valid PD for a loan in arrears.** Rule 10: the model does not read the
  payment state, so a loan two payments behind gets the PD of a loan in order, against observed
  monthly rates of 28.45% and 0.0008%. The **pool** is what was validated -- every loan at risk,
  arrears included, because the anchored level carries the defaults that pass through arrears --
  so the pool is what is scored, and the state is carried as a column for the reader to see.

The regulatory floor of rule 17 is a second object beside the model's PD, never a replacement
for it: ``pd_12m_floored = max(pd_12m, 0.0005)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Final, TypedDict

import numpy as np
import pandas as pd

from creditsurv.backtest.metrics import (
    annualised,
    grade_of,
    weighted_calibration,
    weighted_gini,
)
from creditsurv.backtest.runner import ACCEPTANCE
from creditsurv.config import AGE_BANDS
from creditsurv.data.panel import AGE, WEIGHT, cells_to_episodes, ended_in, month_ordinal
from creditsurv.data.store import load_cells_window
from creditsurv.models.aft import episode_hazards
from creditsurv.models.anchoring import ANCHOR_WINDOW, Anchor, anchor_on_window

if TYPE_CHECKING:
    from collections.abc import Sequence

    from creditsurv.models.aft import FitResult

log = logging.getLogger(__name__)

#: The PD floor for retail exposures secured by residential property: Basel III's final
#: framework (CRE 32) and CRR3's Article 164(1), 0.05%. Rule 17 of `docs/rules.md`, declared
#: before any output carried it. It applies to the twelve-month PD as a regulatory object and to
#: nothing the model is judged on.
REGULATORY_PD_FLOOR: Final = 0.0005

#: The columns of a scored cell that say what the cell is, carried through to the output so a
#: reader can segment the book without the cell file.
_DESCRIBING: Final = (
    "origination_month",
    AGE,
    "credit_score",
    "original_ltv",
    "debt_to_income",
    "purpose",
    "occupancy",
    "harp",
    "mortgage_insurance",
    "buyer_type",
    "delinquency_state",
)


def anchor_streamed(
    fitted: FitResult,
    covariates: Sequence[str],
    *,
    moratorium: str,
    macro: pd.DataFrame,
    window: tuple[str, str] = ANCHOR_WINDOW,
) -> Anchor:
    """Rule 4's multiplier on the anchoring window, read a month at a time.

    `views` reads the three years in one frame, about 13 million cells; summed month by month
    the footprint is one month's, and the multiplier is the same sum. Each month enters
    :func:`anchor_on_window` as one row standing for its loan-months, with its expected and
    realised defaults as rates, so the products the function forms recover the month's sums.
    """
    first, last = (pd.Period(month, freq="M") for month in window)
    loan_months, expected, actual, periods = [], [], [], []
    for period in pd.period_range(first, last, freq="M"):
        month = month_ordinal(period)
        cells = load_cells_window(moratorium, first=month, last=month)
        if cells.empty:
            continue
        frame = cells_to_episodes(cells, macro, covariates=list(covariates))
        hazard = episode_hazards(
            fitted, frame, frame[AGE].to_numpy(dtype=int), columns=list(covariates)
        )
        weight = frame[WEIGHT].to_numpy(dtype=float)
        loan_months.append(float(weight.sum()))
        expected.append(float((hazard * weight).sum()))
        actual.append(float((ended_in(frame) * weight).sum()))
        periods.append(period)
    exposure = np.asarray(loan_months)
    return anchor_on_window(
        np.asarray(expected) / exposure,
        np.asarray(actual) / exposure,
        exposure,
        pd.PeriodIndex(periods),
        window=window,
    )


class BookSummary(TypedDict):
    """The whole book in one row: what was scored, with what, and what came out."""

    month: str
    fit: str
    multiplier: float
    anchoring_window: str
    cells: int
    loans: int
    pd_12m: float
    pd_12m_floored: float
    share_below_floor: float
    share_in_arrears: float


@dataclass(frozen=True)
class ScoredBook:
    """The book at risk in one month, each cell with its PD, and what produced them."""

    month: pd.Period
    fit: str
    anchor: Anchor
    #: One row per cell: what it is, how many loans it stands for, and its PDs.
    cells: pd.DataFrame

    def by_grade(self) -> pd.DataFrame:
        """The book on the master scale: loans, their share, and the mean PD of each grade."""
        cells = self.cells
        weight = cells[WEIGHT].astype(float)
        table = (
            pd.DataFrame(
                {
                    "grade": cells["grade"],
                    "loans": weight,
                    "pd_weighted": cells["pd_12m"] * weight,
                    "floored_weighted": cells["pd_12m_floored"] * weight,
                }
            )
            .groupby("grade")
            .sum()
        )
        table["share"] = table["loans"] / table["loans"].sum()
        table["pd_12m"] = table.pop("pd_weighted") / table["loans"]
        table["pd_12m_floored"] = table.pop("floored_weighted") / table["loans"]
        return table.reset_index()

    def summary(self) -> BookSummary:
        cells = self.cells
        weight = cells[WEIGHT].astype(float)
        loans = float(weight.sum())
        below = cells["pd_12m"] < REGULATORY_PD_FLOOR
        return {
            "month": str(self.month),
            "fit": self.fit,
            "multiplier": round(self.anchor.multiplier, 4),
            "anchoring_window": f"{self.anchor.window[0]} to {self.anchor.window[1]}",
            "cells": len(cells),
            "loans": int(loans),
            "pd_12m": float((cells["pd_12m"] * weight).sum() / loans),
            "pd_12m_floored": float((cells["pd_12m_floored"] * weight).sum() / loans),
            "share_below_floor": float(weight[below].sum() / loans),
            "share_in_arrears": float(
                weight[cells["delinquency_state"].astype(str) != "current"].sum() / loans
            )
            if "delinquency_state" in cells.columns
            else float("nan"),
        }


def _at_risk(
    fitted: FitResult,
    covariates: Sequence[str],
    month: pd.Period,
    anchor: Anchor,
    *,
    moratorium: str,
    macro: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """The episodes at risk in one month and their anchored hazards: one read of the month."""
    ordinal = month_ordinal(month)
    cells = load_cells_window(moratorium, first=ordinal, last=ordinal)
    if cells.empty:
        message = f"No cells are observed in {month}; the table ends earlier."
        raise ValueError(message)
    frame = cells_to_episodes(cells, macro, covariates=list(covariates))
    hazard = anchor.apply(
        episode_hazards(fitted, frame, frame[AGE].to_numpy(dtype=int), columns=list(covariates))
    )
    return frame, hazard


def score_month(
    fitted: FitResult,
    covariates: Sequence[str],
    *,
    fit: str,
    month: pd.Period,
    anchor: Anchor,
    moratorium: str,
    macro: pd.DataFrame,
) -> ScoredBook:
    """Every cell at risk in ``month``, at its own covariates, with the anchored PD.

    The book at risk in a month is the cells observed in it: each stands for the loans of one
    profile that were performing when the month began. That is the population the master scale
    was measured on, so it is the population the PD is valid for.
    """
    frame, hazard = _at_risk(fitted, covariates, month, anchor, moratorium=moratorium, macro=macro)
    twelve = annualised(hazard)
    scored = frame.loc[:, [name for name in _DESCRIBING if name in frame.columns]].copy()
    scored[WEIGHT] = frame[WEIGHT].to_numpy()
    scored["hazard"] = hazard
    scored["pd_12m"] = twelve
    scored["pd_12m_floored"] = np.maximum(twelve, REGULATORY_PD_FLOOR)
    scored["grade"] = grade_of(twelve)
    log.info("%s: %s cells, %s loans", month, f"{len(scored):,}", f"{scored[WEIGHT].sum():,}")
    return ScoredBook(month=month, fit=fit, anchor=anchor, cells=scored.reset_index(drop=True))


@dataclass(frozen=True)
class WindowCriteria:
    """Section 5's out-of-sample criteria for one model, on the declared test window, anchored."""

    first: pd.Period
    last: pd.Period
    anchor: Anchor
    loan_months: float
    actual_defaults: float
    expected_defaults: float
    gini: float
    #: Predicted against realised by decile of predicted risk, as `weighted_calibration` cuts it.
    deciles: pd.DataFrame
    #: Actual over expected by the declared seasoning bands, `config.AGE_BANDS`.
    age_bands: pd.DataFrame

    @property
    def actual_over_expected(self) -> float:
        return self.actual_defaults / self.expected_defaults

    def bands_inside(self) -> int:
        """How many age bands hold inside the acceptance band; rule 5 asks for five of seven."""
        ratios = self.age_bands["actual_over_expected"]
        return int(ratios.between(ACCEPTANCE.ae_low, ACCEPTANCE.ae_high).sum())


def window_criteria(
    fitted: FitResult,
    covariates: Sequence[str],
    *,
    anchor: Anchor,
    moratorium: str,
    macro: pd.DataFrame,
    first: pd.Period,
    last: pd.Period,
) -> WindowCriteria:
    """The overall ratio, the Gini, the deciles and the age bands, read a month at a time.

    The test window is a year and a half of observation, five to six million cells, and
    `views` reads it in one frame. Here each month is read once, scored, and reduced to four
    arrays a cell -- hazard, defaults, weight, age -- which is all the criteria need. On fit
    bc016657b2b40bfc this reproduced every published figure: A/E 0.9065, Gini 0.5461, the
    deciles 0.803 to 1.150 and the age bands 0.8137 to 0.7253, at a 1.40 GB peak.
    """
    hazards, events, weights, ages = [], [], [], []
    for month in pd.period_range(first, last, freq="M"):
        frame, hazard = _at_risk(
            fitted, covariates, month, anchor, moratorium=moratorium, macro=macro
        )
        hazards.append(hazard)
        weights.append(frame[WEIGHT].to_numpy(dtype=float))
        events.append(ended_in(frame).astype(float))
        ages.append(frame[AGE].to_numpy(dtype=int))
    hazard, weight = np.concatenate(hazards), np.concatenate(weights)
    defaulted = np.concatenate(events) * weight
    age = np.concatenate(ages)

    labels = [f"{low} to {high - 1}" for low, high in pairwise(AGE_BANDS)]
    band = pd.cut(age, bins=list(AGE_BANDS), labels=labels, right=False)
    by_band = (
        pd.DataFrame(
            {"age_band": band, "exposure": weight, "events": defaulted, "expected": hazard * weight}
        )
        .groupby("age_band", observed=True)
        .sum()
        .reset_index()
    )
    by_band["actual_over_expected"] = by_band["events"] / by_band["expected"]
    series = {
        name: pd.Series(values)
        for name, values in (("hazard", hazard), ("events", defaulted), ("weight", weight))
    }
    return WindowCriteria(
        first=first,
        last=last,
        anchor=anchor,
        loan_months=float(weight.sum()),
        actual_defaults=float(defaulted.sum()),
        expected_defaults=float((hazard * weight).sum()),
        gini=weighted_gini(series["hazard"], series["events"], series["weight"]),
        deciles=weighted_calibration(series["hazard"], series["events"], series["weight"]),
        age_bands=by_band,
    )
