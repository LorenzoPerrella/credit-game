"""The backtest: fit once on the past, predict forward, compare with what happened.

Deliberately one fit and one comparison. Earlier versions of this module refitted the
model at several reporting dates and again under two macroeconomic assumptions, which
answers a richer question and costs eight fits; on this panel a fit is hours, and the
richer question was not the one being asked.

What remains is the question a credit committee actually puts. **Given a model
calibrated on everything up to the reporting date, how many defaults did it expect
after it, and how many happened?** Everything here is that comparison, cut three ways:
in total, by calendar period, and by decile of predicted risk.

The model is never refitted inside this module. The fit is passed in, which is what
makes "no further calibration in the testing phase" a property of the code rather
than a discipline someone has to remember.

**The scoring uses the macro path that actually occurred.** That is what predicted
against realised means, and it is worth being explicit about what it therefore does
*not* measure: a model used in anger would have had to forecast the economy, and its
error would include that forecast's error. This backtest isolates the credit model and
grants it the economy. Reading it as the performance of the whole system would
overstate what the system can do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from creditsurv.backtest.metrics import (
    actual_versus_expected,
    stability_report,
    weighted_calibration,
    weighted_gini,
)
from creditsurv.backtest.splits import PERIOD, Split, assert_no_lookahead, cell_split
from creditsurv.config import TIME_VARYING_CONTINUOUS
from creditsurv.data.panel import AGE, EVENT, WEIGHT
from creditsurv.models.aft import episode_hazards, fit_aft

if TYPE_CHECKING:
    from collections.abc import Sequence

    from creditsurv.models.aft import FitResult

ORIGINATION = "orig_period"


@dataclass(frozen=True)
class BacktestResult:
    """What one reporting date produced."""

    as_of: pd.Period
    loan_months: int
    expected_defaults: float
    actual_defaults: float
    #: Discrimination, from the exposure-weighted Lorenz curve.
    gini: float
    #: Predicted against realised, by decile of predicted risk.
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: The same, by calendar period of observation.
    over_time: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: How far the covariate distributions moved across the cut.
    stability: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def actual_over_expected(self) -> float:
        """Above one the model under-predicts; below one it over-predicts."""
        if self.expected_defaults <= 0:
            return float("nan")
        return self.actual_defaults / self.expected_defaults

    def summary(self) -> dict[str, object]:
        return {
            "as_of": str(self.as_of),
            "loan_months": self.loan_months,
            "expected_defaults": round(self.expected_defaults, 1),
            "actual_defaults": round(self.actual_defaults, 1),
            "actual_over_expected": round(self.actual_over_expected, 4),
            "gini": round(self.gini, 4),
        }


def predicted_hazard(
    fitted: FitResult, cells: pd.DataFrame, covariates: Sequence[str]
) -> pd.Series:
    """The model's monthly default probability for each cell, at its own covariates.

    No projection and no forward path. A cell already records what its loans looked
    like in the month it covers, so the prediction is evaluated there directly -- which
    is both simpler than projecting a book forward and closer to what is being asked:
    the model said this month carried this much risk, and this is what it carried.
    """
    hazard = episode_hazards(fitted, cells.loc[:, list(covariates)], cells[AGE].to_numpy(dtype=int))
    return pd.Series(hazard, index=cells.index, name="predicted")


def score(
    fitted: FitResult,
    test: pd.DataFrame,
    covariates: Sequence[str],
    *,
    as_of: pd.Period,
    train: pd.DataFrame | None = None,
) -> BacktestResult:
    """Compare what the model expected after ``as_of`` with what happened.

    Expected defaults are the sum of ``hazard times loan-months`` over the test cells;
    actual defaults are the loan-months the data records as ending in default. The
    ratio is the number a credit committee reads: above one the model under-predicted.
    """
    if test.empty:
        message = f"No exposure after {as_of} to score."
        raise ValueError(message)

    hazard = predicted_hazard(fitted, test, covariates)
    exposure = test[WEIGHT].astype(float)
    events = exposure * test[EVENT].astype(bool)

    return BacktestResult(
        as_of=as_of,
        loan_months=int(exposure.sum()),
        expected_defaults=float((hazard * exposure).sum()),
        actual_defaults=float(events.sum()),
        gini=weighted_gini(hazard, events, exposure),
        calibration=weighted_calibration(hazard, events, exposure),
        over_time=actual_versus_expected(hazard, events, exposure, test[PERIOD]),
        stability=pd.DataFrame()
        if train is None
        else stability_report(
            train, test, covariates, time_varying=TIME_VARYING_CONTINUOUS, weights_col=WEIGHT
        ),
    )


def run_backtest(
    episodes: pd.DataFrame,
    as_of: pd.Period,
    covariates: Sequence[str],
    formula: str,
    *,
    distribution: str = "weibull",
    fitted: FitResult | None = None,
) -> tuple[Split, FitResult, BacktestResult]:
    """Split, fit once on the training half, and score the rest.

    ``fitted`` accepts a model already estimated on this split's training half, so a
    caller that needs the same model for its reports does not pay for it twice. It is
    checked against the split rather than trusted: a model fitted on the whole panel
    would score its own training data and return a flattering number with nothing
    visibly wrong.
    """
    split = cell_split(episodes, as_of)
    assert_no_lookahead(split)

    if fitted is None:
        fitted = fit_aft(
            split.train, covariates, formula, distribution=distribution, weights_col=WEIGHT
        )
    elif fitted.n_episodes != len(split.train):
        message = (
            f"The model passed in was fitted on {fitted.n_episodes:,} rows but this "
            f"split's training half has {len(split.train):,}. A model fitted on the "
            "whole panel would be scoring its own training data."
        )
        raise ValueError(message)

    result = score(fitted, split.test, covariates, as_of=as_of, train=split.train)
    return split, fitted, result
