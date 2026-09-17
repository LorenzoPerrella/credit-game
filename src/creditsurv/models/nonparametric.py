"""Kaplan-Meier and Turnbull benchmarks.

These exist to answer a fair question about the parametric model: is the shape it
imposes actually supported by the data? A Kaplan-Meier curve assumes nothing about
the distribution, so a parametric curve that strays outside its confidence band is
being contradicted by the data rather than merely smoothing it.

The comparison also shows *why* the parametric model is needed. Kaplan-Meier
cannot condition on covariates, cannot extrapolate past the last observed event,
and cannot be pushed through a macroeconomic scenario. Lifetime PD requires all
three, so the non-parametric curve is a benchmark rather than an alternative.

One constraint is worth stating. ``KaplanMeierFitter.fit`` supports left
truncation through ``entry``, but ``fit_interval_censoring`` raises
``NotImplementedError`` when ``entry`` is passed, so the Turnbull estimator cannot
be combined with truncation. Loans here are observed from origination, so entry is
zero and the restriction does not bite -- but it would immediately for a portfolio
sampled mid-life.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from creditsurv.data.panel import EVENT, duration_view, to_loan_level
from creditsurv.models.aft import episode_hazards

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lifelines.statistics import StatisticalResult

    from creditsurv.models.aft import FitResult

#: Default label for the pooled curve.
OVERALL: str = "overall"


def kaplan_meier(
    panel: pd.DataFrame, *, label: str = OVERALL, weights_col: str | None = None
) -> KaplanMeierFitter:
    """Fit a Kaplan-Meier curve to the loan-level view of an episode panel.

    ``weights_col`` names the count on an aggregated panel, where a row is a number
    of loan-months rather than a loan. The curve is then estimated on the whole
    population instead of a sample of it, which is the point of aggregating.
    """
    loans = duration_view(panel, weights_col=weights_col)
    fitter = KaplanMeierFitter(label=label)
    fitter.fit(
        loans["duration"],
        event_observed=loans[EVENT].astype(bool),
        weights=None if weights_col is None else loans[weights_col],
    )
    return fitter


def kaplan_meier_by_stratum(panel: pd.DataFrame, stratum: str) -> dict[str, KaplanMeierFitter]:
    """Fit one curve per level of ``stratum``.

    Stratifying is the only way Kaplan-Meier can respond to a covariate at all,
    and it works solely for covariates fixed at origination. A covariate that
    varies over the life of the loan cannot be stratified on without deciding
    which of its values to use, which is precisely the problem the parametric
    time-varying model solves.
    """
    loans = to_loan_level(panel)
    curves: dict[str, KaplanMeierFitter] = {}
    for level, rows in loans.groupby(stratum, observed=True):
        name = str(level)
        fitter = KaplanMeierFitter(label=name)
        fitter.fit(rows["duration"], event_observed=rows[EVENT].astype(bool))
        curves[name] = fitter
    return curves


def logrank_by_stratum(panel: pd.DataFrame, stratum: str) -> StatisticalResult:
    """Test whether survival differs across the levels of ``stratum``."""
    loans = to_loan_level(panel)
    levels = list(loans.groupby(stratum, observed=True).groups)
    if len(levels) != 2:
        message = f"logrank_by_stratum compares exactly two groups; {stratum!r} has {len(levels)}."
        raise ValueError(message)

    first, second = (loans[loans[stratum] == level] for level in levels)
    return logrank_test(
        first["duration"],
        second["duration"],
        event_observed_A=first[EVENT].astype(bool),
        event_observed_B=second[EVENT].astype(bool),
    )


def turnbull(panel: pd.DataFrame, *, weights_col: str | None = None) -> KaplanMeierFitter:
    """Non-parametric estimator honouring the monthly observation interval.

    The Kaplan-Meier counterpart of the parametric interval-censored fit: it makes
    the same admission, that a default is known to a month rather than a moment.

    Left truncation cannot be combined with it -- lifelines raises
    ``NotImplementedError`` if ``entry`` is passed alongside interval bounds --
    which is harmless here only because loans are observed from origination.
    """
    loans = duration_view(panel, weights_col=weights_col)
    defaulted = loans[EVENT].astype(bool).to_numpy()
    duration = loans["duration"].to_numpy(dtype=float)

    horizon = float(duration.max())
    fitter = KaplanMeierFitter(label="turnbull")
    fitter.fit_interval_censoring(
        lower_bound=np.where(defaulted, duration - 1.0, duration),
        upper_bound=np.where(defaulted, duration, np.inf),
        weights=None if weights_col is None else loans[weights_col],
        # Without an explicit timeline lifelines derives one from the bounds, which
        # then contains the infinity used for right censoring. The curve is still
        # correct, but its last row sits at t = inf where survival is zero, and
        # reading it as the end of the horizon is an easy and silent mistake.
        timeline=np.arange(0.0, horizon + 1.0),
    )
    return fitter


def predicted_survival_curve(
    result: FitResult,
    encoded: pd.DataFrame,
    covariates: Sequence[str],
    *,
    age_col: str = "age",
    weights_col: str | None = None,
) -> pd.Series:
    """Portfolio survival implied by the model along the realised covariate paths.

    The obvious approach -- predict each loan's survival curve from its
    origination covariates and average -- is wrong here, and quietly so. Half the
    covariates vary over the life of the loan, and two of them (``ltv_change`` and
    ``unemployment_change``) are zero at origination by construction. Freezing them there
    assumes house prices never move and unemployment never changes, which
    understates risk and overstates survival by more at every further horizon.

    Instead the model's monthly hazard is evaluated at each loan-month's *actual*
    covariates, averaged across the loans at risk at that age, and chained:

        S(t) = prod over a < t of (1 - mean hazard at age a)

    which is the same construction Kaplan-Meier applies to empirical hazards, and
    therefore directly comparable with it.
    """
    ages = encoded[age_col].to_numpy(dtype=int)
    hazard = episode_hazards(result, encoded, ages, columns=list(covariates))

    # On aggregated cells the average has to be over loans at risk, not over distinct
    # covariate combinations: a rare combination would otherwise weigh as much as one
    # carrying a million loan-months. Added up with bincount, because a group-by first
    # builds a frame of the weights and products -- another 1 GB on the training half of
    # the exact key -- to produce a few hundred sums.
    weight = (
        np.ones(len(ages)) if weights_col is None else encoded[weights_col].to_numpy(dtype=float)
    )
    at_risk = np.bincount(ages, weights=weight)
    present = np.flatnonzero(at_risk > 0)
    mean_hazard = np.bincount(ages, weights=weight * hazard)[present] / at_risk[present]
    survival = np.cumprod(1.0 - mean_hazard)
    return pd.Series(survival, index=present + 1.0, name="predicted")


def km_band_contains(
    curve: KaplanMeierFitter, predicted: pd.Series, *, tolerance: float = 0.0
) -> pd.DataFrame:
    """Compare a predicted curve against the Kaplan-Meier confidence band.

    Returns one row per evaluated time with the band, the prediction, whether it falls
    inside, and **by how much it misses**.

    The deviation is not decoration. A confidence band narrows as the square root of
    the sample, and on 48 million loans it collapses: at five years this band runs from
    0.9497 to 0.9498, a width of **one basis point**. Every smooth parametric curve is
    outside a band that narrow, so the in-or-out count stops carrying information and
    starts reading as a catastrophic failure -- it reported 3 of 312 points inside while
    the curve was tracking observed survival to within a third of a percentage point.

    So the count is kept, because it is the honest answer to the question as asked, and
    the magnitude is reported beside it, because that is the question worth asking at
    this sample size: not *is the curve inside the interval* but *how far from the data
    is it*.
    """
    band = curve.confidence_interval_survival_function_
    lower_name, upper_name = band.columns[0], band.columns[1]

    aligned = pd.DataFrame(index=predicted.index)
    aligned["predicted"] = predicted
    aligned["km_lower"] = (
        band[lower_name].reindex(band.index.union(predicted.index)).ffill().reindex(predicted.index)
    )
    aligned["km_upper"] = (
        band[upper_name].reindex(band.index.union(predicted.index)).ffill().reindex(predicted.index)
    )
    aligned["inside"] = (aligned["predicted"] >= aligned["km_lower"] - tolerance) & (
        aligned["predicted"] <= aligned["km_upper"] + tolerance
    )
    observed = (aligned["km_lower"] + aligned["km_upper"]) / 2.0
    aligned["deviation"] = aligned["predicted"] - observed
    aligned["band_width"] = aligned["km_upper"] - aligned["km_lower"]
    return aligned
