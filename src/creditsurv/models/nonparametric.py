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

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import norm

from creditsurv.data.panel import (
    AGE,
    CAUSES,
    DEFAULT_CAUSE,
    EVENT,
    WEIGHT,
    duration_view,
    ended_in,
    to_loan_level,
)
from creditsurv.models.aft import episode_hazards

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lifelines.statistics import StatisticalResult

    from creditsurv.models.aft import FitResult

#: Default label for the pooled curve.
OVERALL: str = "overall"

#: Loan-months an age must carry before its non-parametric curve is worth comparing
#: anything with. Fixed in `docs/rules.md` with the family rule it serves: at 312 months the
#: book holds a handful of loans, and a crossing test that ignored exposure has already
#: fired twice in this project on tails of two loan-months.
EXPOSURE_FLOOR: Final = 100_000.0


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


# --------------------------------------------------------------------------------------
# Competing risks: the cumulative incidence of default when prepayment can intervene
# --------------------------------------------------------------------------------------


def cumulative_incidence(
    cells: pd.DataFrame,
    *,
    causes: Sequence[str] = CAUSES,
    weights_col: str = WEIGHT,
    age_col: str = AGE,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Aalen-Johansen cumulative incidence, read straight off the aggregated cells.

    **Why not one minus Kaplan-Meier.** Treating prepayment as censoring asks what the
    default rate would be *if loans could not be repaid*, and on this book they mostly are:
    prepayment removes loans from the population far faster than default does, and the
    counterfactual answer is not the quantity a lifetime PD is supposed to hold. The
    cumulative incidence asks what actually happens -- the share of loans that have
    defaulted by age t, prepayments and all -- which is the one a provision is calculated
    from, and 1 - KM overstates it, by more at every further horizon.

    The estimator is the textbook one, with the hazards read from the cells:

        S(t) = prod over a <= t of (1 - (d_default(a) + d_prepaid(a)) / n(a))
        F_k(t) = sum over a <= t of S(a-) * d_k(a) / n(a)

    where ``n(a)`` is the loan-months at risk in the episode starting at age ``a``. No
    loan-level frame exists at any point: the three counts come out of ``bincount`` over
    the cell table, which is the whole reason the cells carry a three-state outcome rather
    than a flag.

    The interval is Aalen's, by the delta method (Marubini and Valsecchi, as lifelines
    computes it). Read the *width*, not whether a curve falls inside: at these counts it
    collapses to a hundredth of a percentage point, the way every band in this project
    does.
    """
    ages = cells[age_col].to_numpy(dtype=int)
    weight = cells[weights_col].to_numpy(dtype=float)
    length = int(ages.max()) + 1

    at_risk = np.bincount(ages, weights=weight, minlength=length)
    exits = {
        cause: np.bincount(ages, weights=weight * ended_in(cells, cause), minlength=length)
        for cause in causes
    }
    leaving = sum(exits.values())

    present = np.flatnonzero(at_risk > 0)
    at_risk = at_risk[present]
    exits = {cause: count[present] for cause, count in exits.items()}
    leaving = np.asarray(leaving)[present]

    # S(a-), survival into the episode rather than out of it: the increment at age a is
    # the hazard of a among the loans that reached it.
    survival = np.cumprod(1.0 - leaving / at_risk)
    entering = np.concatenate(([1.0], survival[:-1]))

    table = pd.DataFrame({"age": present, "at_risk": at_risk, "survival": survival})
    quantile = float(norm.ppf(0.5 + confidence / 2.0))
    for cause, count in exits.items():
        hazard = count / at_risk
        incidence = np.cumsum(entering * hazard)
        error = _incidence_error(incidence, entering, at_risk, leaving, count)
        table[cause] = incidence
        table[f"{cause}_lower"] = np.clip(incidence - quantile * error, 0.0, 1.0)
        table[f"{cause}_upper"] = np.clip(incidence + quantile * error, 0.0, 1.0)
        table[f"{cause}_se"] = error
    return table


def predicted_incidence_curve(
    hazards: Mapping[str, np.ndarray],
    ages: np.ndarray,
    *,
    weight: np.ndarray | None = None,
) -> pd.DataFrame:
    """The model's own cumulative incidence, chained from its cause-specific hazards.

    The counterpart of :func:`predicted_survival_curve` for two competing exits, and built
    the same way -- the monthly hazard evaluated at each loan-month's *actual* covariates,
    averaged over the loans at risk at that age, and chained -- so it is directly
    comparable with :func:`cumulative_incidence` computed on the same rows. That
    comparability is the point: rule 2 of `docs/rules.md` chooses the distribution family
    by the gap between these two curves.

    ``hazards`` maps a cause to that cause's predicted monthly hazard, one value per row of
    the same frame the ages and weights come from.
    """
    length = int(ages.max()) + 1
    weights = np.ones(len(ages)) if weight is None else np.asarray(weight, dtype=float)
    at_risk = np.bincount(ages, weights=weights, minlength=length)
    present = np.flatnonzero(at_risk > 0)

    mean = {
        cause: np.bincount(ages, weights=weights * np.asarray(hazard), minlength=length)[present]
        / at_risk[present]
        for cause, hazard in hazards.items()
    }
    leaving = sum(mean.values())
    survival = np.cumprod(1.0 - np.asarray(leaving))
    entering = np.concatenate(([1.0], survival[:-1]))

    table = pd.DataFrame({"age": present, "at_risk": at_risk[present], "survival": survival})
    for cause, hazard in mean.items():
        table[cause] = np.cumsum(entering * hazard)
    return table


def incidence_gap(
    predicted: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    cause: str = DEFAULT_CAUSE,
    exposure_floor: float = EXPOSURE_FLOOR,
) -> pd.DataFrame:
    """Where the model's cumulative incidence sits against the Aalen-Johansen one.

    One row per age carrying at least ``exposure_floor`` loan-months, with both curves and
    the gap in **percentage points**. The floor is why the comparison says anything: the
    tail ages hold a handful of loans, where the non-parametric curve is noise and a
    parametric one is doing the only sensible thing by ignoring it. Every crossing test in
    this project has had to learn that, the last one by firing on tails of two loan-months.
    """
    merged = predicted.merge(observed, on="age", how="inner", suffixes=("_model", "_observed"))
    merged = merged[merged["at_risk_observed"] >= exposure_floor]
    return pd.DataFrame(
        {
            "age": merged["age"],
            "at_risk": merged["at_risk_observed"],
            "model": merged[f"{cause}_model"],
            "observed": merged[f"{cause}_observed"],
            "gap_pp": 100.0 * (merged[f"{cause}_model"] - merged[f"{cause}_observed"]),
        }
    )


def _incidence_error(
    incidence: np.ndarray,
    entering: np.ndarray,
    at_risk: np.ndarray,
    leaving: np.ndarray,
    count: np.ndarray,
) -> np.ndarray:
    """Standard error of a cumulative incidence, by the delta method.

    Three terms, each of which would be a double sum over ages written directly:

        sum over a <= t of (F(t) - F(a))^2 * d(a) / (n(a) * (n(a) - d(a)))
      + sum over a <= t of S(a-)^2 * (n(a) - d_k(a)) / n(a) * d_k(a) / n(a)^2
      - 2 * sum over a <= t of (F(t) - F(a)) * S(a-) * d_k(a) / n(a)^2

    ``F(t) - F(a)`` depends on both ends, so each square is expanded and accumulated as
    running sums instead: three hundred ages here, but 312 x 312 terms a segment is a
    different proposition when every view is computed per grade and per vintage.
    """
    safe = np.where(at_risk > leaving, at_risk * (at_risk - leaving), np.inf)
    first = leaving / safe
    second = entering**2 * (at_risk - count) * count / at_risk**3
    third = entering * count / at_risk**2

    # (F(t) - F(a))^2 = F(t)^2 - 2 F(t) F(a) + F(a)^2, and the same trick for the cross term.
    squared = (
        incidence**2 * np.cumsum(first)
        - 2.0 * incidence * np.cumsum(first * incidence)
        + np.cumsum(first * incidence**2)
        + np.cumsum(second)
        - 2.0 * (incidence * np.cumsum(third) - np.cumsum(third * incidence))
    )
    return np.sqrt(np.clip(squared, 0.0, None))
