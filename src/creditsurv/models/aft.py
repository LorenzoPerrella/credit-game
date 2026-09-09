"""Parametric accelerated failure time models with time-varying covariates.

lifelines presents these as alternatives rather than companions: time-varying
covariates are steered towards the semi-parametric ``CoxTimeVaryingFitter``, and
``fit_interval_censoring`` is documented one row per subject. On an episode-split
panel they combine, because the interval-censored likelihood with left truncation
telescopes into the discrete-time likelihood with time-varying covariates. See
:mod:`creditsurv.data.panel` for the encoding.

A parametric model is not a stylistic preference here. Lifetime PD needs a
survival curve beyond the observation window, a response to macroeconomic
scenarios, and a smooth term structure. A Cox model supplies none of the three.

Two likelihoods are offered over the *same* episodes:

``interval_censored``
    The faithful statement of monthly reporting: the default month is known, the
    day is not.
``right_censored``
    The common approximation, which places the default at month end.

Fitting both on identical rows is what turns "interval censoring matters" from an
assertion into a measurement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter

from creditsurv.data.panel import (
    AGE_START,
    AGE_STOP,
    EVENT,
    EXACT_OBSERVATION,
    LOWER_BOUND,
    UPPER_BOUND,
    model_frame,
    right_censored_frame,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from lifelines.fitters import ParametericAFTRegressionFitter


class Likelihood(StrEnum):
    """How the observation interval of an episode is treated."""

    INTERVAL_CENSORED = "interval_censored"
    RIGHT_CENSORED = "right_censored"


#: Distributions available for the baseline hazard, by name.
#:
#: The generalised gamma is deliberately absent: it is a
#: ``ParametricRegressionFitter`` rather than an AFT fitter, taking per-parameter
#: ``regressors`` instead of ``formula``/``ancillary``. It is wrapped separately
#: in :mod:`creditsurv.models.selection`, where it is used to *test* the
#: distributional family rather than to fit a final model.
FITTERS: Final[dict[str, type[ParametericAFTRegressionFitter]]] = {
    "weibull": WeibullAFTFitter,
    "lognormal": LogNormalAFTFitter,
    "loglogistic": LogLogisticAFTFitter,
}


@dataclass(frozen=True)
class FitResult:
    """A fitted model together with what it was fitted on.

    The counts are carried because they are needed to read the fit and are not
    recoverable from the fitter afterwards. In particular ``AIC_`` is computed on
    *episodes*, not loans, so it is comparable across distributions on one panel
    and meaningless across different panel constructions.
    """

    fitter: ParametericAFTRegressionFitter
    distribution: str
    likelihood: Likelihood
    formula: str
    n_episodes: int
    n_events: int
    elapsed_seconds: float

    @property
    def aic(self) -> float:
        """Akaike information criterion, on the episode scale."""
        return float(self.fitter.AIC_)

    @property
    def log_likelihood(self) -> float:
        return float(self.fitter.log_likelihood_)


def fit_aft(
    encoded: pd.DataFrame,
    covariates: Sequence[str],
    formula: str,
    *,
    distribution: str = "weibull",
    likelihood: Likelihood = Likelihood.INTERVAL_CENSORED,
    penalizer: float = 0.0,
    weights_col: str | None = None,
    ancillary: str | bool | None = None,
) -> FitResult:
    """Fit a parametric AFT model to an encoded episode panel.

    ``encoded`` must come from :func:`creditsurv.data.panel.to_interval_censored`.
    ``covariates`` lists the raw columns the formula may read; narrowing them
    explicitly keeps identifiers and raw macro out of the design matrix.

    ``ancillary`` lets the shape parameter depend on covariates, relaxing the
    assumption that a covariate shifts the timing of default without changing the
    shape of the hazard over the life of the loan.
    """
    if distribution not in FITTERS:
        message = f"Unknown distribution {distribution!r}; expected one of {sorted(FITTERS)}."
        raise ValueError(message)

    fitter = FITTERS[distribution](penalizer=penalizer)
    started = time.perf_counter()

    if likelihood is Likelihood.INTERVAL_CENSORED:
        frame = model_frame(encoded, covariates)
        if weights_col is not None:
            frame[weights_col] = encoded[weights_col].to_numpy()
        fitter.fit_interval_censoring(
            frame,
            lower_bound_col=LOWER_BOUND,
            upper_bound_col=UPPER_BOUND,
            event_col=EXACT_OBSERVATION,
            entry_col=AGE_START,
            formula=formula,
            weights_col=weights_col,
            ancillary=ancillary,
        )
    else:
        frame = right_censored_frame(encoded, covariates)
        if weights_col is not None:
            frame[weights_col] = encoded[weights_col].to_numpy()
        fitter.fit(
            frame,
            duration_col=AGE_STOP,
            event_col=EVENT,
            entry_col=AGE_START,
            formula=formula,
            weights_col=weights_col,
            ancillary=ancillary,
        )

    elapsed = time.perf_counter() - started
    n_events = (
        int(encoded[EVENT].sum()) if EVENT in encoded.columns else int(fitter.event_observed.sum())
    )
    return FitResult(
        fitter=fitter,
        distribution=distribution,
        likelihood=likelihood,
        formula=formula,
        n_episodes=len(encoded),
        n_events=n_events,
        elapsed_seconds=elapsed,
    )


def coefficient_table(result: FitResult) -> pd.DataFrame:
    """Coefficients with standard errors, confidence intervals and time ratios.

    ``exp(coef)`` on an AFT model is a **time ratio**, not a hazard ratio: a value
    above one lengthens expected survival and therefore lowers risk. Reading it as
    a hazard ratio inverts the sign of every conclusion, so the column is named
    for what it is.
    """
    summary = result.fitter.summary.copy()
    summary = summary.rename(columns={"exp(coef)": "time_ratio"})
    keep = [
        column
        for column in ("coef", "se(coef)", "time_ratio", "coef lower 95%", "coef upper 95%", "p")
        if column in summary.columns
    ]
    table: pd.DataFrame = summary.loc[:, keep]
    return table
