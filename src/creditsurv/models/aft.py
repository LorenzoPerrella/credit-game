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

import numpy as np
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
#: ``lognormal`` is offered but **does not converge on this panel structure**.
#: Observed across 900-3,000 loans, with and without a penalizer, under both
#: L-BFGS-B and SLSQP, with left truncation on and off, and with durations
#: rescaled -- every combination raises ``ConvergenceError`` while ``weibull``
#: and ``loglogistic`` fit the identical rows without complaint. The cause was
#: not established, so no mechanism is claimed here; the fact is recorded and
#: pinned by a strict xfail in ``tests/test_aft.py``, which will fail if a later
#: lifelines makes it work.
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

#: Distributions that reliably converge on episode panels, used as the default
#: candidate set for model selection.
CONVERGENT_DISTRIBUTIONS: Final[tuple[str, ...]] = ("weibull", "loglogistic")


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


def _check_frequency_weights(weights: pd.Series, name: str) -> None:
    """Reject anything that is not a count of observations.

    Grouped estimation is only valid with frequency weights: integer counts of
    identical rows. The tempting alternative -- weighting by the outstanding
    amount of each position, as loss models do -- answers a different question. It
    estimates a value-weighted default rate rather than a borrower probability of
    default, and Basel and IFRS 9 both define PD per obligor. It also breaks
    inference, because lifelines derives the standard errors by treating weights
    as replication counts and warns that non-integer weights bias them.

    If exposure should influence the model, ``log_orig_upb`` is already a
    covariate, and loss severity belongs in a separate model.
    """
    values = weights.to_numpy(dtype=float)
    if (values <= 0).any():
        message = f"Weight column {name!r} contains non-positive values."
        raise ValueError(message)
    if not np.array_equal(values, np.round(values)):
        message = (
            f"Weight column {name!r} is not integer-valued. Grouped estimation "
            "expects frequency weights: counts of identical loan-months. Weighting "
            "by exposure estimates a value-weighted default rate rather than a "
            "borrower PD, and biases the standard errors. Use log_orig_upb as a "
            "covariate if exposure should matter."
        )
        raise ValueError(message)


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
    show_progress: bool = False,
) -> FitResult:
    """Fit a parametric AFT model to an encoded episode panel.

    ``encoded`` must come from :func:`creditsurv.data.panel.to_interval_censored`.
    ``covariates`` lists the raw columns the formula may read; narrowing them
    explicitly keeps identifiers and raw macro out of the design matrix.

    ``ancillary`` lets the shape parameter depend on covariates, relaxing the
    assumption that a covariate shifts the timing of default without changing the
    shape of the hazard over the life of the loan.

    ``show_progress`` prints the optimiser's iterations. On a table of this size a fit
    is tens of minutes, and the difference between "converging slowly" and "not
    converging" is worth being able to see without waiting for the answer.
    """
    if distribution not in FITTERS:
        message = f"Unknown distribution {distribution!r}; expected one of {sorted(FITTERS)}."
        raise ValueError(message)

    if weights_col is not None:
        _check_frequency_weights(encoded[weights_col], weights_col)

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
            show_progress=show_progress,
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
            show_progress=show_progress,
        )

    elapsed = time.perf_counter() - started
    # Weighted whenever there is a weight. The first version weighted only in the branch
    # with no event column, so on the aggregated panel -- which carries both -- it
    # counted the *cells* flagged as a default rather than the loan-months they stand
    # for. methodology.md published 935,939 defaults where the fit had seen 1,866,441,
    # contradicting the backtesting report generated by the same run.
    counts = (
        encoded[weights_col].to_numpy(dtype=float)
        if weights_col is not None
        else np.ones(len(encoded))
    )
    if EVENT in encoded.columns:
        defaulted = encoded[EVENT].to_numpy(dtype=bool)
    else:
        # No event column: a row is a default when its interval is bounded above.
        defaulted = np.isfinite(encoded[UPPER_BOUND].to_numpy(dtype=float))
    n_events = int(counts[defaulted].sum())
    return FitResult(
        fitter=fitter,
        distribution=distribution,
        likelihood=likelihood,
        formula=formula,
        n_episodes=len(encoded),
        n_events=n_events,
        elapsed_seconds=elapsed,
    )


#: Memory the row-by-time prediction grid may occupy, in bytes. 256 MB leaves room for
#: the copies lifelines makes internally while staying large enough that the per-block
#: overhead is negligible against the arithmetic.
_PREDICTION_BUDGET: Final = 256 * 1024 * 1024


def episode_hazards(
    result: FitResult,
    frame: pd.DataFrame,
    ages: np.ndarray,
    *,
    budget_bytes: int = _PREDICTION_BUDGET,
) -> np.ndarray:
    """Monthly conditional failure probability for each row, at its own age.

    ``1 - S(a+1 | x) / S(a | x)`` evaluated with the covariates the row actually
    carries. This is the quantity the episode likelihood is built from, and the
    building block for every survival curve in the project.

    lifelines predicts a full row-by-time grid, of which exactly two cells per row are
    wanted. Asking for it in one call is the obvious implementation and it does not
    survive contact with this panel: 15.9 million rows over a 327-month grid is a
    **41.5 GB** array, on a machine with 16 GB. The process is killed by the kernel
    with no traceback, two and a half hours into a run, at the first line that touches
    a full-size prediction -- which is a memorable way to discover an O(rows x horizon)
    allocation.

    So the rows are processed in blocks sized to a memory budget rather than to a row
    count: the wider the age grid, the fewer rows per block. The arithmetic inside a
    block is unchanged, so the result is identical to the single-call version -- which
    a test asserts, because "identical apart from chunking" is exactly the kind of
    claim that quietly stops being true.
    """
    horizon = int(ages.max()) + 2
    grid = np.arange(0.0, float(horizon))
    block = _block_size(horizon, budget_bytes)

    hazard = np.empty(len(frame), dtype=float)
    for start in range(0, len(frame), block):
        stop = min(start + block, len(frame))
        rows = frame.iloc[start:stop]
        at = ages[start:stop]
        cumulative = result.fitter.predict_cumulative_hazard(rows, times=grid).to_numpy()
        columns = np.arange(cumulative.shape[1])
        increment = cumulative[at + 1, columns] - cumulative[at, columns]
        hazard[start:stop] = 1.0 - np.exp(-increment)
    return hazard


def _block_size(horizon: int, budget_bytes: int = _PREDICTION_BUDGET) -> int:
    """How many rows can be predicted at once within the budget."""
    return max(int(budget_bytes / (horizon * 8)), 1)


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
