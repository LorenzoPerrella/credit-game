"""Parameter recovery against a known data-generating process.

The strongest evidence in the project that the encoding is right.

The generator draws from *exactly* the estimand the interval-censored likelihood
maximises -- the per-episode conditional failure probability of a Weibull AFT --
so the model is not merely plausible for this data, it is correct for it. A
coefficient that misses its confidence interval therefore indicates a bug in the
encoding, not a misspecified model. Without that equivalence the test would only
show the fit was reasonable, which is worth much less.

Two design notes:

*Reduced specification.* Three covariates, no categoricals, no prepayment. The
full specification produces roughly 500 events against 21 parameters, where the
intercept trades off against ``log_original_balance`` and ``origination_spread`` and several
coefficients are genuinely weakly identified. That is a property of the sample
size, not of the encoding, and a test that fails for it would be testing the
wrong thing.

*Pinned seed, and enough of them.* With five parameters at 95% coverage, roughly
one seed in four shows a miss by chance, so the seed is pinned rather than the
assertion loosened -- a weaker assertion would stop catching the bugs this test
exists for. The sample was also raised to 5,000 loans (about 1,500 defaults):
checked across five seeds the estimates scatter around the truth with no
systematic bias, so a miss is sampling noise, and the cure for sampling noise is
information rather than a friendlier threshold.

*Through the loader.* The book is written as Freddie Mac files and read back with
the real loader, so this test also covers the parsing, the missing-value
sentinels and the truncation at the first terminating month.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from creditsurv.data.panel import to_interval_censored
from creditsurv.models.aft import FitResult, Likelihood, fit_aft
from fixtures import DEFAULT_PARAMS, build_panel

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

RECOVERY_SEED = 17
COVARIATES = ["credit_score", "ltv_change", "unemployment_change"]
FORMULA = "credit_score + ltv_change + unemployment_change"

TRUE_COEFFICIENTS = {"credit_score": 0.0068, "ltv_change": -0.020, "unemployment_change": -0.105}
TRUE_INTERCEPT = 0.24

#: The score the intercept is read at: the middle of the book, not a score of zero.
SCORE_CENTRE = 700.0

#: Prepayment is switched off by pushing its scale far out. The recovery claim is
#: about the default model; censoring is exercised elsewhere.
RECOVERY_PARAMS = replace(
    DEFAULT_PARAMS,
    intercept=TRUE_INTERCEPT,
    continuous=TRUE_COEFFICIENTS,
    categorical={},
    prepayment_intercept=50.0,
)


@pytest.fixture(scope="module")
def fitted(book_dir: Path, macro_module: pd.DataFrame) -> FitResult:
    panel, _ = build_panel(
        book_dir, macro_module, n_loans=5000, seed=RECOVERY_SEED, params=RECOVERY_PARAMS
    )
    encoded = to_interval_censored(panel)
    return fit_aft(encoded, COVARIATES, FORMULA, likelihood=Likelihood.INTERVAL_CENSORED)


def _interval(fitted: FitResult, parameter: str, covariate: str) -> tuple[float, float]:
    lower, upper = fitted.fitter.confidence_intervals_.loc[(parameter, covariate)]
    return float(lower), float(upper)


def _estimate(fitted: FitResult, parameter: str, covariate: str) -> float:
    return float(fitted.fitter.params_[(parameter, covariate)])


@pytest.mark.parametrize(("covariate", "truth"), list(TRUE_COEFFICIENTS.items()))
def test_covariate_coefficients_are_recovered(
    fitted: FitResult, covariate: str, truth: float
) -> None:
    lower, upper = _interval(fitted, "lambda_", covariate)

    assert lower <= truth <= upper, (
        f"{covariate}: true {truth:+.4f} outside 95% CI "
        f"[{lower:+.4f}, {upper:+.4f}] (estimate {_estimate(fitted, 'lambda_', covariate):+.4f})"
    )


def test_intercept_is_recovered(fitted: FitResult) -> None:
    """At a score of 700, where the data are, rather than at a score of zero.

    The credit score is in points, so the intercept alone is the log scale of a loan scored
    zero -- 700 points outside the data, where it trades off almost exactly against the
    score's coefficient. The quantity the encoding determines is the intercept at the
    centre of the book, ``Intercept + 700 * credit_score``, which is the intercept this
    test held when the score was stored as ``(score - 700) / 50``.
    """
    params = fitted.fitter.params_
    covariance = fitted.fitter.variance_matrix_
    intercept, score = ("lambda_", "Intercept"), ("lambda_", "credit_score")
    estimate = float(params[intercept] + SCORE_CENTRE * params[score])
    variance = float(
        covariance.loc[intercept, intercept]
        + SCORE_CENTRE**2 * covariance.loc[score, score]
        + 2.0 * SCORE_CENTRE * covariance.loc[intercept, score]
    )
    truth = TRUE_INTERCEPT + SCORE_CENTRE * TRUE_COEFFICIENTS["credit_score"]

    assert abs(estimate - truth) <= 1.96 * variance**0.5


def test_shape_parameter_is_recovered(fitted: FitResult) -> None:
    """The shape is the parameter most sensitive to a broken encoding.

    Mis-stating when an episode starts or ends distorts the hazard's shape over
    loan age long before it moves a covariate coefficient.
    """
    lower, upper = _interval(fitted, "rho_", "Intercept")

    assert lower <= RECOVERY_PARAMS.log_rho <= upper


def test_covariate_signs_match_the_generating_process(fitted: FitResult) -> None:
    """A weaker claim than recovery, but one that must never fail.

    Sign errors survive every structural test and invert the model's meaning.
    """
    for covariate, truth in TRUE_COEFFICIENTS.items():
        estimate = _estimate(fitted, "lambda_", covariate)
        assert estimate * truth > 0, f"{covariate}: sign flipped ({estimate:+.4f})"
