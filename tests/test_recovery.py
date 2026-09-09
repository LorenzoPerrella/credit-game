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
intercept trades off against ``log_orig_upb`` and ``orig_spread`` and several
coefficients are genuinely weakly identified. That is a property of the sample
size, not of the encoding, and a test that fails for it would be testing the
wrong thing.

*Pinned seed.* Everything is seeded, so the test is deterministic rather than
flaky. It is worth being explicit that with five parameters at 95% coverage,
roughly one seed in four would show a miss by chance. The seed is pinned rather
than the assertion loosened, because a weaker assertion would stop catching the
bugs this test exists for.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from creditsurv.data.panel import to_interval_censored
from creditsurv.data.synthetic import DEFAULT_PARAMS, build_synthetic_panel
from creditsurv.models.aft import FitResult, Likelihood, fit_aft

if TYPE_CHECKING:
    import pandas as pd

RECOVERY_SEED = 17
COVARIATES = ["fico_s", "cltv_drift", "unemp_gap"]
FORMULA = "fico_s + cltv_drift + unemp_gap"

TRUE_COEFFICIENTS = {"fico_s": 0.34, "cltv_drift": -0.020, "unemp_gap": -0.105}
TRUE_INTERCEPT = 5.0

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
def fitted(macro_module: pd.DataFrame) -> FitResult:
    panel, _ = build_synthetic_panel(
        macro_module, n_loans=3000, seed=RECOVERY_SEED, params=RECOVERY_PARAMS
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
    lower, upper = _interval(fitted, "lambda_", "Intercept")

    assert lower <= TRUE_INTERCEPT <= upper


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
