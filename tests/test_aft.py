"""Tests for the AFT fitter wrapper."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from lifelines.exceptions import ConvergenceError

from creditsurv.data.panel import to_interval_censored
from creditsurv.models.aft import (
    CONVERGENT_DISTRIBUTIONS,
    FITTERS,
    Likelihood,
    coefficient_table,
    fit_aft,
)
from fixtures import DEFAULT_PARAMS, build_panel

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

COVARIATES = ["fico_s", "cltv_drift", "unemp_gap"]
FORMULA = "fico_s + cltv_drift + unemp_gap"

SMALL_PARAMS = replace(
    DEFAULT_PARAMS,
    intercept=4.7,
    continuous={"fico_s": 0.34, "cltv_drift": -0.020, "unemp_gap": -0.105},
    categorical={},
    prepayment_intercept=50.0,
)


@pytest.fixture(scope="module")
def encoded(book_dir: Path, macro_module: pd.DataFrame) -> pd.DataFrame:
    panel, _ = build_panel(book_dir, macro_module, n_loans=900, seed=23, params=SMALL_PARAMS)
    return to_interval_censored(panel)


@pytest.mark.parametrize("distribution", CONVERGENT_DISTRIBUTIONS)
def test_every_distribution_fits(encoded: pd.DataFrame, distribution: str) -> None:
    result = fit_aft(encoded, COVARIATES, FORMULA, distribution=distribution)

    assert result.n_episodes == len(encoded)
    assert result.n_events > 0
    assert result.distribution == distribution


@pytest.mark.xfail(raises=ConvergenceError, strict=True, reason="see FITTERS docstring")
def test_lognormal_does_not_converge_on_episode_panels(encoded: pd.DataFrame) -> None:
    """Characterisation test for a known limitation.

    The lognormal AFT fails on this panel structure across every remedy tried:
    sample sizes from 900 to 3,000 loans, with and without a penalizer, under
    L-BFGS-B and SLSQP, with left truncation on and off, and with durations
    rescaled. Weibull and log-logistic fit the identical rows. The cause was not
    established, so this pins the behaviour rather than explaining it -- and being
    strict, it will fail loudly if a later lifelines fixes the problem.
    """
    fit_aft(encoded, COVARIATES, FORMULA, distribution="lognormal")


def test_every_distribution_is_registered() -> None:
    assert set(CONVERGENT_DISTRIBUTIONS) <= set(FITTERS)


def test_unknown_distribution_is_rejected(encoded: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown distribution"):
        fit_aft(encoded, COVARIATES, FORMULA, distribution="gompertz")


def test_both_likelihoods_fit_the_same_episodes(encoded: pd.DataFrame) -> None:
    interval = fit_aft(encoded, COVARIATES, FORMULA, likelihood=Likelihood.INTERVAL_CENSORED)
    right = fit_aft(encoded, COVARIATES, FORMULA, likelihood=Likelihood.RIGHT_CENSORED)

    assert interval.n_episodes == right.n_episodes
    assert interval.likelihood is Likelihood.INTERVAL_CENSORED
    assert right.likelihood is Likelihood.RIGHT_CENSORED


def test_the_two_likelihoods_broadly_agree(encoded: pd.DataFrame) -> None:
    """Placing a default at month end rather than inside the month is a mild
    approximation when the monthly hazard is small, and the two fits should not
    diverge materially. A large gap here would mean one of the encodings is
    wrong, not that the approximation is expensive."""
    interval = fit_aft(encoded, COVARIATES, FORMULA, likelihood=Likelihood.INTERVAL_CENSORED)
    right = fit_aft(encoded, COVARIATES, FORMULA, likelihood=Likelihood.RIGHT_CENSORED)

    for covariate in COVARIATES:
        a = interval.fitter.params_[("lambda_", covariate)]
        b = right.fitter.params_[("lambda_", covariate)]
        standard_error = interval.fitter.standard_errors_[("lambda_", covariate)]
        assert abs(a - b) < 2.0 * standard_error


def test_coefficient_table_names_the_time_ratio(encoded: pd.DataFrame) -> None:
    """exp(coef) on an AFT model is a time ratio, not a hazard ratio.

    Reading it as a hazard ratio inverts the sign of every conclusion, so the
    column must not carry a name that invites the mistake.
    """
    table = coefficient_table(fit_aft(encoded, COVARIATES, FORMULA))

    assert "time_ratio" in table.columns
    assert "exp(coef)" not in table.columns
    assert {"coef", "se(coef)", "p"} <= set(table.columns)


def test_aic_is_reported_on_the_episode_scale(encoded: pd.DataFrame) -> None:
    """AIC counts episodes, not loans, so it compares distributions on one panel
    and nothing across panels."""
    result = fit_aft(encoded, COVARIATES, FORMULA)

    assert result.aic == pytest.approx(-2 * result.log_likelihood + 2 * 5, rel=1e-6)
