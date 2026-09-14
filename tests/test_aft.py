"""Tests for the AFT fitter wrapper."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
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


def test_chunked_prediction_matches_a_single_call(encoded: pd.DataFrame) -> None:
    """Blocking must not move a single number.

    The blocks exist because one call allocates rows x horizon, which on the real
    panel is 41.5 GB against 16 GB of RAM. That is a memory fix, not a modelling one,
    so "identical apart from chunking" has to be asserted -- it is exactly the kind of
    claim that quietly stops being true.
    """
    from creditsurv.models import aft

    covariates = ["fico_s", "cltv_drift", "unemp_gap"]
    result = fit_aft(encoded, covariates, " + ".join(covariates))

    frame = encoded.loc[:, covariates]
    ages = encoded["age"].to_numpy(dtype=int)

    whole = aft.episode_hazards(result, frame, ages)
    # A budget small enough to force many blocks on a fixture this size.
    blocked = aft.episode_hazards(result, frame, ages, budget_bytes=8 * 1024)

    assert aft._block_size(int(ages.max()) + 2) > 1
    # Not bit-for-bit: BLAS multiplies differently shaped matrices along different
    # blockings, so the last bit of a dot product moves. Measured at 8e-16 absolute on
    # a hazard of order 1e-3 -- a tolerance this tight would still catch any change
    # that alters an answer anyone asks of it.
    np.testing.assert_allclose(blocked, whole, rtol=1e-12, atol=1e-15)


def test_the_block_size_shrinks_as_the_horizon_grows() -> None:
    """The budget is on memory, not on rows: a wider age grid means fewer rows."""
    from creditsurv.models.aft import _block_size

    assert _block_size(60) > _block_size(327)
    assert _block_size(10**9) == 1


def test_the_event_count_is_weighted_whenever_there_is_a_weight(encoded: pd.DataFrame) -> None:
    """A row of an aggregated panel is a number of loan-months, not one.

    The count was weighted only in the branch with no event column, so on the production
    panel -- which carries both a weight and an event flag -- it counted *cells*.
    methodology.md published 935,939 defaults where the fit had seen 1,866,441, and the
    backtesting report from the same run said otherwise.
    """
    unweighted = fit_aft(encoded, COVARIATES, FORMULA)
    weighted = fit_aft(encoded.assign(n=3), COVARIATES, FORMULA, weights_col="n")

    assert unweighted.n_events == int(encoded["event"].sum())
    assert weighted.n_events == 3 * unweighted.n_events
