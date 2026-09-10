"""Tests for lifetime PD, the term structure and macro scenarios."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from creditsurv.data.panel import at_origination, to_interval_censored
from creditsurv.models.aft import FitResult, fit_aft
from creditsurv.models.lifetime_pd import (
    ADVERSE,
    BASELINE,
    Scenario,
    conditional_pd,
    extend_macro,
    pd_term_structure,
    project_panel,
    scenario_lifetime_pd,
    survival_along_path,
)
from fixtures import DEFAULT_PARAMS, build_panel

if TYPE_CHECKING:
    from pathlib import Path

COVARIATES = ["fico_s", "cltv_drift", "unemp_gap"]
FORMULA = " + ".join(COVARIATES)
HORIZON = 60

PARAMS = replace(
    DEFAULT_PARAMS,
    intercept=4.9,
    continuous={"fico_s": 0.34, "cltv_drift": -0.020, "unemp_gap": -0.105},
    categorical={},
    prepayment_intercept=50.0,
)


@pytest.fixture(scope="module")
def panel(book_dir: Path, macro_module: pd.DataFrame) -> pd.DataFrame:
    built, _ = build_panel(book_dir, macro_module, n_loans=900, seed=81, params=PARAMS)
    return built


@pytest.fixture(scope="module")
def fitted(panel: pd.DataFrame) -> FitResult:
    return fit_aft(to_interval_censored(panel), COVARIATES, FORMULA)


@pytest.fixture(scope="module")
def new_business(panel: pd.DataFrame, macro_module: pd.DataFrame) -> pd.DataFrame:
    """Loans written at the reporting date, so the whole path is forward-looking."""
    loans = at_origination(panel).head(300).copy()
    loans["age"] = 0
    loans["orig_period"] = macro_module.index.max() + 1
    loans["period"] = macro_module.index.max() + 1
    return loans


@pytest.fixture(scope="module")
def forward_survival(
    fitted: FitResult, new_business: pd.DataFrame, macro_module: pd.DataFrame
) -> pd.DataFrame:
    extended = extend_macro(macro_module, HORIZON + 2, BASELINE)
    projected = project_panel(new_business, extended, horizon_months=HORIZON)
    return survival_along_path(fitted, projected, COVARIATES)


def test_survival_is_a_probability_and_decreases(forward_survival: pd.DataFrame) -> None:
    values = forward_survival.to_numpy()

    assert ((values >= 0.0) & (values <= 1.0)).all()
    assert (np.diff(values, axis=1) <= 1e-12).all()


def test_projected_paths_are_balanced(forward_survival: pd.DataFrame) -> None:
    """Every loan present at every age, which the term structure requires."""
    assert not forward_survival.isna().to_numpy().any()
    assert forward_survival.shape[1] == HORIZON


def test_pd_is_bounded_and_grows_with_horizon(forward_survival: pd.DataFrame) -> None:
    twelve = conditional_pd(forward_survival, horizon_months=12)
    lifetime = conditional_pd(forward_survival)

    assert ((twelve >= 0.0) & (twelve <= 1.0)).all()
    assert (lifetime >= twelve - 1e-12).all()


def test_conditional_pd_matches_the_survival_ratio(forward_survival: pd.DataFrame) -> None:
    """1 - S(t+h)/S(t) is the definition; this guards the indexing around it."""
    computed = conditional_pd(forward_survival, as_of_month=12, horizon_months=12)
    expected = 1.0 - forward_survival[24] / forward_survival[12]

    pd.testing.assert_series_equal(computed, expected.rename("pd"))


def test_conditioning_on_survival_raises_the_pd(forward_survival: pd.DataFrame) -> None:
    """A seasoned loan faces a higher hazard here, because the baseline hazard
    rises with age over this horizon. The point is that the two differ at all:
    quoting a lifetime PD without saying what it is conditioned on is ambiguous."""
    from_origination = conditional_pd(forward_survival, as_of_month=0, horizon_months=12)
    from_month_twelve = conditional_pd(forward_survival, as_of_month=12, horizon_months=12)

    assert from_month_twelve.mean() > from_origination.mean()


def test_horizon_beyond_the_projection_is_rejected(forward_survival: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="beyond the projected"):
        conditional_pd(forward_survival, horizon_months=HORIZON + 12)


def test_term_structure_is_internally_consistent(forward_survival: pd.DataFrame) -> None:
    table = pd_term_structure(forward_survival)

    assert table["cumulative_pd"].is_monotonic_increasing
    assert table["survival"].is_monotonic_decreasing
    assert np.allclose(table["survival"] + table["cumulative_pd"], 1.0)
    assert table["marginal_pd"].sum() == pytest.approx(table["cumulative_pd"].iloc[-1])


def test_term_structure_rejects_a_ragged_table(fitted: FitResult, panel: pd.DataFrame) -> None:
    """Averaging a ragged table mixes different loans at different ages and shows a
    falling hazard near the horizon that is only a change of population."""
    observed = survival_along_path(fitted, to_interval_censored(panel), COVARIATES)

    with pytest.raises(ValueError, match="missing cells"):
        pd_term_structure(observed)


def test_extend_macro_appends_a_random_walk(macro_module: pd.DataFrame) -> None:
    extended = extend_macro(macro_module, 6, BASELINE)

    assert len(extended) == len(macro_module) + 6
    last_observed = macro_module.iloc[-1]
    for column in macro_module.columns:
        assert extended[column].tail(6).eq(last_observed[column]).all()


def test_adverse_scenario_moves_the_macro_series(macro_module: pd.DataFrame) -> None:
    extended = extend_macro(macro_module, 36, ADVERSE)
    last_observed = macro_module.iloc[-1]

    assert extended["unemployment_rate"].iloc[-1] > last_observed["unemployment_rate"]
    assert extended["hpi"].iloc[-1] < last_observed["hpi"]


def test_house_price_shocks_are_proportional(macro_module: pd.DataFrame) -> None:
    """A house price index is a level, so a twenty percent fall must mean the same
    thing whatever the index happens to be."""
    scenario = Scenario(name="crash", shocks={"hpi": [-0.5]})

    extended = extend_macro(macro_module, 3, scenario)

    expected = float(macro_module["hpi"].iloc[-1]) * 0.5
    assert extended["hpi"].iloc[-1] == pytest.approx(expected)


def test_projection_requires_an_age_column(
    new_business: pd.DataFrame, macro_module: pd.DataFrame
) -> None:
    extended = extend_macro(macro_module, 12, BASELINE)

    with pytest.raises(ValueError, match="age"):
        project_panel(new_business.drop(columns=["age"]), extended, horizon_months=6)


def test_projection_requires_macro_to_reach_the_horizon(
    new_business: pd.DataFrame, macro_module: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="Extend it with extend_macro"):
        project_panel(new_business, macro_module, horizon_months=24)


def test_adverse_scenario_raises_lifetime_pd(
    fitted: FitResult, new_business: pd.DataFrame, macro_module: pd.DataFrame
) -> None:
    """The test that would have caught projecting from origination instead of from
    the reporting date: that bug left the two scenarios almost identical."""
    result = scenario_lifetime_pd(fitted, new_business, macro_module, COVARIATES, horizon_months=36)

    assert result["adverse"].mean() > result["baseline"].mean() * 1.25
    assert (result["adverse"] >= result["baseline"]).mean() > 0.95


def test_lifelines_conditional_after_matches_the_survival_ratio(
    fitted: FitResult, new_business: pd.DataFrame
) -> None:
    """lifelines' own conditioning, checked against its definition.

    ``conditional_after`` is the built-in way to ask for survival given the loan
    has already lasted ``t`` months, and for **fixed** covariates it is exactly
    ``S(t+h)/S(t)``. This test pins that.

    It is also the reason the rest of this module does not use it. The identity
    holds with the covariates held at one set of values, and half of ours move over
    the life of the loan -- two of them starting at zero by construction. Applied
    to a time-varying model it silently answers a question about a loan whose
    circumstances never change.
    """
    subject = new_business.head(20).loc[:, COVARIATES]
    as_of, horizon = 12.0, 24.0

    conditioned = fitted.fitter.predict_survival_function(
        subject, times=[horizon], conditional_after=np.full(len(subject), as_of)
    ).to_numpy()[0]
    unconditional = fitted.fitter.predict_survival_function(
        subject, times=[as_of, as_of + horizon]
    ).to_numpy()
    ratio = unconditional[1] / unconditional[0]

    assert np.allclose(conditioned, ratio, rtol=1e-6)
