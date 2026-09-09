"""Tests for distribution selection and the non-parametric benchmarks.

The recovery test proves the encoding is right. These prove the *selection
procedure* is right, by checking it identifies the family the data was generated
from rather than merely producing a ranking.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from creditsurv.data.panel import at_origination, to_interval_censored
from creditsurv.data.synthetic import DEFAULT_PARAMS, build_synthetic_panel
from creditsurv.models.aft import FitResult, fit_aft
from creditsurv.models.nonparametric import (
    kaplan_meier,
    kaplan_meier_by_stratum,
    km_band_contains,
    predicted_survival_curve,
    turnbull,
)
from creditsurv.models.selection import (
    distribution_comparison,
    likelihood_ratio_test,
    marginal_comparison,
    shape_depends_on_covariates,
)

COVARIATES = ["fico_s", "cltv_drift", "unemp_gap"]
FORMULA = " + ".join(COVARIATES)

#: The generator is Weibull with a constant shape. Selection should say so.
PARAMS = replace(
    DEFAULT_PARAMS,
    intercept=4.9,
    continuous={"fico_s": 0.34, "cltv_drift": -0.020, "unemp_gap": -0.105},
    categorical={},
    prepayment_intercept=50.0,
)


@pytest.fixture(scope="module")
def panel(macro_module: pd.DataFrame) -> pd.DataFrame:
    built, _ = build_synthetic_panel(macro_module, n_loans=1500, seed=71, params=PARAMS)
    return built


@pytest.fixture(scope="module")
def encoded(panel: pd.DataFrame) -> pd.DataFrame:
    return to_interval_censored(panel)


@pytest.fixture(scope="module")
def fitted(encoded: pd.DataFrame) -> FitResult:
    return fit_aft(encoded, COVARIATES, FORMULA)


def test_marginal_comparison_identifies_the_generating_family(panel: pd.DataFrame) -> None:
    """The generator is Weibull, so the ranking should put Weibull first."""
    table = marginal_comparison(panel)

    assert table.iloc[0]["distribution"] == "weibull"
    assert table.iloc[0]["delta_aic"] == 0.0
    assert (table["delta_aic"] >= 0).all()


def test_regression_comparison_prefers_the_generating_family(
    encoded: pd.DataFrame,
) -> None:
    table = distribution_comparison(encoded, COVARIATES, FORMULA)

    assert table.iloc[0]["distribution"] == "weibull"
    assert set(table["n_episodes"]) == {len(encoded)}


def test_shape_test_does_not_reject_a_constant_shape(encoded: pd.DataFrame) -> None:
    """The generator uses one shape for every loan, so the extra parameter should
    fail to earn its place. A test that rejected here would be finding structure
    that is not in the data."""
    result = shape_depends_on_covariates(encoded, COVARIATES, FORMULA, "fico_s")

    assert result.iloc[0]["p_value"] > 0.05
    assert result.iloc[0]["added_parameters"] == 1


def test_likelihood_ratio_test_rejects_a_degenerate_comparison() -> None:
    with pytest.raises(ValueError, match="more parameters"):
        likelihood_ratio_test(-100.0, -90.0, 0)


def test_likelihood_ratio_test_matches_a_known_value() -> None:
    statistic, p_value = likelihood_ratio_test(-100.0, -98.0, 1)

    assert statistic == pytest.approx(4.0)
    assert p_value == pytest.approx(0.0455, abs=1e-3)


def test_fitted_curve_stays_inside_the_kaplan_meier_band(
    panel: pd.DataFrame, encoded: pd.DataFrame, fitted: FitResult
) -> None:
    """The strongest distribution-free evidence that the imposed shape holds.

    This is the test that caught predicting from origination covariates: doing so
    left only half the horizon inside the band.
    """
    curve = kaplan_meier(panel)
    predicted = predicted_survival_curve(fitted, encoded, COVARIATES)

    comparison = km_band_contains(curve, predicted)
    assert comparison["inside"].all()


def test_survival_curve_is_monotone_and_bounded(encoded: pd.DataFrame, fitted: FitResult) -> None:
    predicted = predicted_survival_curve(fitted, encoded, COVARIATES)

    assert predicted.is_monotonic_decreasing
    assert ((predicted >= 0.0) & (predicted <= 1.0)).all()


def test_turnbull_agrees_with_kaplan_meier(panel: pd.DataFrame) -> None:
    """One-month intervals, so the two should be close but not identical."""
    km = kaplan_meier(panel).survival_function_
    tb = turnbull(panel).survival_function_

    assert float(tb.loc[60.0].iloc[0]) == pytest.approx(float(km.loc[60.0].iloc[0]), abs=0.02)


def test_turnbull_timeline_is_finite(panel: pd.DataFrame) -> None:
    """Without an explicit timeline the last row sits at t = inf, where survival
    is zero, and is easily mistaken for the end of the horizon."""
    assert np.isfinite(turnbull(panel).survival_function_.index.to_numpy()).all()


def test_stratified_curves_separate_by_credit_quality(panel: pd.DataFrame) -> None:
    """Stratification is all Kaplan-Meier can do with a covariate, and only for
    one fixed at origination."""
    banded = panel.copy()
    banded["quality"] = pd.cut(banded["fico_s"], bins=[-10.0, 0.0, 10.0], labels=["weak", "strong"])

    curves = kaplan_meier_by_stratum(banded, "quality")

    weak = float(curves["weak"].survival_function_.iloc[-1].iloc[0])
    strong = float(curves["strong"].survival_function_.iloc[-1].iloc[0])
    assert weak < strong


def test_at_origination_returns_the_first_month(panel: pd.DataFrame) -> None:
    origination = at_origination(panel)

    assert (origination["age"] == 0).all()
    assert len(origination) == panel["loan_id"].nunique()
