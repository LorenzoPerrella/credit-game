"""Tests for distribution selection and the non-parametric benchmarks.

The recovery test proves the encoding is right. These prove the *selection
procedure* is right, by checking it identifies the family the data was generated
from rather than merely producing a ranking.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from creditsurv.data.panel import at_origination, to_interval_censored
from creditsurv.models.aft import FitResult, fit_aft
from creditsurv.models.nonparametric import (
    kaplan_meier,
    kaplan_meier_by_stratum,
    km_band_contains,
    predicted_survival_curve,
    turnbull,
)
from creditsurv.models.selection import (
    backward_elimination,
    distribution_comparison,
    likelihood_ratio_test,
    marginal_comparison,
    shape_depends_on_covariates,
    stepwise_vif,
    univariate_screening,
    variance_inflation,
)
from fixtures import DEFAULT_PARAMS, build_panel

if TYPE_CHECKING:
    from pathlib import Path

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
def panel(book_dir: Path, macro_module: pd.DataFrame) -> pd.DataFrame:
    built, _ = build_panel(book_dir, macro_module, n_loans=1500, seed=71, params=PARAMS)
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


# --------------------------------------------------------------------------------------
# Variable selection
# --------------------------------------------------------------------------------------


def test_vif_is_one_when_covariates_are_independent() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {"a": rng.normal(size=500), "b": rng.normal(size=500), "c": rng.normal(size=500)}
    )

    table = variance_inflation(frame, ["a", "b", "c"]).set_index("covariate")

    assert float(table.loc[table.index == "a", "vif"].iloc[0]) < 1.1


def test_vif_detects_a_near_duplicate() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=500)
    frame = pd.DataFrame(
        {"a": base, "b": base + rng.normal(scale=0.01, size=500), "c": rng.normal(size=500)}
    )

    table = variance_inflation(frame, ["a", "b", "c"])

    assert float(table.iloc[0]["vif"]) > 100.0
    assert str(table.iloc[0]["covariate"]) in {"a", "b"}


def test_vif_is_weighted_by_exposure() -> None:
    """Unweighted on an aggregated panel would measure collinearity among cells,
    which is a property of the binning rather than of the data."""
    rng = np.random.default_rng(2)
    base = rng.normal(size=400)
    frame = pd.DataFrame(
        {
            "a": base,
            "b": base * 0.5 + rng.normal(scale=0.02, size=400),
            "n": rng.integers(1, 1000, 400),
        }
    )

    weighted = variance_inflation(frame, ["a", "b"], weight="n")
    unweighted = variance_inflation(frame, ["a", "b"])

    assert float(weighted.iloc[0]["vif"]) != float(unweighted.iloc[0]["vif"])


def test_stepwise_vif_drops_until_under_the_threshold() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(size=600)
    frame = pd.DataFrame(
        {"keep": rng.normal(size=600), "a": base, "b": base + rng.normal(scale=0.01, size=600)}
    )

    log, surviving = stepwise_vif(frame, ["keep", "a", "b"], threshold=10.0)

    assert len(log) == 1
    assert len(surviving) == 2
    assert "keep" in surviving


def test_priority_decides_which_collinear_covariate_survives() -> None:
    """The procedure has no view on which of two collinear covariates the model is
    for. Left alone it drops whichever has the larger factor, which is arbitrary and
    unstable across samples."""
    rng = np.random.default_rng(4)
    base = rng.normal(size=600)
    frame = pd.DataFrame({"a": base, "b": base + rng.normal(scale=0.01, size=600)})

    _, keeps_b = stepwise_vif(frame, ["a", "b"], priority=["a", "b"])
    _, keeps_a = stepwise_vif(frame, ["a", "b"], priority=["b", "a"])

    assert keeps_b == ["b"]
    assert keeps_a == ["a"]


def test_stepwise_vif_leaves_independent_covariates_alone() -> None:
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})

    log, surviving = stepwise_vif(frame, ["a", "b"])

    assert log.empty
    assert surviving == ["a", "b"]


def test_a_backwards_sign_is_eliminated_even_when_significant(
    encoded: pd.DataFrame,
) -> None:
    """A wrong sign is a symptom, usually of collinearity, not a weak result.

    A model asserting that higher credit scores default sooner fits its sample and
    no other, so significance does not save it.
    """
    flipped = encoded.copy()
    flipped["fico_s"] = -flipped["fico_s"]

    log, surviving, _ = backward_elimination(flipped, ["fico_s", "cltv_drift"])

    assert "fico_s" not in surviving
    assert str(log.iloc[0]["reason"]) == "wrong sign"


def test_backward_elimination_keeps_covariates_that_earn_their_place(
    encoded: pd.DataFrame,
) -> None:
    log, surviving, result = backward_elimination(encoded, ["fico_s", "cltv_drift"])

    assert set(surviving) == {"fico_s", "cltv_drift"}
    assert log.empty
    assert result.n_events > 0


def test_univariate_screening_ranks_by_significance(encoded: pd.DataFrame) -> None:
    table = univariate_screening(encoded, ["fico_s", "cltv_drift"])

    assert list(table.columns) >= ["covariate", "coef", "p", "aic", "keep"]
    assert table["p"].is_monotonic_increasing
    assert table["keep"].any()


def test_the_exponential_is_tested_without_a_second_fit(
    panel: pd.DataFrame,
) -> None:
    """The exponential nests inside the Weibull, so it costs a Wald test, not a fit.

    Where families nest, an information criterion is both more expensive and weaker
    than the test that is available.
    """
    from creditsurv.models.selection import exponential_is_rejected

    fitted = fit_aft(to_interval_censored(panel), COVARIATES, FORMULA)
    verdict = exponential_is_rejected(fitted)

    assert verdict["rho"] == pytest.approx(np.exp(verdict["log_rho"]))
    assert verdict["z"] == pytest.approx(verdict["log_rho"] / verdict["standard_error"])
    # The fixture draws from a Weibull with a rising hazard, so a constant one is wrong.
    assert verdict["p_value"] < 0.05


def test_the_exponential_test_refuses_the_wrong_family() -> None:
    """It nests inside the Weibull and nowhere else, so asking of another family is
    a mistake rather than an approximation."""
    from creditsurv.models.selection import exponential_is_rejected

    class _Stub:
        distribution = "loglogistic"

    with pytest.raises(ValueError, match="nests inside the Weibull"):
        exponential_is_rejected(_Stub())  # type: ignore[arg-type]


def test_every_candidate_has_an_economic_dimension_fixed_in_advance() -> None:
    """The stability rule only compares covariates of one dimension.

    A candidate without a dimension could never be judged by it, and one assigned a
    dimension after the fits would be judged by a rule written to fit the result.
    """
    from creditsurv.config import (
        ECONOMIC_DIMENSION,
        MACRO_CANDIDATES,
        MACRO_ELIMINATION_PRIORITY,
        ORDINAL,
        STATIC_CONTINUOUS,
    )

    candidates = {*MACRO_CANDIDATES, *STATIC_CONTINUOUS, *ORDINAL}
    assert candidates <= set(ECONOMIC_DIMENSION), sorted(candidates - set(ECONOMIC_DIMENSION))
    assert set(MACRO_ELIMINATION_PRIORITY) == set(MACRO_CANDIDATES)


def test_factors_read_back_from_a_saved_correlation_are_the_ones_the_rows_give() -> None:
    """The notebook shows the variance inflation from the correlation the selection saved,
    so the two routes have to agree."""
    from creditsurv.explore import weighted_correlation
    from creditsurv.models.selection import inflation_from_covariance

    rng = np.random.default_rng(8)
    base = rng.normal(size=4000)
    frame = pd.DataFrame(
        {
            "a": base,
            "b": base + 0.1 * rng.normal(size=4000),
            "c": rng.normal(size=4000),
            "n": rng.integers(1, 5, 4000).astype(float),
        }
    )

    rows = variance_inflation(frame, ["a", "b", "c"], weight="n").set_index("covariate")["vif"]
    saved = weighted_correlation(frame, ["a", "b", "c"], weight="n")
    table = inflation_from_covariance(saved).set_index("covariate")["vif"]

    np.testing.assert_allclose(table.loc[rows.index], rows, rtol=1e-8)
