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
    distribution_comparison,
    likelihood_ratio_test,
    marginal_comparison,
    shape_depends_on_covariates,
    shape_formula,
    stepwise_vif,
    variance_inflation,
)
from fixtures import DEFAULT_PARAMS, build_panel

if TYPE_CHECKING:
    from pathlib import Path

COVARIATES = ["credit_score", "ltv_change", "unemployment_change"]
FORMULA = " + ".join(COVARIATES)

#: The generator is Weibull with a constant shape. Selection should say so.
PARAMS = replace(
    DEFAULT_PARAMS,
    intercept=0.14,
    continuous={"credit_score": 0.0068, "ltv_change": -0.020, "unemployment_change": -0.105},
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
    # The generator's own family points every declared prior the right way.
    assert table.iloc[0]["signs_against_prior"] == ""


def test_shape_test_does_not_reject_a_constant_shape(encoded: pd.DataFrame) -> None:
    """The generator uses one shape for every loan, so the extra parameter should
    fail to earn its place. A test that rejected here would be finding structure
    that is not in the data."""
    result = shape_depends_on_covariates(encoded, COVARIATES, FORMULA, "credit_score")

    assert result.iloc[0]["p_value"] > 0.05
    assert result.iloc[0]["added_parameters"] == 1


def test_the_shape_test_takes_a_categorical_into_the_shape(encoded: pd.DataFrame) -> None:
    """What ``report --extra-fits`` now runs: ``occupancy`` in the shape parameter, against
    its reference, through the block engine. Three levels, so two shape parameters, and the
    larger model can never fit worse than the one nested in it."""
    covariates = [*COVARIATES, "occupancy"]
    references = {"occupancy": "owner_occupied"}
    formula = f"{FORMULA} + C(occupancy, Treatment('owner_occupied'))"
    relaxed, ancillary = shape_formula(covariates, references)

    result = shape_depends_on_covariates(encoded, covariates, formula, ancillary)

    assert relaxed == "occupancy"
    assert encoded["occupancy"].nunique() == 3
    assert result.iloc[0]["added_parameters"] == 2
    assert result.iloc[0]["full_log_likelihood"] >= result.iloc[0]["restricted_log_likelihood"]


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
    banded["quality"] = pd.cut(
        banded["credit_score"], bins=[0.0, 700.0, 1000.0], labels=["weak", "strong"]
    )

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
            "loan_months": rng.integers(1, 1000, 400),
        }
    )

    weighted = variance_inflation(frame, ["a", "b"], weight="loan_months")
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
            "loan_months": rng.integers(1, 5, 4000).astype(float),
        }
    )

    rows = variance_inflation(frame, ["a", "b", "c"], weight="loan_months").set_index("covariate")[
        "vif"
    ]
    saved = weighted_correlation(frame, ["a", "b", "c"], weight="loan_months")
    table = inflation_from_covariance(saved).set_index("covariate")["vif"]

    np.testing.assert_allclose(table.loc[rows.index], rows, rtol=1e-8)


def test_the_shape_test_relaxes_the_covariate_whose_curves_cross() -> None:
    """S3: the report relaxed the first covariate, ``credit_score``, while the stratum whose curves
    cross -- the assumption the test exists for -- was ``occupancy``."""
    covariates = ["credit_score", "ltv_change", "purpose", "occupancy"]
    references = {"purpose": "purchase", "occupancy": "owner_occupied"}

    assert shape_formula(covariates, references) == (
        "occupancy",
        "C(occupancy, Treatment('owner_occupied'))",
    )
    assert shape_formula(["credit_score", "purpose"], {"purpose": "purchase"}) == (
        "credit_score",
        "credit_score",
    )


def test_the_comparison_fits_through_the_function_it_is_given(encoded: pd.DataFrame) -> None:
    """The report passes a cached fit, so a family already estimated costs nothing again;
    and it keeps every family's fit, to hold each against Kaplan-Meier."""
    calls: list[str] = []

    def counted(*args: object, **kwargs: object) -> FitResult:
        calls.append(str(kwargs["distribution"]))
        return fit_aft(*args, **kwargs)  # type: ignore[arg-type]

    weibull = fit_aft(encoded, COVARIATES, FORMULA)
    kept: dict[str, FitResult] = {}
    distribution_comparison(encoded, COVARIATES, FORMULA, fitted=weibull, fit=counted, fits=kept)

    assert calls == ["loglogistic"]
    assert set(kept) == {"weibull", "loglogistic"}
    assert kept["weibull"] is weibull


# --------------------------------------------------------------------------------------
# Competing risks
# --------------------------------------------------------------------------------------


def _competing_cells(
    rng: np.random.Generator, loans: int = 400
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A book where loans default, prepay or survive, as cells and as loans.

    The cells are what the pipeline holds -- one row per loan-month, carrying the outcome
    of that month -- and the loan-level frame is what lifelines needs to be asked the same
    question.
    """
    from creditsurv.data.panel import AGE, OUTCOME, WEIGHT

    duration = rng.integers(1, 25, size=loans)
    cause = rng.choice(["default", "prepayment", "none"], size=loans, p=[0.2, 0.5, 0.3])
    rows = []
    for months, ending in zip(duration, cause, strict=True):
        for age in range(int(months)):
            last = age == int(months) - 1
            rows.append({AGE: age, OUTCOME: ending if last else "none", WEIGHT: 1})
    cells = pd.DataFrame(rows)
    loan_level = pd.DataFrame({"duration": duration.astype(float), "cause": cause})
    return cells, loan_level


def test_the_cumulative_incidence_is_what_lifelines_computes_from_the_loans() -> None:
    """Estimated from counts over the cells, with no loan-level frame at any point.

    On distinct exit times, because that is the only ground lifelines can be asked to
    stand on: ``AalenJohansenFitter`` cannot handle tied event times and **jitters the
    data randomly** to break them. A monthly panel is nothing but ties -- every loan
    exits on a month boundary -- so the estimator here groups them exactly instead, and
    the agreement is checked where the two definitions coincide.
    """
    from lifelines import AalenJohansenFitter

    from creditsurv.data.panel import AGE, OUTCOME, WEIGHT
    from creditsurv.models.nonparametric import cumulative_incidence

    rng = np.random.default_rng(11)
    duration = np.arange(1, 41)
    cause = rng.choice(["default", "prepayment", "none"], size=len(duration), p=[0.3, 0.4, 0.3])
    cells = pd.DataFrame(
        [
            {AGE: age, OUTCOME: ending if age == months - 1 else "none", WEIGHT: 1}
            for months, ending in zip(duration, cause, strict=True)
            for age in range(int(months))
        ]
    )
    codes = pd.Series(cause).map({"none": 0, "default": 1, "prepayment": 2})

    table = cumulative_incidence(cells)

    for name, code in (("default", 1), ("prepayment", 2)):
        fitter = AalenJohansenFitter(calculate_variance=False)
        fitter.fit(duration.astype(float), codes, event_of_interest=code)
        theirs = fitter.cumulative_density_.iloc[:, 0]
        ours = pd.Series(table[name].to_numpy(), index=table["age"].to_numpy() + 1.0)
        aligned = theirs.reindex(theirs.index.union(ours.index)).ffill().reindex(ours.index)
        np.testing.assert_allclose(ours.to_numpy(), aligned.to_numpy(), atol=1e-12)


def test_the_two_incidences_and_survival_account_for_every_loan() -> None:
    """The identity that makes a competing-risks model a model of one population:
    a loan has defaulted, prepaid, or is still there.
    """
    from creditsurv.models.nonparametric import cumulative_incidence

    cells, _ = _competing_cells(np.random.default_rng(5))

    table = cumulative_incidence(cells)

    total = table["survival"] + table["default"] + table["prepayment"]
    np.testing.assert_allclose(total.to_numpy(), 1.0, atol=1e-12)


def test_treating_prepayment_as_censoring_overstates_the_default_incidence() -> None:
    """One minus Kaplan-Meier answers what would happen if loans could not be repaid,
    which on this book is not a hypothetical worth provisioning against: prepayment
    removes loans far faster than default does.
    """
    from lifelines import KaplanMeierFitter

    from creditsurv.models.nonparametric import cumulative_incidence

    cells, loans = _competing_cells(np.random.default_rng(3))
    table = cumulative_incidence(cells)

    censoring = KaplanMeierFitter()
    censoring.fit(loans["duration"], (loans["cause"] == "default").astype(int))
    naive = 1.0 - float(censoring.survival_function_.iloc[-1, 0])

    assert naive > float(table["default"].iloc[-1])


def test_the_incidence_band_is_reported_and_is_as_narrow_as_the_counts_make_it() -> None:
    from creditsurv.models.nonparametric import cumulative_incidence

    cells, _ = _competing_cells(np.random.default_rng(7), loans=4_000)

    table = cumulative_incidence(cells)

    assert (table["default_lower"] <= table["default"]).all()
    assert (table["default"] <= table["default_upper"]).all()
    assert (table["default_se"] >= 0).all()
    # The band narrows as the square root of the count: four thousand loans already
    # bring it under two percentage points, and the book is four million.
    assert float(table["default_se"].iloc[-1]) < 0.02


def test_the_family_rule_excludes_a_broken_model_before_it_ranks_anything() -> None:
    """Rule 2 of docs/rules.md, first clause: a model that turns a declared sign is not a
    worse model, it is a broken one, and no likelihood or distance redeems it.
    """
    from creditsurv.models.selection import family_by_the_rule, family_comparison

    gaps = {
        "weibull": pd.DataFrame({"gap_pp": [0.9, -1.1]}),
        "loglogistic": pd.DataFrame({"gap_pp": [0.1, -0.1]}),
    }
    comparison = family_comparison(gaps, {"loglogistic": ["financial_conditions"]})

    chosen, why = family_by_the_rule(comparison)

    assert chosen == "weibull"
    assert "loglogistic excluded on a declared sign" in why
    assert bool(comparison.set_index("distribution").loc["loglogistic", "excluded"])


def test_the_family_rule_keeps_the_weibull_on_a_tie() -> None:
    """Within a tenth of a percentage point the two are not distinguishable by this data,
    and the Weibull's hazard does not fall at long ages -- which is where the families
    differ most and where a lifetime PD spends its time.
    """
    from creditsurv.models.selection import family_by_the_rule, family_comparison

    gaps = {
        "weibull": pd.DataFrame({"gap_pp": [1.05, -1.05]}),
        "loglogistic": pd.DataFrame({"gap_pp": [1.0, -1.0]}),
    }

    chosen, why = family_by_the_rule(family_comparison(gaps, {}))

    assert chosen == "weibull"
    assert "inside the 0.1 pp tie" in why


def test_the_family_rule_takes_the_closer_model_when_the_gap_is_real() -> None:
    from creditsurv.models.selection import family_by_the_rule, family_comparison

    gaps = {
        "weibull": pd.DataFrame({"gap_pp": [2.0, -2.0]}),
        "loglogistic": pd.DataFrame({"gap_pp": [1.0, -1.0]}),
    }

    chosen, why = family_by_the_rule(family_comparison(gaps, {}))

    assert chosen == "loglogistic"
    assert "1.000 pp closer" in why


def test_the_predicted_incidence_is_comparable_with_the_observed_one() -> None:
    """Both curves are built the same way -- a mean hazard by age, chained -- which is what
    makes the gap between them a statement about the model rather than about two different
    constructions.
    """
    from creditsurv.data.panel import AGE, OUTCOME, WEIGHT
    from creditsurv.models.nonparametric import (
        cumulative_incidence,
        incidence_gap,
        predicted_incidence_curve,
    )

    rng = np.random.default_rng(23)
    duration = rng.integers(1, 30, size=3_000)
    cause = rng.choice(["default", "prepayment", "none"], size=3_000, p=[0.25, 0.45, 0.3])
    cells = pd.DataFrame(
        [
            {AGE: age, OUTCOME: ending if age == months - 1 else "none", WEIGHT: 1}
            for months, ending in zip(duration, cause, strict=True)
            for age in range(int(months))
        ]
    )
    observed = cumulative_incidence(cells)

    # A "model" that knows the answer: the observed hazard of each cause at each age.
    ages = cells[AGE].to_numpy(dtype=int)
    hazards = {}
    for name in ("default", "prepayment"):
        ended = (cells[OUTCOME] == name).to_numpy(dtype=float)
        at_risk = np.bincount(ages)
        hazards[name] = (np.bincount(ages, weights=ended) / at_risk)[ages]
    predicted = predicted_incidence_curve(hazards, ages)

    np.testing.assert_allclose(
        predicted["default"].to_numpy(), observed["default"].to_numpy(), atol=1e-12
    )
    gap = incidence_gap(predicted, observed, exposure_floor=100.0)
    assert not gap.empty
    assert gap["at_risk"].min() >= 100.0
    assert float(gap["gap_pp"].abs().max()) < 1e-9
