"""Tests for splits, metrics and the backtest runner.

The split tests matter most. A leaking split produces excellent numbers and no
error, so the absence of look-ahead is asserted structurally rather than trusted.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from creditsurv.backtest.metrics import (
    brier_score,
    calibration_slope_intercept,
    calibration_table,
    discrimination,
    population_stability_index,
    stability_report,
)
from creditsurv.backtest.runner import MacroMode, macro_mode_gap, run_backtest, run_split
from creditsurv.backtest.splits import (
    as_of_split,
    assert_no_lookahead,
    out_of_sample,
    out_of_time,
    walk_forward,
)
from creditsurv.data.synthetic import DEFAULT_PARAMS, build_synthetic_panel

COVARIATES = ["fico_s", "cltv_drift", "unemp_gap"]
FORMULA = " + ".join(COVARIATES)
AS_OF = pd.Period("2008-12", freq="M")

PARAMS = replace(
    DEFAULT_PARAMS,
    intercept=4.9,
    continuous={"fico_s": 0.34, "cltv_drift": -0.020, "unemp_gap": -0.105},
    categorical={},
    prepayment_intercept=50.0,
)


@pytest.fixture(scope="module")
def panel(macro_module: pd.DataFrame) -> pd.DataFrame:
    built, _ = build_synthetic_panel(macro_module, n_loans=1500, seed=91, params=PARAMS)
    return built


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [as_of_split, out_of_time, out_of_sample])
def test_no_split_leaks_the_future_into_training(panel: pd.DataFrame, factory: object) -> None:
    """The failure this guards against produces excellent numbers and no error."""
    split = factory(panel, AS_OF)  # type: ignore[operator]

    assert_no_lookahead(split)
    assert split.train["period"].max() <= AS_OF


def test_as_of_split_holds_only_loans_on_the_books(panel: pd.DataFrame) -> None:
    """Written by the reporting date and not yet terminated.

    Dropping the origination condition lets later vintages in, whose test window
    would open years after the reporting date -- so their covariates would describe
    a different economy from the one being scored.
    """
    split = as_of_split(panel, AS_OF)

    first_test_period = split.test.groupby("loan_id", observed=True)["period"].min()
    assert (first_test_period == AS_OF + 1).all()


def test_out_of_time_holds_only_later_vintages(panel: pd.DataFrame) -> None:
    split = out_of_time(panel, AS_OF)

    assert (split.test["orig_period"] > AS_OF).all()
    assert not set(split.train["loan_id"]) & set(split.test["loan_id"])


def test_out_of_sample_separates_loans_not_periods(panel: pd.DataFrame) -> None:
    split = out_of_sample(panel, AS_OF, test_fraction=0.3, seed=1)

    assert not set(split.train["loan_id"]) & set(split.test["loan_id"])
    assert split.test["period"].max() <= AS_OF


def test_walk_forward_expands_the_training_window(panel: pd.DataFrame) -> None:
    dates = [pd.Period(p, freq="M") for p in ("2006-12", "2008-12", "2010-12")]

    splits = walk_forward(panel, dates)

    sizes = [len(split.train) for split in splits]
    assert sizes == sorted(sizes)
    assert [split.as_of for split in splits] == dates


def test_lookahead_check_catches_a_corrupted_split(panel: pd.DataFrame) -> None:
    split = as_of_split(panel, AS_OF)
    leaked = replace(split, train=panel)

    with pytest.raises(ValueError, match="beyond the reporting date"):
        assert_no_lookahead(leaked)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def test_discrimination_is_above_a_half_for_a_useful_score() -> None:
    durations = pd.Series([5.0, 10.0, 15.0, 20.0])
    events = pd.Series([True, True, True, False])
    predicted = pd.Series([0.9, 0.6, 0.3, 0.1])

    result = discrimination(durations, events, predicted)

    assert result["concordance"] > 0.9
    assert result["gini"] == pytest.approx(2 * result["concordance"] - 1)


def test_reversing_the_score_reverses_concordance() -> None:
    """Guards the negation: a sign error reads as a broken model, not a flipped one."""
    durations = pd.Series([5.0, 10.0, 15.0, 20.0])
    events = pd.Series([True, True, True, False])

    good = discrimination(durations, events, pd.Series([0.9, 0.6, 0.3, 0.1]))
    bad = discrimination(durations, events, pd.Series([0.1, 0.3, 0.6, 0.9]))

    assert good["concordance"] + bad["concordance"] == pytest.approx(1.0)


def test_brier_score_rewards_confident_correctness() -> None:
    observed = pd.Series([True, False, True, False])

    confident = brier_score(observed, pd.Series([0.95, 0.05, 0.95, 0.05]))
    hedged = brier_score(observed, pd.Series([0.5, 0.5, 0.5, 0.5]))

    assert confident < hedged


def test_calibration_table_buckets_by_predicted_risk() -> None:
    rng = np.random.default_rng(0)
    predicted = pd.Series(rng.uniform(0.01, 0.4, 500))
    observed = pd.Series(rng.uniform(size=500) < predicted)

    table = calibration_table(observed, predicted, n_buckets=5)

    assert len(table) == 5
    assert table["expected"].is_monotonic_increasing
    assert table["loans"].sum() == 500


def test_calibration_reports_actual_over_expected() -> None:
    predicted = pd.Series([0.1] * 100)
    observed = pd.Series([True] * 20 + [False] * 80)

    result = calibration_slope_intercept(observed, predicted)

    assert result["expected"] == pytest.approx(0.1)
    assert result["actual"] == pytest.approx(0.2)
    assert result["actual_over_expected"] == pytest.approx(2.0)


def test_psi_is_zero_for_an_unchanged_distribution() -> None:
    rng = np.random.default_rng(1)
    sample = pd.Series(rng.normal(size=2000))

    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_when_a_distribution_moves() -> None:
    rng = np.random.default_rng(2)
    reference = pd.Series(rng.normal(size=2000))
    shifted = pd.Series(rng.normal(loc=1.5, size=2000))

    assert population_stability_index(reference, shifted) > 0.25


def test_time_varying_covariates_are_labelled_not_flagged() -> None:
    """Their index is large whenever the economy moved, which is not a defect.

    Scoring them against thresholds meant for application characteristics would
    trigger a model review every time anything happened.
    """
    rng = np.random.default_rng(3)
    train = pd.DataFrame({"fico_s": rng.normal(size=800), "unemp_gap": rng.normal(size=800)})
    test = pd.DataFrame(
        {"fico_s": rng.normal(size=800), "unemp_gap": rng.normal(loc=3.0, size=800)}
    )

    report = stability_report(train, test, ["fico_s", "unemp_gap"], time_varying=["unemp_gap"])
    by_covariate = report.set_index("covariate")

    assert by_covariate.loc["unemp_gap", "interpretation"] == "expected to move"
    assert by_covariate.loc["fico_s", "interpretation"] == "stable"


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------


def test_a_split_can_be_scored_end_to_end(panel: pd.DataFrame, macro_module: pd.DataFrame) -> None:
    result = run_split(
        as_of_split(panel, AS_OF), macro_module, COVARIATES, FORMULA, horizon_months=12
    )

    assert result.n_loans > 0
    assert result.n_defaults > 0
    assert 0.0 <= result.metrics["concordance"] <= 1.0
    assert not result.calibration.empty


def test_the_model_discriminates_better_than_chance(
    panel: pd.DataFrame, macro_module: pd.DataFrame
) -> None:
    result = run_split(
        as_of_split(panel, AS_OF), macro_module, COVARIATES, FORMULA, horizon_months=12
    )

    assert result.metrics["concordance"] > 0.55


def test_both_macro_modes_run_and_are_reported(
    panel: pd.DataFrame, macro_module: pd.DataFrame
) -> None:
    splits = walk_forward(panel, [AS_OF, pd.Period("2010-12", freq="M")])

    summary, results = run_backtest(splits, macro_module, COVARIATES, FORMULA, horizon_months=12)

    assert len(summary) == 4
    assert set(summary["macro_mode"]) == {"conditional", "unconditional"}
    assert len(results) == 4


def test_macro_mode_gap_is_reported_on_calibration(
    panel: pd.DataFrame, macro_module: pd.DataFrame
) -> None:
    """Discrimination barely moves between the modes; the level does.

    A macro path shifts every loan's PD in the same direction, so the ranking
    survives and the calibration does not.
    """
    splits = walk_forward(panel, [AS_OF])
    summary, _ = run_backtest(splits, macro_module, COVARIATES, FORMULA, horizon_months=12)

    concordance_gap = macro_mode_gap(summary, "concordance")
    calibration_gap = macro_mode_gap(summary, "actual_over_expected")

    assert abs(float(concordance_gap["gap"].iloc[0])) < 0.02
    assert "gap" in calibration_gap.columns


def test_unconditional_mode_cannot_see_the_future_macro(
    panel: pd.DataFrame, macro_module: pd.DataFrame
) -> None:
    """The two modes must actually differ, or the unconditional path is not being
    built from a truncated panel at all."""
    split = as_of_split(panel, AS_OF)

    conditional = run_split(
        split, macro_module, COVARIATES, FORMULA, macro_mode=MacroMode.CONDITIONAL
    )
    unconditional = run_split(
        split, macro_module, COVARIATES, FORMULA, macro_mode=MacroMode.UNCONDITIONAL
    )

    assert conditional.metrics["expected"] != unconditional.metrics["expected"]
