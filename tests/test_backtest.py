"""Tests for splits, metrics and the backtest runner.

The split tests matter most. A leaking split produces excellent numbers and no
error, so the absence of look-ahead is asserted structurally rather than trusted.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from creditsurv.backtest.metrics import (
    actual_versus_expected,
    brier_score,
    calibration_slope_intercept,
    calibration_table,
    discrimination,
    population_stability_index,
    stability_report,
    weighted_calibration,
    weighted_gini,
)
from creditsurv.backtest.runner import MacroMode, macro_mode_gap, run_backtest, run_split
from creditsurv.backtest.splits import (
    as_of_split,
    assert_no_lookahead,
    cell_split,
    out_of_sample,
    out_of_time,
    walk_forward,
)
from fixtures import DEFAULT_PARAMS, build_panel

if TYPE_CHECKING:
    from pathlib import Path

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
def panel(book_dir: Path, macro_module: pd.DataFrame) -> pd.DataFrame:
    built, _ = build_panel(book_dir, macro_module, n_loans=1500, seed=91, params=PARAMS)
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


def test_a_fold_with_no_defaults_is_reported_not_raised(
    panel: pd.DataFrame, macro_module: pd.DataFrame
) -> None:
    """lifelines raises ZeroDivisionError from concordance_index when a fold holds
    no admissible pairs, and it descends from ArithmeticError rather than
    ValueError -- so catching ValueError alone lets it through.

    That is exactly how a small nightly run died while every test stayed green: the
    suite always used panels large enough to have defaults in every fold.
    """
    late = pd.Period("2014-11", freq="M")
    splits = walk_forward(panel.head(4000), [late])

    summary, results = run_backtest(splits, macro_module, COVARIATES, FORMULA, horizon_months=1)

    assert len(summary) == 2
    assert all(result.n_loans == 0 or result.metrics for result in results)
    for result in results:
        if not result.metrics:
            assert result.error, "a fold that produced nothing must say why"


# --------------------------------------------------------------------------------------
# Aggregated panels
# --------------------------------------------------------------------------------------


def _cells() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": pd.PeriodIndex(
                ["2023-06", "2023-12", "2024-06", "2024-12", "2025-06"], freq="M"
            ),
            "age": [0, 6, 12, 24, 36],
            "event": [False, True, False, True, False],
            "n": [1000, 10, 900, 8, 700],
        }
    )


def test_cell_split_divides_on_calendar_time() -> None:
    """Aggregated cells have no loan identifier -- that is what aggregating means --
    so the loan-level split does not apply."""
    split = cell_split(_cells(), pd.Period("2024-06", freq="M"))

    assert (split.train["period"] <= pd.Period("2024-06", freq="M")).all()
    assert (split.test["period"] > pd.Period("2024-06", freq="M")).all()
    assert len(split.train) == 3


def test_cell_split_reports_exposure_not_loans() -> None:
    """Exposure is the quantity both panel shapes can report, and the one a backtest
    is actually sized by."""
    described = cell_split(_cells(), pd.Period("2024-06", freq="M")).describe()

    assert described["train_loan_months"] == 1910
    assert described["test_loan_months"] == 708


def test_cell_split_rejects_a_date_before_the_data() -> None:
    with pytest.raises(ValueError, match="No exposure"):
        cell_split(_cells(), pd.Period("2000-01", freq="M"))


def test_weighted_gini_rewards_a_correct_ordering() -> None:
    """A cell is not a subject, so a concordance index has no pairs to form. The
    Lorenz curve asks the equivalent question of the data that does exist."""
    exposure = pd.Series([1000.0] * 4)
    events = pd.Series([40.0, 20.0, 10.0, 5.0])
    good = pd.Series([0.04, 0.02, 0.01, 0.005])
    reversed_order = pd.Series([0.005, 0.01, 0.02, 0.04])

    assert weighted_gini(good, events, exposure) > 0.3
    assert weighted_gini(reversed_order, events, exposure) < 0


def test_weighted_gini_is_zero_without_ordering() -> None:
    exposure = pd.Series([1000.0] * 4)
    events = pd.Series([20.0] * 4)

    assert weighted_gini(pd.Series([0.01, 0.02, 0.03, 0.04]), events, exposure) == pytest.approx(
        0.0, abs=1e-9
    )


def test_weighted_gini_is_undefined_without_events() -> None:
    exposure = pd.Series([1000.0, 1000.0])
    events = pd.Series([0.0, 0.0])

    assert np.isnan(weighted_gini(pd.Series([0.01, 0.02]), events, exposure))


def test_weighted_calibration_buckets_by_exposure() -> None:
    """Buckets are weighted so each is a comparable slice of the book, not of the
    cell table -- which is an artefact of the binning."""
    predicted = pd.Series(np.linspace(0.001, 0.05, 100))
    exposure = pd.Series(np.full(100, 1000.0))
    events = predicted * exposure

    table = weighted_calibration(predicted, events, exposure, n_buckets=5)

    assert len(table) == 5
    assert table["expected"].is_monotonic_increasing
    assert int(table["loan_months"].sum()) == 100_000
    assert np.allclose(table["ratio"], 1.0, atol=0.05)


def test_actual_versus_expected_flags_under_prediction() -> None:
    """Above one the model under-predicts; below one it over-predicts."""
    frame = pd.DataFrame(
        {
            "predicted": [0.01, 0.01],
            "events": [20.0, 5.0],
            "exposure": [1000.0, 1000.0],
            "vintage": ["2006", "2012"],
        }
    )

    table = actual_versus_expected(
        frame["predicted"], frame["events"], frame["exposure"], frame["vintage"]
    ).set_index("group")

    assert cell(table, "2006", "actual_over_expected") == pytest.approx(2.0)
    assert cell(table, "2012", "actual_over_expected") == pytest.approx(0.5)


def cell(frame: pd.DataFrame, row: object, column: str) -> float:
    """Read one numeric cell by label; pandas-stubs cannot narrow a label lookup."""
    return float(frame.loc[frame.index == row, column].to_numpy(dtype=float)[0])
