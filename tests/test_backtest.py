"""Tests for the split, the metrics and the backtest runner.

The split test matters most. A leaking split produces excellent numbers and no error,
so the absence of look-ahead is asserted structurally rather than trusted.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from creditsurv.backtest.metrics import (
    actual_versus_expected,
    population_stability_index,
    stability_report,
    weighted_calibration,
    weighted_gini,
)
from creditsurv.backtest.runner import run_backtest, score
from creditsurv.backtest.splits import Split, assert_no_lookahead, cell_split
from creditsurv.models.aft import fit_aft
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


# --------------------------------------------------------------------------------------
# Population stability
# --------------------------------------------------------------------------------------


def test_psi_is_zero_for_an_unchanged_distribution() -> None:
    values = pd.Series(np.random.default_rng(0).normal(size=2000))

    assert population_stability_index(values, values) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_when_a_distribution_moves() -> None:
    rng = np.random.default_rng(1)
    reference = pd.Series(rng.normal(size=4000))

    near = population_stability_index(reference, pd.Series(rng.normal(0.1, size=4000)))
    far = population_stability_index(reference, pd.Series(rng.normal(1.5, size=4000)))

    assert far > near


def test_psi_weights_by_exposure_not_by_row() -> None:
    """A cell stands for a number of loan-months, so an unweighted index would
    describe the binning rather than the book."""
    reference = pd.Series([0.0, 1.0])
    comparison = pd.Series([0.0, 1.0])
    # Same two values on both sides, but almost all the exposure has moved across.
    moved = population_stability_index(
        reference,
        comparison,
        reference_weights=pd.Series([999.0, 1.0]),
        comparison_weights=pd.Series([1.0, 999.0]),
    )
    unmoved = population_stability_index(reference, comparison)

    assert unmoved == pytest.approx(0.0, abs=1e-9)
    assert moved > 1.0


def test_time_varying_covariates_are_labelled_not_flagged() -> None:
    """The PSI thresholds were devised for application characteristics. A covariate
    designed to move with the economy is large whenever anything happened, which is
    not news and must not read as a model defect."""
    rng = np.random.default_rng(3)
    train = pd.DataFrame(
        {"fico_s": rng.normal(size=3000), "unemp_gap": rng.normal(size=3000), "n": 1.0}
    )
    test = pd.DataFrame(
        {"fico_s": rng.normal(size=3000), "unemp_gap": rng.normal(4.0, size=3000), "n": 1.0}
    )

    table = stability_report(
        train, test, ["fico_s", "unemp_gap"], time_varying=["unemp_gap"], weights_col="n"
    ).set_index("covariate")

    assert table.loc["unemp_gap", "interpretation"] == "expected to move"
    assert table.loc["fico_s", "interpretation"] == "stable"


# --------------------------------------------------------------------------------------
# The backtest, end to end
# --------------------------------------------------------------------------------------


def _encoded(panel: pd.DataFrame) -> pd.DataFrame:
    """The fixture book as a weighted, interval-censored panel."""
    from creditsurv.data.panel import to_interval_censored

    return to_interval_censored(panel.assign(n=1))


def test_lookahead_check_catches_a_corrupted_split(panel: pd.DataFrame) -> None:
    """A leaking split produces excellent numbers and no error, so it is caught
    structurally rather than trusted."""
    encoded = _encoded(panel)
    split = cell_split(encoded, AS_OF)
    corrupted = Split(as_of=split.as_of, train=encoded, test=split.test)

    with pytest.raises(ValueError, match="beyond the reporting date"):
        assert_no_lookahead(corrupted)


def test_the_backtest_fits_once_and_scores_what_follows(panel: pd.DataFrame) -> None:
    """One fit, then predicted against realised. Nothing is calibrated on the test half."""
    split, fitted, result = run_backtest(_encoded(panel), AS_OF, COVARIATES, FORMULA)

    assert (split.train["period"] <= AS_OF).all()
    assert (split.test["period"] > AS_OF).all()
    assert fitted.n_episodes == len(split.train)
    assert result.expected_defaults > 0
    assert result.loan_months == int(split.test["n"].sum())


def test_a_model_from_the_wrong_panel_is_refused(panel: pd.DataFrame) -> None:
    """The guarantee that "no further calibration" is a property of the code.

    A model fitted on the whole panel would score its own training data and return a
    flattering number with nothing visibly wrong, so passing one in is an error rather
    than a shortcut.
    """
    encoded = _encoded(panel)
    everything = fit_aft(encoded, COVARIATES, FORMULA, weights_col="n")

    with pytest.raises(ValueError, match="scoring its own training data"):
        run_backtest(encoded, AS_OF, COVARIATES, FORMULA, fitted=everything)


def test_passing_the_fitted_model_in_does_not_refit(panel: pd.DataFrame) -> None:
    encoded = _encoded(panel)
    split = cell_split(encoded, AS_OF)
    fitted = fit_aft(split.train, COVARIATES, FORMULA, weights_col="n")

    _, returned, _ = run_backtest(encoded, AS_OF, COVARIATES, FORMULA, fitted=fitted)

    assert returned is fitted


def test_scoring_recovers_the_realised_default_count(panel: pd.DataFrame) -> None:
    """On a book drawn from a known process, expected and realised should be close.

    Not a calibration claim about the model -- the process is known and the model is
    correctly specified for it -- but a check that the two sides of the comparison are
    counting the same thing. An expected count in the wrong units would be off by
    orders of magnitude, not by a few percent.
    """
    _, _, result = run_backtest(_encoded(panel), AS_OF, COVARIATES, FORMULA)

    assert result.actual_defaults > 0
    assert 0.5 < result.actual_over_expected < 2.0


def test_the_over_time_table_covers_the_whole_test_window(panel: pd.DataFrame) -> None:
    _, _, result = run_backtest(_encoded(panel), AS_OF, COVARIATES, FORMULA)

    assert (result.over_time["group"] > AS_OF).all()
    assert result.over_time["exposure"].sum() == pytest.approx(result.loan_months)


def test_score_refuses_an_empty_test_half(panel: pd.DataFrame) -> None:
    encoded = _encoded(panel)
    split = cell_split(encoded, AS_OF)
    fitted = fit_aft(split.train, COVARIATES, FORMULA, weights_col="n")

    with pytest.raises(ValueError, match="No exposure after"):
        score(fitted, split.test.iloc[:0], COVARIATES, as_of=AS_OF)


# --------------------------------------------------------------------------------------
# Acceptance, and the in-sample dispersion an out-of-time number is read against
# --------------------------------------------------------------------------------------


def test_a_backtest_inside_every_threshold_passes() -> None:
    from creditsurv.backtest.runner import ACCEPTANCE, BacktestResult

    result = BacktestResult(
        as_of=AS_OF,
        loan_months=1_000_000,
        expected_defaults=100.0,
        actual_defaults=102.0,
        gini=0.55,
        calibration=pd.DataFrame({"ratio": [0.90, 1.00, 1.10, 1.20]}),
    )

    assert ACCEPTANCE.passed(result)
    assert ACCEPTANCE.assess(result)["passed"].all()


@pytest.mark.parametrize(
    ("overall", "gini", "deciles", "failing"),
    [
        (0.75, 0.55, [0.90, 1.00], "actual / expected, overall"),
        (1.00, 0.40, [0.90, 1.00], "Gini, exposure-weighted"),
        # The first out-of-time result this model produced fails here and only here: its
        # overall 0.84 sits inside the band, its lowest decile at 0.54 does not. A single
        # overall figure would have passed a model whose calibration fails by decile.
        (1.00, 0.55, [0.54, 1.00], "actual / expected, every decile"),
    ],
)
def test_each_criterion_fails_on_its_own(
    overall: float, gini: float, deciles: list[float], failing: str
) -> None:
    """A criterion that cannot fail is not a criterion, so each is shown failing while
    the other two hold."""
    from creditsurv.backtest.runner import ACCEPTANCE, BacktestResult

    result = BacktestResult(
        as_of=AS_OF,
        loan_months=1_000_000,
        expected_defaults=100.0,
        actual_defaults=100.0 * overall,
        gini=gini,
        calibration=pd.DataFrame({"ratio": deciles}),
    )
    table = ACCEPTANCE.assess(result).set_index("criterion")

    assert not ACCEPTANCE.passed(result)
    assert not bool(table.loc[table.index == failing, "passed"].iloc[0])
    assert int(table["passed"].sum()) == len(table) - 1


def test_the_in_sample_years_arrive_beside_the_out_of_time_result(panel: pd.DataFrame) -> None:
    """An out-of-time actual-over-expected means nothing without the dispersion the model
    shows on data it has already seen, so the backtest carries both."""
    split, _, result = run_backtest(_encoded(panel), AS_OF, COVARIATES, FORMULA)

    table = result.in_sample_by_year
    training_years = set(pd.PeriodIndex(split.train["period"]).year)

    assert set(table["group"]) == training_years
    assert table["exposure"].sum() == pytest.approx(float(split.train["n"].sum()))
    assert (table["group"] <= AS_OF.year).all(), "in-sample must stop at the reporting date"


def test_splitting_the_cells_first_gives_the_halves_splitting_the_panel_would(
    macro: pd.DataFrame,
) -> None:
    """The halves are built from the cells so the whole panel never exists beside them.

    Expanding everything and then splitting holds the panel and both halves at once --
    ~12 GB on the exact calendar key -- and the two routes must agree to the row,
    including on the cells too early for their macro history to be built.
    """
    from creditsurv.backtest.splits import split_cells
    from creditsurv.data.panel import cells_to_episodes

    vintages = [1997 * 12, 2006 * 12, 2007 * 12]
    cells = pd.DataFrame(
        {
            "orig_month": [month for month in vintages for _ in range(8)],
            "purpose": pd.Categorical(["purchase", "refinance_cashout"] * 12),
            "fico_s": [0.4] * 24,
            "orig_ltv": [85.0] * 24,
            "age": list(range(8)) * 3,
            "event": ([False] * 7 + [True]) * 3,
            "n": [50] * 24,
        }
    )
    as_of = pd.Period("2007-03", freq="M")

    halves = split_cells(cells, macro, as_of)
    reference = cell_split(cells_to_episodes(cells, macro), as_of)

    pd.testing.assert_frame_equal(halves.train, reference.train)
    pd.testing.assert_frame_equal(halves.test, reference.test)


def test_group_totals_are_the_totals_a_group_by_gives() -> None:
    """The table is added up with bincount now, and must not move by a row or a digit."""
    rng = np.random.default_rng(11)
    size = 5_000
    predicted = pd.Series(rng.uniform(0.0, 0.01, size))
    exposure = pd.Series(rng.integers(1, 50, size).astype(float))
    observed = pd.Series(rng.binomial(1, 0.005, size) * exposure)
    by = pd.Series(
        pd.Categorical(rng.choice(["b", "a", "c"], size), categories=["c", "b", "a", "unused"])
    )

    table = actual_versus_expected(predicted, observed, exposure, by)

    frame = pd.DataFrame(
        {"group": by, "expected": predicted * exposure, "events": observed, "exposure": exposure}
    )
    columns = ["expected", "events", "exposure"]
    reference = frame.groupby("group", observed=True)[columns].sum().reset_index()
    pd.testing.assert_frame_equal(table[["group", *columns]], reference, rtol=1e-12)
