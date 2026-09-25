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
    covered_months,
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

COVARIATES = ["credit_score", "ltv_change", "unemployment_change"]
FORMULA = " + ".join(COVARIATES)
AS_OF = pd.Period("2008-12", freq="M")

PARAMS = replace(
    DEFAULT_PARAMS,
    intercept=0.14,
    continuous={"credit_score": 0.0068, "ltv_change": -0.020, "unemployment_change": -0.105},
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
            "loan_months": [1000, 10, 900, 8, 700],
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


def test_the_expected_rate_of_a_bucket_is_weighted_by_its_loan_months() -> None:
    """Cells stand for different numbers of loan-months, and the expected rate counts them."""
    predicted = pd.Series([0.001, 0.009])
    exposure = pd.Series([9000.0, 1000.0])
    events = pd.Series([9.0, 9.0])

    table = weighted_calibration(predicted, events, exposure, n_buckets=1)

    # 18 expected defaults over 10,000 loan-months; the plain mean of the two hazards is 0.005.
    assert table["expected"].iloc[0] == pytest.approx(0.0018)
    assert table["ratio"].iloc[0] == pytest.approx(1.0)


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
        {
            "credit_score": rng.normal(size=3000),
            "unemployment_change": rng.normal(size=3000),
            "loan_months": 1.0,
        }
    )
    test = pd.DataFrame(
        {
            "credit_score": rng.normal(size=3000),
            "unemployment_change": rng.normal(4.0, size=3000),
            "loan_months": 1.0,
        }
    )

    table = stability_report(
        train,
        test,
        ["credit_score", "unemployment_change"],
        time_varying=["unemployment_change"],
        weights_col="loan_months",
    ).set_index("covariate")

    assert table.loc["unemployment_change", "interpretation"] == "expected to move"
    assert table.loc["credit_score", "interpretation"] == "stable"


# --------------------------------------------------------------------------------------
# The backtest, end to end
# --------------------------------------------------------------------------------------


def _encoded(panel: pd.DataFrame) -> pd.DataFrame:
    """The fixture book as a weighted, interval-censored panel."""
    from creditsurv.data.panel import to_interval_censored

    return to_interval_censored(panel.assign(loan_months=1))


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
    assert result.loan_months == int(split.test["loan_months"].sum())


def test_a_model_from_the_wrong_panel_is_refused(panel: pd.DataFrame) -> None:
    """The guarantee that "no further calibration" is a property of the code.

    A model fitted on the whole panel would score its own training data and return a
    flattering number with nothing visibly wrong, so passing one in is an error rather
    than a shortcut.
    """
    encoded = _encoded(panel)
    everything = fit_aft(encoded, COVARIATES, FORMULA, weights_col="loan_months")

    with pytest.raises(ValueError, match="scoring its own training data"):
        run_backtest(encoded, AS_OF, COVARIATES, FORMULA, fitted=everything)


def test_passing_the_fitted_model_in_does_not_refit(panel: pd.DataFrame) -> None:
    encoded = _encoded(panel)
    split = cell_split(encoded, AS_OF)
    fitted = fit_aft(split.train, COVARIATES, FORMULA, weights_col="loan_months")

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
    fitted = fit_aft(split.train, COVARIATES, FORMULA, weights_col="loan_months")

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
    assert table["exposure"].sum() == pytest.approx(float(split.train["loan_months"].sum()))
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
            "origination_month": [month for month in vintages for _ in range(8)],
            "purpose": pd.Categorical(["purchase", "cash_out_refinance"] * 12),
            "credit_score": [0.4] * 24,
            "original_ltv": [85.0] * 24,
            "age": list(range(8)) * 3,
            "event": ([False] * 7 + [True]) * 3,
            "loan_months": [50] * 24,
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


def test_months_the_release_barely_covers_are_set_aside_by_name() -> None:
    """The published backtest plunged to zero in its last month, on 8 loan-months of
    exposure, and read as the model diverging. Such months leave the picture, named."""
    table = pd.DataFrame(
        {
            "group": pd.PeriodIndex(["2026-01", "2026-02", "2026-03", "2026-04"], freq="M"),
            "exposure": [12_000_000.0, 3_900_000.0, 51_341.0, 8.0],
        }
    )

    covered, thin = covered_months(table)

    assert list(covered["group"].astype(str)) == ["2026-01", "2026-02"]
    assert list(thin["group"].astype(str)) == ["2026-03", "2026-04"]


# --------------------------------------------------------------------------------------
# The master scale
# --------------------------------------------------------------------------------------


def test_the_master_scale_doubles_and_covers_the_line() -> None:
    """Eight grades on geometric thresholds, declared in docs/rules.md before the run: a
    scale drawn after seeing the distribution is a scale drawn to pass.
    """
    from creditsurv.backtest.metrics import GRADES, grade_of, master_scale

    edges = master_scale()

    assert len(edges) == GRADES - 1
    np.testing.assert_allclose(edges[1:] / edges[:-1], 2.0)
    assert grade_of(np.array([0.0])) == 1, "the safest grade has no floor"
    assert grade_of(np.array([1.0])) == GRADES, "the riskiest has no ceiling"
    assert grade_of(np.array([edges[0]]))[0] == 1, "the edges are closed on the left"
    assert grade_of(np.array([edges[0] * 1.001]))[0] == 2


def test_the_jeffreys_interval_says_something_where_a_normal_one_would_not() -> None:
    """The top grades hold few defaults and sometimes none, where a normal interval has
    zero width and no grade can fail. That is the reason the validation asked for this one.
    """
    from creditsurv.backtest.metrics import jeffreys_interval

    lower, upper = jeffreys_interval(np.array([0.0, 5.0]), np.array([10_000.0, 10_000.0]))

    # Not zero, as a Clopper-Pearson lower bound would be: Beta(1/2, n + 1/2) puts a little
    # mass below any rate, which is the prior doing what it is there for.
    assert 0 < lower[0] < 1e-6
    assert 0 < upper[0] < 0.001, "no defaults in ten thousand still bounds the rate"
    assert lower[1] < 5.0 / 10_000.0 < upper[1]
    # The interval narrows as the exposure grows.
    wide = jeffreys_interval(5.0, 10_000.0)
    narrow = jeffreys_interval(500.0, 1_000_000.0)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_a_grade_passes_when_its_prediction_falls_inside_the_realised_interval() -> None:
    from creditsurv.backtest.metrics import (
        annualised,
        grade_backtest,
        master_scale_passed,
    )

    hazard = np.concatenate(
        [np.full(2_000, 0.00005), np.full(2_000, 0.0005), np.full(2_000, 0.005)]
    )
    exposure = pd.Series(np.full(len(hazard), 12_000.0))
    # Realised defaults exactly at the predicted rate: every populated grade must hold.
    events = pd.Series(annualised(hazard) * exposure.to_numpy() / 12.0)

    table = grade_backtest(pd.Series(hazard), events, exposure)

    assert master_scale_passed(table)
    assert table["passed"].all()
    assert table["grade"].is_monotonic_increasing
    assert (table["predicted_pd"].diff().dropna() > 0).all()

    # A model predicting a tenth of what happens fails, and says so grade by grade.
    broken = grade_backtest(pd.Series(hazard), events * 10.0, exposure)
    assert not master_scale_passed(broken)
    assert not broken["passed"].any()


def test_the_predicted_pd_and_the_realised_rate_are_in_the_same_unit() -> None:
    """Obligor-years on both sides: the denominator is loan-months over twelve, which is
    the exposure the defaults were earned on and the unit the annualised hazard is in.
    """
    from creditsurv.backtest.metrics import grade_backtest

    hazard = pd.Series(np.full(100, 0.001))
    exposure = pd.Series(np.full(100, 1_200.0))
    events = pd.Series(np.full(100, 1.2))

    table = grade_backtest(hazard, events, exposure)

    assert len(table) == 1
    row = table.iloc[0]
    assert row["obligor_years"] == pytest.approx(100 * 1_200.0 / 12.0)
    assert row["actual_pd"] == pytest.approx(120.0 / 10_000.0)
    assert row["predicted_pd"] == pytest.approx(1.0 - 0.999**12)


# --------------------------------------------------------------------------------------
# Windows with an end
# --------------------------------------------------------------------------------------


def test_the_declared_cuts_each_carry_twenty_four_months() -> None:
    """Three cuts rather than one, and each closed. The previous model cut once, at
    2024-12, and was judged on fifteen quiet months; open-ended, the 2018 cut would be
    judged on the pandemic as well and the three would not be three regimes.
    """
    from creditsurv.backtest.runner import (
        BACKTEST_CUTS,
        BACKTEST_WINDOW_MONTHS,
        backtest_windows,
    )

    windows = backtest_windows()

    assert len(windows) == len(BACKTEST_CUTS) == 3
    for (cut, until), declared in zip(windows, BACKTEST_CUTS, strict=True):
        assert str(cut) == declared
        assert (until - cut).n == BACKTEST_WINDOW_MONTHS
    assert [str(cut) for cut, _ in windows] == ["2018-12", "2020-12", "2022-12"]


def test_a_closed_window_scores_only_the_months_it_covers() -> None:
    from creditsurv.backtest.splits import cell_split

    months = pd.period_range("2018-01", "2022-12", freq="M")
    cells = pd.DataFrame(
        {
            "period": months,
            "event": False,
            "loan_months": 10,
        }
    )

    closed = cell_split(cells, pd.Period("2018-12", freq="M"), until=pd.Period("2020-12", freq="M"))
    open_ended = cell_split(cells, pd.Period("2018-12", freq="M"))

    assert closed.until == pd.Period("2020-12", freq="M")
    assert list(closed.test["period"]) == list(pd.period_range("2019-01", "2020-12", freq="M"))
    assert len(open_ended.test) > len(closed.test)
    assert closed.describe()["until"] == "2020-12"
    # The training half is the same either way: an end bounds what is scored, never what
    # the model saw.
    assert len(closed.train) == len(open_ended.train)


def test_the_split_taken_from_cells_honours_the_same_end(
    macro: pd.DataFrame,
) -> None:
    """The halves are taken before either is expanded, so the end has to be applied there
    too -- expanding first and trimming after is what put the moratorium comparison at a
    17.3 GB footprint.
    """
    from creditsurv.backtest.splits import split_cells

    months = pd.PeriodIndex(macro.index[-30:])
    cells = pd.DataFrame(
        {
            "credit_score": 720.0,
            "original_ltv": 80.0,
            "origination_month": [month.year * 12 + month.month - 1 for month in months],
            "age": 0,
            "outcome": "none",
            "loan_months": 5,
        }
    )
    as_of = months[9]
    until = months[19]

    split = split_cells(cells, macro, as_of, covariates=["credit_score"], until=until)

    assert split.until == until
    assert split.test["period"].max() == until
    assert split.test["period"].min() == as_of + 1
    assert len(split.test) == 10


# --------------------------------------------------------------------------------------
# Anchoring the level
# --------------------------------------------------------------------------------------


def _anchoring_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.PeriodIndex]:
    """Four years of loan-months at a known hazard, defaulting at twice the rate."""
    months = pd.period_range("2021-01", "2025-12", freq="M")
    hazard = np.full(len(months), 0.001)
    weight = np.full(len(months), 100_000.0)
    events = np.where(pd.PeriodIndex(months).year >= 2022, 0.002, 0.004)
    return hazard, events, weight, pd.PeriodIndex(months)


def test_the_anchor_reads_the_anchoring_window_and_nothing_else() -> None:
    """The development window ends in 2021-12 and the test window begins in 2025-01, so a
    multiplier that read either would be re-fitting the intercept or marking its own
    homework. The 2021 months here default at four times the expected rate and must not
    move the answer.
    """
    from creditsurv.models.anchoring import ANCHOR_WINDOW, anchor_on_window

    hazard, events, weight, months = _anchoring_frame()

    anchor = anchor_on_window(hazard, events, weight, months)

    assert anchor.window == ANCHOR_WINDOW
    assert anchor.multiplier == pytest.approx(2.0)
    assert anchor.loan_months == pytest.approx(36 * 100_000.0), "36 months, not 60"


def test_anchoring_puts_actual_over_expected_at_one_on_its_own_window() -> None:
    from creditsurv.models.anchoring import anchor_on_window

    hazard, events, weight, months = _anchoring_frame()
    inside = (months >= pd.Period("2022-01", freq="M")) & (months <= pd.Period("2024-12", freq="M"))

    anchor = anchor_on_window(hazard, events, weight, months)
    anchored = anchor.apply(hazard)

    expected = (anchored[inside] * weight[inside]).sum()
    actual = (events[inside] * weight[inside]).sum()
    assert actual / expected == pytest.approx(1.0)


def test_anchoring_changes_the_level_and_leaves_the_ranking_alone() -> None:
    """One multiplier on every hazard: the order of loans is exactly what it was, which is
    why discrimination cannot change and why the segment views still mean something. A
    per-segment adjustment would fix every segment's level by absorbing the model's errors
    into the cuts the model is examined through.
    """
    from creditsurv.models.anchoring import Anchor

    hazard = np.array([0.0001, 0.002, 0.0005, 0.01])
    anchor = Anchor(1.7, ("2022-01", "2024-12"), 17.0, 10.0, 1_000.0)

    anchored = anchor.apply(hazard)

    np.testing.assert_allclose(anchored, hazard * 1.7)
    assert list(np.argsort(anchored)) == list(np.argsort(hazard))
    assert anchor.apply(np.array([0.9])) == pytest.approx(1.0), "a hazard stays a probability"


def test_a_level_too_far_out_to_be_a_level_is_refused() -> None:
    """Out by more than four times, the specification is what is wrong, and multiplying
    would hide that behind a number that then looks calibrated.
    """
    from creditsurv.models.anchoring import anchor_on_window

    hazard, events, weight, months = _anchoring_frame()

    with pytest.raises(ValueError, match="not a level to be scaled"):
        anchor_on_window(hazard, events * 6.0, weight, months)
    with pytest.raises(ValueError, match="No exposure in the anchoring window"):
        anchor_on_window(hazard[:6], events[:6], weight[:6], months[:6])


# --------------------------------------------------------------------------------------
# The prepayment backtest
# --------------------------------------------------------------------------------------


def test_a_monthly_prepayment_rate_is_quoted_as_the_market_quotes_it() -> None:
    from creditsurv.backtest.metrics import conditional_prepayment_rate

    # 0.5% a month is the 5.84% CPR every prepayment table in the world would print.
    assert conditional_prepayment_rate(np.array([0.005]))[0] == pytest.approx(
        1.0 - 0.995**12, rel=1e-12
    )
    assert conditional_prepayment_rate(np.array([0.0]))[0] == 0.0
    assert conditional_prepayment_rate(np.array([0.02]))[0] == pytest.approx(0.2153, abs=1e-4)


def test_prepayment_is_scored_month_by_month_rather_than_over_a_window() -> None:
    """A rate cut can double prepayment in a quarter, so a model can hit the average of a
    period it has exactly backwards within it. The monthly path is what says which.
    """
    from creditsurv.backtest.metrics import prepayment_by_month

    months = pd.period_range("2022-01", "2022-06", freq="M")
    predicted = pd.Series(np.full(6, 0.01))
    # Realised: half the predicted rate for three months, then double it.
    observed = pd.Series([0.005, 0.005, 0.005, 0.02, 0.02, 0.02])
    exposure = pd.Series(np.full(6, 50_000.0))

    table = prepayment_by_month(predicted, observed, exposure, pd.Series(months))

    assert len(table) == 6
    assert table["predicted_cpr"].iloc[0] == pytest.approx(1.0 - 0.99**12)
    assert list(table["ratio"].round(3)) == [0.5, 0.5, 0.5, 2.0, 2.0, 2.0]
    # Over the window as a whole the model looks unbiased, which is the point.
    overall = table["prepayments"].sum() / table["expected_prepayments"].sum()
    assert overall == pytest.approx(1.25)


def test_prepayment_calibration_by_decile_reads_the_same_table_as_default_does() -> None:
    """The decile machinery is the same, and deliberately: two versions of one bucketing
    would put the same loan-month in different deciles.
    """
    from creditsurv.backtest.metrics import weighted_calibration

    rng = np.random.default_rng(44)
    predicted = pd.Series(rng.uniform(0.001, 0.03, size=2_000))
    exposure = pd.Series(rng.integers(50, 5_000, size=2_000).astype(float))
    observed = pd.Series(predicted.to_numpy() * 1.1)

    table = weighted_calibration(predicted, observed * exposure, exposure)

    assert len(table) == 10
    np.testing.assert_allclose(table["ratio"].to_numpy(), 1.1, rtol=1e-9)


def test_the_windows_command_cuts_anchors_and_grades_end_to_end(
    tmp_path: Path, macro_module: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole of rule 1, 4, 5 and 7 on a book small enough for the suite.

    The command's own numbers belong to the population, not to seven hundred loans; what
    this holds is that every stage runs and that the report says which windows it used --
    the failure mode of a command like this is dying at minute twenty of a four-hour run.
    """
    from typer.testing import CliRunner

    from creditsurv.cli import app
    from creditsurv.data.aggregate import build_cells
    from creditsurv.data.ingest import ingest
    from creditsurv.data.store import save_cells
    from fixtures import write_book_archives

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CREDITSURV_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(
        "creditsurv.data.fred.load_macro_panel", lambda *_args, **_kwargs: macro_module
    )
    write_book_archives(tmp_path / "FREDDIE MAC", macro_module, n_loans=700, seed=45)
    ingest()
    save_cells(build_cells())

    result = CliRunner().invoke(
        app,
        [
            "windows",
            "--cuts",
            "2012-06",
            "--as-of",
            "2013-06",
            "--anchor-window",
            "2013-07,2014-06",
        ],
    )

    assert result.exit_code == 0, result.output
    body = (tmp_path / "reports" / "windows.md").read_text()
    assert "2012-06" in body, "the report states the cut it used"
    # The criteria of the published model are those of the model as published, so the
    # anchored row is there beside the unanchored one.
    assert "anchored" in body and "unanchored" in body
    assert "2013-07 to 2014-06" in body, "and the window the level was anchored on"
    assert "multiplier" in body
    assert "Twelve-month PD by grade" in body
    assert "The cycle, in sample" in body
