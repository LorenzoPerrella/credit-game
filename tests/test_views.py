"""The tables behind the site's views, held to the estimators they claim to be."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from creditsurv.backtest.metrics import exposure_buckets, weighted_calibration
from creditsurv.backtest.runner import predicted_hazard
from creditsurv.backtest.splits import Split, cell_split
from creditsurv.cli import app
from creditsurv.data.panel import WEIGHT, to_interval_censored
from creditsurv.models.aft import FitResult, fit_aft
from creditsurv.models.lifetime_pd import origination_book
from creditsurv.models.nonparametric import kaplan_meier, predicted_survival_curve
from creditsurv.views.calibration import WHOLE_BOOK, actual_expected, survival_by_age
from creditsurv.views.model import (
    calibration_views,
    coefficient_view,
    covariates_over_time,
    projection_views,
)
from creditsurv.views.segments import SEGMENTS, age_bands, available, calendar_years
from creditsurv.views.tables import View, load_manifest, load_view, write_views
from fixtures import DEFAULT_PARAMS, build_panel

if TYPE_CHECKING:
    from pathlib import Path

COVARIATES = ["credit_score", "ltv_change", "unemployment_change"]
FORMULA = " + ".join(COVARIATES)

PARAMS = replace(
    DEFAULT_PARAMS,
    intercept=0.14,
    continuous={"credit_score": 0.0068, "ltv_change": -0.020, "unemployment_change": -0.105},
    categorical={},
    prepayment_intercept=50.0,
)


@pytest.fixture(scope="module")
def encoded(book_dir: Path, macro_module: pd.DataFrame) -> pd.DataFrame:
    panel, _ = build_panel(book_dir, macro_module, n_loans=1500, seed=71, params=PARAMS)
    return to_interval_censored(panel).assign(**{WEIGHT: 1})


@pytest.fixture(scope="module")
def fitted(encoded: pd.DataFrame) -> FitResult:
    return fit_aft(encoded, COVARIATES, FORMULA, weights_col=WEIGHT)


@pytest.fixture(scope="module")
def hazard(fitted: FitResult, encoded: pd.DataFrame) -> np.ndarray:
    return predicted_hazard(fitted, encoded, COVARIATES).to_numpy()


def test_the_kaplan_meier_from_the_episodes_is_the_one_lifelines_fits(
    encoded: pd.DataFrame,
) -> None:
    """Product-limit on the episodes' risk sets, against lifelines on the loan durations."""
    table = survival_by_age(encoded)
    curve = kaplan_meier(encoded, weights_col=WEIGHT)

    reference = curve.survival_function_at_times(table["age"].to_numpy() + 1.0).to_numpy()

    assert set(table["group"]) == {WHOLE_BOOK}
    assert np.allclose(table["km_survival"].to_numpy(), reference, atol=1e-12)
    assert (table["km_lower"] <= table["km_survival"]).all()
    assert (table["km_survival"] <= table["km_upper"]).all()


def test_the_model_curve_is_the_portfolio_curve_split_by_group(
    fitted: FitResult, encoded: pd.DataFrame, hazard: np.ndarray
) -> None:
    table = survival_by_age(encoded, hazard)
    reference = predicted_survival_curve(fitted, encoded, COVARIATES, weights_col=WEIGHT)

    assert np.allclose(table["predicted_survival"].to_numpy(), reference.to_numpy(), atol=1e-12)
    assert np.allclose(
        table["deviation"], table["predicted_survival"] - table["km_survival"], atol=1e-15
    )


def test_the_groups_of_a_segment_add_up_to_the_book(
    encoded: pd.DataFrame, hazard: np.ndarray
) -> None:
    segment = SEGMENTS["purpose"]
    split = survival_by_age(encoded, hazard, groups=segment.label(encoded))
    whole = survival_by_age(encoded, hazard).set_index("age")

    summed = split.groupby("age")[["at_risk", "defaults", "expected"]].sum()

    assert split["group"].nunique() > 1
    assert np.allclose(summed.to_numpy(), whole.loc[summed.index, summed.columns].to_numpy())


def test_actual_against_expected_totals_do_not_depend_on_the_split(
    encoded: pd.DataFrame, hazard: np.ndarray
) -> None:
    purpose = SEGMENTS["purpose"].label(encoded)
    by_one = actual_expected(encoded, hazard, {"purpose": purpose})
    by_two = actual_expected(encoded, hazard, {"purpose": purpose, "year": calendar_years(encoded)})

    for column in ("exposure", "events", "expected"):
        assert by_one[column].sum() == pytest.approx(by_two[column].sum())
        assert by_one[column].sum() == pytest.approx(
            {
                "exposure": encoded[WEIGHT].sum(),
                "events": encoded["event"].sum(),
                "expected": hazard.sum(),
            }[column]
        )
    assert list(by_two.columns[:2]) == ["purpose", "year"]
    assert by_two["actual_over_expected"].notna().all()


def test_buckets_of_predicted_risk_hold_equal_exposure() -> None:
    rng = np.random.default_rng(5)
    predicted = rng.random(10_000)
    exposure = np.ones(10_000)

    buckets = exposure_buckets(predicted, exposure, n_buckets=10)

    counts = np.bincount(buckets, minlength=10)
    assert counts.min() >= 999
    assert counts.max() <= 1001
    # Riskier rows never sit in a lower bucket than safer ones.
    order = np.argsort(predicted)
    assert np.all(np.diff(buckets[order]) >= 0)


def test_the_backtest_deciles_are_the_shared_buckets(
    encoded: pd.DataFrame, hazard: np.ndarray
) -> None:
    exposure = encoded[WEIGHT].astype(float)
    events = exposure * encoded["event"].astype(bool)

    table = weighted_calibration(pd.Series(hazard), events, exposure)
    buckets = exposure_buckets(hazard, exposure.to_numpy())

    assert table["loan_months"].tolist() == np.bincount(buckets, weights=exposure).tolist()


def test_credit_score_bands_are_named_as_scores() -> None:
    frame = pd.DataFrame({"credit_score": [600.0, 675.0, 720.0, 750.0, 800.0, 875.0]})

    labels = SEGMENTS["fico"].label(frame)

    assert labels.tolist()[0].startswith("580")
    assert labels.tolist()[-1].endswith("820")
    assert labels.notna().all()


def test_age_bands_cover_every_age() -> None:
    frame = pd.DataFrame({"age": [0, 11, 12, 59, 60, 359]})

    assert age_bands(frame).notna().all()
    assert age_bands(frame).tolist()[:3] == ["0 to 11", "0 to 11", "12 to 23"]


def test_only_the_segments_a_frame_carries_are_offered(encoded: pd.DataFrame) -> None:
    names = {segment.name for segment in available(encoded)}

    assert {"purpose", "fico", "ltv"} <= names
    assert all(SEGMENTS[name].available(encoded) for name in names)


def test_a_view_is_written_with_its_manifest_and_read_back(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "month": pd.period_range("2020-01", periods=3, freq="M"),
            "group": pd.Categorical(["a", "b", "a"]),
            "rate": [0.1, 0.2, 0.3],
        }
    )

    write_views([View("rates", "Rates", "A test view.", frame, source="fit")], tmp_path, fit="abc")

    back = load_view("rates", tmp_path)
    manifest = load_manifest(tmp_path)
    assert back["month"].tolist() == ["2020-01", "2020-02", "2020-03"]
    assert back["group"].tolist() == ["a", "b", "a"]
    assert manifest["rates"]["fit"] == "abc"
    assert manifest["rates"]["rows"] == 3


def test_a_view_that_identifies_loans_is_refused(tmp_path: Path) -> None:
    frame = pd.DataFrame({"loan_id": ["F1"], "rate": [0.1]})

    with pytest.raises(ValueError, match="aggregates"):
        write_views([View("bad", "Bad", "", frame, source="cells")], tmp_path)


# ----- views that need the model ---------------------------------------------------------


@pytest.fixture(scope="module")
def split(encoded: pd.DataFrame) -> Split:
    last = pd.PeriodIndex(encoded["period"]).max()
    return cell_split(encoded, last - 12)


@pytest.fixture(scope="module")
def split_fit(split: Split) -> FitResult:
    return fit_aft(split.train, COVARIATES, FORMULA, weights_col=WEIGHT)


def test_every_calibration_view_opens_by_segment_and_adds_up(
    split: Split, split_fit: FitResult
) -> None:
    train_hazard = predicted_hazard(split_fit, split.train, COVARIATES).to_numpy()
    test_hazard = predicted_hazard(split_fit, split.test, COVARIATES).to_numpy()

    views = {
        view.name: view.frame
        for view in calibration_views(
            split,
            train_hazard=train_hazard,
            test_hazard=test_hazard,
            families={"weibull": train_hazard},
        )
    }

    assert {"km_vs_model", "ae_by_year", "backtest_by_month", "acceptance_by_segment"} <= set(views)
    for name in ("ae_by_year", "ae_by_vintage", "ae_by_age_band", "ae_by_decile"):
        table = views[name]
        assert list(table.columns[:2]) == ["segment", "group"]
        totals = table.groupby("segment")["events"].sum()
        assert totals.nunique() == 1, name
        assert totals.iloc[0] == pytest.approx(split.train["event"].sum())
    months = views["backtest_by_month"]
    assert months.loc[months["segment"] == "all", "exposure"].sum() == pytest.approx(
        split.test[WEIGHT].sum()
    )
    whole = views["acceptance_by_segment"].set_index("segment").loc["all"]
    assert whole["actual_over_expected"] == pytest.approx(
        split.test["event"].sum() / (test_hazard * split.test[WEIGHT]).sum()
    )
    assert set(views["families_vs_km"]["distribution"]) == {"weibull"}


def test_the_coefficient_view_puts_covariates_on_one_scale(
    split: Split, split_fit: FitResult
) -> None:
    table = coefficient_view(split_fit, split.train, COVARIATES).frame
    fico = table[(table["parameter"] == "lambda_") & (table["term"] == "credit_score")].iloc[0]

    assert fico["effect_1sd"] == pytest.approx(fico["coef"] * fico["one_sd"])
    assert fico["one_sd"] > 0


def test_covariates_over_time_are_monthly_exposure_weighted_means(split: Split) -> None:
    table = covariates_over_time(split, ["ltv_change"]).frame
    frame = pd.concat([split.train, split.test])
    first = str(pd.PeriodIndex(frame["period"]).min())
    rows = frame[pd.PeriodIndex(frame["period"]).astype(str) == first]

    value = table[(table["month"] == first) & (table["covariate"] == "ltv_change")]["mean"]
    assert value.iloc[0] == pytest.approx(np.average(rows["ltv_change"], weights=rows[WEIGHT]))


def test_projections_by_segment_weight_back_to_the_book(
    split: Split, split_fit: FitResult, macro_module: pd.DataFrame
) -> None:
    book = origination_book(split.train, macro_module, 40)
    views = {
        view.name: view.frame
        for view in projection_views(split_fit, book, macro_module, COVARIATES, horizon_months=24)
    }

    scenarios = views["scenarios_by_segment"]
    whole = scenarios[scenarios["segment"] == "all"].iloc[0]
    purpose = scenarios[scenarios["segment"] == "purpose"]
    recombined = (purpose["pd_12m"] * purpose["book_weight"]).sum() / purpose["book_weight"].sum()

    assert recombined == pytest.approx(whole["pd_12m"])
    assert (scenarios["adverse_multiple"] > 1).all()
    structure = views["term_structure_by_segment"]
    curve = structure[structure["segment"] == "all"].sort_values("month")
    assert curve["cumulative_pd"].is_monotonic_increasing
    assert curve["month"].max() == 24


def test_the_views_command_publishes_the_selection_record_as_long_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports, tables = tmp_path / "reports", tmp_path / "tables"
    reports.mkdir()
    pd.DataFrame(
        [[1.0, -0.8], [-0.8, 1.0]],
        index=["mortgage_rate_decline", "policy_rate_change"],
        columns=["mortgage_rate_decline", "policy_rate_change"],
    ).to_csv(reports / "selection_correlation.csv")
    monkeypatch.setenv("CREDITSURV_REPORTS_DIR", str(reports))
    monkeypatch.setenv("CREDITSURV_TABLES_DIR", str(tables))

    result = CliRunner().invoke(app, ["views", "--no-model", "--no-portfolio"])

    assert result.exit_code == 0, result.output
    assert set(load_manifest(tables)) == {"selection_correlation"}
    table = load_view("selection_correlation", tables)
    assert len(table) == 4
    pair = table[
        (table["first"] == "mortgage_rate_decline") & (table["second"] == "policy_rate_change")
    ]
    assert pair["correlation"].iloc[0] == pytest.approx(-0.8)


# --------------------------------------------------------------------------------------
# Competing risks, the new levels, the grades, the anchoring and the windows
# --------------------------------------------------------------------------------------


def _outcome_cells(rng: np.random.Generator, months: int = 24, loans: int = 600) -> pd.DataFrame:
    """Cells with three outcomes and both new levels of the key."""
    duration = rng.integers(1, months, size=loans)
    cause = rng.choice(["default", "prepayment", "none"], size=loans, p=[0.2, 0.5, 0.3])
    harp = rng.choice(["standard", "harp"], size=loans, p=[0.8, 0.2])
    rows = [
        {
            "age": age,
            "outcome": ending if age == int(length) - 1 else "none",
            "harp": flag,
            "delinquency_state": "current" if age < int(length) - 1 else "one_month",
            "loan_months": 1,
        }
        for length, ending, flag in zip(duration, cause, harp, strict=True)
        for age in range(int(length))
    ]
    return pd.DataFrame(rows)


def test_the_competing_view_sets_the_observed_incidence_beside_the_model() -> None:
    from creditsurv.views.competing import competing_incidence

    cells = _outcome_cells(np.random.default_rng(51))
    hazards = {
        "default": np.full(len(cells), 0.01),
        "prepayment": np.full(len(cells), 0.03),
    }

    table = competing_incidence(cells, hazards)

    assert {"observed_default", "observed_prepayment", "model_default"} <= set(table.columns)
    # Both sides account for the whole book at every age.
    observed = table["observed_survival"] + table["observed_default"] + table["observed_prepayment"]
    modelled = table["model_survival"] + table["model_default"] + table["model_prepayment"]
    np.testing.assert_allclose(observed.to_numpy(), 1.0, atol=1e-12)
    np.testing.assert_allclose(modelled.to_numpy(), 1.0, atol=1e-12)
    # Without a model it is the observed curve alone, and says so.
    assert "model_default" not in competing_incidence(cells).columns


def test_a_level_of_the_key_is_opened_with_its_exposure_and_both_exits() -> None:
    """The table the HARP level and the payment state exist for: how much of the book each
    holds, and how differently it behaves.
    """
    from creditsurv.views.competing import exposure_by_level

    cells = _outcome_cells(np.random.default_rng(52))

    table = exposure_by_level(cells, "harp")

    assert set(table["harp"]) == {"standard", "harp"}
    assert table["loan_months"].sum() == cells["loan_months"].sum()
    assert table["share"].sum() == pytest.approx(1.0)
    assert (table["default_rate"] >= 0).all()
    assert table["default"].sum() == float((cells["outcome"] == "default").sum())


def test_the_cycle_criterion_counts_the_years_inside_the_band() -> None:
    """The previous model ran 0.47 to 1.60 across years and nothing said whether that was
    acceptable. Rule 5 asks for 70% of years in the band, and this is where it is counted.
    """
    from creditsurv.views.competing import cycle_in_band

    good = pd.DataFrame({"actual_over_expected": [0.9, 1.0, 1.1, 1.2, 0.85, 1.3, 0.95, 1.05]})
    bad = pd.DataFrame({"actual_over_expected": [0.47, 1.60, 1.1, 0.5, 1.9, 1.0]})

    assert bool(cycle_in_band(good)["passed"].iloc[0])
    assert float(cycle_in_band(good)["share_in_band"].iloc[0]) == pytest.approx(7 / 8)
    assert not bool(cycle_in_band(bad)["passed"].iloc[0])
    assert float(cycle_in_band(bad)["lowest"].iloc[0]) == 0.47


def test_the_anchoring_view_shows_the_same_rows_before_and_after_one_multiplier() -> None:
    from creditsurv.models.anchoring import Anchor
    from creditsurv.views.competing import anchoring_view

    anchor = Anchor(1.2, ("2022-01", "2024-12"), 120.0, 100.0, 1_000_000.0)
    before = pd.DataFrame({"year": [2025, 2026], "actual_over_expected": [1.2, 1.1]})
    after = pd.DataFrame({"year": [2025, 2026], "actual_over_expected": [1.0, 0.917]})

    view = anchoring_view(anchor, before=before, after=after)

    assert set(view.frame["model"]) == {"unanchored", "anchored"}
    assert view.frame["anchor_multiplier"].unique().tolist() == [1.2]
    assert "1.2000" in view.description
    assert len(view.frame) == 4


def test_the_grade_view_names_what_the_scale_did() -> None:
    from creditsurv.views.competing import grade_view

    cells = _outcome_cells(np.random.default_rng(53), loans=4_000)
    cells["loan_months"] = 50
    hazard = np.full(len(cells), 0.002)

    view = grade_view(hazard, cells)

    assert view.name == "pd_by_grade"
    assert "grades hold" in view.description
    assert {"grade", "predicted_pd", "actual_pd", "lower", "upper", "passed"} <= set(
        view.frame.columns
    )


# --------------------------------------------------------------------------------------
# The same tables, accumulated over the cell file
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def streamed_cells(tmp_path_factory: pytest.TempPathFactory, macro_module: pd.DataFrame) -> Path:
    from creditsurv.data.aggregate import build_cells
    from creditsurv.data.ingest import ingest
    from creditsurv.data.store import save_cells
    from fixtures import write_book_archives

    root = tmp_path_factory.mktemp("views_streamed")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CREDITSURV_DATA_DIR", str(root))
        write_book_archives(root / "FREDDIE MAC", macro_module, n_loans=700, seed=51)
        ingest()
        return save_cells(build_cells())


def test_the_curves_are_the_same_accumulated_as_computed_whole(
    streamed_cells: Path, macro_module: pd.DataFrame
) -> None:
    """Every column of a survival table is a function of three sums, and sums add over
    batches. This is the claim that lets the views come off the expanded panel.
    """
    from creditsurv.data.panel import CellBlocks, cells_to_episodes
    from creditsurv.models.aft import fit_aft
    from creditsurv.views.calibration import survival_by_age
    from creditsurv.views.streamed import risk_sets_over

    covariates = ["credit_score", "original_ltv"]
    formula = " + ".join(covariates)
    cells = pd.read_parquet(streamed_cells)
    held = cells_to_episodes(cells, macro_module, covariates=covariates)
    fitted = fit_aft(held, covariates, formula, weights_col=WEIGHT)

    whole = survival_by_age(held, predicted_hazard(fitted, held, covariates).to_numpy())
    source = CellBlocks(
        str(streamed_cells),
        macro_module,
        tuple(covariates),
        rows=len(cells) // 6 + 1,
        model_only=False,
    ).prepared()
    accumulated = risk_sets_over(source(), fitted, covariates)

    assert list(accumulated.columns) == list(whole.columns)
    assert len(accumulated) == len(whole)
    for column in ("at_risk", "defaults", "km_survival", "predicted_survival", "km_lower"):
        np.testing.assert_allclose(
            accumulated[column].to_numpy(), whole[column].to_numpy(), rtol=1e-10
        )


def test_actual_against_expected_is_the_same_accumulated_as_computed_whole(
    streamed_cells: Path, macro_module: pd.DataFrame
) -> None:
    from creditsurv.data.panel import CellBlocks, cells_to_episodes
    from creditsurv.models.aft import fit_aft
    from creditsurv.views.calibration import actual_expected
    from creditsurv.views.segments import calendar_years
    from creditsurv.views.streamed import totals_over

    covariates = ["credit_score", "original_ltv"]
    cells = pd.read_parquet(streamed_cells)
    held = cells_to_episodes(cells, macro_module, covariates=covariates)
    fitted = fit_aft(held, covariates, " + ".join(covariates), weights_col=WEIGHT)

    whole = actual_expected(
        held,
        predicted_hazard(fitted, held, covariates).to_numpy(),
        {"year": calendar_years(held)},
    )
    source = CellBlocks(
        str(streamed_cells),
        macro_module,
        tuple(covariates),
        rows=len(cells) // 6 + 1,
        model_only=False,
    ).prepared()
    accumulated = totals_over(
        source(), fitted, covariates, lambda frame: {"year": calendar_years(frame)}
    )

    merged = whole.merge(accumulated, on="year", suffixes=("_whole", "_streamed"))
    assert len(merged) == len(whole)
    for column in ("exposure", "events", "expected", "actual_over_expected"):
        np.testing.assert_allclose(
            merged[f"{column}_streamed"].to_numpy(), merged[f"{column}_whole"].to_numpy(), rtol=1e-9
        )


def test_the_decile_boundaries_divide_the_book_into_equal_exposure(
    streamed_cells: Path, macro_module: pd.DataFrame
) -> None:
    """A decile is the one quantity that is not a sum: it needs the whole distribution before
    a row can be assigned, so it is taken in two passes off a weighted histogram.
    """
    from creditsurv.data.panel import CellBlocks, cells_to_episodes
    from creditsurv.models.aft import fit_aft
    from creditsurv.views.streamed import decile_boundaries, deciles_of

    covariates = ["credit_score", "original_ltv"]
    cells = pd.read_parquet(streamed_cells)
    held = cells_to_episodes(cells, macro_module, covariates=covariates)
    fitted = fit_aft(held, covariates, " + ".join(covariates), weights_col=WEIGHT)
    source = CellBlocks(
        str(streamed_cells),
        macro_module,
        tuple(covariates),
        rows=len(cells) // 6 + 1,
        model_only=False,
    ).prepared()

    boundaries = decile_boundaries(source(), fitted, covariates)
    hazard = predicted_hazard(fitted, held, covariates).to_numpy()
    buckets = deciles_of(hazard, boundaries)
    exposure = held[WEIGHT].to_numpy(dtype=float)
    shares = np.bincount(buckets, weights=exposure, minlength=10) / exposure.sum()

    assert len(boundaries) == 9
    assert (np.diff(boundaries) > 0).all(), "the boundaries are increasing"
    # Within a percentage point of a tenth each, on a book of seven hundred loans.
    np.testing.assert_allclose(shares, 0.1, atol=0.01)


def test_the_in_sample_views_are_the_same_accumulated_as_computed_whole(
    streamed_cells: Path, macro_module: pd.DataFrame
) -> None:
    """The published tables, name for name and column for column, from one pass over the cell
    file instead of from the expanded training half.

    Segment by segment, because that is where the saving is: `calibration_views` opens every
    table by every segment, and each of those was a pass over the panel it held.
    """
    from creditsurv.data.panel import CellBlocks, cells_to_episodes
    from creditsurv.models.aft import fit_aft
    from creditsurv.views.model import calibration_views, in_sample_recipes, in_sample_views
    from creditsurv.views.streamed import accumulate, decile_boundaries

    covariates = ["credit_score", "original_ltv"]
    cells = pd.read_parquet(streamed_cells)
    held = cells_to_episodes(cells, macro_module, covariates=covariates)
    fitted = fit_aft(held, covariates, " + ".join(covariates), weights_col=WEIGHT)
    source = CellBlocks(
        str(streamed_cells),
        macro_module,
        tuple(covariates),
        rows=len(cells) // 5 + 1,
        model_only=False,
    ).prepared()

    hazard = predicted_hazard(fitted, held, covariates).to_numpy()
    split = Split(as_of=pd.PeriodIndex(held["period"]).max(), train=held, test=held.iloc[:0])
    whole = {
        view.name: view.frame
        for view in calibration_views(split, train_hazard=hazard, test_hazard=hazard[:0])
        if view.name.startswith(("km_vs_model", "ae_by_"))
    }

    boundaries = decile_boundaries(source(), fitted, covariates)
    accumulated = accumulate(
        source(),
        {"weibull": (fitted, covariates)},
        in_sample_recipes(primary="weibull", boundaries=boundaries),
    )
    streamed = {view.name: view.frame for view in in_sample_views(accumulated, as_of="2014-12")}

    assert set(streamed) == set(whole)
    for name, table in streamed.items():
        reference = whole[name]
        assert list(table.columns) == list(reference.columns), name
        keys = [column for column in ("segment", "group") if column in table.columns]
        keys += [
            column
            for column in ("age", "year", "vintage_year", "age_band", "decile")
            if column in table.columns
        ]
        merged = reference.merge(table, on=keys, suffixes=("_whole", "_streamed"))
        assert len(merged) == len(reference), name

        # The decile table is the one that cannot agree row by row, and the reason is stated
        # where it is built: a decile needs the whole distribution, so its boundaries come off
        # a weighted histogram, which lands within a bin of the true quantile and **cannot
        # split a tie**. Cells sharing a key share a hazard, so a boundary inside such a
        # cluster puts all of it on one side. What must still hold exactly is that the deciles
        # partition the same rows -- every loan-month and every default is in one of them.
        if name == "ae_by_decile":
            for segment, rows in table.groupby("segment", observed=True):
                assert set(rows["decile"]) <= set(range(1, 11)), segment
            for column in ("exposure", "events", "expected"):
                by_segment = table.groupby("segment", observed=True)[column].sum()
                expected_by_segment = reference.groupby("segment", observed=True)[column].sum()
                np.testing.assert_allclose(
                    by_segment.to_numpy(),
                    expected_by_segment.reindex(by_segment.index).to_numpy(),
                    rtol=1e-9,
                    err_msg=f"{name}.{column} over the deciles",
                )
            continue

        for column in ("exposure", "events", "expected", "at_risk", "defaults", "km_survival"):
            if f"{column}_whole" not in merged.columns:
                continue
            np.testing.assert_allclose(
                merged[f"{column}_streamed"].to_numpy(),
                merged[f"{column}_whole"].to_numpy(),
                rtol=1e-9,
                err_msg=f"{name}.{column}",
            )


def test_the_views_command_scores_from_the_cell_file(
    streamed_cells: Path,
    macro_module: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command end to end, on a book aggregated for real.

    It reads a cached fit, takes two passes over the cell file and writes every table the site
    reads. What it must not do is hold the training half: that is what the passes are for, and
    what this test would catch the loss of only by getting slower -- so what it checks is that
    the tables arrive, named as the site expects them.
    """
    from typer.testing import CliRunner

    from creditsurv.cli import _fit_description, app, default_covariates
    from creditsurv.config import default_formula
    from creditsurv.data.panel import CellBlocks
    from creditsurv.data.store import cells_path, fit_fingerprint, save_cells, save_fit
    from creditsurv.models.aft import fit_streamed

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CREDITSURV_TABLES_DIR", str(tmp_path / "tables"))
    monkeypatch.setenv("CREDITSURV_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(
        "creditsurv.data.fred.load_macro_panel", lambda *_args, **_kwargs: macro_module
    )
    save_cells(pd.read_parquet(streamed_cells))

    as_of = "2014-06"
    covariates, formula = default_covariates(), default_formula()
    cut = 2014 * 12 + 6 - 1
    source = CellBlocks(
        str(cells_path("exclude")),
        macro_module,
        tuple(covariates),
        rows=20_000,
        months=(None, cut),
    ).prepared()
    fitted = fit_streamed(source, covariates, formula, weights_col=WEIGHT)
    record = fitted.blocks
    assert record is not None
    described = _fit_description(
        (fitted.n_episodes, int(record.loan_months)),
        formula,
        as_of=as_of,
        moratorium="exclude",
    )
    save_fit(fitted, fit_fingerprint(**described), described)

    result = CliRunner().invoke(
        app, ["views", "--no-portfolio", "--as-of", as_of, "--block-rows", "20000", "--loans", "50"]
    )

    assert result.exit_code == 0, result.output
    manifest = load_manifest(tmp_path / "tables")
    for name in (
        "km_vs_model",
        "ae_by_year",
        "ae_by_vintage",
        "ae_by_age_band",
        "ae_by_decile",
        "coefficients",
        "covariates_over_time",
        "backtest_by_month",
        "acceptance_by_segment",
    ):
        assert name in manifest, f"{name} missing: {sorted(manifest)}"
    # One fit behind every model view, as the site's build insists.
    assert len({entry["fit"] for entry in manifest.values() if entry["fit"]}) == 1
    curves = load_view("km_vs_model", tmp_path / "tables")
    assert {"segment", "group", "age", "km_survival", "predicted_survival"} <= set(curves.columns)
