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
