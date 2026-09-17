"""Tests for the report builder, charts and CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from creditsurv.backtest.metrics import population_stability_index
from creditsurv.cli import app, default_covariates
from creditsurv.reporting import charts
from creditsurv.reporting.builder import Report, markdown_table, provenance

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def test_markdown_table_has_a_header_and_separator() -> None:
    frame = pd.DataFrame({"name": ["a", "b"], "value": [1.5, 2.25]})

    rendered = markdown_table(frame, decimals=2).splitlines()

    assert rendered[0] == "| name | value |"
    assert rendered[1] == "|---|---|"
    assert rendered[2] == "| a | 1.50 |"


def test_markdown_table_renders_missing_values_as_a_dash() -> None:
    frame = pd.DataFrame({"value": [float("nan")]})

    assert "| - |" in markdown_table(frame)


def test_markdown_table_falls_back_to_significant_figures() -> None:
    """A coefficient of 3e-07 formatted to four decimals reads as zero."""
    frame = pd.DataFrame({"value": [0.0000003]})

    assert "3e-07" in markdown_table(frame)


def test_report_assembles_sections_in_order(tmp_path: Path) -> None:
    report = Report("Title", subtitle="Sub")
    report.heading("Section").text("Body.").bullets(["one", "two"])
    report.key_values({"metric": 0.5})

    path = report.write(tmp_path / "r.md")
    content = path.read_text()

    assert content.index("# Title") < content.index("## Section")
    assert content.index("Body.") < content.index("- one")
    assert "**metric**: 0.5000" in content


def test_figure_links_are_relative_to_the_report(tmp_path: Path) -> None:
    """Reports live in reports/ and figures in reports/figures/, so the link has to
    survive the pair being served from anywhere."""
    figure = tmp_path / "figures" / "chart.png"
    figure.parent.mkdir(parents=True)
    figure.touch()

    content = Report("T").figure(figure, "alt").write(tmp_path / "r.md").read_text()

    assert "![alt](figures/chart.png)" in content


def test_provenance_records_how_a_report_was_made() -> None:
    assert "How this was produced" in provenance(["a command"])


def test_charts_write_files(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "expected": np.linspace(0.01, 0.2, 10),
            "actual": np.linspace(0.02, 0.18, 10),
            "loans": [50] * 10,
        }
    )

    path = charts.calibration(table, tmp_path / "c.png")

    assert path.exists()
    assert path.stat().st_size > 0


def test_coefficient_chart_omits_intercepts(tmp_path: Path) -> None:
    """The scale intercept sits near 6 while covariates are inside +/-0.5.

    Plotted together the covariates compress into an unreadable sliver, which is
    what the first version of this chart did.
    """
    index = pd.MultiIndex.from_tuples(
        [("lambda_", "Intercept"), ("lambda_", "credit_score"), ("rho_", "Intercept")],
        names=["param", "covariate"],
    )
    table = pd.DataFrame(
        {
            "coef": [6.0, 0.34, 0.26],
            "coef lower 95%": [3.0, 0.20, 0.10],
            "coef upper 95%": [9.0, 0.48, 0.42],
        },
        index=index,
    )

    figure = charts.coefficients(table, tmp_path / "coef.png")

    assert figure.exists()


def test_formulaic_labels_are_made_readable() -> None:
    label = charts._label("lambda_", "C(purpose, Treatment('purchase'))[T.cash_out_refinance]")

    assert label == "purpose: cash_out_refinance"


def test_intercept_labels_name_their_parameter_block() -> None:
    """Both blocks own an intercept, so two rows would otherwise read identically."""
    assert charts._label("rho_", "Intercept") == "rho: Intercept"


def test_psi_handles_categorical_variables() -> None:
    """Comparing the mix of business written is the original use of the index.

    Passing a categorical to the numeric path raises deep inside numpy, which is how
    every backtest fold failed before this was handled.
    """
    reference = pd.Series(["a"] * 70 + ["b"] * 30, dtype="category")
    same = pd.Series(["a"] * 70 + ["b"] * 30, dtype="category")
    shifted = pd.Series(["a"] * 30 + ["b"] * 70, dtype="category")

    assert population_stability_index(reference, same) == pytest.approx(0.0, abs=1e-9)
    assert population_stability_index(reference, shifted) > 0.25


def test_default_covariates_match_the_formula() -> None:
    from creditsurv.config import default_formula

    formula = default_formula()

    for name in default_covariates():
        assert name in formula


def test_cli_exposes_every_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "fetch-macro",
        "ingest",
        "portfolio",
        "profile",
        "aggregate",
        "fit",
        "compare",
        "backtest",
        "report",
    ):
        assert command in result.stdout


def test_fit_without_cells_explains_how_to_build_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing artefact should say what to run, not raise a bare path error."""
    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))

    result = runner.invoke(app, ["fit"])

    assert result.exit_code != 0
    assert "creditsurv aggregate" in str(result.exception)


def test_time_varying_covariates_have_a_marginal_effect(
    book_dir: Path, macro_module: pd.DataFrame
) -> None:
    """Regression test for a silent zero.

    Projecting a panel recomputes the macro-derived covariates from the macro
    series, so a shock applied to the loan record is overwritten. The first version
    of this table reported exactly 0.0000 percentage points for all four
    time-varying covariates -- flatly contradicting their own coefficients, and
    entirely plausible to anyone not checking.
    """
    from dataclasses import replace

    from creditsurv.config import STATIC_CONTINUOUS, TIME_VARYING_CONTINUOUS
    from creditsurv.data.panel import at_origination, to_interval_censored
    from creditsurv.models.aft import fit_aft
    from creditsurv.reporting.calibration import marginal_effects
    from fixtures import DEFAULT_PARAMS, build_panel

    covariates = ["credit_score", "ltv_change", "unemployment_change"]
    params = replace(
        DEFAULT_PARAMS,
        intercept=0.14,
        continuous={"credit_score": 0.0068, "ltv_change": -0.020, "unemployment_change": -0.105},
        categorical={},
        prepayment_intercept=50.0,
    )
    panel, _ = build_panel(book_dir, macro_module, n_loans=600, seed=13, params=params)
    fitted = fit_aft(to_interval_censored(panel), covariates, " + ".join(covariates))

    book = at_origination(panel).head(150).copy()
    book["age"] = 0
    book["period"] = macro_module.index.max() + 1

    effects = marginal_effects(
        fitted, book, macro_module, covariates, [*STATIC_CONTINUOUS, *TIME_VARYING_CONTINUOUS]
    ).set_index("covariate")

    changes = effects["change_pp"].astype(float)
    for name in ("ltv_change", "unemployment_change"):
        assert abs(changes[name]) > 1e-6, f"{name} shows no effect"
    assert effects.loc["ltv_change", "kind"] == "time-varying"


def test_a_macro_level_moves_by_its_deviation_in_the_fitting_data(
    book_dir: Path, macro_module: pd.DataFrame
) -> None:
    """The second silent zero in the same table.

    Under the random-walk baseline a macro *level* is identical for every loan and flat across the
    projection, so its deviation on the projected panel is zero and the row was skipped. The
    validation found the table omitting ``equity_volatility`` -- the macro covariate with the
    largest standardised effect -- for that reason alone. The step now comes from the data the model
    was fitted to, and the table says which step it used.
    """
    from dataclasses import replace

    from creditsurv.data.panel import at_origination, to_interval_censored
    from creditsurv.models.aft import fit_aft
    from creditsurv.reporting.calibration import covariate_steps, marginal_effects
    from fixtures import DEFAULT_PARAMS, build_panel

    covariates = ["credit_score", "ltv_change", "equity_volatility"]
    params = replace(
        DEFAULT_PARAMS,
        intercept=0.14,
        continuous={"credit_score": 0.0068, "ltv_change": -0.020},
        categorical={},
        prepayment_intercept=50.0,
    )
    panel, _ = build_panel(book_dir, macro_module, n_loans=600, seed=13, params=params)
    encoded = to_interval_censored(panel)
    fitted = fit_aft(encoded, covariates, " + ".join(covariates))

    book = at_origination(panel).head(150).copy()
    book["age"] = 0
    book["period"] = macro_module.index.max() + 1

    steps = covariate_steps(encoded, covariates)
    effects = marginal_effects(
        fitted, book, macro_module, covariates, covariates, steps=steps
    ).set_index("covariate")

    assert "equity_volatility" in effects.index, "a macro level must not drop out of the table"
    assert effects.loc["equity_volatility", "kind"] == "time-varying"
    vix = effects.loc[effects.index == "equity_volatility"]
    assert float(vix["one_sd"].to_numpy(dtype=float)[0]) == pytest.approx(
        steps["equity_volatility"]
    )
    assert abs(float(vix["change_pp"].to_numpy(dtype=float)[0])) > 0.0


def test_a_saved_fit_comes_back_and_a_changed_specification_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fit on the whole population is two and a half hours, so losing one matters.

    The run that produced this was killed while writing its reports, throwing away the
    expensive part and keeping nothing. Saving happens the moment the fit succeeds,
    before anything downstream can fail -- and the fingerprint has to separate models
    that differ, or a later run would report on the wrong one.
    """
    from creditsurv.data.store import fit_fingerprint, load_fit, save_fit

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))

    described = {"as_of": "2024-12", "formula": "credit_score", "rows": 1000}
    fingerprint = fit_fingerprint(**described)
    save_fit({"marker": 7}, fingerprint, described)

    assert load_fit(fingerprint) == {"marker": 7}
    assert fit_fingerprint(**described) == fingerprint, "the fingerprint must be stable"

    for changed in (
        {"as_of": "2023-12"},
        {"formula": "credit_score + debt_to_income"},
        {"rows": 1001},
    ):
        other = fit_fingerprint(**{**described, **changed})
        assert other != fingerprint, f"{changed} must not share a cache entry"
        assert load_fit(other) is None


def test_an_unreadable_cached_fit_is_a_miss_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pickle is tied to the lifelines and numpy that wrote it, so an upgrade should
    cost a refit rather than a traceback."""
    from creditsurv.data.store import fit_path, load_fit

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    path = fit_path("deadbeefdeadbeef")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a pickle")

    assert load_fit("deadbeefdeadbeef") is None


def test_the_distribution_reading_follows_the_numbers() -> None:
    """The first version of this section asserted that a family fitting worse and turning a
    prior around is rejected twice over, and the run it was published with had the
    log-logistic fitting better. The reading is now written from the tables."""
    from creditsurv.reporting.methodology import _against_kaplan_meier, _comparison_reading

    months = pd.Index([1.0, 2.0, 3.0])
    weibull = pd.DataFrame(
        {"predicted": [0.99, 0.98, 0.90], "deviation": [0.0, -0.01, -0.03]}, index=months
    )
    loglogistic = pd.DataFrame(
        {"predicted": [0.99, 0.98, 0.92], "deviation": [0.0, -0.005, -0.01]}, index=months
    )
    against = pd.DataFrame(
        [
            _against_kaplan_meier("weibull", weibull),
            _against_kaplan_meier("loglogistic", loglogistic),
        ]
    )
    regression = pd.DataFrame(
        {
            "distribution": ["loglogistic", "weibull"],
            "delta_aic": [0.0, 83961.0],
            "signs_against_prior": ["financial_conditions", ""],
        }
    )

    reading = _comparison_reading(regression, against, "weibull")

    assert "**loglogistic** has the better likelihood, by 83,961 AIC points" in reading
    assert "turns `financial_conditions` against its declared prior" in reading
    assert "Against Kaplan-Meier the **loglogistic** is closer" in reading
    assert "kept against a better likelihood" in reading
    assert against.loc[0, "deviation_at_last_month"] == pytest.approx(-3.0)
    assert against.loc[0, "last_month"] == 3
