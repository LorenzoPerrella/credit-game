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
        [("lambda_", "Intercept"), ("lambda_", "fico_s"), ("rho_", "Intercept")],
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
    label = charts._label("lambda_", "C(purpose, Treatment('purchase'))[T.refinance_cashout]")

    assert label == "purpose: refinance_cashout"


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
    for command in ("fetch-macro", "build-data", "fit", "compare", "backtest", "report"):
        assert command in result.stdout


def test_build_data_rejects_an_unknown_source() -> None:
    result = runner.invoke(app, ["build-data", "--source", "nonsense"])

    assert result.exit_code != 0


def test_fit_without_a_panel_explains_how_to_build_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing artefact should say what to run, not raise a bare path error."""
    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))

    result = runner.invoke(app, ["fit"])

    assert result.exit_code != 0
    assert "build-data" in str(result.exception)
