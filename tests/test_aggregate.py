"""Tests for the DuckDB aggregation.

Small fixtures, real SQL. The point is not volume but that the event definition, the
truncation and the collapse are right — none of which needs a billion rows to check,
and all of which are impossible to check on one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from creditsurv.data.aggregate import CellSpec, build_cells, cardinality_report
from creditsurv.data.ingest import Quarter, ingest_quarter
from fixtures import origination_row, performance_row, write_archives

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))


def _ingested(tmp_path: Path, origination: list[str], performance: list[str]) -> None:
    write_archives(tmp_path / "FREDDIE MAC", 2015, {1: (origination, performance)})
    ingest_quarter(2015, 1)


def _sources(tmp_path: Path) -> tuple[str, str]:
    return (
        str(Quarter(2015, 1).parquet_path("perf")),
        str(Quarter(2015, 1).parquet_path("orig")),
    )


def test_identical_loans_collapse_into_one_cell(tmp_path: Path) -> None:
    """The claim the whole pipeline rests on."""
    origination = [origination_row(f"F{i:09d}") for i in range(20)]
    performance = [
        performance_row(f"F{i:09d}", "201503", str(age)) for i in range(20) for age in range(3)
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    # Twenty identical loans over three months, all inside the first age band:
    # one cell, weight sixty. The age banding is doing most of the work here, and
    # that is the point -- monthly ages compressed 2.2x on real data, bands 30x.
    assert len(cells) == 1
    assert int(cells["n"].iloc[0]) == 60


def test_weights_account_for_every_loan_month(tmp_path: Path) -> None:
    origination = [origination_row(f"F{i:09d}", fico=str(620 + i * 20)) for i in range(6)]
    performance = [
        performance_row(f"F{i:09d}", "201503", str(age)) for i in range(6) for age in range(4)
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert int(cells["n"].sum()) == 24


def test_default_is_flagged_once_and_the_loan_is_cut(tmp_path: Path) -> None:
    """Servicing files keep reporting through foreclosure; the loan must stop."""
    origination = [origination_row("F000000001")]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000001", "201504", "1", delinquency="1"),
        performance_row("F000000001", "201505", "2", delinquency="3"),
        performance_row("F000000001", "201506", "3", delinquency="4"),
        performance_row("F000000001", "201507", "4", delinquency="RA", zero_balance="09"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert int(cells["n"].sum()) == 3, "the loan should stop at its first defaulted month"
    assert int(cells.loc[cells["event"], "n"].sum()) == 1
    # Ages 0-5 share a band, whose lower edge is 0.
    assert int(cells.loc[cells["event"], "age"].iloc[0]) == 0


def test_an_reo_code_counts_even_when_delinquency_is_alphanumeric(tmp_path: Path) -> None:
    """`RA` casts to null, which compares false and reads as performing.

    The zero-balance code has to be checked alongside the delinquency status, not
    instead of it, or an REO acquisition silently becomes a surviving loan.
    """
    origination = [origination_row("F000000001")]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000001", "201504", "1", delinquency="RA", zero_balance="09"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert int(cells.loc[cells["event"], "n"].sum()) == 1


def test_prepayment_is_censoring_not_an_event(tmp_path: Path) -> None:
    origination = [origination_row("F000000001")]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000001", "201504", "1", zero_balance="01"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert not cells["event"].any()
    assert int(cells["n"].sum()) == 2


def test_sentinel_values_drop_the_loan(tmp_path: Path) -> None:
    """9999 is not a credit score, and a loan without one cannot be modelled."""
    origination = [origination_row("F000000001", fico="9999"), origination_row("F000000002")]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000002", "201503", "0"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert int(cells["n"].sum()) == 1


def test_negative_ages_are_dropped(tmp_path: Path) -> None:
    origination = [origination_row("F000000001")]
    performance = [
        performance_row("F000000001", "201502", "-1"),
        performance_row("F000000001", "201503", "0"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert int(cells["n"].sum()) == 1
    assert int(cells["age"].min()) == 0


def test_binning_puts_neighbouring_loans_in_one_cell(tmp_path: Path) -> None:
    """Coarse classing is what makes the collapse possible: without it every
    loan-month is its own cell and the aggregation saves nothing."""
    # Two scores a point apart fall in the same band.
    origination = [
        origination_row("F000000001", fico="742"),
        origination_row("F000000002", fico="743"),
    ]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000002", "201503", "0"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert len(cells) == 1
    assert int(cells["n"].iloc[0]) == 2


def test_term_is_reduced_to_fifteen_or_thirty_years(tmp_path: Path) -> None:
    origination = [
        origination_row("F000000001", term="180"),
        origination_row("F000000002", term="360"),
    ]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000002", "201503", "0"),
    ]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert set(cells["term_years"]) == {15, 30}


def test_cardinality_report_measures_the_collapse(tmp_path: Path) -> None:
    """Run before fixing the grain, not after: the decision needs a number."""
    origination = [origination_row(f"F{i:09d}") for i in range(10)]
    performance = [
        performance_row(f"F{i:09d}", "201503", str(age)) for i in range(10) for age in range(5)
    ]
    _ingested(tmp_path, origination, performance)

    report = cardinality_report(*_sources(tmp_path))

    assert int(report["loan_months"].iloc[0]) == 50
    assert int(report["weight_total"].iloc[0]) == 50
    # Ten identical loans over five months, all in one age band: a single cell.
    assert int(report["cells"].iloc[0]) == 1
    assert float(report["compression"].iloc[0]) == pytest.approx(50.0)


def test_no_ingested_data_says_what_to_run() -> None:
    with pytest.raises(FileNotFoundError, match="creditsurv ingest"):
        build_cells()


def test_spec_rejects_an_unknown_continuous_covariate() -> None:
    spec = CellSpec(continuous={"not_a_covariate": (0.0, 1.0)}, categorical=())

    with pytest.raises(ValueError, match="Unknown continuous"):
        spec.validate()


def test_spec_rejects_an_unknown_categorical_covariate() -> None:
    spec = CellSpec(continuous={}, categorical=("not_a_covariate",))

    with pytest.raises(ValueError, match="Unknown categorical"):
        spec.validate()


def test_age_bands_carry_their_lower_edge(tmp_path: Path) -> None:
    """The band's value is its lower edge in months, not an index, so the episode
    bounds can be read straight off it."""
    origination = [origination_row("F000000001")]
    performance = [performance_row("F000000001", "201503", str(age)) for age in (0, 7, 30, 100)]
    _ingested(tmp_path, origination, performance)

    cells = build_cells(*_sources(tmp_path))

    assert sorted(cells["age"]) == [0, 6, 24, 96]


def test_a_narrower_spec_collapses_harder(tmp_path: Path) -> None:
    """The specification is the cardinality: it is the product of the band counts.

    This is why it has to be chosen after variable selection rather than before --
    aggregating on everything available produced a table too large to fit.
    """
    origination = [origination_row(f"F{i:09d}", fico=str(600 + i * 30)) for i in range(8)]
    performance = [
        performance_row(f"F{i:09d}", "201503", str(age)) for i in range(8) for age in range(3)
    ]
    _ingested(tmp_path, origination, performance)

    wide = build_cells(*_sources(tmp_path))
    narrow = build_cells(
        *_sources(tmp_path),
        spec=CellSpec(continuous={"fico_s": (-3.0, 3.0)}, categorical=()),
    )

    assert len(narrow) < len(wide)
    assert int(narrow["n"].sum()) == int(wide["n"].sum()) == 24
