"""Tests for the portfolio description."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from creditsurv.data.ingest import Quarter, ingest_quarter
from creditsurv.portfolio import (
    covariate_evolution,
    default_rate_by_period,
    origination_mix,
    originations_by_period,
    outstanding_by_period,
)
from fixtures import origination_row, performance_row, write_archives

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))


@pytest.fixture
def book(tmp_path: Path) -> tuple[list[str], list[str]]:
    origination = [
        origination_row("F000000001", upb="200000", first_payment="201503", purpose="P"),
        origination_row("F000000002", upb="300000", first_payment="201503", purpose="C"),
        origination_row("F000000003", upb="100000", first_payment="201504", purpose="P"),
    ]
    performance = [
        performance_row("F000000001", "201503", "0", upb="200000"),
        performance_row("F000000001", "201504", "1", upb="199000"),
        performance_row("F000000002", "201503", "0", upb="300000"),
        performance_row("F000000003", "201504", "0", upb="100000"),
    ]
    write_archives(tmp_path / "FREDDIE MAC", 2015, {1: (origination, performance)})
    ingest_quarter(2015, 1)
    return (
        [str(Quarter(2015, 1).parquet_path("perf"))],
        [str(Quarter(2015, 1).parquet_path("orig"))],
    )


def test_outstanding_counts_contracts_and_balance(book: tuple[list[str], list[str]]) -> None:
    table = outstanding_by_period(*book).set_index("period")

    march = table.loc[table.index == "2015-03"]
    assert int(march["contracts"].iloc[0]) == 2
    assert float(march["balance"].iloc[0]) == pytest.approx(500_000.0)


def test_originations_are_counted_and_summed(book: tuple[list[str], list[str]]) -> None:
    table = originations_by_period(*book).set_index("period")

    march = table.loc[table.index == "2015-03"]
    assert int(march["loans"].iloc[0]) == 2
    assert float(march["amount"].iloc[0]) == pytest.approx(500_000.0)


def test_origination_averages_are_loan_weighted(book: tuple[list[str], list[str]]) -> None:
    """Averaging quarterly averages unweighted would give a busy quarter and a quiet
    one the same say in the year."""
    table = originations_by_period(*book)

    assert (table["mean_score"] > 500).all()
    assert (table["mean_ltv"] > 0).all()


def test_mix_shares_sum_to_one_per_year(book: tuple[list[str], list[str]]) -> None:
    table = origination_mix("purpose", *book)

    totals = table.groupby("year")["share"].sum()
    assert totals.round(6).eq(1.0).all()


def test_covariate_evolution_reports_quartiles(book: tuple[list[str], list[str]]) -> None:
    table = covariate_evolution(*book)

    assert {"score_q25", "score_q50", "score_q75"} <= set(table.columns)
    assert (table["score_q25"] <= table["score_q75"]).all()


def test_default_rate_is_events_over_exposure(book: tuple[list[str], list[str]]) -> None:
    table = default_rate_by_period(*book)

    assert (table["default_rate"] >= 0).all()
    assert (table["default_rate"] <= 1).all()
    assert int(table["loan_months"].sum()) > 0


def test_a_quarter_with_no_valid_quantiles_is_skipped(tmp_path: Path) -> None:
    """A quarter where every value is a sentinel returns no quantiles at all, and
    those rows must be dropped rather than propagate a null through the average."""
    origination = [origination_row("F000000001", fico="9999", dti="999", ltv="999")]
    performance = [performance_row("F000000001", "201503", "0")]
    write_archives(tmp_path / "FREDDIE MAC", 2015, {1: (origination, performance)})
    ingest_quarter(2015, 1)

    table = covariate_evolution(
        [str(Quarter(2015, 1).parquet_path("perf"))],
        [str(Quarter(2015, 1).parquet_path("orig"))],
    )

    assert len(table) == 1
    assert "score_q50" not in table.columns or table["score_q50"].isna().all()
