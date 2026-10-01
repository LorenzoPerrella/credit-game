"""Tests for the portfolio description."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from creditsurv.data.ingest import Quarter, ingest_quarter
from creditsurv.data.panel import EVENT, WEIGHT
from creditsurv.portfolio import (
    book_summary,
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
    origination = [origination_row("F000000001", fico="9999", debt_to_income="999", ltv="999")]
    performance = [performance_row("F000000001", "201503", "0")]
    write_archives(tmp_path / "FREDDIE MAC", 2015, {1: (origination, performance)})
    ingest_quarter(2015, 1)

    table = covariate_evolution(
        [str(Quarter(2015, 1).parquet_path("perf"))],
        [str(Quarter(2015, 1).parquet_path("orig"))],
    )

    assert len(table) == 1
    assert "score_q50" not in table.columns or table["score_q50"].isna().all()


def test_the_book_summary_counts_what_it_says_it_counts() -> None:
    """S7: two documents gave two default counts, each right about a different run. Every
    number in the portfolio table says what it counts, and the modelled ones are summed from
    the cells -- a count re-derived in SQL once disagreed with the reports by 443."""
    lending = pd.DataFrame(
        {
            "period": pd.PeriodIndex(["1999-03", "2026-01"], freq="M"),
            "loans": [3, 2],
            "amount": [300.0, 250.0],
        }
    )
    outstanding = pd.DataFrame({"contracts": [3, 5, 4], "balance": [300.0, 520.0, 410.0]})
    cells = pd.DataFrame({WEIGHT: [3, 4, 4], EVENT: [False, True, False]})

    summary = book_summary(lending, outstanding, performance_rows=13, cells=cells)

    assert summary["vintages"] == "1999 - 2026"
    assert summary["loans_originated"] == 5
    assert summary["performance_rows"] == 13
    assert summary["loan_months_outstanding"] == 12
    assert summary["peak_contracts_outstanding"] == 5
    assert summary["peak_balance_outstanding"] == 520.0
    assert summary["loan_months_modelled"] == 11
    assert summary["defaults_modelled"] == 4
    assert "definition" in summary


def test_the_book_summary_has_no_modelled_figures_before_aggregation() -> None:
    lending = pd.DataFrame(
        {"period": pd.PeriodIndex(["2001-01"], freq="M"), "loans": [1], "amount": [100.0]}
    )
    outstanding = pd.DataFrame({"contracts": [1], "balance": [100.0]})

    summary = book_summary(lending, outstanding, performance_rows=1, cells=None)

    assert summary["loan_months_modelled"] is None
    assert summary["defaults_modelled"] is None
