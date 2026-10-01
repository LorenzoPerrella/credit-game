"""The portfolio views, on a book written in Freddie Mac's own format and ingested for real."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from creditsurv.data.aggregate import build_cells
from creditsurv.data.ingest import Quarter, ingest_quarter
from creditsurv.views.portfolio import (
    book_by_segment,
    lending_by_segment,
    underwriting_by_vintage,
    vintage_curves,
)
from fixtures import origination_row, performance_row, write_archives

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Three loans: one performing, one defaulting in its second month, one prepaying."""
    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    origination = [
        origination_row("F000000001", upb="200000", first_payment="201503", purpose="P"),
        origination_row("F000000002", upb="300000", first_payment="201503", purpose="P"),
        origination_row("F000000003", upb="100000", first_payment="201503", purpose="C"),
    ]
    performance = [
        performance_row("F000000001", "201503", "0"),
        performance_row("F000000001", "201504", "1"),
        performance_row("F000000001", "201505", "2"),
        performance_row("F000000002", "201503", "0"),
        performance_row("F000000002", "201504", "1", delinquency="3"),
        performance_row("F000000003", "201503", "0"),
        performance_row("F000000003", "201504", "1", zero_balance="01", upb="0"),
    ]
    write_archives(tmp_path / "FREDDIE MAC", 2015, {1: (origination, performance)})
    ingest_quarter(2015, 1)
    return (
        str(Quarter(2015, 1).parquet_path("perf")),
        str(Quarter(2015, 1).parquet_path("orig")),
    )


def test_the_book_counts_loan_months_defaults_and_prepayments_by_month(
    sources: tuple[str, str],
) -> None:
    table = book_by_segment(*sources)
    whole = table[table["segment"] == "all"].set_index("month")

    assert whole["loan_months"].to_dict() == {"2015-03": 3, "2015-04": 3, "2015-05": 1}
    assert whole.loc["2015-04", "defaults"] == 1
    assert whole.loc["2015-04", "prepayments"] == 1
    assert whole.loc["2015-04", "smm"] == pytest.approx(1 / 3)
    assert whole.loc["2015-04", "cpr"] == pytest.approx(1 - (2 / 3) ** 12)
    assert whole.loc["2015-04", "default_rate_bp"] == pytest.approx(1e4 / 3)


def test_every_segment_adds_up_to_the_book_in_every_month(sources: tuple[str, str]) -> None:
    table = book_by_segment(*sources)
    whole = table[table["segment"] == "all"].set_index("month")["loan_months"]

    for segment, rows in table[table["segment"] != "all"].groupby("segment"):
        summed = rows.groupby("month")["loan_months"].sum()
        assert summed.to_dict() == whole.to_dict(), segment


def test_segments_are_labelled_as_the_calibration_views_label_them(
    sources: tuple[str, str],
) -> None:
    table = book_by_segment(*sources)

    purposes = set(table.loc[table["segment"] == "purpose", "group"])
    scores = set(table.loc[table["segment"] == "fico", "group"])
    assert {"purchase", "cash_out_refinance"} <= purposes
    assert all(" to " in label for label in scores)


def test_lending_counts_each_loan_once_with_its_amount(sources: tuple[str, str]) -> None:
    table = lending_by_segment(*sources)
    whole = table[table["segment"] == "all"].set_index("year")
    purpose = table[table["segment"] == "purpose"]

    assert whole.loc[2015, "loans"] == 3
    assert whole.loc[2015, "amount"] == pytest.approx(600_000)
    assert purpose["loan_share"].sum() == pytest.approx(1.0)
    assert purpose.set_index("group").loc["purchase", "loans"] == 2


def test_vintage_curves_cumulate_default_on_each_vintage_s_risk_sets(
    sources: tuple[str, str],
) -> None:
    curves = vintage_curves(build_cells(*sources))
    vintage = curves[curves["vintage_year"] == 2015].set_index("age")

    assert vintage.loc[0, "cumulative_default"] == pytest.approx(0.0)
    assert vintage.loc[1, "at_risk"] == pytest.approx(3.0)
    assert vintage.loc[1, "cumulative_default"] == pytest.approx(1 / 3)
    assert isinstance(curves, pd.DataFrame)


def test_underwriting_quartiles_are_one_long_table_by_measure(sources: tuple[str, str]) -> None:
    table = underwriting_by_vintage(sources[1])

    assert set(table["measure"]) == {"score", "ltv", "dti"}
    assert (table["loans"] == 3).all()
    assert (table["q25"] <= table["q50"]).all()
    assert (table["q50"] <= table["q75"]).all()
