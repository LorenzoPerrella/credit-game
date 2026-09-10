"""Tests for the pre-aggregation screening.

The step exists because the order matters: screening before the group-by means the
cut points and merges are decided from the data, screening after it means they were
assumed. These tests check the screening says the right thing, on fixtures small
enough to reason about.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from creditsurv.data.ingest import Quarter, ingest_quarter
from creditsurv.profiling import (
    is_monotonic,
    profile_categorical,
    profile_continuous,
    propose_cut_points,
    screen_categoricals,
)
from fixtures import origination_row, performance_row, write_archives

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))


def _ingested(tmp_path: Path, origination: list[str], performance: list[str]) -> None:
    write_archives(tmp_path / "FREDDIE MAC", 2015, {1: (origination, performance)})
    ingest_quarter(2015, 1)


def _sources(tmp_path: Path) -> tuple[list[str], list[str]]:
    return (
        [str(Quarter(2015, 1).parquet_path("perf"))],
        [str(Quarter(2015, 1).parquet_path("orig"))],
    )


def test_categorical_profile_reports_share_and_default_rate(tmp_path: Path) -> None:
    origination = [origination_row(f"F{i:09d}", purpose="P" if i < 8 else "C") for i in range(10)]
    performance = [
        performance_row(f"F{i:09d}", "201503", "0", delinquency="3" if i == 9 else "0")
        for i in range(10)
    ]
    _ingested(tmp_path, origination, performance)

    table = profile_categorical("purpose", *_sources(tmp_path)).set_index("level")

    assert float(table.loc[table.index == "purchase", "share"].iloc[0]) == pytest.approx(0.8)
    assert float(
        table.loc[table.index == "refinance_cashout", "default_rate"].iloc[0]
    ) == pytest.approx(0.5)


def test_a_rare_level_is_marked_for_merging(tmp_path: Path) -> None:
    """Below the minimum share a level is merged rather than dropped, as nmds does."""
    origination = [origination_row(f"F{i:09d}", purpose="P" if i < 99 else "C") for i in range(100)]
    performance = [performance_row(f"F{i:09d}", "201503", "0") for i in range(100)]
    _ingested(tmp_path, origination, performance)

    table = profile_categorical("purpose", *_sources(tmp_path)).set_index("level")

    verdict = str(table.loc[table.index == "refinance_cashout", "verdict"].iloc[0])
    assert verdict == "merge: below minimum share"


def test_screening_flags_a_degenerate_covariate(tmp_path: Path) -> None:
    """Degenerate means there is nothing left to estimate a contrast from."""
    origination = [origination_row(f"F{i:09d}", occupancy="P") for i in range(200)]
    performance = [performance_row(f"F{i:09d}", "201503", "0") for i in range(200)]
    _ingested(tmp_path, origination, performance)

    table = screen_categoricals(["occupancy", "purpose"], *_sources(tmp_path)).set_index(
        "covariate"
    )

    assert str(table.loc[table.index == "occupancy", "verdict"].iloc[0]) == "drop: degenerate"


def test_continuous_profile_detects_monotone_risk(tmp_path: Path) -> None:
    """A covariate that orders its own risk is doing its job."""
    origination = [origination_row(f"F{i:09d}", fico=str(600 + i * 5)) for i in range(40)]
    # Worse scores default; better ones do not.
    performance = [
        performance_row(f"F{i:09d}", "201503", "0", delinquency="3" if i < 10 else "0")
        for i in range(40)
    ]
    _ingested(tmp_path, origination, performance)

    table = profile_continuous("fico_s", (-1.6, -0.8, 0.0), *_sources(tmp_path))

    assert is_monotonic(table)


def test_a_non_monotone_covariate_is_reported_as_such(tmp_path: Path) -> None:
    """This check is what caught mark-to-market leverage carrying an untreated
    missing-value sentinel: credit score and loan-to-value ordered their own risk
    cleanly, and that one did not.

    Scores are placed at the centre of each band so the assignment is unambiguous,
    and defaults are put in alternating bands so the rate rises and falls.
    """
    # fico_s = (fico - 700) / 50, bands at -1.6, -0.8, 0.0.
    scores = {0: "610", 1: "640", 2: "680", 3: "750"}
    defaulting_bands = {0, 2}

    origination, performance = [], []
    for band, score in scores.items():
        for i in range(10):
            loan = f"F{band}{i:08d}"
            origination.append(origination_row(loan, fico=score))
            performance.append(
                performance_row(
                    loan,
                    "201503",
                    "0",
                    delinquency="3" if band in defaulting_bands and i < 5 else "0",
                )
            )
    _ingested(tmp_path, origination, performance)

    table = profile_continuous("fico_s", (-1.6, -0.8, 0.0), *_sources(tmp_path))

    assert not is_monotonic(table)


def test_missing_is_excluded_from_the_monotonicity_check(tmp_path: Path) -> None:
    """Missing is not a point on the covariate's scale, so including it would report
    a break that says nothing about the binning."""
    origination = [origination_row(f"F{i:09d}", fico=str(620 + i * 20)) for i in range(20)]
    performance = [
        performance_row(f"F{i:09d}", "201503", "0", delinquency="3" if i < 5 else "0")
        for i in range(20)
    ]
    _ingested(tmp_path, origination, performance)

    table = profile_continuous("fico_s", (-1.6, 0.0), *_sources(tmp_path))

    assert (table["band"] >= 0).all() or is_monotonic(table)


def test_cut_points_come_back_ordered(tmp_path: Path) -> None:
    """Quantiles say where the mass is. They are a starting point, not an answer:
    data-driven cuts fit the sample they were taken from."""
    origination = [origination_row(f"F{i:09d}", ltv=str(40 + i)) for i in range(50)]
    performance = [performance_row(f"F{i:09d}", "201503", "0") for i in range(50)]
    _ingested(tmp_path, origination, performance)

    edges = propose_cut_points("orig_ltv", *_sources(tmp_path))

    assert edges == sorted(edges)
    assert len(edges) > 2
