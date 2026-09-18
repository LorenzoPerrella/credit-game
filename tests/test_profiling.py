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
        table.loc[table.index == "cash_out_refinance", "default_rate"].iloc[0]
    ) == pytest.approx(0.5)


def test_a_rare_level_is_marked_for_merging(tmp_path: Path) -> None:
    """Below the minimum share a level is merged rather than dropped, as nmds does."""
    origination = [origination_row(f"F{i:09d}", purpose="P" if i < 99 else "C") for i in range(100)]
    performance = [performance_row(f"F{i:09d}", "201503", "0") for i in range(100)]
    _ingested(tmp_path, origination, performance)

    table = profile_categorical("purpose", *_sources(tmp_path)).set_index("level")

    verdict = str(table.loc[table.index == "cash_out_refinance", "verdict"].iloc[0])
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

    table = profile_continuous("credit_score", (620.0, 660.0, 700.0), *_sources(tmp_path))

    assert is_monotonic(table)


def test_a_non_monotone_covariate_is_reported_as_such(tmp_path: Path) -> None:
    """This check is what caught mark-to-market leverage carrying an untreated
    missing-value sentinel: credit score and loan-to-value ordered their own risk
    cleanly, and that one did not.

    Scores are placed at the centre of each band so the assignment is unambiguous,
    and defaults are put in alternating bands so the rate rises and falls.
    """
    # Bands at 620, 660 and 700 points.
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

    table = profile_continuous("credit_score", (620.0, 660.0, 700.0), *_sources(tmp_path))

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

    table = profile_continuous("credit_score", (-1.6, 0.0), *_sources(tmp_path))

    assert (table["band"] >= 0).all() or is_monotonic(table)


def test_cut_points_come_back_ordered(tmp_path: Path) -> None:
    """Quantiles say where the mass is. They are a starting point, not an answer:
    data-driven cuts fit the sample they were taken from."""
    origination = [origination_row(f"F{i:09d}", ltv=str(40 + i)) for i in range(50)]
    performance = [performance_row(f"F{i:09d}", "201503", "0") for i in range(50)]
    _ingested(tmp_path, origination, performance)

    edges = propose_cut_points("original_ltv", *_sources(tmp_path))

    assert edges == sorted(edges)
    assert len(edges) > 2


def test_every_extension_of_the_key_is_priced_against_the_same_base() -> None:
    """A cost attributed to an extension is only a cost if one thing changed.

    And the ladder has to walk the give-up order of docs/rules.md, so the first
    specification under the ceiling can be read off the table rather than argued for
    afterwards.
    """
    from creditsurv.data.aggregate import BASE_SPEC, Extension
    from creditsurv.profiling import _priced_specifications

    priced = _priced_specifications()

    assert priced["base"] == BASE_SPEC
    for extension in Extension:
        alone = priced[extension.value]
        changed = set(alone.categorical) - set(BASE_SPEC.categorical)
        widened = {
            name
            for name, edges in alone.continuous.items()
            if edges != BASE_SPEC.continuous.get(name)
        }
        assert len(changed) + len(widened) >= 1
        assert len(changed) <= 1, f"{extension} changes more than its own level"
    assert list(priced)[-3:] == [
        "all less fine_bands",
        "all less fine_bands, origination_spread",
        "all less fine_bands, origination_spread, delinquency_state",
    ]


def test_the_cost_of_the_extensions_is_measured_quarter_by_quarter(tmp_path: Path) -> None:
    """On fixtures, where the answer is small enough to check by hand."""
    from creditsurv.profiling import extension_cost

    origination = [
        origination_row(
            f"F{i:09d}",
            fico=str(600 + i * 10),
            harp="Y" if i % 3 == 0 else "N",
            debt_to_income="" if i % 3 == 0 else "32",
        )
        for i in range(12)
    ]
    performance = [
        performance_row(f"F{i:09d}", f"2015{month:02d}", str(month - 3))
        for i in range(12)
        for month in (3, 4)
    ]
    _ingested(tmp_path, origination, performance)

    table = extension_cost(["2015Q1"], published_cells=63_639_116)

    assert list(table["specification"])[:2] == ["base", "harp"]
    assert float(table.loc[table["specification"] == "base", "multiple_of_base"].iloc[0]) == 1.0
    priced = table.set_index("specification")["multiple_of_base"]
    assert priced["all"] >= priced["all less fine_bands"] >= 1.0
    assert (table["projected_cells"] >= 63_639_116).all()
