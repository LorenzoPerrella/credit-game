"""Tests for the archive ingest.

Built against synthetic archives rather than the real 40 GB download: the nesting,
the column selection and the idempotence are what matter, and none of them need
real data to exercise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from creditsurv.data.ingest import (
    ORIGINATION_KEEP,
    PERFORMANCE_KEEP,
    Quarter,
    discover_years,
    ingest,
    ingest_quarter,
    load_manifest,
)
from fixtures import origination_row, performance_row, write_archives

if TYPE_CHECKING:
    from pathlib import Path


def _quarter_rows(tag: str, loans: int) -> tuple[list[str], list[str]]:
    origination = [origination_row(f"F{tag}{i:06d}") for i in range(loans)]
    performance = [
        performance_row(f"F{tag}{i:06d}", "201503", str(age))
        for i in range(loans)
        for age in range(3)
    ]
    return origination, performance


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))


@pytest.fixture
def archives(tmp_path: Path) -> Path:
    root = tmp_path / "FREDDIE MAC"
    write_archives(root, 2015, {1: _quarter_rows("2015Q1", 4), 2: _quarter_rows("2015Q2", 3)})
    return root


def test_discovers_every_vintage_archive(archives: Path) -> None:
    assert discover_years() == [2015]


def test_missing_archive_directory_says_what_to_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Loan-Level Dataset"):
        discover_years(tmp_path / "absent")


def test_a_quarter_converts_to_parquet(archives: Path) -> None:
    counts = ingest_quarter(2015, 1)

    assert counts == {"orig": 4, "perf": 12}
    assert Quarter(2015, 1).parquet_path("orig").exists()
    assert Quarter(2015, 1).parquet_path("perf").exists()


def test_only_the_wanted_columns_survive(archives: Path) -> None:
    """Selection happens during the parse, so the discarded columns are never
    materialised — which is most of why this fits in memory at all."""
    ingest_quarter(2015, 1)

    origination = pq.read_table(Quarter(2015, 1).parquet_path("orig"))
    performance = pq.read_table(Quarter(2015, 1).parquet_path("perf"))

    assert tuple(origination.column_names) == ORIGINATION_KEEP
    assert tuple(performance.column_names) == PERFORMANCE_KEEP
    assert "seller_name" not in origination.column_names
    assert "net_sales_proceeds" not in performance.column_names


def test_values_survive_the_round_trip(archives: Path) -> None:
    ingest_quarter(2015, 1)

    table = pq.read_table(Quarter(2015, 1).parquet_path("perf"))
    ages = table.column("loan_age").to_pylist()

    assert sorted(set(ages)) == ["0", "1", "2"]


def test_ingest_is_idempotent(archives: Path) -> None:
    """An interrupted run must cost only the quarter it was in the middle of."""
    first = ingest_quarter(2015, 1)
    second = ingest_quarter(2015, 1)

    assert first == {"orig": 4, "perf": 12}
    assert second == {}, "a completed quarter should be skipped"


def test_force_reconverts(archives: Path) -> None:
    ingest_quarter(2015, 1)

    assert ingest_quarter(2015, 1, force=True) == {"orig": 4, "perf": 12}


def test_ingest_records_a_manifest(archives: Path) -> None:
    """The manifest is what later lets the archives be deleted with something
    better than optimism."""
    ingest()

    manifest = load_manifest()
    assert set(manifest) == {"2015Q1", "2015Q2"}
    assert manifest["2015Q1"] == {"orig": 4, "perf": 12}
    assert manifest["2015Q2"] == {"orig": 3, "perf": 9}


def test_absent_quarters_are_skipped_not_failed(archives: Path) -> None:
    """A partial vintage year genuinely has fewer than four quarters."""
    manifest = ingest([2015])

    assert "2015Q3" not in manifest
    assert "2015Q4" not in manifest


def test_a_missing_quarter_inside_an_archive_raises(archives: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not contain"):
        ingest_quarter(2015, 3)
