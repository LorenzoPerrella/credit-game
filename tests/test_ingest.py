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


def test_an_archive_is_safe_only_when_every_quarter_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting the archives is the one irreversible step, so the check is three
    conditions rather than the existence of a file."""
    from creditsurv.data.ingest import audit_archives

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    write_archives(tmp_path / "FREDDIE MAC", 2015, {1: _one_loan_quarter()})
    ingest([2015])

    audit = audit_archives([2015])[0]
    assert audit.safe_to_delete
    assert audit.quarters == ("2015Q1",)


def test_a_truncated_parquet_blocks_the_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row count that no longer matches the manifest is the condition the existence
    of a file cannot catch -- a parquet overwritten or truncated since it was written."""
    import pyarrow.parquet as pq

    from creditsurv.data.ingest import Quarter, audit_archives

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    write_archives(tmp_path / "FREDDIE MAC", 2015, {1: _one_loan_quarter()})
    ingest([2015])

    path = Quarter(2015, 1).parquet_path("perf")
    table = pq.read_table(path)
    pq.write_table(table.slice(0, max(table.num_rows - 1, 0)), path)

    audit = audit_archives([2015])[0]
    assert not audit.safe_to_delete
    assert audit.mismatched == ("2015Q1/perf",)


def test_a_missing_parquet_blocks_the_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from creditsurv.data.ingest import Quarter, audit_archives

    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    write_archives(tmp_path / "FREDDIE MAC", 2015, {1: _one_loan_quarter()})
    ingest([2015])
    Quarter(2015, 1).parquet_path("orig").unlink()

    audit = audit_archives([2015])[0]
    assert not audit.safe_to_delete
    assert audit.missing == ("2015Q1/orig",)


def _one_loan_quarter() -> tuple[list[str], list[str]]:
    return (
        [origination_row("F15Q1000001")],
        [
            performance_row("F15Q1000001", "201503", "0"),
            performance_row("F15Q1000001", "201504", "1"),
            performance_row("F15Q1000001", "201505", "2"),
        ],
    )


def test_every_performance_field_is_either_kept_or_refused_with_a_reason() -> None:
    """The most expensive defect in this project was three fields dropped in silence.

    `delinquency_due_to_disaster`, `borrower_assistance_plan` and
    `payment_deferral_flag` are what distinguishes a statutory payment holiday from a
    credit default, and they were absent from the parse without appearing in any list
    of exclusions. The CARES Act required forbearance to be reported as delinquency, so
    17% of the events the model was fitted on were not credit at all.

    A field not kept must therefore be a decision recorded on the exclusion list, never
    an omission from the inclusion one.
    """
    from creditsurv.data.freddiemac import PERFORMANCE_COLUMNS
    from creditsurv.data.ingest import PERFORMANCE_DROPPED, PERFORMANCE_KEEP

    decided = set(PERFORMANCE_KEEP) | set(PERFORMANCE_DROPPED)
    undecided = [name for name in PERFORMANCE_COLUMNS if name not in decided]
    assert not undecided, f"no decision recorded for {undecided}"

    unknown = decided - set(PERFORMANCE_COLUMNS)
    assert not unknown, f"{unknown} are not in the published layout"

    overlap = set(PERFORMANCE_KEEP) & set(PERFORMANCE_DROPPED)
    assert not overlap, f"{overlap} are both kept and dropped"

    # The three that cost 17% of the dependent variable.
    for name in (
        "delinquency_due_to_disaster",
        "borrower_assistance_plan",
        "payment_deferral_flag",
    ):
        assert name in PERFORMANCE_KEEP, f"{name} is what tells a moratorium from a default"


def test_every_origination_field_is_either_kept_or_refused_with_a_reason() -> None:
    """The same rule on the origination side, where `super_conforming_flag` reached
    the parquet without ever being screened or listed as degenerate."""
    from creditsurv.data.freddiemac import ORIGINATION_COLUMNS
    from creditsurv.data.ingest import ORIGINATION_DROPPED, ORIGINATION_KEEP

    decided = set(ORIGINATION_KEEP) | set(ORIGINATION_DROPPED)
    undecided = [name for name in ORIGINATION_COLUMNS if name not in decided]
    assert not undecided, f"no decision recorded for {undecided}"
    assert not (decided - set(ORIGINATION_COLUMNS))
    assert not (set(ORIGINATION_KEEP) & set(ORIGINATION_DROPPED))


def test_a_failed_write_leaves_the_previous_parquet_intact(
    archives: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reboot during `ingest --force` left 2013Q2 truncated, because the writer went
    straight onto the only copy. Written to a sibling file and moved into place once
    closed, a crash leaves the old file or the new one -- never half of either."""
    ingest_quarter(2015, 1)
    paths = {kind: Quarter(2015, 1).parquet_path(kind) for kind in ("orig", "perf")}
    before = {kind: pq.read_table(path).num_rows for kind, path in paths.items()}

    def fail(*_: object, **__: object) -> None:
        message = "the machine rebooted"
        raise OSError(message)

    monkeypatch.setattr(pq.ParquetWriter, "write_batch", fail)
    with pytest.raises(OSError, match="rebooted"):
        ingest_quarter(2015, 1, force=True)

    for kind, path in paths.items():
        assert pq.read_table(path).num_rows == before[kind], f"{kind} was damaged"
    assert not list(paths["perf"].parent.parent.rglob("*.partial")), "a half-file was left"


def test_an_unreadable_parquet_blocks_the_deletion_instead_of_crashing(tmp_path: Path) -> None:
    """The audit read each footer and raised on one that was gone -- which would have
    taken `prune-archives` down with a traceback about magic bytes. An unreadable file
    is the strongest possible reason to keep the archive, and is reported as one."""
    from creditsurv.data.ingest import audit_archives

    write_archives(tmp_path / "FREDDIE MAC", 2015, {1: _one_loan_quarter()})
    ingest([2015])

    path = Quarter(2015, 1).parquet_path("perf")
    content = path.read_bytes()
    path.write_bytes(content[: len(content) // 2])

    audit = audit_archives([2015])[0]
    assert not audit.safe_to_delete
    assert audit.unreadable == ("2015Q1/perf",)
