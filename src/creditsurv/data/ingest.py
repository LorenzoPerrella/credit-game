"""Turning 40 GB of nested archives into columnar files that can actually be queried.

The dataset ships as one zip per vintage year, each holding four quarterly zips,
each holding two headerless pipe-delimited files. Roughly 1.75 billion performance
rows in total, about 245 GB once extracted — far past what fits in memory, and past
what is pleasant to re-parse on every run.

Nothing here ever holds a quarter in memory. The inner archive is streamed to a
temporary file, its member is read in batches by pyarrow, and each batch is written
straight out to parquet. Peak memory is one batch, so a 4 GB quarter costs the same
as a 40 MB one.

Two decisions keep the output small. Only the columns the model can use are parsed
at all — ``include_columns`` does the selection during the parse rather than after,
so the discarded ones are never materialised. And the row counts are recorded in a
manifest as the data goes past, which is what later lets the archives be deleted
with something better than optimism.

Measured: pyarrow parses this format at about 2 million rows a second, so the whole
dataset costs roughly fifteen minutes of parsing and half an hour end to end.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

from creditsurv.config import data_dir
from creditsurv.data.freddiemac import ORIGINATION_COLUMNS, PERFORMANCE_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_LOGGER: Final = logging.getLogger(__name__)

#: Where the downloaded archives live.
ARCHIVE_DIRNAME: Final = "FREDDIE MAC"

#: Rows per parse batch. Large enough to keep pyarrow busy, small enough that peak
#: memory is measured in hundreds of megabytes rather than gigabytes.
_BATCH_ROWS: Final = 500_000

_ARCHIVE_PATTERN: Final = re.compile(r"historical_data_(\d{4})\.zip$")

#: Origination fields worth keeping. The rest are either empty in every vintage
#: (``vantagescore_4``, ``pre_harp_loan_sequence_number``) or carry no signal for a
#: default model (postal code, seller name).
ORIGINATION_KEEP: Final[tuple[str, ...]] = (
    "loan_identifier",
    "classic_fico",
    "first_payment_date",
    "first_time_homebuyer_indicator",
    "maturity_date",
    "occupancy_status",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "channel",
    "property_state",
    "property_type",
    "number_of_units",
    "number_of_borrowers",
    "loan_purpose",
    "original_loan_term",
    "mortgage_insurance_percentage",
    "super_conforming_flag",
    "interest_only_indicator",
    "amortization_type",
    # Kept from September 2026: HARP refinances carry no debt-to-income, so the
    # complete-case rule dropped them -- 18% of the 2009Q2 to 2019Q1 vintages, at about
    # three times the default rate of the loans kept. With the flag in the cells they
    # become a level of their own instead of a silent omission.
    "harp_indicator",
)

#: Origination fields read from the layout and deliberately not kept, with the reason.
#:
#: Same rule as ``PERFORMANCE_DROPPED``: a field not kept is a decision on this list,
#: never an omission from the one above.
ORIGINATION_DROPPED: Final[dict[str, str]] = {
    "msa": "geography at a finer grain than the four census regions the model uses",
    "postal_code": "same, and higher cardinality still",
    "seller_name": "high cardinality; who sold the loan is not a borrower attribute",
    "prepayment_penalty_indicator": "near-constant N on conforming loans",
    "pre_harp_loan_sequence_number": "empty in every vintage checked",
    "special_eligibility_program": "affordable-lending programme, out of scope",
    "property_valuation_method": "how the value was obtained, not what it is",
    "vantagescore_4": "empty in every vintage checked",
}

#: Performance fields worth keeping.
#:
#: The last three were absent, and their absence was the most expensive thing in this
#: pipeline. The event definition is "90+ days delinquent or a loss zero-balance
#: code", and the **CARES Act required loans in forbearance to be reported as
#: delinquent** -- so a payment holiday granted by statute reads as a credit default.
#: It is 17% of all events: 330,056 of them, peaking at 90.4 bp a month in May 2020
#: against a 2019 baseline of 3.07 bp, a factor of 29 where the 2008 crisis managed
#: 24.5 bp. Of the loans first reaching 90+ in 2020, **99.5% returned to performing**.
#: Hurricanes add another 27,000.
#:
#: These three fields are what tells a moratorium from a default, and they were being
#: dropped at the parse without appearing in any list of exclusions. See
#: ``aggregate.MoratoriumPolicy`` for what is done with them.
#:
#: The loss columns are deliberately absent: they are populated only for defaulted
#: loans and only matter for LGD, which is out of scope.
PERFORMANCE_KEEP: Final[tuple[str, ...]] = (
    "loan_identifier",
    "period",
    "current_actual_upb",
    "current_loan_delinquency_status",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "zero_balance_code",
    "current_interest_rate",
    "estimated_loan_to_value",
    "modification_flag",
    "delinquency_due_to_disaster",
    "borrower_assistance_plan",
    "payment_deferral_flag",
)

#: Performance fields read from the layout and deliberately not kept, with the reason.
#:
#: Recorded because the three fields above were dropped silently and cost the project
#: 17% of its dependent variable. A field that is not kept should be a decision on this
#: list, never an omission from the one above.
PERFORMANCE_DROPPED: Final[dict[str, str]] = {
    # Loss accounting. Every one is populated only for defaulted loans and only feeds
    # LGD, which is out of scope: this project models PD alone.
    "mi_recoveries": "loss accounting, needs LGD",
    "net_sales_proceeds": "loss accounting, needs LGD",
    "non_mi_recoveries": "loss accounting, needs LGD",
    "total_expenses": "loss accounting, needs LGD",
    "legal_costs": "loss accounting, needs LGD",
    "maintenance_and_preservation_costs": "loss accounting, needs LGD",
    "taxes_and_insurance": "loss accounting, needs LGD",
    "miscellaneous_expenses": "loss accounting, needs LGD",
    "actual_loss": "loss accounting, needs LGD",
    "delinquent_accrued_interest": "loss accounting, needs LGD",
    "bankruptcy_cramdown_costs": "loss accounting, needs LGD",
    # Modification cost detail. A modification already ends observation, so what it
    # cost is never read.
    "cumulative_modification_costs": "modification already censors; cost not read",
    "current_period_modification_costs": "modification already censors; cost not read",
    "interest_rate_step_indicator": "modification detail; modification already censors",
    # Balance components. The model reads no balances -- the weight is a count of
    # loan-months, never an amount -- so only current_actual_upb is kept, for the
    # portfolio description.
    "current_non_interest_bearing_upb": "a balance component; the model reads no balances",
    "current_interest_bearing_upb": "a balance component; the model reads no balances",
    "zero_balance_removal_upb": "a balance component; the model reads no balances",
    # Redundant with something already kept.
    "zero_balance_effective_date": "redundant with the period the code appears in",
    "due_date_of_last_paid_installment": "redundant with the delinquency counter",
    # Not a state of the loan.
    "defect_settlement_date": "a repurchase settlement, not a borrower outcome",
    "mortgage_insurance_cancellation_indicator": "insurance administration, not risk",
    "servicer_name": "high cardinality, and servicer identity is not a risk driver here",
}
#: Everything is read as string and converted later. The dataset encodes missing
#: values as sentinels (9999, 999) and delinquency status as alphanumeric, so an
#: eager numeric cast here would either fail or quietly invent data.
_STRING_TYPES: Final = pa.string()


@dataclass(frozen=True)
class Quarter:
    """One vintage quarter, and where its files land."""

    year: int
    quarter: int

    @property
    def tag(self) -> str:
        return f"{self.year}Q{self.quarter}"

    def parquet_path(self, kind: str) -> Path:
        return interim_dir() / kind / f"{self.tag}.parquet"


def archive_dir() -> Path:
    return data_dir() / ARCHIVE_DIRNAME


def interim_dir() -> Path:
    return data_dir() / "interim"


def manifest_path() -> Path:
    return interim_dir() / "manifest.json"


def discover_years(directory: Path | None = None) -> list[int]:
    """Vintage years present as archives, in order."""
    source = directory or archive_dir()
    if not source.exists():
        message = (
            f"No archive directory at {source}. Download the Single-Family "
            "Loan-Level Dataset and place the historical_data_YYYY.zip files there."
        )
        raise FileNotFoundError(message)
    years = [
        int(match.group(1))
        for path in source.iterdir()
        if (match := _ARCHIVE_PATTERN.search(path.name))
    ]
    return sorted(years)


def _inner_archives(outer: Path) -> list[str]:
    with zipfile.ZipFile(outer) as archive:
        return sorted(name for name in archive.namelist() if name.endswith(".zip"))


def _stream_inner(outer: Path, member: str, destination: Path) -> None:
    """Extract one inner archive without holding it in memory.

    ``unzip -p`` writes the member to stdout, which is redirected straight to disk.
    Python's zipfile would need the whole member resident before it could be opened
    as an archive in turn, and the largest are several gigabytes.
    """
    with destination.open("wb") as sink:
        subprocess.run(
            ["/usr/bin/unzip", "-p", str(outer), member],
            stdout=sink,
            stderr=subprocess.DEVNULL,
            check=True,
        )


def _convert_member(
    archive: zipfile.ZipFile,
    member: str,
    columns: tuple[str, ...],
    keep: tuple[str, ...],
    destination: Path,
) -> int:
    """Parse one member in batches and write it to parquet, atomically.

    Written to a sibling ``.partial`` file and moved into place only once the writer has
    closed. The first version wrote straight onto the destination, which is fine until
    an ingest runs with ``--force`` over quarters that already exist -- and the machine
    rebooted during one. 2013Q2's performance file was left truncated ("magic bytes not
    found in footer"), the only intact copy of that quarter was gone, and the manifest
    still recorded the row count of the file it had replaced, so ``completed_files``
    would have handed the truncated file to the aggregation.

    ``Path.replace`` is ``os.replace``, atomic on one filesystem: a crash now leaves the
    old file or the new one, never half of either.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    rows = 0
    try:
        with archive.open(member) as handle:
            reader = pv.open_csv(
                handle,
                read_options=pv.ReadOptions(column_names=list(columns), block_size=1 << 24),
                parse_options=pv.ParseOptions(delimiter="|"),
                convert_options=pv.ConvertOptions(
                    include_columns=list(keep),
                    column_types=dict.fromkeys(columns, _STRING_TYPES),
                    strings_can_be_null=True,
                ),
            )
            writer: pq.ParquetWriter | None = None
            try:
                for batch in reader:
                    if writer is None:
                        writer = pq.ParquetWriter(partial, batch.schema, compression="zstd")
                    writer.write_batch(batch)
                    rows += batch.num_rows
            finally:
                if writer is not None:
                    writer.close()
        if writer is not None:
            partial.replace(destination)
    finally:
        # A no-op after a successful replace; after a failure, it removes the half-file
        # and leaves the destination exactly as it was.
        partial.unlink(missing_ok=True)
    return rows


def ingest_quarter(year: int, quarter: int, *, force: bool = False) -> dict[str, int]:
    """Convert one quarter to parquet. Returns the row counts written."""
    target = Quarter(year, quarter)
    outputs = {kind: target.parquet_path(kind) for kind in ("orig", "perf")}
    if not force and all(path.exists() for path in outputs.values()):
        _LOGGER.debug("%s already ingested", target.tag)
        return {}

    outer = archive_dir() / f"historical_data_{year}.zip"
    member = f"historical_data_{target.tag}.zip"
    if member not in _inner_archives(outer):
        message = f"{outer.name} does not contain {member}."
        raise FileNotFoundError(message)

    counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as scratch:
        inner = Path(scratch) / member
        _stream_inner(outer, member, inner)
        with zipfile.ZipFile(inner) as archive:
            names = archive.namelist()
            for kind, columns, keep in (
                ("orig", ORIGINATION_COLUMNS, ORIGINATION_KEEP),
                ("perf", PERFORMANCE_COLUMNS, PERFORMANCE_KEEP),
            ):
                name = next(entry for entry in names if entry.startswith(kind))
                counts[kind] = _convert_member(archive, name, columns, keep, outputs[kind])
    return counts


def quarters_for(years: list[int]) -> Iterator[Quarter]:
    for year in years:
        for quarter in (1, 2, 3, 4):
            yield Quarter(year, quarter)


def completed_files(kind: str) -> list[str]:
    """Parquet paths for the quarters the manifest records as finished.

    Globbing the directory instead would pick up a file still being written -- an
    ingest running in one terminal and an aggregation in another is not a hypothetical
    -- and a half-written parquet fails with a message about magic bytes that says
    nothing about the cause. The manifest is written only after a quarter closes, so
    it is the only honest answer to "what is complete".
    """
    manifest = load_manifest()
    paths = [interim_dir() / kind / f"{tag}.parquet" for tag in sorted(manifest)]
    return [str(path) for path in paths if path.exists()]


def load_manifest() -> dict[str, dict[str, int]]:
    path = manifest_path()
    if not path.exists():
        return {}
    loaded: dict[str, dict[str, int]] = json.loads(path.read_text())
    return loaded


def save_manifest(manifest: dict[str, dict[str, int]]) -> Path:
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def ingest(
    years: list[int] | None = None,
    *,
    force: bool = False,
) -> dict[str, dict[str, int]]:
    """Convert every requested vintage to parquet, recording what was written.

    Idempotent: a quarter whose parquet already exists is skipped, so an interrupted
    run costs only the quarter it was in the middle of. The manifest accumulates
    rather than being rewritten, so a partial run still leaves a usable record.
    """
    requested = years if years is not None else discover_years()
    manifest = load_manifest()

    for target in quarters_for(requested):
        outer = archive_dir() / f"historical_data_{target.year}.zip"
        if not outer.exists():
            continue
        # A partial vintage year genuinely has fewer than four quarters.
        if f"historical_data_{target.tag}.zip" not in _inner_archives(outer):
            continue

        counts = ingest_quarter(target.year, target.quarter, force=force)
        if counts:
            manifest[target.tag] = counts
            save_manifest(manifest)
            _LOGGER.info(
                "%s: %d origination rows, %d performance rows",
                target.tag,
                counts["orig"],
                counts["perf"],
            )
    return manifest


@dataclass(frozen=True)
class ArchiveAudit:
    """What a vintage archive's quarters look like against the manifest and the parquet."""

    year: int
    path: Path
    quarters: tuple[str, ...]
    missing: tuple[str, ...]
    mismatched: tuple[str, ...]
    #: Present but unparseable -- a footer gone, as a crash mid-write leaves it.
    unreadable: tuple[str, ...]
    bytes_on_disk: int

    @property
    def safe_to_delete(self) -> bool:
        return (
            not self.missing and not self.mismatched and not self.unreadable and bool(self.quarters)
        )

    def describe(self) -> dict[str, object]:
        return {
            "year": self.year,
            "quarters": len(self.quarters),
            "missing": ", ".join(self.missing) or "-",
            "mismatched": ", ".join(self.mismatched) or "-",
            "unreadable": ", ".join(self.unreadable) or "-",
            "gigabytes": round(self.bytes_on_disk / 1024**3, 2),
            "safe": self.safe_to_delete,
        }


def audit_archives(years: Sequence[int] | None = None) -> list[ArchiveAudit]:
    """Check, archive by archive, that nothing would be lost by deleting it.

    Deleting the archives is the only irreversible step in this pipeline, and
    re-downloading them takes hours behind a manual registration. So the check is not
    "is there a parquet file" -- a half-written one is also a file -- but three separate
    conditions, each of which has to hold for every quarter of the year:

    * the manifest records the quarter as finished, which only happens after it closed;
    * the parquet files exist, both origination and performance;
    * **their row counts still match what the manifest recorded** when they were
      written. That is the one that catches a file truncated or overwritten since.

    Recounting reads the parquet footers rather than the data, so the whole audit is
    seconds even across 2.9 billion rows.
    """

    manifest = load_manifest()
    selected = years or discover_years()
    audits = []

    for year in selected:
        path = archive_dir() / f"historical_data_{year}.zip"
        if not path.exists():
            continue
        tags = [tag for tag in sorted(manifest) if tag.startswith(str(year))]
        missing, mismatched, unreadable = [], [], []
        for tag in tags:
            for kind in ("orig", "perf"):
                parquet = interim_dir() / kind / f"{tag}.parquet"
                if not parquet.exists():
                    missing.append(f"{tag}/{kind}")
                    continue
                # A file whose footer is gone -- what a crash mid-write leaves -- raises
                # on the first read. It is the strongest possible reason to keep the
                # archive, so it is recorded as one rather than allowed to abort the
                # audit with a traceback about magic bytes.
                try:
                    counted = pq.ParquetFile(parquet).metadata.num_rows
                except (OSError, pa.ArrowException):
                    unreadable.append(f"{tag}/{kind}")
                    continue
                if counted != manifest[tag][kind]:
                    mismatched.append(f"{tag}/{kind}")
        audits.append(
            ArchiveAudit(
                year=year,
                path=path,
                quarters=tuple(tags),
                missing=tuple(missing),
                mismatched=tuple(mismatched),
                unreadable=tuple(unreadable),
                bytes_on_disk=path.stat().st_size,
            )
        )
    return audits
