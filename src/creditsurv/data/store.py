"""Persisting the aggregated cells.

Parquet rather than CSV: it round-trips the monthly ``Period`` columns and the
categorical dtypes without a schema of its own, and the table is large enough that
the difference in size and load time is worth having.

Only the cells are stored. An intermediate loan-month panel used to be written here
too, from a single quarter; it stopped being read once every command took the
aggregated path, and a saved artefact nothing loads is a way to end up fitting last
week's data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from typing import TYPE_CHECKING, Final, cast

import pandas as pd

from creditsurv.config import processed_dir

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_LOGGER: Final = logging.getLogger(__name__)

#: The moratorium policy a cell table is built under when none is named.
DEFAULT_POLICY: Final = "exclude"


def cells_path(policy: str = DEFAULT_POLICY) -> Path:
    """Where the cells built under ``policy`` live.

    One file per moratorium policy. The two treatments are built to be compared on
    coefficients and on the backtest, and a single ``cells.parquet`` would let the second
    aggregation overwrite the first without a word -- after which any comparison is of a
    table with itself.
    """
    return processed_dir() / f"cells_{policy}.parquet"


def save_cells(cells: pd.DataFrame, policy: str = DEFAULT_POLICY) -> Path:
    """Persist the aggregated cells, atomically.

    Written to a sibling file and moved into place, for the reason the ingest learned by
    losing a quarter to a reboot: an aggregation is hours of work, and a crash while
    writing should leave the previous table intact rather than half of a new one.
    """
    path = cells_path(policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        cells.to_parquet(partial)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def load_cells(
    policy: str = DEFAULT_POLICY, *, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Read the cells built under ``policy``, or say how to build them.

    ``columns`` reads only those.
    """
    path = cells_path(policy)
    if not path.exists():
        message = (
            f"No aggregated cells at {path}. Build them first:\n"
            "  uv run creditsurv ingest\n"
            f"  uv run creditsurv aggregate --moratorium {policy}"
        )
        raise FileNotFoundError(message)
    cells = pd.read_parquet(path, columns=None if columns is None else list(columns))
    # A table saved before its text keys were categorical comes back as Python strings.
    # Categorised here, once, over the whole table -- which is also what keeps the levels
    # of every later slice identical.
    for column in cells.columns:
        if cells[column].dtype == object:
            cells[column] = cells[column].astype("category")
    return cells


def load_cells_window(
    policy: str = DEFAULT_POLICY,
    *,
    first: int | None = None,
    last: int | None = None,
) -> pd.DataFrame:
    """The cells observed between two month ordinals, read without the rest of the table.

    ``first`` and ``last`` are ``year * 12 + month - 1``, inclusive, on the **observation**
    month -- which is ``origination_month + age`` and therefore not a column, so the filter
    cannot be pushed into parquet by name. DuckDB evaluates it while reading instead, and
    what comes back is the window alone.

    That is the difference between a backtest window and the book: 91.6 million cells are a
    few gigabytes as a frame, and two years of observation are about 3% of them. The whole
    table is read only where the whole table is the question.
    """
    import duckdb

    path = cells_path(policy)
    if not path.exists():
        message = (
            f"No aggregated cells at {path}. Build them first:\n"
            "  uv run creditsurv ingest\n"
            f"  uv run creditsurv aggregate --moratorium {policy}"
        )
        raise FileNotFoundError(message)

    bounds = []
    if first is not None:
        bounds.append(f"origination_month + age >= {int(first)}")
    if last is not None:
        bounds.append(f"origination_month + age <= {int(last)}")
    where = f"WHERE {' AND '.join(bounds)}" if bounds else ""
    cells: pd.DataFrame = (
        duckdb.connect().execute(f"SELECT * FROM read_parquet('{path}') {where}").df()
    )
    for column in cells.columns:
        if cells[column].dtype == object:
            cells[column] = cells[column].astype("category")
    return cells


def load_largest_cells(
    policy: str = DEFAULT_POLICY, *, age: int = 0, limit: int = 500
) -> pd.DataFrame:
    """The ``limit`` heaviest cells at one loan age, read without the rest of the table.

    At age zero these are the origination profiles the book was written in, and their weights
    say how much of it each accounts for -- which is what the projections are scored on. A
    few hundred rows out of 91.6 million: the ordering and the limit belong in the reader.
    """
    import duckdb

    path = cells_path(policy)
    cells: pd.DataFrame = (
        duckdb.connect()
        .execute(
            f"""
            SELECT * FROM read_parquet('{path}')
            WHERE age = {int(age)} ORDER BY loan_months DESC LIMIT {int(limit)}
            """
        )
        .df()
    )
    for column in cells.columns:
        if cells[column].dtype == object:
            cells[column] = cells[column].astype("category")
    return cells


def outcomes_by_age(
    policy: str = DEFAULT_POLICY, *, first: int | None = None, last: int | None = None
) -> pd.DataFrame:
    """Loan-months by loan age and outcome, aggregated inside the parquet reader.

    Everything the Aalen-Johansen estimator needs -- the exposure at each age and the exits
    of each kind among it -- as a few hundred rows. The cells themselves are 91.6 million,
    and a non-parametric curve is a statement about sums, so there is no reason to hold them
    to compute one.
    """
    import duckdb

    path = cells_path(policy)
    bounds = []
    if first is not None:
        bounds.append(f"origination_month + age >= {int(first)}")
    if last is not None:
        bounds.append(f"origination_month + age <= {int(last)}")
    where = f"WHERE {' AND '.join(bounds)}" if bounds else ""
    frame: pd.DataFrame = (
        duckdb.connect()
        .execute(
            f"""
            SELECT age, outcome, SUM(loan_months) AS loan_months
            FROM read_parquet('{path}') {where}
            GROUP BY ALL ORDER BY age
            """
        )
        .df()
    )
    frame["outcome"] = frame["outcome"].astype("category")
    return frame


def cells_identity(policy: str = DEFAULT_POLICY) -> str:
    """The cell table's name, size and time of writing.

    What a selection fit is cached under, so it is never reused for a table rebuilt since,
    even one that happens to have as many rows.
    """
    path = cells_path(policy)
    source = path.stat()
    return f"{path.name}:{source.st_size}:{source.st_mtime_ns}"


#: Where fitted models are cached, under the processed directory.
FITS_DIRNAME = "fits"


def fit_fingerprint(**parts: object) -> str:
    """A stable name for a fit, from everything that determines it.

    Everything: the distribution, the likelihood, the formula, the weight column, the
    reporting date, and the size of the panel it saw. Two fits agreeing on all of those
    are the same fit; two differing in any are not, and must not share a file.

    The panel size is in there as a cheap stand-in for the panel's contents. It will
    not catch a change that leaves the row count alone -- a re-aggregation with
    different cut points, say -- which is why the cache is opt-in rather than automatic.
    """
    rendered = "|".join(f"{key}={parts[key]!r}" for key in sorted(parts))
    return hashlib.sha256(rendered.encode()).hexdigest()[:16]


def fit_path(fingerprint: str) -> Path:
    return processed_dir() / FITS_DIRNAME / f"{fingerprint}.pickle"


def save_fit(result: object, fingerprint: str, description: dict[str, object]) -> Path:
    """Persist a fitted model so a later run need not spend the hours again.

    A fit on the whole population is two and a half hours. A run that completes one and
    then fails while writing its reports has thrown away the expensive part and kept
    nothing, which is how this function came to exist.

    The description is written beside the pickle as readable JSON, because a directory
    of hashed filenames is unusable otherwise.
    """
    path = fit_path(fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    path.with_suffix(".json").write_text(json.dumps(description, indent=2, default=str) + "\n")
    return path


def find_fits(**criteria: object) -> list[tuple[str, dict[str, object]]]:
    """Every cached fit whose description matches ``criteria``, most recent first.

    The fingerprint is a hash of everything that determines a fit, **including the row and
    loan-month counts**, so a caller that wants a fit it did not make has to know the counts
    to name it -- which meant expanding the rows to count them, to find the model that would
    have scored them. The descriptions are written beside the pickles for exactly this: the
    directory of hashed filenames is unusable otherwise, and it is also searchable.

    A criterion of ``None`` requires the key to be **absent**, which is how a report's fit is
    told from a selection's: one carries ``purpose`` and the other does not.
    """
    directory = processed_dir() / FITS_DIRNAME
    if not directory.exists():
        return []
    found: list[tuple[float, str, dict[str, object]]] = []
    for path in directory.glob("*.json"):
        try:
            described = cast("dict[str, object]", json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
        if any(
            (key in described) if wanted is None else (described.get(key) != wanted)
            for key, wanted in criteria.items()
        ):
            continue
        found.append((path.stat().st_mtime, path.stem, described))
    return [(fingerprint, described) for _, fingerprint, described in sorted(found, reverse=True)]


def load_fit(fingerprint: str) -> object | None:
    """Read a cached fit, or ``None`` if there is not a usable one.

    An unreadable cache is a miss, not an error. A pickle is tied to the versions of
    lifelines and numpy that wrote it, so an upgrade should cost a refit rather than a
    traceback.
    """
    path = fit_path(fingerprint)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            loaded: object = pickle.load(handle)
        return loaded
    except Exception:
        _LOGGER.warning("Cached fit at %s could not be read; refitting.", path)
        return None
