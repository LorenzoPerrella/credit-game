"""An encoding, written once and read back without being rebuilt.

A reading of the production table is **10.9 minutes** and 1.09 GB of fifteen-byte rows, and
there is nothing in it that depends on a formula: it is parquet, the macro family, and the key
coding. A selection pays three of them -- the training half and step 9's two origination-year
halves -- and the four runs rule 2 needs pay twelve. Written to disk and mapped back, the second
and every later reading is seconds.

**And it survives a restart**, which on a job that takes a day matters more than the minutes. A
run that stops picks up at the fit it was on; it used to pick up at the reading as well.

The format is five arrays, two key frames and a little JSON:

* the rows as five ``.npy`` files, **concatenated across the blocks** with their lengths beside
  them. One file per column rather than one archive, because an archive cannot be mapped; and
  concatenated rather than one file per block, because 443 blocks times five columns is a
  directory nobody wants. Each block is then a *view* of the mapping, and a view of a memory
  map is a memory map -- the gigabyte is never copied into the process;
* the two key frames as parquet, which round-trips the categorical levels the codes mean;
* the counts, which are all the column moments need;
* and the scalars, the column lists and the declared levels as JSON, so a stale encoding can be
  read by a person.

Restoring the factorisation is **replaying the keys, not the rows**: the key frames come back in
the order the codes were handed out in, so re-registering them in that order gives every
combination the index it had. Nothing is recomputed and nothing is renumbered.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from creditsurv.data.artefacts import ENCODINGS, fingerprint
from creditsurv.data.panel import month_ordinal
from creditsurv.models.engine.scan import Encoding
from creditsurv.models.kernel.factorisation import Factorisation, Rows

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

log: Final = logging.getLogger(__name__)

#: The five columns of a row, in the order they are written. Fifteen bytes between them.
_COLUMNS: Final = ("i", "j", "age", "event", "weight")

#: What the row arrays are, so a reader does not have to trust the file's own header.
_DTYPES: Final = {
    "i": np.uint32,
    "j": np.uint32,
    "age": np.uint16,
    "event": np.bool_,
    "weight": np.uint32,
}


def encoding_fingerprint(
    *,
    identity: str,
    loan: Sequence[str],
    calendar: Sequence[str],
    age_column: str,
    cause: str,
    parity: int | None,
    months: tuple[int | None, int | None] | None,
    block_rows: int,
    macro: pd.DataFrame,
    lag_months: int,
) -> str:
    """A name for a reading, from everything that determines which rows it holds and how.

    ``block_rows`` is in there and has to be: the blocks are where the sums are cut, and
    floating-point addition is not associative, so a reading cut differently is a different
    objective in its last digits -- which an optimiser turns into a different search.

    **And the macro panel is in there**, as a digest of its own numbers rather than as a path or
    a date. Every calendar-side covariate is computed from it while the cells are read, so a
    revised series is a different calendar key on the same cell file -- which the cell table's
    identity cannot see. The panel is a few hundred rows by a dozen columns, so hashing the
    values themselves costs nothing and cannot be fooled by a file that was rewritten without
    changing.
    """
    return fingerprint(
        identity=identity,
        loan=tuple(loan),
        calendar=tuple(calendar),
        age_column=age_column,
        cause=cause,
        parity=parity,
        months=months,
        block_rows=block_rows,
        macro=hashlib.sha256(_readable(macro, months, lag_months).to_csv().encode()).hexdigest(),
        lag_months=lag_months,
    )


def _readable(
    macro: pd.DataFrame, months: tuple[int | None, int | None] | None, lag_months: int
) -> pd.DataFrame:
    """The macro rows a reading of this window can actually reach.

    The window cuts on the **observation** month and every series is lagged, so a cell observed
    in month `m` reads the panel at `m - lag` and at its origination month less the lag, both
    earlier still. The last row any reading of a window ending at `cut` can touch is therefore
    **`cut - lag`**, and rule "every macro series is lagged three months" is what makes that
    exact rather than approximate.

    Hashing a row above that would retire a correct 3.6-minute reading every time FRED publishes
    or revises a month the reading could not have read -- which is three readings on a selection
    and six on rule 2's four runs. **It happened**: the clip stopped at the cut rather than at
    the cut less the lag, a revision landed in those three months, and a second 1.02 GB reading
    of the same 72,671,500 cells was taken and written beside the first. The two were compared
    and are identical in every row array and both key frames, which is how the three months were
    found.

    There is no full-sample normalisation anywhere in the macro path -- `fred.py` averages within
    a month and nothing standardises over a series -- so a later month cannot reach back and
    change an earlier value either.
    """
    last = None if months is None else months[1]
    if last is None:
        return macro
    reachable = last - lag_months
    return macro.loc[[month_ordinal(period) <= reachable for period in macro.index]]


def save_encoding(encoding: Encoding, name: str, description: dict[str, object]) -> Path:
    """Write a reading, atomically, without ever holding a second copy of the rows.

    The arrays are created at full size and filled block by block through a memory map, so the
    peak does not double at the moment of writing -- which on a 1.09 GB encoding and a 16 GB
    machine is the difference between this being free and this being the new peak.
    """
    loan_keys, calendar_keys = encoding.factorisation.keys()
    lengths = np.array([block.rows for block in encoding.rows], dtype=np.int64)
    with ENCODINGS.writing(name) as folder:
        folder.mkdir(parents=True, exist_ok=True)
        for column in _COLUMNS:
            mapped = np.lib.format.open_memmap(
                folder / f"{column}.npy",
                mode="w+",
                dtype=_DTYPES[column],
                shape=(int(lengths.sum()),),
            )
            at = 0
            for block in encoding.rows:
                values = getattr(block, column)
                mapped[at : at + len(values)] = values
                at += len(values)
            mapped.flush()
            del mapped
        np.save(folder / "lengths.npy", lengths)
        np.save(folder / "loan_counts.npy", encoding.factorisation.loan_counts)
        np.save(folder / "calendar_counts.npy", encoding.factorisation.calendar_counts)
        loan_keys.to_parquet(folder / "loan_keys.parquet", index=False)
        calendar_keys.to_parquet(folder / "calendar_keys.parquet", index=False)
        encoding.bounds.rename("weight").reset_index().to_parquet(
            folder / "bounds.parquet", index=False
        )
        (folder / "meta.json").write_text(
            json.dumps(
                {
                    "events": encoding.events,
                    "weight": encoding.weight,
                    "names": list(encoding.names),
                    "loan": list(encoding.factorisation.loan),
                    "calendar": list(encoding.factorisation.calendar),
                    "age_column": encoding.factorisation.age_column,
                    "categories": {
                        column: [str(level) for level in levels]
                        for column, levels in encoding.categories.items()
                    },
                },
                indent=2,
            )
            + "\n"
        )
    ENCODINGS.describe(name, description)
    log.info("encoding %s written to %s", name, ENCODINGS.path(name))
    return ENCODINGS.path(name)


def load_encoding(name: str) -> Encoding | None:
    """Map a reading back, or ``None`` if there is not a usable one.

    An unreadable cache is a miss, not an error: a reading costs eleven minutes and a traceback
    costs the run. The rows come back as views of a memory map, so what the process holds is the
    page table rather than the gigabyte.
    """
    folder = ENCODINGS.path(name)
    if not folder.is_dir():
        return None
    try:
        return _mapped(folder)
    except Exception:
        log.warning("Cached encoding at %s could not be read; reading the cells.", folder)
        return None


def _levelled(keys: pd.DataFrame, categories: dict[str, pd.Index]) -> pd.DataFrame:
    """The key frame with its categorical columns back on their **declared** levels.

    Not left to parquet. What a formula's expansion reads off a key frame is its *dtypes*: a
    categorical column whose levels came back in another order, or as plain strings sorted by
    pandas, would hand `C(purpose)` a different reference level -- a different model, fitted
    without a word. The levels are saved beside the rows and re-applied here, by the same cast
    `CellBlocks` makes while reading the cell file, so a restored key frame is built the way a
    fresh one was rather than the way a file format happened to store it.
    """
    restored = keys.copy()
    for column, levels in categories.items():
        if column in restored.columns:
            restored[column] = pd.Categorical(restored[column].astype(str), categories=levels)
    return restored


def _mapped(folder: Path) -> Encoding:
    """The encoding at this path, with its rows mapped rather than read."""
    meta = json.loads((folder / "meta.json").read_text())
    lengths = np.load(folder / "lengths.npy")
    mapped = {column: np.load(folder / f"{column}.npy", mmap_mode="r") for column in _COLUMNS}
    edges = np.concatenate([[0], np.cumsum(lengths)])
    blocks = tuple(
        Rows(**{column: mapped[column][start:stop] for column in _COLUMNS})
        for start, stop in itertools.pairwise(edges)
    )
    bounds = pd.read_parquet(folder / "bounds.parquet").set_index(["lower", "upper", "entry"])[
        "weight"
    ]
    categories = {column: pd.Index(levels) for column, levels in meta["categories"].items()}
    factorisation = Factorisation.restored(
        loan=meta["loan"],
        calendar=meta["calendar"],
        age_column=meta["age_column"],
        loan_keys=_levelled(pd.read_parquet(folder / "loan_keys.parquet"), categories),
        calendar_keys=_levelled(pd.read_parquet(folder / "calendar_keys.parquet"), categories),
        loan_counts=np.load(folder / "loan_counts.npy"),
        calendar_counts=np.load(folder / "calendar_counts.npy"),
    )
    return Encoding(
        factorisation=factorisation,
        rows=blocks,
        bounds=bounds,
        events=float(meta["events"]),
        weight=float(meta["weight"]),
        categories=categories,
        names=(
            meta["names"][0],
            meta["names"][1],
            meta["names"][2],
            meta["names"][3],
            meta["names"][4],
        ),
    )


@dataclass(frozen=True)
class StoredReading:
    """One reading on disk, with the cell table it was taken from and what it occupies."""

    name: str
    identity: str
    current: bool
    bytes_on_disk: int
    cells: int | None
    cause: str | None
    parity: int | None
    as_of: str | None

    def describe(self) -> dict[str, object]:
        """One row of what `creditsurv prune-encodings` shows before it deletes anything.

        **The reporting date is in it, and it is the column that does the work.** A reading is
        stale by construction when its cell table has been rebuilt, and the sweep finds those;
        a reading of the table on disk at a date nobody will ask about again -- a short-window
        rehearsal, say -- is indistinguishable from the one every run wants unless the date is
        shown. That is why `--name` exists beside the sweep.
        """
        return {
            "reading": self.name,
            "table": self.identity.split(":")[0],
            "current": self.current,
            "as_of": self.as_of,
            "cause": self.cause,
            "parity": "whole" if self.parity is None else self.parity,
            "cells": self.cells,
            "gigabytes": round(self.bytes_on_disk / 1024**3, 2),
        }


def _count(value: object) -> int | None:
    """A cell count out of a description, or ``None`` where it is not a number."""
    return int(value) if isinstance(value, int | float) else None


def audit_readings() -> list[StoredReading]:
    """Every reading on disk, and whether its cell table is still the one on disk.

    A reading is named by the table's identity -- its file, size and time of writing -- so a
    rebuild makes every reading of the old table dead weight that nothing will ever hit again.
    At **1.0 GB each** and six per campaign of rule 2 that is worth being able to find, and
    worth never deleting by accident: a current one costs 11.6 minutes to take again.

    The identity is read from the description beside the arrays, so a reading whose description
    is unreadable is reported as **not** current rather than guessed at -- the conservative
    direction, since it cannot then be deleted by the default sweep.
    """
    from creditsurv.data.book import MoratoriumPolicy
    from creditsurv.data.store import cells_identity, cells_path

    live = {
        cells_identity(policy.value)
        for policy in MoratoriumPolicy
        if cells_path(policy.value).exists()
    }
    readings = []
    for name, described in ENCODINGS.find():
        folder = ENCODINGS.path(name)
        identity = str(described.get("identity", ""))
        readings.append(
            StoredReading(
                name=name,
                identity=identity,
                current=identity in live,
                bytes_on_disk=sum(
                    child.stat().st_size for child in folder.rglob("*") if child.is_file()
                )
                if folder.is_dir()
                else 0,
                cells=_count(described.get("cells")),
                cause=cause if isinstance(cause := described.get("cause"), str) else None,
                parity=parity if isinstance(parity := described.get("parity"), int) else None,
                as_of=when if isinstance(when := described.get("as_of"), str) else None,
            )
        )
    return readings


def remove_reading(name: str) -> None:
    """Delete one reading and the description beside it. Irreversible, and 11.6 minutes."""
    ENCODINGS.remove(name)
