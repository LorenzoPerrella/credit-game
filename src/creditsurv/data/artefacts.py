"""A directory of artefacts, each named by everything that determines it.

Two kinds of thing in this project are expensive enough that losing one costs hours, and they
want the same treatment: a **fit**, which is minutes to an hour on the production table, and an
**encoding** of the rows, which is eleven minutes and which every fit of a selection then reads.
What they share is not the format -- one is a pickle, the other a directory of arrays and two
parquet files -- but the handling:

* a **fingerprint** of everything that determines the artefact, so two that agree on all of it
  are the same artefact and two differing in any cannot share a name;
* a **readable description** written beside the bytes, because a directory of hashed filenames
  is unusable otherwise -- and because it makes the directory *searchable*, which is how a
  caller finds an artefact it did not make without knowing the counts that named it;
* an **atomic** write, for the reason the ingest learned by losing a quarter to a reboot: a
  crash while writing should leave the previous artefact intact rather than half of a new one;
* and a **miss rather than an error** on anything unreadable. A pickle is tied to the versions
  that wrote it, so an upgrade should cost a recomputation and not a traceback.

The format stays with whatever knows it. This module knows where things go and how to find them
again, and it is in `data/` because that is where the files are; what an `Encoding` or a
`FitResult` *is* belongs to `models/`, which is above it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from creditsurv.config import processed_dir

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

log: Final = logging.getLogger(__name__)

#: How many hex digits of the hash name an artefact. Sixteen is 64 bits: on a directory of a
#: few hundred, a collision is not a thing that happens, and a name a person can read back to
#: a log line is worth more than the rest of the digest.
_DIGITS: Final = 16


def fingerprint(**parts: object) -> str:
    """A stable name from everything that determines an artefact.

    Rendered rather than hashed field by field, so the name changes if a field is *added*: an
    artefact determined by one more thing than before is a different artefact, and silently
    reusing the old one under the old name is the failure this guards against.
    """
    rendered = "|".join(f"{key}={parts[key]!r}" for key in sorted(parts))
    return hashlib.sha256(rendered.encode()).hexdigest()[:_DIGITS]


@dataclass(frozen=True)
class Store:
    """One kind of artefact, under its own directory of the processed data.

    ``suffix`` is what the payload is called: ``.pickle`` for a fit, nothing for an encoding,
    which is a directory. The description is always ``<fingerprint>.json`` beside it, which is
    what makes a store searchable whatever its payload is.
    """

    kind: str
    suffix: str = ""

    @property
    def directory(self) -> Path:
        return processed_dir() / self.kind

    def path(self, name: str) -> Path:
        """Where the payload of this artefact goes."""
        return self.directory / f"{name}{self.suffix}"

    def described(self, name: str) -> Path:
        """Where its description goes: readable JSON, beside the bytes."""
        return self.directory / f"{name}.json"

    def describe(self, name: str, description: dict[str, object]) -> None:
        """Write the description, which is also what makes the store searchable."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self.described(name).write_text(json.dumps(description, indent=2, default=str) + "\n")

    @contextmanager
    def writing(self, name: str) -> Iterator[Path]:
        """A path to write the payload to, moved into place only once writing has finished.

        The caller is handed a sibling path and the move happens on a clean exit, so a crash
        half way leaves whatever was there before. A directory payload is replaced wholesale:
        the previous one is removed only after the new one is complete.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        final = self.path(name)
        partial = final.with_name(final.name + ".partial")
        _remove(partial)
        try:
            yield partial
            _remove(final)
            partial.replace(final)
        finally:
            _remove(partial)

    def find(self, **criteria: object) -> list[tuple[str, dict[str, object]]]:
        """Every artefact whose description matches, most recent first.

        A criterion of ``None`` requires the key to be **absent**, which is how a report's fit
        is told from a selection's: one carries ``purpose`` and the other does not.
        """
        if not self.directory.exists():
            return []
        found: list[tuple[float, str, dict[str, object]]] = []
        for path in self.directory.glob("*.json"):
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
        return [(name, described) for _, name, described in sorted(found, reverse=True)]

    def remove(self, name: str) -> None:
        """Delete one artefact and the description beside it.

        Irreversible, and both of the things this store holds are expensive to make again --
        minutes to an hour for a fit, 11.6 for a reading of the production table. So it is
        here for a command that asks first, and there is nothing in this module that calls it.
        """
        _remove(self.path(name))
        _remove(self.described(name))


def _remove(path: Path) -> None:
    """Whatever is at this path, file or directory, gone."""
    if path.is_dir() and not path.is_symlink():
        for child in sorted(path.rglob("*"), reverse=True):
            child.rmdir() if child.is_dir() else child.unlink()
        path.rmdir()
    elif path.exists() or path.is_symlink():
        path.unlink()


#: The fits a run must not lose. A fit on the whole population is minutes to hours, and one run
#: completed a 154-minute fit and was then killed writing its reports, keeping nothing.
FITS: Final = Store("fits", suffix=".pickle")

#: The readings of the cell file. Eleven minutes each on the production table, 1.09 GB of
#: fifteen-byte rows, and every fit of a selection reads one.
ENCODINGS: Final = Store("encodings")
