"""The training rows, held compactly and expanded only while they are evaluated.

A stock lifelines fit keeps about 680 bytes a row of autograd tape and design copies, which on
this book would be 45-50 GB. What is stored here is 30 bytes a row: each design column as the
narrowest thing that reproduces it exactly -- single-byte codes into its own distinct values
where there are few of them, ``float32`` where the round trip is exact, ``float64`` otherwise --
and the bounds, the entry and the weight the same way.

:class:`_Slicer` is lifelines' own ``DataframeSlicer`` over numpy, and it is here because the
likelihood filters the design several times per call over masks that are properties of the data.
Two of those filters select every row or none, because no observation on this panel is exact,
and answering them without copying took 76% off a value-and-gradient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from lifelines import utils

if TYPE_CHECKING:
    #: What the parent asks a worker for, and what a worker sends back.
    Command = tuple[str, np.ndarray | None]
    Prepared = tuple[pd.MultiIndex, np.ndarray, float, dict[str, np.ndarray]]
    Answer = dict[str, Any] | tuple[float, np.ndarray] | np.ndarray
    #: A worker's answer with the part it read, so the parent can add them in one order.
    Tagged = tuple[int, Answer]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredColumn:
    """One column of one block, in the smallest representation that is still exact.

    Most columns of this design take few distinct values -- indicators, binned scores,
    loan ages, the bounds -- and are kept as codes into those values. The rest are the
    macro covariates, built as ``float32`` upstream, which survive a round trip through
    ``float32`` exactly. Only a column that does neither stays ``float64``. Expanding
    therefore returns the ``float64`` column lifelines would have used, bit for bit.
    """

    codes: np.ndarray | None = None
    levels: np.ndarray | None = None
    values: np.ndarray | None = None

    @classmethod
    def of(cls, column: np.ndarray) -> StoredColumn:
        exact = np.asarray(column, dtype=np.float64)
        levels, codes = np.unique(exact, return_inverse=True)
        if len(levels) <= 1 << 8:
            return cls(codes=codes.astype(np.uint8), levels=levels)
        if len(levels) <= 1 << 16:
            return cls(codes=codes.astype(np.uint16), levels=levels)
        narrow = exact.astype(np.float32)
        if np.array_equal(narrow, exact):
            return cls(values=narrow)
        return cls(values=np.ascontiguousarray(exact))

    def __len__(self) -> int:
        stored = self.codes if self.codes is not None else self.values
        return 0 if stored is None else len(stored)

    def expand_into(self, out: np.ndarray) -> None:
        if self.codes is not None and self.levels is not None:
            np.take(self.levels, self.codes, out=out, mode="clip")
        elif self.values is not None:
            out[...] = self.values

    def expand(self) -> np.ndarray:
        out = np.empty(len(self), dtype=np.float64)
        self.expand_into(out)
        return out

    @property
    def nbytes(self) -> int:
        parts = (self.codes, self.levels, self.values)
        return sum(part.nbytes for part in parts if part is not None)


@dataclass(frozen=True)
class _Block:
    """One block of training rows, stored compactly, expanded only while evaluated."""

    design: tuple[StoredColumn, ...]
    lower: StoredColumn
    upper: StoredColumn
    entry: StoredColumn
    weight: StoredColumn
    exact: np.ndarray
    weight_sum: float

    @property
    def rows(self) -> int:
        return len(self.exact)

    @property
    def nbytes(self) -> int:
        stored = (*self.design, self.lower, self.upper, self.entry, self.weight)
        return sum(column.nbytes for column in stored) + self.exact.nbytes

    def arguments(self, columns: pd.MultiIndex, scale: np.ndarray) -> tuple[Any, ...]:
        """The block as lifelines' likelihood takes it: ``(Ts, E, W, entries, Xs)``."""
        # Column-major, so each column is contiguous to write into and the slicer below
        # can take its columns without copying them.
        design = np.empty((self.rows, len(self.design)), dtype=np.float64, order="F")
        for position, column in enumerate(self.design):
            column.expand_into(design[:, position])
        design /= scale
        bounds = (self.lower.expand(), self.upper.expand())
        return (
            bounds,
            self.exact,
            self.weight.expand(),
            self.entry.expand(),
            _Slicer(design, columns),
        )


def _column_slices(columns: pd.MultiIndex) -> dict[str, slice | np.ndarray]:
    """Each parameter's columns as a slice where they are adjacent, positions where not.

    A slice of an F-ordered array is a view of contiguous columns; a list of positions is a
    copy. The design is built in the MultiIndex's own order, so the slice is the usual case and
    the fallback is there so an unusual order is slow rather than wrong.
    """
    outer = columns.get_level_values(0)
    found: dict[str, slice | np.ndarray] = {}
    for name in outer.unique():
        positions = np.flatnonzero(outer == name)
        adjacent = positions[-1] - positions[0] + 1 == len(positions)
        found[str(name)] = (
            slice(int(positions[0]), int(positions[-1]) + 1) if adjacent else positions
        )
    return found


class _Slicer:
    """lifelines' ``DataframeSlicer``, over numpy and remembering what it has been asked.

    The likelihood filters the design four times per call -- the exact observations, the
    censored rows twice, the delayed entries -- and lifelines' slicer answers each with a
    pandas take, which copies the whole block. Those masks are properties of the *data*: they
    are the same on every evaluation of every iteration, and so is the design.

    Measured on a 250,000-row block of the production table, a value-and-gradient spent **32%
    of its time in ``pandas.take`` and another 20% copying**, against 42% in autograd's tape
    and a minority in the arithmetic. Two things remove most of that, and neither touches the
    likelihood -- which is what keeps `tests/test_blocks.py` able to hold this engine to
    lifelines' own answer:

    * a parameter's columns are **adjacent** in the design, because the design is built in the
      order of the MultiIndex, so asking for them is a slice and a slice is a view. pandas
      copied them on every call;
    * the masks are properties of the data, so a repeated filter is answered from a cache.

    The filtered copy is kept in Fortran order like the design it came from. Row-indexing an
    F-ordered array returns a C-ordered copy, and the likelihood then works on non-contiguous
    columns: the first version of this class did that and was **60% slower** than the pandas it
    replaced, which is the sort of thing only a measurement finds.
    """

    __slots__ = ("_cache", "_columns", "_design", "_positions")

    def __init__(self, design: np.ndarray, columns: pd.MultiIndex) -> None:
        self._design = design
        self._columns = columns
        self._positions = _column_slices(columns)
        self._cache: dict[bytes, _Slicer] = {}

    def __getitem__(self, key: str) -> np.ndarray:
        taken: np.ndarray = self._design[:, self._positions[str(key)]]
        return taken

    def filter(self, mask: np.ndarray) -> _Slicer:
        """The rows the mask selects. The same mask twice costs nothing the second time.

        **And a mask that selects every row costs nothing at all, which is the common case
        here.** lifelines filters by the event flag and its complement, and on this panel the
        event flag is `exact_observation`, which `panel.py` sets `False` with no condition:
        the reporting interval tells us the month, never the day. So `Xs.filter(E)` selects no
        rows and `Xs.filter(~E)` selects all of them -- and the second was answered by boolean
        fancy-indexing the whole design into a new Fortran-ordered array, 50 MB on a
        242,000-row block, at every evaluation. The filtered design with every row *is* this
        design, so it is this slicer.

        Measured on one block of the production table at 26 parameters, a value-and-gradient
        is 232.9 ms, of which 38.7 is expanding the design and only 44.3 is autograd: the
        other 177.7 ms is copies of data that does not change between evaluations.
        """
        flat = np.asarray(mask)
        if flat.dtype != bool:
            return _Slicer(np.asfortranarray(self._design[flat]), self._columns)
        kept = int(np.count_nonzero(flat))
        if kept == len(flat):
            return self
        if kept == 0:
            # An empty slice of an F-ordered array, rather than a boolean take of nothing.
            return _Slicer(self._design[:0], self._columns)
        key = flat.tobytes()
        cached = self._cache.get(key)
        if cached is None:
            cached = _Slicer(np.asfortranarray(self._design[flat]), self._columns)
            self._cache[key] = cached
        return cached

    def groupby(self, *args: object, **kwargs: object) -> object:
        """Only the fitter's reporting path asks for this, never the likelihood."""
        frame = pd.DataFrame(self._design, columns=self._columns, copy=False)
        return utils.DataframeSlicer(frame).groupby(*args, **kwargs)

    @property
    def size(self) -> int:
        return int(self._design.shape[0])


#: Rows evaluated at once. One block's autograd tape costs about 700 bytes a row, so a
#: million rows is 0.7 GB above the stored data: small against the machine, and large
#: enough that the Python overhead per block is noise against the arithmetic.
DEFAULT_BLOCK_ROWS: Final = 1_000_000

#: What lifelines substitutes for an infinite upper bound before evaluating anything.
INFINITY_STAND_IN: Final = 1e25

#: Below this a column's standard deviation counts as zero, as it does in lifelines.
_CONSTANT: Final = 1e-8
