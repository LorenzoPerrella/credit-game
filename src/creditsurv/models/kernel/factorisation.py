"""The design as two small tables, and the rows as two indices into them.

Measured on the training window of 72,671,500 cells: every column of the design is a function
of the **loan** combination (3,001 of them) or of the **calendar** key (152,565), and never of
both. So the scale's linear predictor is ``eta = A[i] + B[j]`` -- a lookup plus a lookup -- and
the 26-column design that costs 38.7 ms a block to expand and another 124 to filter need never
exist. A row is fifteen bytes: two indices, an age, an exit and a weight.

The partition is **declared** by the caller -- which covariates the cell key carries and which
are functions of the calendar -- and **verified** here, because a declaration that is not
checked is a comment. A column that is a function of neither side is a term coupling the two,
which a sum of two tables cannot represent, and it is refused by name.

And because the key values themselves are kept, a *different* model's tables are built by
putting its formula through 3,001 and 152,565 rows instead of reading the 72 million again.
That is what lets one reading serve a whole selection, and it is why the rows are encoded at
all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from creditsurv.models.kernel.likelihood import (
    _MAX_AGE,
)

#: A row's five columns, in the order they are written, read and handed over.
_ROW_COLUMNS: Final = ("i", "j", "age", "event", "weight")

#: And what each one is. Fifteen bytes between them, and a compiled backend reads the bytes:
#: a column that arrived as `int64` would be refused at the boundary rather than misread, but
#: it is refused *here*, by name, before a fit has spent an hour getting there.
_ROW_DTYPES: Final = {
    "i": np.uint32,
    "j": np.uint32,
    "age": np.uint16,
    "event": np.bool_,
    "weight": np.uint32,
}


@dataclass(frozen=True)
class Rows:
    """A block of rows as the kernel reads them: four narrow columns and two indices.

    ``i`` says which **loan** combination a row has and ``j`` which **calendar** key, so the
    scale's linear predictor is a lookup plus a lookup. On the production table that is
    3,001 and 153,309 table entries against 53 million rows, and the row itself is ten bytes
    where the expanded design is 208.
    """

    i: np.ndarray
    j: np.ndarray
    age: np.ndarray
    event: np.ndarray
    weight: np.ndarray

    @property
    def rows(self) -> int:
        return len(self.i)

    @property
    def nbytes(self) -> int:
        """What the rows cost: fifteen bytes each, and the two tables counted once elsewhere.

        Four for each index, two for the age, one for the exit and four for the weight -- which
        is a count of loan-months and an integer, never an amount. The expanded design the
        present engine hands the likelihood is 208 bytes a row at 26 columns, and the whole
        training half fits here in 0.8 GB.
        """
        return int(
            self.i.nbytes + self.j.nbytes + self.age.nbytes + self.event.nbytes + self.weight.nbytes
        )


def _stable_codes(frame: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    """The named columns as a float matrix whose numbers mean the same thing in every block.

    A categorical contributes its **declared** level codes, never codes discovered in this
    block: the tables are global, so a level absent from one batch would otherwise shift every
    code after it and two different combinations would be handed the same index.
    ``blocks._check_categories`` is what makes the levels the same everywhere.
    """
    columns: list[np.ndarray] = []
    for name in names:
        values = frame[name]
        if isinstance(values.dtype, pd.CategoricalDtype):
            columns.append(values.cat.codes.to_numpy(dtype=float))
            continue
        if values.dtype == object:
            message = (
                f"Column {name!r} is plain text, so its codes would be this block's own. "
                "Restore the declared categorical levels before encoding."
            )
            raise ValueError(message)
        columns.append(values.to_numpy(dtype=float))
    return np.column_stack(columns) if columns else np.zeros((len(frame), 1))


def _distinct(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``np.unique(values, axis=0, return_inverse=True)``, found by hashing instead of sorting.

    The same two arrays, to the element: the distinct rows in lexicographic order and each row's
    index into them. What changes is how they are found. `np.unique(axis=0)` views each row as a
    composite scalar and **lexicographically sorts all of them**, which on the production table
    was 9.5 of the encoding's 12.1 seconds and two thirds of a whole reading -- and a block
    carries about 230,000 rows and at most 24,000 distinct combinations, so that sort does ten
    to fifty times the work it needs to.

    Hashed instead: each column is factorised on its own (a hash, not a sort), the codes are
    folded into one integer, and the fold is re-factorised whenever the radix would overflow.
    Only the **distinct** rows are then sorted, which is the k the caller actually needs
    ordered. The order has to be preserved exactly, because it is the order new combinations are
    registered in and therefore which index each one gets.
    """
    if len(values) == 0:
        return values[:0], np.zeros(0, dtype=np.intp)
    folded = np.zeros(len(values), dtype=np.int64)
    radix = 1
    for column in range(values.shape[1]):
        codes, _ = pd.factorize(values[:, column], use_na_sentinel=False)
        width = int(codes.max()) + 1 if len(codes) else 1
        if radix * width > 2**62:
            folded, _ = pd.factorize(folded, use_na_sentinel=False)
            folded = folded.astype(np.int64)
            radix = int(folded.max()) + 1 if len(folded) else 1
        folded = folded * width + codes.astype(np.int64)
        radix *= width
    # One pass over the folded key: the groups, a representative row of each, and which group
    # every row belongs to. All three come off a single 1-D unique.
    _, first, by_group = np.unique(folded, return_index=True, return_inverse=True)
    distinct = values[first]
    order = np.lexsort(distinct.T[::-1])
    rank = np.empty(len(order), dtype=np.intp)
    rank[order] = np.arange(len(order))
    return distinct[order], rank[np.asarray(by_group).ravel()]


def _counts(weight: np.ndarray) -> np.ndarray:
    """The weight as the count of loan-months it is, in four bytes rather than eight.

    `aft._check_weights` already refuses a weight that is not a positive whole number, because
    lifelines' variance estimates treat it as a replication count and Basel defines the
    probability per obligor rather than per dollar. So it is an integer, and storing it as one
    takes four bytes off every row of the training half -- 212 MB.
    """
    rounded = np.rint(weight)
    if (
        not np.array_equal(rounded, weight)
        or rounded.min() < 1
        or rounded.max() > np.iinfo(np.uint32).max
    ):
        message = (
            "The weight is not a count of loan-months that fits in four bytes. Grouped "
            "estimation expects frequency weights: whole numbers, at least one, and the "
            "largest cell of the production table holds far fewer than four billion."
        )
        raise ValueError(message)
    return np.asarray(rounded.astype(np.uint32))


def _first_seen(codes: np.ndarray, held: int) -> dict[int, int]:
    """Where each combination new to this block first appears in it.

    Codes are handed out in order of first appearance, so the new ones are exactly the
    indices from ``held`` up. Finding them by walking the block in Python cost **minutes**:
    two passes over 250,000 rows for each of 443 blocks is 221 million interpreted
    iterations, and it made an encoding of the production table slower than the design-building
    scan it replaces. `np.unique` sorts instead, and returns the first position of every
    distinct code in one call.
    """
    if not len(codes):
        return {}
    distinct, first = np.unique(codes, return_index=True)
    fresh = distinct >= held
    return dict(zip(distinct[fresh].tolist(), first[fresh].tolist(), strict=True))


def _tallied(counts: np.ndarray, codes: np.ndarray, size: int) -> np.ndarray:
    """The running count of rows per combination, grown to hold the newest ones."""
    if size > len(counts):
        counts = np.concatenate([counts, np.zeros(size - len(counts), dtype=np.int64)])
    counts += np.bincount(codes, minlength=size).astype(np.int64)
    return counts


def _a_function_of(design: np.ndarray, codes: np.ndarray, size: int) -> np.ndarray:
    """Which of the design's columns are functions of ``codes``, one column at a time.

    Written into a table indexed by the codes and read back: a column that is a function of
    them comes back unchanged, and one that is not comes back with whichever row was written
    last. Exact equality, not a tolerance -- the same combination carries literally the same
    float -- and one pass over the block.
    """
    table = np.zeros((size, design.shape[1]))
    table[codes] = design
    return np.asarray(np.all(table[codes] == design, axis=0))


class Factorisation:
    """The two design tables, grown block by block, and the rows that index into them.

    The partition is **declared** by the caller -- which covariates the cell key carries and
    which are functions of the calendar -- and **verified** here, because a declaration that
    is not checked is a comment. Every column of the design has to be a function of the loan
    combination or of the calendar key; one that is a function of neither is a term coupling
    the two sides, which this factorisation cannot represent, and it is refused by name rather
    than quietly fitted as something else.

    The tables are kept **unscaled**. lifelines optimises each coefficient multiplied by its
    column's standard deviation, and dividing two tables of a few thousand rows once is
    nothing next to dividing a design of 53 million.
    """

    def __init__(
        self,
        *,
        loan: Sequence[str],
        calendar: Sequence[str],
        age_column: str,
        columns: Sequence[str] = (),
    ) -> None:
        self._loan = tuple(loan)
        self._declared_calendar = tuple(calendar)
        self._calendar = (*calendar, age_column)
        self._age_column = age_column
        self._columns = tuple(columns)
        self._loan_codes = _Growing()
        self._calendar_codes = _Growing()
        self._loan_positions: np.ndarray | None = None
        self._calendar_positions: np.ndarray | None = None
        self._loan_rows: list[np.ndarray] = []
        self._calendar_rows: list[np.ndarray] = []
        # The key values themselves, one row per combination, kept so that a **different**
        # model's two tables can be built by putting its formula through 3,001 and 153,309
        # rows instead of reading the 72 million again. This is what makes one scan serve a
        # whole selection, and it is the reason the rows are encoded at all.
        self._loan_keys: list[pd.DataFrame] = []
        self._calendar_keys: list[pd.DataFrame] = []
        # And how many rows each combination carries, which is all the column moments need:
        # the sum of a loan-side column over every row is the sum over combinations of its
        # value times its count.
        self._loan_counts = np.zeros(0, dtype=np.int64)
        self._calendar_counts = np.zeros(0, dtype=np.int64)

    def add(
        self,
        frame: pd.DataFrame,
        design: np.ndarray | None = None,
        *,
        event: np.ndarray,
        weight: np.ndarray,
    ) -> Rows:
        """Encode one block, growing the tables with whatever combinations are new to it.

        With a ``design`` the partition is also decided from it and **verified against these
        rows**, which is what makes a declared split a fact. Without one, nothing but the keys
        and the counts is kept, and :meth:`expand` decides the partition from the key frames
        when a model asks for its tables -- which is the point of encoding at all: the rows are
        read once and no formula is involved, so there is no design to expand, no moments to
        accumulate over 26 columns and no 250,000-row block to put through formulaic.
        """
        i = self._loan_codes.of(_stable_codes(frame, self._loan))
        j = self._calendar_codes.of(_stable_codes(frame, self._calendar))
        if design is not None and self._loan_positions is None:
            self._decide(design, i, j)
        if design is not None:
            assert self._loan_positions is not None
            assert self._calendar_positions is not None
            fresh = self._grow(self._loan_rows, design[:, self._loan_positions], i)
            self._keys(self._loan_keys, frame, self._loan, fresh)
            fresh = self._grow(self._calendar_rows, design[:, self._calendar_positions], j)
            self._keys(self._calendar_keys, frame, self._calendar, fresh)
        else:
            self._keys(self._loan_keys, frame, self._loan, self._unseen(self._loan_keys, i))
            self._keys(
                self._calendar_keys, frame, self._calendar, self._unseen(self._calendar_keys, j)
            )
        self._loan_counts = _tallied(self._loan_counts, i, len(self._loan_codes))
        self._calendar_counts = _tallied(self._calendar_counts, j, len(self._calendar_codes))
        if design is not None:
            self._verify(design, i, j)
        age = frame[self._age_column].to_numpy()
        if age.min() < 0 or age.max() > _MAX_AGE:
            message = (
                f"The loan ages run {age.min()} to {age.max()}, outside the 0 to {_MAX_AGE} the "
                "log-time tables cover. `panel.MAX_AGE_MONTHS` is 360 and the production table "
                "reaches 326, so this is either a different time scale or a broken age."
            )
            raise ValueError(message)
        if not np.array_equal(age, np.rint(age)):
            message = "The loan ages are not whole months, so they cannot index a table."
            raise ValueError(message)
        return Rows(
            i=i.astype(np.uint32),
            j=j.astype(np.uint32),
            age=age.astype(np.uint16),
            event=np.asarray(event, dtype=bool),
            weight=_counts(np.asarray(weight)),
        )

    @property
    def loan(self) -> tuple[str, ...]:
        """The covariates the cell key carries, which the loan combination is of."""
        return self._loan

    @property
    def calendar(self) -> tuple[str, ...]:
        """The covariates that are functions of the calendar, without the age beside them.

        The key itself carries the age -- the interval bounds are a function of it -- but the
        *declaration* is the covariate list the caller gave, and that is what a reader wants
        back and what a restored factorisation is given.
        """
        return self._declared_calendar

    @property
    def age_column(self) -> str:
        return self._age_column

    @property
    def loan_counts(self) -> np.ndarray:
        """How many rows each loan combination carries. The column moments need nothing else."""
        return self._loan_counts

    @property
    def calendar_counts(self) -> np.ndarray:
        return self._calendar_counts

    @classmethod
    def restored(
        cls,
        *,
        loan: Sequence[str],
        calendar: Sequence[str],
        age_column: str,
        loan_keys: pd.DataFrame,
        calendar_keys: pd.DataFrame,
        loan_counts: np.ndarray,
        calendar_counts: np.ndarray,
    ) -> Factorisation:
        """A factorisation rebuilt from its keys rather than from the rows.

        **Replaying the keys, not the rows.** The key frames come back in the order the codes
        were handed out in, so registering them in that order gives every combination the index
        it had -- which is what makes a saved reading usable at all, since the indices are in
        the rows. Nothing is recomputed and nothing is renumbered, and a restored factorisation
        can be added to like any other.
        """
        restored = cls(loan=loan, calendar=calendar, age_column=age_column)
        restored._loan_codes.restore(_stable_codes(loan_keys, restored._loan))
        restored._calendar_codes.restore(_stable_codes(calendar_keys, restored._calendar))
        restored._loan_keys.append(loan_keys.reset_index(drop=True))
        restored._calendar_keys.append(calendar_keys.reset_index(drop=True))
        restored._loan_counts = np.asarray(loan_counts, dtype=np.int64)
        restored._calendar_counts = np.asarray(calendar_counts, dtype=np.int64)
        return restored

    def keys(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The two key frames, one row per combination, in their codes' own order."""
        if not self._loan_keys or not self._calendar_keys:
            message = "Nothing has been encoded, so there are no keys."
            raise ValueError(message)
        return (
            pd.concat(self._loan_keys, ignore_index=True),
            pd.concat(self._calendar_keys, ignore_index=True),
        )

    def probe(self) -> pd.DataFrame:
        """One frame carrying every covariate column, for a formula to be built against.

        The loan combinations with the calendar held at one key. What it is for is lifelines'
        ``CovariateParameterMappings``, which reads the dtypes and the categorical levels to
        decide the design's columns, and must see the levels the rows were encoded against.
        """
        loan_keys, calendar_keys = self.keys()
        return _completed(loan_keys, calendar_keys)

    def expand(
        self, transform: Callable[[pd.DataFrame], pd.DataFrame], *, primary: str
    ) -> Expanded:
        """A model's two design tables and column moments, from the key frames alone.

        ``transform`` is the model's own expansion -- lifelines'
        ``CovariateParameterMappings.transform_df`` -- applied to each key frame with the other
        side held at one row. A column that does not move when only the calendar moves is a
        function of the loan combination; one that does not move when only the loan moves is a
        function of the calendar key; the intercept is both and goes to the loan side. A column
        that moves on both couples the two sides, which a sum of two tables cannot represent,
        and it is refused by name.

        ``primary`` names the **scale's** parameter block, and the tables hold only its
        columns. lifelines' design carries the shape's column too -- a constant, and therefore
        a function of both sides -- so without this the shape's coefficient would be read once
        as the shape and again as a loan-side column, and the fit would stop 2.1 standard
        errors out while reporting 2.5e-4. The moments still cover every column, because that
        is what gives each coefficient its scale.
        """
        loan_keys, calendar_keys = self.keys()
        on_loan = transform(_completed(loan_keys, calendar_keys))
        on_calendar = transform(_completed(calendar_keys, loan_keys))
        if not on_loan.columns.equals(on_calendar.columns):
            message = "The formula expanded to different columns on the two key frames."
            raise ValueError(message)
        columns = on_loan.columns
        by_loan = on_loan.to_numpy(dtype=np.float64)
        by_calendar = on_calendar.to_numpy(dtype=np.float64)
        held = _unchanging(by_calendar)
        moving = _unchanging(by_loan)
        neither = ~held & ~moving
        if neither.any():
            named = ", ".join(str(columns[int(position)]) for position in np.flatnonzero(neither))
            message = (
                f"The design column(s) {named} move with the loan *and* with the calendar, so "
                "the linear predictor is not a sum of two tables. A term coupling a loan "
                "characteristic to the calendar belongs in the calendar key, or the fit "
                "belongs on the autograd evaluator."
            )
            raise ValueError(message)
        everywhere_loan = np.flatnonzero(held)
        everywhere_calendar = np.flatnonzero(~held & moving)
        width = len(columns)
        first, second = np.zeros(width), np.zeros(width)
        low, high = np.zeros(width), np.zeros(width)
        for positions, table, counts in (
            (everywhere_loan, by_loan[:, everywhere_loan], self._loan_counts),
            (everywhere_calendar, by_calendar[:, everywhere_calendar], self._calendar_counts),
        ):
            if not positions.size:
                continue
            seen = counts > 0
            first[positions] = counts @ table
            second[positions] = counts @ (table * table)
            low[positions] = table[seen].min(axis=0)
            high[positions] = table[seen].max(axis=0)
        scale = _only(cast("pd.MultiIndex", columns), primary)
        loan_positions = _within(everywhere_loan, scale)
        calendar_positions = _within(everywhere_calendar, scale)
        loan = by_loan[:, scale.start + loan_positions]
        calendar = by_calendar[:, scale.start + calendar_positions]
        return Expanded(
            columns=cast("pd.MultiIndex", columns),
            loan=loan,
            calendar=calendar,
            loan_positions=loan_positions,
            calendar_positions=calendar_positions,
            first=first,
            second=second,
            low=low,
            high=high,
        )

    @property
    def rows(self) -> int:
        """How many rows were encoded, from the counts the two sides agree on."""
        return int(self._loan_counts.sum())

    def tables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The loan table, the calendar table, and which design columns each one holds."""
        if self._loan_positions is None or self._calendar_positions is None:
            message = "Nothing has been encoded, so there are no tables."
            raise ValueError(message)
        return (
            np.asarray(self._loan_rows, dtype=np.float64),
            np.asarray(self._calendar_rows, dtype=np.float64),
            self._loan_positions,
            self._calendar_positions,
        )

    def _decide(self, design: np.ndarray, i: np.ndarray, j: np.ndarray) -> None:
        loan = _a_function_of(design, i, len(self._loan_codes))
        calendar = _a_function_of(design, j, len(self._calendar_codes))
        neither = ~loan & ~calendar
        if neither.any():
            named = ", ".join(self._columns[position] for position in np.flatnonzero(neither))
            message = (
                f"The design column(s) {named} are a function of neither the loan combination "
                "nor the calendar key, so the linear predictor is not a sum of the two and "
                "this kernel cannot fit it. A term coupling a loan characteristic to the "
                "calendar belongs in the calendar key, or the fit belongs on the autograd "
                "evaluator."
            )
            raise ValueError(message)
        # A constant column -- the intercept -- is a function of both, and has to go to
        # exactly one side. The loan side, which is also where lifelines puts it first.
        self._loan_positions = np.flatnonzero(loan)
        self._calendar_positions = np.flatnonzero(~loan & calendar)

    @staticmethod
    def _grow(rows: list[np.ndarray], design: np.ndarray, codes: np.ndarray) -> dict[int, int]:
        """Store a design row for each combination this block is the first to carry.

        Codes are handed out in order of first appearance, so a block's new ones are exactly
        the indices past the end of the table, and any row carrying one of them will do: the
        invariant `_verify` checks is that they all carry the same values. What comes back is
        which row of this block was taken for each new combination, so the key values can be
        taken from the same rows.
        """
        taken = _first_seen(codes, len(rows))
        if not taken:
            return taken
        rows.extend(np.zeros(design.shape[1]) for _ in taken)
        for code, position in taken.items():
            rows[code] = design[position].copy()
        return taken

    @staticmethod
    def _unseen(pieces: list[pd.DataFrame], codes: np.ndarray) -> dict[int, int]:
        """Which combinations this block is the first to carry, and where they are in it.

        The same accounting `_grow` does while storing a design row, for the path that stores
        no design.
        """
        return _first_seen(codes, sum(len(piece) for piece in pieces))

    @staticmethod
    def _keys(
        pieces: list[pd.DataFrame],
        frame: pd.DataFrame,
        columns: Sequence[str],
        taken: dict[int, int],
    ) -> None:
        """The key values of the combinations new to this block, in their code's order."""
        if not taken:
            return
        wanted = [taken[code] for code in sorted(taken)]
        pieces.append(frame.iloc[wanted][list(columns)].reset_index(drop=True))

    def _verify(self, design: np.ndarray, i: np.ndarray, j: np.ndarray) -> None:
        """Every row's design is the two tables read at its two indices. Exactly.

        This is the check that makes the declared partition a fact rather than a claim, and it
        runs on every block rather than the first: a covariate that is a function of the key on
        one quarter of the book and not on another would pass a test of the first block alone.
        """
        assert self._loan_positions is not None
        assert self._calendar_positions is not None
        loan = np.asarray(self._loan_rows, dtype=np.float64)
        calendar = np.asarray(self._calendar_rows, dtype=np.float64)
        for side, table, codes, positions in (
            ("loan", loan, i, self._loan_positions),
            ("calendar", calendar, j, self._calendar_positions),
        ):
            if positions.size and not np.array_equal(table[codes], design[:, positions]):
                wrong = np.flatnonzero(~np.all(table[codes] == design[:, positions], axis=0))
                named = ", ".join(self._columns[positions[position]] for position in wrong)
                message = (
                    f"The design column(s) {named} are not a function of the {side} key after "
                    "all: the same combination carries different values in different blocks. "
                    "The factorisation would fit a different model, so it is refused."
                )
                raise ValueError(message)


class _Growing:
    """A table of distinct rows, giving the same index to the same row in every block."""

    __slots__ = ("_rows", "_seen")

    def __init__(self) -> None:
        self._seen: dict[bytes, int] = {}
        self._rows = 0

    def restore(self, values: np.ndarray) -> None:
        """Register these rows as codes 0, 1, 2 ... in the order they are given.

        Not through :meth:`of`, which sorts within a call and would renumber them: a restored
        table has to hand out the indices the saved rows already point at.
        """
        for index, row in enumerate(values):
            self._seen[np.ascontiguousarray(row).tobytes()] = index
        self._rows = len(values)

    def of(self, values: np.ndarray) -> np.ndarray:
        distinct, inverse = _distinct(values)
        mapped = np.empty(len(distinct), dtype=np.int64)
        for position, row in enumerate(distinct):
            key = np.ascontiguousarray(row).tobytes()
            index = self._seen.get(key)
            if index is None:
                index = self._rows
                self._seen[key] = index
                self._rows += 1
            mapped[position] = index
        return np.asarray(mapped[inverse])

    def __len__(self) -> int:
        return self._rows


@dataclass(frozen=True)
class Expanded:
    """One model's design as two tables, built from the key frames rather than from the rows.

    This is what makes a scan serve a whole selection. The rows carry two indices into the
    combinations of a **widest** key, and a candidate model's design is a function of those
    combinations, so its tables come from putting its formula through 3,001 and 153,309 rows.
    Nothing re-reads the 72 million, nothing rebuilds the macro family, and nothing expands a
    design again.

    The moments come the same way. Every column is a function of one side, so the sum of it
    over every row is the sum over combinations of its value times how many rows carry that
    combination -- which is the one thing the scan has to count.
    """

    columns: pd.MultiIndex
    loan: np.ndarray
    calendar: np.ndarray
    loan_positions: np.ndarray
    calendar_positions: np.ndarray
    first: np.ndarray
    second: np.ndarray
    low: np.ndarray
    high: np.ndarray


def _only(columns: pd.MultiIndex, parameter: str) -> slice:
    """Where one parameter's columns sit in the design, as a slice.

    The design is built in the order of its MultiIndex, so a parameter's columns are adjacent;
    an order that broke that would make the slice wrong rather than slow, so it is refused.
    """
    outer = columns.get_level_values(0)
    positions = np.flatnonzero(outer == parameter)
    if not positions.size:
        message = f"The design has no columns for {parameter!r}."
        raise ValueError(message)
    if positions[-1] - positions[0] + 1 != len(positions):
        message = f"The columns of {parameter!r} are not adjacent in the design."
        raise ValueError(message)
    return slice(int(positions[0]), int(positions[-1]) + 1)


def _within(positions: np.ndarray, scale: slice) -> np.ndarray:
    """Those positions that fall inside the slice, numbered from its start."""
    inside = positions[(positions >= scale.start) & (positions < scale.stop)]
    return np.asarray(inside - scale.start)


def _unchanging(values: np.ndarray) -> np.ndarray:
    """Which columns never move. Exact equality, as `_standard_deviation` reads constancy."""
    return np.asarray(values.max(axis=0) == values.min(axis=0))


def _completed(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """``base`` with the other side's columns held at its first row, so a formula can read it.

    Repeating a row through ``iloc`` rather than assigning scalars, because the categoricals
    have to keep their declared levels: a dummy column exists for a level the probe never
    shows, and the design's width has to be the width the rows were encoded against.
    """
    filler = other.iloc[[0] * len(base)].reset_index(drop=True)
    return pd.concat([base.reset_index(drop=True), filler], axis=1)


def joined(blocks: Sequence[Rows]) -> Rows:
    """The blocks as one buffer, which is what the compiled backend takes in a single call.

    **Without a copy where there is already one buffer.** A reading that came off the disk cache
    is five memory maps with the blocks cut out of them as views, so concatenating would copy
    1.09 GB that is already contiguous -- and the gate for the compiled kernel is a memory gate
    as much as a time one. Where the blocks are adjacent views of a single array in order, that
    array is handed over as it stands; anywhere else the columns are concatenated, which is a
    one-off per fit and not per evaluation.

    The order is the blocks' own, which is the order the sums are taken in.
    """
    return Rows(**{column: _one_buffer(blocks, column) for column in _ROW_COLUMNS})


def _one_buffer(blocks: Sequence[Rows], column: str) -> np.ndarray:
    """One column of every block, as a single array, shared if it already is one.

    What comes back is **C-contiguous and of the column's declared dtype**, whichever branch
    runs: a compiled backend takes a slice of it and refuses anything else, and a refusal in
    the middle of an hour-old fit is a worse way to learn this than a cast here.
    """
    parts = [getattr(block, column) for block in blocks]
    rows = sum(len(part) for part in parts)
    whole = parts[0].base if parts else None
    if isinstance(whole, np.ndarray) and len(whole) == rows and _laid_out(parts, whole):
        shared = np.asarray(whole)
    elif len(parts) == 1:
        shared = np.ascontiguousarray(parts[0])
    else:
        shared = np.concatenate(parts)
    if shared.dtype != _ROW_DTYPES[column]:
        message = (
            f"The rows' {column!r} column is {shared.dtype}, where this kernel reads "
            f"{np.dtype(_ROW_DTYPES[column])}. `Factorisation.add` is what casts it."
        )
        raise TypeError(message)
    return shared


def _laid_out(parts: Sequence[np.ndarray], whole: np.ndarray) -> bool:
    """Whether these parts are exactly ``whole``, in order, as C-contiguous views of it."""
    at = whole.__array_interface__["data"][0]
    for part in parts:
        if part.base is not whole or not part.flags.c_contiguous:
            return False
        if part.__array_interface__["data"][0] != at:
            return False
        at += part.nbytes
    return True
