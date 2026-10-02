"""The interval-censored likelihood and its derivatives, written out rather than traced.

This module is the second implementation of lifelines' likelihood that
:mod:`creditsurv.models.blocks` was built to avoid, and it is here because the measurement
said so. At the 26 parameters rule 12 produced, a value-and-gradient on a 250,000-cell block
of the production table is 166.8 ms and a Hessian 1,373 -- so a cold fit is an hour and a
selection run is a day, and the finer bands the calibration needs cost a full re-selection for
two families. ``docs/reports/engine.md`` carries the attribution.

**What makes writing it out cheap is the shape of the data, not cleverness.** Measured on the
training window:

* every column of the design is a function of the **loan** combination (3,001 of them) or of
  the **calendar** key (153,309), and never of both, so the scale's linear predictor is
  ``eta = A[i] + B[j]`` -- two small tables rebuilt per evaluation for about a million flops,
  against expanding a 26-column design over 53 million rows;
* the interval is always ``[a, a+1]`` for an exit and ``[a+1, INFINITY_STAND_IN)`` for a
  survivor, the entry is always the age, and exact observations never occur, so every time a
  row needs is a lookup into a table of at most 361 ages;
* the shape is a single number -- no production fit gives it covariates -- so **a row depends
  on exactly two scalars**: its own ``eta`` and the shape's coefficient ``r``.

That last point is what this module is built around. The derivatives of a function of two
scalars are six numbers, whatever the number of parameters, and they are carried here through a
second-order forward chain (:class:`_Jet`) rather than derived by hand: the formulas below read
like the likelihood, and the chain makes their first and second derivatives exact by
construction. ``tests/test_kernel.py`` holds every one of them to autograd's own answer,
including inside the clipped regions, because a wall that moves by an epsilon moves
``blocks._outside_the_domain`` with it and changes what the guards do.

**Matching lifelines exactly means matching its clips, including where they are wrong.**
``safe_exp`` caps its argument and then reports the derivative of the *uncapped* exponential,
which is what autograd's custom VJP does; the Weibull's survival function is unclipped while
the log-logistic's is clipped to ``[1e-12, 1-1e-12]`` with a derivative of zero outside,
because the Weibull fitter overrides the method and the log-logistic does not; the interval
probability is clipped to ``[1e-25, 1-1e-25]`` in the likelihood itself, and the
left-truncation term is added to it **unclipped**, which is the whole reason the objective is
unbounded below and the reason this engine has a floor and a wall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: lifelines' ``safe_exp`` ceiling: ``exp`` is never asked for more than this, and its
#: derivative is reported as the capped value rather than as zero.
MAX_EXPONENT: Final = float(np.log(np.finfo(float).max) - 75)

#: The interval probability's clip, from ``_log_likelihood_interval_censoring``.
INTERVAL_FLOOR: Final = 1e-25
INTERVAL_CEILING: Final = 1.0 - 1e-25

#: The survival function's clip, from ``ParametricRegressionFitter._survival_function``. The
#: Weibull fitter overrides that method and does **not** clip; the log-logistic inherits it.
SURVIVAL_FLOOR: Final = 1e-12
SURVIVAL_CEILING: Final = 1.0 - 1e-12

#: What each family floors the time at before taking its logarithm. Different numbers, from
#: the two ``_cumulative_hazard`` implementations, and they matter at age zero.
TIME_FLOOR: Final[dict[str, float]] = {"weibull": 1e-100, "loglogistic": 1e-25}

#: What ``blocks`` puts in place of an infinite upper bound, and therefore the last time a
#: survivor's interval reaches. Kept here so the log of it is a table entry like any other.
INFINITY_STAND_IN: Final = 1e25

#: A structural zero: a derivative that is zero by construction rather than by arithmetic.
#: Carried as this float so the chain can skip the term instead of multiplying 53 million
#: rows by it.
_ZERO: Final = 0.0


def _is_zero(value: np.ndarray | float) -> bool:
    return isinstance(value, float) and value == 0.0


def _plus(left: np.ndarray | float, right: np.ndarray | float) -> np.ndarray | float:
    if _is_zero(left):
        return right
    if _is_zero(right):
        return left
    return left + right


def _times(left: np.ndarray | float, right: np.ndarray | float) -> np.ndarray | float:
    if _is_zero(left) or _is_zero(right):
        return _ZERO
    if isinstance(left, float) and left == 1.0:
        return right
    if isinstance(right, float) and right == 1.0:
        return left
    return left * right


def _negate(value: np.ndarray | float) -> np.ndarray | float:
    return _ZERO if _is_zero(value) else -value


@dataclass(frozen=True)
class _Jet:
    """A quantity and its first and second derivatives in the two scalars a row depends on.

    ``eta`` is the row's own scale predictor and ``r`` the shape's single coefficient. Six
    components, independent of how many parameters the model has, which is why the Hessian
    costs what a gradient costs here instead of forty-nine times a value.

    A component held as the float ``0.0`` is a **structural** zero -- zero for every row, by
    construction -- and the arithmetic drops the term rather than allocating an array for it.
    That is not an approximation: ``log t - eta`` has a second derivative of exactly zero, and
    multiplying 53 million rows by it would be 400 MB of nothing.
    """

    v: np.ndarray | float
    de: np.ndarray | float = _ZERO
    dr: np.ndarray | float = _ZERO
    dee: np.ndarray | float = _ZERO
    der: np.ndarray | float = _ZERO
    drr: np.ndarray | float = _ZERO
    #: Whether the second derivatives are wanted at all. An optimiser evaluation asks for a
    #: value and a gradient and nothing else, and the second-order components are **six of the
    #: eleven multiplications** a chain rule performs and nine of the fourteen a product does:
    #: measured, carrying them through a gradient-only evaluation costs 420 ns a row against
    #: 211. Seeded once by `row_likelihood` and carried by every operation, so a jet that was
    #: never asked for curvature cannot acquire it half way through.
    curved: bool = True

    def __add__(self, other: _Jet) -> _Jet:
        curved = self.curved and other.curved
        return _Jet(
            _plus(self.v, other.v),
            _plus(self.de, other.de),
            _plus(self.dr, other.dr),
            _plus(self.dee, other.dee) if curved else _ZERO,
            _plus(self.der, other.der) if curved else _ZERO,
            _plus(self.drr, other.drr) if curved else _ZERO,
            curved,
        )

    def __sub__(self, other: _Jet) -> _Jet:
        return self + (-other)

    def __neg__(self) -> _Jet:
        return _Jet(
            _negate(self.v),
            _negate(self.de),
            _negate(self.dr),
            _negate(self.dee),
            _negate(self.der),
            _negate(self.drr),
            self.curved,
        )

    def __mul__(self, other: _Jet) -> _Jet:
        curved = self.curved and other.curved
        value = _times(self.v, other.v)
        de = _plus(_times(self.de, other.v), _times(self.v, other.de))
        dr = _plus(_times(self.dr, other.v), _times(self.v, other.dr))
        if not curved:
            return _Jet(value, de, dr, _ZERO, _ZERO, _ZERO, False)
        dee = _plus(
            _plus(_times(self.dee, other.v), _times(self.v, other.dee)),
            _times(2.0, _times(self.de, other.de)),
        )
        der = _plus(
            _plus(_times(self.der, other.v), _times(self.v, other.der)),
            _plus(_times(self.de, other.dr), _times(self.dr, other.de)),
        )
        drr = _plus(
            _plus(_times(self.drr, other.v), _times(self.v, other.drr)),
            _times(2.0, _times(self.dr, other.dr)),
        )
        return _Jet(value, de, dr, dee, der, drr, True)

    def chain(
        self,
        value: np.ndarray | float,
        first: np.ndarray | float,
        second: np.ndarray | float,
    ) -> _Jet:
        """``f(self)``, given ``f`` at this value and its first two derivatives there.

        The second-order terms are grouped as ``(f'' * d) * d`` rather than ``f'' * (d * d)``,
        which is not a style: the two differ where one factor overflows. Far out on the
        prepayment surface a cumulative hazard reaches 1.3e275 and its derivative -2.6e276, so
        the square is 6.8e552 and is `inf` -- and a survival of exactly zero times `inf` is
        `nan`, where autograd's own association gives zero. The test that found it compares a
        Hessian at eta = -500 with the shape on its bound.
        """
        de = _times(first, self.de)
        dr = _times(first, self.dr)
        if not self.curved:
            return _Jet(value, de, dr, _ZERO, _ZERO, _ZERO, False)
        return _Jet(
            value,
            de,
            dr,
            _plus(_times(first, self.dee), _times(_times(second, self.de), self.de)),
            _plus(_times(first, self.der), _times(_times(second, self.de), self.dr)),
            _plus(_times(first, self.drr), _times(_times(second, self.dr), self.dr)),
            True,
        )

    def scaled(self, by: np.ndarray | float) -> _Jet:
        """This jet times a constant -- a mask or a weight, nothing that carries derivatives."""
        return _Jet(
            _times(self.v, by),
            _times(self.de, by),
            _times(self.dr, by),
            _times(self.dee, by),
            _times(self.der, by),
            _times(self.drr, by),
            self.curved,
        )


def _exp(jet: _Jet) -> _Jet:
    value = np.exp(jet.v)
    return jet.chain(value, value, value)


def _safe_exp(jet: _Jet) -> _Jet:
    """lifelines' ``safe_exp``: the argument is capped, the derivative is not zeroed.

    ``defvjp(safe_exp, lambda ans, x: lambda g: g * ans)`` -- so where the cap binds, autograd
    reports the derivative of an exponential that was never evaluated. Mathematically the
    derivative there is zero; reproducing autograd matters more, because the cap is one of the
    walls the optimiser backtracks from and moving it changes which fits are refused.
    """
    value = np.exp(np.minimum(jet.v, MAX_EXPONENT))
    return jet.chain(value, value, value)


def _log(jet: _Jet) -> _Jet:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.log(jet.v)
        first = 1.0 / jet.v
    return jet.chain(value, first, -first * first)


def _clipped(jet: _Jet, floor: float, ceiling: float) -> _Jet:
    """``anp.clip``, whose gradient autograd zeroes wherever the answer is a bound.

    ``g * logical_and(ans != a_min, ans != a_max)``, so a value sitting exactly on a bound
    counts as clipped even if it arrived there without being cut.
    """
    value = np.clip(jet.v, floor, ceiling)
    inside = (value != floor) & (value != ceiling)
    return jet.chain(value, inside, _ZERO)


def _logaddexp_zero(jet: _Jet) -> _Jet:
    """``logaddexp(self, 0)`` -- the log-logistic's cumulative hazard.

    The value comes from numpy so it matches lifelines digit for digit; the derivatives are
    the logistic function and its own derivative, which is what the chain needs.
    """
    value = np.logaddexp(jet.v, 0.0)
    first = 1.0 / (1.0 + np.exp(-jet.v))
    return jet.chain(value, first, first * (1.0 - first))


def log_times(distribution: str, ages: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The logarithms every row needs, as tables over the loan ages in the data.

    Three of them: the entry (the age itself), the next month, and the stand-in for infinity
    that an unbounded interval is clipped to. The age is floored exactly as the family's
    ``_cumulative_hazard`` floors its time argument -- ``1e-100`` for the Weibull and ``1e-25``
    for the log-logistic -- which is what keeps age zero finite instead of ``-inf``.
    """
    floor = TIME_FLOOR[distribution]
    times = np.asarray(ages, dtype=float)
    entry = np.log(np.clip(times, floor, np.inf))
    following = np.log(np.clip(times + 1.0, floor, np.inf))
    return entry, following, float(np.log(np.clip(INFINITY_STAND_IN, floor, np.inf)))


def _weibull_terms(log_time: np.ndarray | float, eta: _Jet, shape: _Jet) -> tuple[_Jet, _Jet]:
    """The Weibull's cumulative hazard at ``log_time``, and the survival beside it.

    ``H = safe_exp(rho * (log t - log lambda))`` with ``rho = safe_exp(r)``, and
    ``S = safe_exp(-H)`` -- the fitter overrides the base class's survival function, so there
    is no clip on ``S`` here. The shape arrives as its **coefficient** and is exponentiated
    here, because the two families do it differently: this one through ``safe_exp`` and the
    log-logistic through a plain one. It is a single number, so doing it per call is free.
    """
    cumulative = _safe_exp(_safe_exp(shape) * (_Jet(log_time) - eta))
    return cumulative, _safe_exp(-cumulative)


def _loglogistic_terms(log_time: np.ndarray | float, eta: _Jet, shape: _Jet) -> tuple[_Jet, _Jet]:
    """The log-logistic's cumulative hazard at ``log_time``, and the survival beside it.

    ``H = logaddexp(beta * (log t - log(safe_exp(eta))), 0)`` with ``beta = exp(r)`` -- a plain
    exponential, not the safe one -- and ``S = clip(exp(-H), 1e-12, 1-1e-12)``, because this
    fitter does **not** override the base class's survival function. The log of the capped
    exponential is written as lifelines writes it rather than simplified to ``eta``: the two
    differ in the last digits, and above the cap they differ altogether.
    """
    log_scale = _log(_safe_exp(eta))
    cumulative = _logaddexp_zero(_exp(shape) * (_Jet(log_time) - log_scale))  # beta_, plain exp
    survival = _clipped(_exp(-cumulative), SURVIVAL_FLOOR, SURVIVAL_CEILING)
    return cumulative, survival


_FAMILIES: Final[dict[str, Callable[[np.ndarray | float, _Jet, _Jet], tuple[_Jet, _Jet]]]] = {
    "weibull": _weibull_terms,
    "loglogistic": _loglogistic_terms,
}

#: The families this kernel can fit. The others keep the autograd evaluator.
FAMILIES: Final = tuple(sorted(_FAMILIES))


def row_likelihood(
    distribution: str,
    *,
    log_entry: np.ndarray,
    log_start: np.ndarray,
    log_stop: np.ndarray,
    truncated: np.ndarray,
    eta: np.ndarray,
    shape: float,
    curvature: bool = True,
) -> _Jet:
    """One row's log-likelihood and its derivatives in ``eta`` and the shape's coefficient.

    The interval-censored term and the left-truncation term, exactly as
    ``_log_likelihood_interval_censoring`` writes them for a panel where no observation is
    exact::

        log(clip(S(start) - S(stop), 1e-25, 1 - 1e-25))  +  H(entry) if entry > 0

    The second term is **not** inside the clip, and that asymmetry is the whole of the problem
    this engine works around: clipped, the first term stops falling; unclipped, the second
    keeps rising, so the objective -- a mean negative log-likelihood, which cannot be negative
    -- goes negative and flat far from the data. It needs only 0.0176 on the mean to fabricate
    a minimum lower than the true one.
    """
    if distribution not in _FAMILIES:
        message = (
            f"The written-out kernel fits {' and '.join(FAMILIES)}, not {distribution!r}. "
            "Pass the autograd evaluator for anything else."
        )
        raise ValueError(message)
    family = _FAMILIES[distribution]
    scale = _Jet(eta, de=1.0, curved=curvature)
    coefficient = _Jet(shape, dr=1.0, curved=curvature)

    entry, _ = family(log_entry, scale, coefficient)
    _, opened = family(log_start, scale, coefficient)
    _, closed = family(log_stop, scale, coefficient)

    interval = _clipped(opened - closed, INTERVAL_FLOOR, INTERVAL_CEILING)
    return _log(interval) + entry.scaled(truncated)


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
    return rounded.astype(np.uint32)


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
        columns: Sequence[str],
    ) -> None:
        self._loan = tuple(loan)
        self._calendar = (*calendar, age_column)
        self._age_column = age_column
        self._columns = tuple(columns)
        self._loan_codes = _Growing()
        self._calendar_codes = _Growing()
        self._loan_positions: np.ndarray | None = None
        self._calendar_positions: np.ndarray | None = None
        self._loan_rows: list[np.ndarray] = []
        self._calendar_rows: list[np.ndarray] = []

    def add(
        self, frame: pd.DataFrame, design: np.ndarray, *, event: np.ndarray, weight: np.ndarray
    ) -> Rows:
        """Encode one block, growing the tables with whatever combinations are new to it."""
        i = self._loan_codes.of(_stable_codes(frame, self._loan))
        j = self._calendar_codes.of(_stable_codes(frame, self._calendar))
        if self._loan_positions is None:
            self._decide(design, i, j)
        assert self._loan_positions is not None
        assert self._calendar_positions is not None
        self._grow(self._loan_rows, design[:, self._loan_positions], i)
        self._grow(self._calendar_rows, design[:, self._calendar_positions], j)
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
    def _grow(rows: list[np.ndarray], design: np.ndarray, codes: np.ndarray) -> None:
        """Store a design row for each combination this block is the first to carry.

        Codes are handed out in order of first appearance, so a block's new ones are exactly
        the indices past the end of the table, and any row carrying one of them will do: the
        invariant `_verify` checks is that they all carry the same values.
        """
        wanted = int(codes.max()) + 1 if len(codes) else 0
        missing = set(range(len(rows), wanted))
        if not missing:
            return
        rows.extend(np.zeros(design.shape[1]) for _ in missing)
        for position, code in enumerate(codes):
            index = int(code)
            if index in missing:
                rows[index] = design[position].copy()
                missing.discard(index)
                if not missing:
                    return

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

    def of(self, values: np.ndarray) -> np.ndarray:
        distinct, inverse = np.unique(values, axis=0, return_inverse=True)
        mapped = np.empty(len(distinct), dtype=np.int64)
        for position, row in enumerate(distinct):
            key = np.ascontiguousarray(row).tobytes()
            index = self._seen.get(key)
            if index is None:
                index = self._rows
                self._seen[key] = index
                self._rows += 1
            mapped[position] = index
        return mapped[np.asarray(inverse).ravel()]

    def __len__(self) -> int:
        return self._rows


@dataclass(frozen=True)
class _Totals:
    """The objective, its gradient, and its curvature where one was asked for."""

    value: float
    gradient: np.ndarray
    curvature: np.ndarray | None


#: The oldest loan age the log-time tables cover. `panel.MAX_AGE_MONTHS` is 360 and the
#: production table reaches 326; a few more costs three floats.
_MAX_AGE: Final = 400


@dataclass(frozen=True)
class Kernel:
    """The objective over every row, as a sum over two small tables.

    ``loan`` and ``calendar`` are the design's two sides, **already divided by lifelines'
    column standard deviations**, so this works in the same scaled space the optimiser does.
    ``loan_index``, ``calendar_index`` and ``shape_index`` say where each side's coefficients
    sit in the parameter vector, so what comes back is a gradient and a curvature in the
    order lifelines reads them.

    What it computes is lifelines' own objective: the **mean negative** log-likelihood over
    every loan-month, weights counted as the replications they are. The penalty is not here;
    it is a function of the parameters alone and the caller adds it once.

    The work per row is two table lookups, three cumulative hazards and two survivals, and the
    gradient and the curvature cost two scatter-adds and a handful of small matrix products on
    top -- so a Hessian is about three times a gradient rather than the 5.8 autograd charges,
    and neither grows with the number of parameters.
    """

    distribution: str
    loan: np.ndarray
    calendar: np.ndarray
    loan_index: np.ndarray
    calendar_index: np.ndarray
    shape_index: int
    blocks: tuple[Rows, ...]
    total_weight: float
    #: Rows evaluated at a time, and the measurement rather than a guess: swept from 2,048 to
    #: 1,048,576 on five million rows, a value-and-gradient runs 289, 347, 241, **212**, 217
    #: and 265 ns a row, so the curve is shallow and 131,072 is the floor of it. Small chunks
    #: pay numpy's per-call overhead and large ones leave cache; neither effect is worth more
    #: than about 30%. What the chunking is really for is a working set that does not grow
    #: with the table, and boundaries fixed by the block's length -- which is what keeps the
    #: summation order, and so the answer, identical on every run.
    chunk: int = 1 << 17

    def __call__(self, x: np.ndarray, *, curvature: bool = False) -> _Totals:
        """The value, the gradient, and the curvature when it is asked for."""
        entry, following, far = log_times(self.distribution, np.arange(_MAX_AGE + 1))
        loans, calendars = len(self.loan), len(self.calendar)
        total = 0.0
        by_loan = np.zeros(loans)
        by_calendar = np.zeros(calendars)
        shape_first = 0.0
        curved_loan = np.zeros(loans) if curvature else None
        curved_calendar = np.zeros(calendars) if curvature else None
        crossed = np.zeros((loans, self.calendar.shape[1])) if curvature else None
        mixed_loan = np.zeros(loans) if curvature else None
        mixed_calendar = np.zeros(calendars) if curvature else None
        shape_second = 0.0

        eta_loan = self.loan @ x[self.loan_index]
        eta_calendar = self.calendar @ x[self.calendar_index]
        shape = float(x[self.shape_index])

        for block in self.blocks:
            for start in range(0, block.rows, self.chunk):
                stop = start + self.chunk
                i = block.i[start:stop].astype(np.intp)
                j = block.j[start:stop].astype(np.intp)
                age = block.age[start:stop].astype(np.intp)
                weight = block.weight[start:stop].astype(np.float64)
                event = block.event[start:stop]
                jet = row_likelihood(
                    self.distribution,
                    log_entry=entry[age],
                    log_start=np.where(event, entry[age], following[age]),
                    log_stop=np.where(event, following[age], far),
                    truncated=(age > 0).astype(np.float64),
                    eta=eta_loan[i] + eta_calendar[j],
                    shape=shape,
                    curvature=curvature,
                )
                total += float(np.dot(weight, np.asarray(jet.v)))
                scaled = weight * np.asarray(jet.de)
                by_loan += np.bincount(i, weights=scaled, minlength=loans)
                by_calendar += np.bincount(j, weights=scaled, minlength=calendars)
                shape_first += float(np.dot(weight, np.broadcast_to(jet.dr, weight.shape)))
                if not curvature:
                    continue
                assert curved_loan is not None
                assert curved_calendar is not None
                assert crossed is not None
                assert mixed_loan is not None
                assert mixed_calendar is not None
                second = weight * np.asarray(jet.dee)
                curved_loan += np.bincount(i, weights=second, minlength=loans)
                curved_calendar += np.bincount(j, weights=second, minlength=calendars)
                for column in range(self.calendar.shape[1]):
                    crossed[:, column] += np.bincount(
                        i, weights=second * self.calendar[j, column], minlength=loans
                    )
                mixing = weight * np.broadcast_to(jet.der, weight.shape)
                mixed_loan += np.bincount(i, weights=mixing, minlength=loans)
                mixed_calendar += np.bincount(j, weights=mixing, minlength=calendars)
                shape_second += float(np.dot(weight, np.broadcast_to(jet.drr, weight.shape)))

        value = -float(total) / self.total_weight
        gradient = np.zeros(len(x))
        gradient[self.loan_index] = -(self.loan.T @ by_loan) / self.total_weight
        gradient[self.calendar_index] = -(self.calendar.T @ by_calendar) / self.total_weight
        gradient[self.shape_index] = -shape_first / self.total_weight
        if not curvature:
            return _Totals(value, gradient, None)
        assert curved_loan is not None
        assert curved_calendar is not None
        assert crossed is not None
        assert mixed_loan is not None
        assert mixed_calendar is not None
        hessian = np.zeros((len(x), len(x)))
        loan_block = self.loan.T @ (self.loan * curved_loan[:, None])
        calendar_block = self.calendar.T @ (self.calendar * curved_calendar[:, None])
        cross = self.loan.T @ crossed
        hessian[np.ix_(self.loan_index, self.loan_index)] = loan_block
        hessian[np.ix_(self.calendar_index, self.calendar_index)] = calendar_block
        hessian[np.ix_(self.loan_index, self.calendar_index)] = cross
        hessian[np.ix_(self.calendar_index, self.loan_index)] = cross.T
        hessian[self.loan_index, self.shape_index] = self.loan.T @ mixed_loan
        hessian[self.shape_index, self.loan_index] = self.loan.T @ mixed_loan
        hessian[self.calendar_index, self.shape_index] = self.calendar.T @ mixed_calendar
        hessian[self.shape_index, self.calendar_index] = self.calendar.T @ mixed_calendar
        hessian[self.shape_index, self.shape_index] = shape_second
        return _Totals(value, gradient, -hessian / self.total_weight)

    @property
    def rows(self) -> int:
        return sum(block.rows for block in self.blocks)

    @property
    def nbytes(self) -> int:
        tables = self.loan.nbytes + self.calendar.nbytes
        return tables + sum(block.nbytes for block in self.blocks)
