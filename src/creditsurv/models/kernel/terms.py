"""The objective over every row, as a sum over two small tables.

**This is the port a compiled implementation replaces.** It takes the encoded rows, the two
design tables and a parameter vector, and returns a scalar, a gradient and -- when it is asked
for -- a curvature. Nothing else: no configuration, no panel, no lifelines, no I/O. The
architecture test checks that this package imports nothing from ``creditsurv`` at all, because
that is the property that makes the replacement a drop-in rather than a fork.

What it computes is lifelines' own objective: the **mean negative** log-likelihood over every
loan-month, weights counted as the replications they are. The penalty is not here; it is a
function of the parameters alone and the caller adds it once.

The work per row is two table lookups, three cumulative hazards and two survivals, and the
gradient and the curvature cost two scatter-adds and a handful of small matrix products on top
-- so a Hessian is about twice a value-and-gradient rather than the forty-nine times a value
autograd charges, and neither grows with the number of parameters.

**And there is a second implementation behind it.** `crates/creditsurv-kernel` is the same
arithmetic as a fused loop, built by `uv sync --extra kernel` and **optional at import**: this
module is normative, a missing extension is the normal case and not an error, and
`tests/test_kernel.py` compares whatever backends are present. Rule 13 of `docs/rules.md`
declared the gate it had to pass and the segregation it has to keep before any of it was
written.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:
    from creditsurv.models.kernel.factorisation import (
        Rows,
    )

from creditsurv.models.kernel.factorisation import joined
from creditsurv.models.kernel.likelihood import (
    _MAX_AGE,
    log_times,
    row_likelihood,
)

#: The compiled backend, where the optional extra is installed. `None` is the ordinary case:
#: this module computes the same numbers and nothing needs a Rust toolchain.
#: An assignment rather than an import alias, so it is a name this module owns: mypy's
#: `no_implicit_reexport` treats an aliased import as private to the file that made it, and the
#: tests have to be able to switch the backend off.
_compiled: Any
try:  # pragma: no cover - which branch runs is which extras are installed
    import creditsurv_kernel

    _compiled = creditsurv_kernel
except ImportError:  # pragma: no cover
    _compiled = None

#: How the compiled backend names the two families: a number rather than a string, because only
#: fixed-dtype numbers cross that boundary. Written out rather than derived from `FAMILIES`,
#: which is **sorted** and would have handed the log-logistic the Weibull's code -- a different
#: family fitted under the right name, which is the one mistake this boundary can make silently.
#: `tests/test_kernel.py` holds it to the crate's own constants.
_FAMILY_CODE: Final = {"weibull": 0, "loglogistic": 1}


@dataclass(frozen=True)
class _Totals:
    """The objective, its gradient, and its curvature where one was asked for."""

    value: float
    gradient: np.ndarray
    curvature: np.ndarray | None


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
    #: Threads the row loop is cut into, and **part of what determines the answer**.
    #:
    #: The rows go into this many contiguous parts, each part sums its own in its own order, and
    #: the partials are added in the parts' own order rather than as they finish. So two runs at
    #: the same count agree bit for bit and a run at a different count agrees to the last digits
    #: of a sum over 72.7 million terms -- which is why the count is carried here and recorded
    #: with the fit, not taken from whatever machine happens to run it. Rule 13 of
    #: `docs/rules.md` declares that, and the reason is a pooled fit that once took its shares
    #: from whichever worker finished first.
    #:
    #: One by default, because the NumPy backend is single-threaded and is what the equivalence
    #: tests compare against; a caller that wants the cores asks for them.
    threads: int = 1
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
        if _compiled is not None and self.distribution in _FAMILY_CODE:
            return self._compiled(x, curvature=curvature)
        return self._written(x, curvature=curvature)

    @cached_property
    def _order(self) -> np.ndarray:
        """Where the compiled backend's parameters sit in lifelines' own vector.

        It returns `[loan..., calendar..., shape]`, in the order the two design tables give
        their columns, because which index a coefficient sits at is a fact about lifelines and
        not about the arithmetic. One permutation puts it back.
        """
        return np.concatenate([self.loan_index, self.calendar_index, [self.shape_index]]).astype(
            np.intp
        )

    @cached_property
    def _rows(self) -> Rows:
        """Every block as one buffer, for the single call the compiled backend takes."""
        return joined(self.blocks)

    def _compiled(self, x: np.ndarray, *, curvature: bool) -> _Totals:
        """The same objective, evaluated by the crate: arrays in, three results out."""
        rows = self._rows
        value, gradient, hessian = _compiled.evaluate(
            _FAMILY_CODE[self.distribution],
            np.ascontiguousarray(self.loan, dtype=np.float64),
            np.ascontiguousarray(self.calendar, dtype=np.float64),
            np.ascontiguousarray(x[self.loan_index], dtype=np.float64),
            np.ascontiguousarray(x[self.calendar_index], dtype=np.float64),
            float(x[self.shape_index]),
            *log_times(self.distribution, np.arange(_MAX_AGE + 1)),
            rows.i,
            rows.j,
            rows.age,
            rows.event,
            rows.weight,
            self.total_weight,
            curvature,
            self.threads,
        )
        whole = np.zeros(len(x))
        whole[self._order] = gradient
        if hessian is None:
            return _Totals(value, whole, None)
        curved = np.zeros((len(x), len(x)))
        curved[np.ix_(self._order, self._order)] = hessian
        return _Totals(value, whole, curved)

    def _written(self, x: np.ndarray, *, curvature: bool = False) -> _Totals:
        """The objective written out in NumPy, which is the normative implementation."""
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
                i = block.i[start:stop]
                j = block.j[start:stop]
                age = block.age[start:stop]
                weight = block.weight[start:stop].astype(np.float64)
                event = block.event[start:stop]
                # Each table read once. `entry[age]` was gathered twice -- once as the
                # truncation time and once inside the interval's own `where` -- which is a
                # megabyte of gather per chunk for a number already in hand.
                opens, closes = entry[age], following[age]
                jet = row_likelihood(
                    self.distribution,
                    log_entry=opens,
                    log_start=np.where(event, opens, closes),
                    log_stop=np.where(event, closes, far),
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
