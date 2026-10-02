"""One row's log-likelihood and its derivatives, written out rather than traced.

A row depends on exactly **two scalars** -- its own linear predictor and the shape's single
coefficient, since no production fit gives the shape covariates -- so its derivatives are six
numbers whatever the parameter count. They are carried through a second-order forward chain
(:class:`_Jet`) rather than derived by hand: the formulas below read like the likelihood, and
the chain makes their first and second derivatives exact by construction.

**Matching lifelines means matching its clips, including where they are wrong.** ``safe_exp``
caps its argument and then reports the derivative of the *uncapped* exponential, which is what
autograd's custom VJP does; the Weibull's survival function is unclipped while the
log-logistic's is clipped to ``[1e-12, 1-1e-12]`` with a derivative of zero outside, because the
Weibull fitter overrides the method and the log-logistic does not; the interval probability is
clipped to ``[1e-25, 1-1e-25]`` in the likelihood itself, and the left-truncation term is added
to it **unclipped**, which is the whole reason the objective is unbounded below and the reason
the engine has a floor and a wall.

``tests/test_kernel.py`` holds every one of them to autograd's own answer, including inside the
clipped regions, because a wall that moves by an epsilon moves the engine's guards with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


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

#: The oldest loan age the log-time tables cover. `panel.MAX_AGE_MONTHS` is 360 and the
#: production table reaches 326; a few more costs three floats.
_MAX_AGE: Final = 400

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
