"""What an evaluation of the objective means, and what ends a fit that cannot finish.

This is the engine's port. Everything above it -- the optimiser, the damped Newton polish, the
method chain -- is algebra on a handful of numbers and does not know where the rows are or how
the arithmetic was done. Three implementations satisfy it: autograd over a stored design, the
written-out likelihood of :mod:`creditsurv.models.kernel`, and the pool that adds up what
several processes computed.

With it live the things that are properties of the *surface* rather than of any one
implementation, because every implementation needs them and none owns them:

* **the bounds**, and the fact that the shape's matter and the coefficients' do not. The
  cumulative hazard is ``exp(rho * (log t - log lambda))``, so the shape sits in an exponent
  and needs only reach ``exp(5)`` to take the objective with it, while a scale coefficient of
  the same size does nothing of the kind;
* **the wall**: lifelines clips the interval probability and adds the left-truncation term
  unclipped, so beyond a ridge the objective -- a mean *negative* log-likelihood, which cannot
  be negative -- falls away into a region that is not a likelihood at all. At face value that
  region is the most attractive place on the surface. Returned as ``inf`` with a zero gradient
  it is a wall, and every method backtracks from it;
* **the floor**: a nested model cannot fit better than its parent, so the parent's optimum
  bounds every point of the child, and handing that bound to the objective makes the clipped
  region unreachable *while the fit runs*;
* **the guards** that end a fit which is circling a boundary it cannot cross, and which tell a
  refusal by the floor from a refusal by the clipping -- because the first says the same thing
  to every optimiser and every starting point, and the second does not.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np
from lifelines import exceptions

if TYPE_CHECKING:
    #: What the parent asks a worker for, and what a worker sends back.
    import pandas as pd

    Command = tuple[str, np.ndarray | None]
    Prepared = tuple[pd.MultiIndex, np.ndarray, float, dict[str, np.ndarray]]
    Answer = dict[str, Any] | tuple[float, np.ndarray] | np.ndarray
    #: A worker's answer with the part it read, so the parent can add them in one order.
    Tagged = tuple[int, Answer]

log = logging.getLogger(__name__)


class _Evaluator(Protocol):
    """The objective as the optimiser and the polish use it, wherever the rows are."""

    total_weight: float
    evaluations: int
    pinned: _Pinned

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]: ...

    def hessian(self, x: np.ndarray) -> np.ndarray: ...


class Pinned(exceptions.ConvergenceError):  # type: ignore[misc]  # lifelines is untyped
    """The optimiser could not leave a boundary, so no starting point will help.

    Told apart from every other way a fit fails to converge because the caller's remedy differs.
    A warm start that diverges has failed *as a starting point* and a cold fit is the answer; an
    optimiser pinned against the parent's optimum has found where lifelines' clipped region
    begins, and that edge is a property of the surface. Starting somewhere else and walking back
    to the same maximum costs another hour to meet the same edge, which the prepayment model paid
    twice before this was told apart.
    """


class _Pinned:
    """Whether the optimiser is stuck against a boundary it cannot leave.

    Two readings of the same thing: a run of refusals, and a window of them without progress.
    The first catches an optimiser that has walked into the wall and stopped; the second catches
    one that is circling it, which is what the prepayment model did for four hours.
    """

    def __init__(self) -> None:
        self.refusals = 0
        self._window: deque[bool] = deque(maxlen=_PINNED_WINDOW)
        self._against_the_floor: deque[bool] = deque(maxlen=_PINNED_WINDOW)
        self._best = np.inf
        self._best_at = 0
        self._seen = 0

    def saw(self, *, refused: bool, value: float) -> None:
        self._seen += 1
        self.refusals = self.refusals + 1 if refused else 0
        self._window.append(refused)
        # A refusal of a value a likelihood could take is the floor's doing; an impossible one is
        # lifelines' clipping. The polish needs them apart, because only the first is a statement
        # about the parent's optimum.
        self._against_the_floor.append(refused and _possible(value))
        if not refused and value < self._best:
            self._best, self._best_at = value, self._seen

    @property
    def against_the_floor(self) -> bool:
        """Whether what turned the optimiser back was the parent's optimum, not the clipping.

        Most of the **refusals**, not most of the evaluations: a damped Newton alternates a
        refused step with an accepted one, so the refusals are barely half of what it does. On
        the prepayment model's warm phase, 9 of 11 refusals were the floor and 22 evaluations
        were made -- a majority of the evaluations would have wanted 12, and the fit went on to
        spend three more hours reaching the same answer.
        """
        refusals = sum(self._window)
        return refusals > 0 and 2 * sum(self._against_the_floor) > refusals

    @property
    def circling(self) -> bool:
        """A full window mostly refused, and no better point found inside it."""
        if len(self._window) < _PINNED_WINDOW:
            return False
        if sum(self._window) / len(self._window) < _PINNED_SHARE:
            return False
        return self._seen - self._best_at >= _PINNED_WINDOW


def _check_pinned(pinned: _Pinned, floor: float | None) -> None:
    """Give up once the optimiser is pinned against the boundary, in a row or in a window."""
    where = "below the parent's optimum" if floor is not None else "outside the likelihood"
    if pinned.refusals >= _PINNED_REFUSALS:
        message = (
            f"The optimiser has been refused {pinned.refusals} times in a row, every point "
            f"{where}. The maximum of the likelihood as lifelines computes it lies in the region "
            "it cannot compute, so this specification cannot be fitted."
        )
        raise Pinned(message)
    if pinned.circling:
        message = (
            f"Of the last {_PINNED_WINDOW} evaluations at least "
            f"{_PINNED_SHARE:.0%} were refused, every one {where}, and none of the rest "
            "improved on the best point already found. The optimiser is circling a boundary it "
            "cannot cross, so this specification cannot be fitted."
        )
        raise Pinned(message)


def _outside_the_domain(
    value: float, x: np.ndarray, floor: float | None = None
) -> tuple[float, np.ndarray] | None:
    """``(inf, 0)`` where the objective is not one a likelihood can take, else ``None``.

    The objective is a **mean negative log-likelihood** and cannot be negative. lifelines clips
    the interval probability at 1e-25 but adds the left-truncation term unclipped, so beyond a
    ridge the surface falls away into a region that is not a likelihood at all -- the worst
    point seen here read -8.97e+69, and `rho_` needs only reach exp(5) for the cumulative hazard
    to get there.

    Reported at face value, that region is the most attractive place on the surface and every
    optimiser walks into it: on the prepayment model, **six attempts in a row** -- warm and
    cold, SLSQP, L-BFGS-B and trust-constr alike -- ended there, and bounding the coefficients
    at 100 did not help, because it is the *shape* parameter that makes the hazard explode.
    Reported as infinite, it is a wall: every method backtracks from it, which is how a domain
    boundary is meant to be told to an optimiser.

    ``floor`` is the value the objective cannot go below on this specification, and for a
    **nested** model there is one: its parameters are the parent's with a coefficient held at
    zero, so every point of the child is a point of the parent and the parent's maximum bounds
    all of them. A child reporting better than that is reporting a wrong number, and the floor
    makes the whole artefact unreachable *during* the fit rather than refusing it afterwards --
    which is the difference between a fit that converges and one that runs for three hours and is
    thrown away.

    The gradient is zero because there is nothing there to differentiate; the optimisers only
    use it to shorten a step they are already rejecting.
    """
    if _possible(value) and (floor is None or value >= floor):
        return None
    return float("inf"), np.zeros_like(x)


def _possible(value: float) -> bool:
    """Whether a mean negative log-likelihood could take this value: finite and not negative."""
    return bool(np.isfinite(value)) and value >= 0.0


def _limits(columns: pd.MultiIndex, primary: str) -> list[tuple[float, float]]:
    """A bound for every parameter: wide on the scale's coefficients, tight on the shape.

    The two are not comparable. A scale coefficient of 100 is absurd but harmless to evaluate;
    the shape sits in an exponent, and the same number overflows the cumulative hazard and takes
    the objective with it.
    """
    shape = [
        _SHAPE_BOUND if name != primary else _PARAMETER_BOUND
        for name in columns.get_level_values(0)
    ]
    return [(-bound, bound) for bound in shape]


def _check_interior(x: np.ndarray, limits: list[tuple[float, float]]) -> None:
    """Refuse a point sitting on a bound: that is not a maximum of the likelihood.

    The bounds keep the optimiser inside the region where lifelines' objective is a likelihood;
    they are not constraints on the model, and on this book they cannot bind -- 125 converged
    fits put the shape six times inside its own. A fit that ends on one is telling us the model
    is not identified, and saying so is more use than a coefficient of exactly 100.
    """
    values = np.asarray(x, dtype=float)
    edges = np.array([bound for _, bound in limits])
    at_bound = np.flatnonzero(np.abs(values) >= edges * 0.99)
    if at_bound.size:
        message = (
            f"Parameter(s) {at_bound.tolist()} reached their bound "
            f"({edges[at_bound].tolist()}), so the fit is at the edge of where the likelihood "
            "can be evaluated rather than at a maximum. The specification is not identified."
        )
        raise exceptions.ConvergenceError(message)


#: How large a coefficient the optimiser may consider, on standardised covariates.
#:
#: lifelines leaves an AFT model's coefficients unbounded -- its ``_bounds`` are for the
#: univariate fitters -- and that is where the runs went wrong. The likelihood it writes clips
#: the interval probability at 1e-25 but adds the truncation term unclipped, so far from the
#: data the objective -- a mean *negative* log-likelihood, which cannot be negative -- goes
#: negative and flat. Every failure of the prepayment model ended there: SLSQP at coefficients
#: of 1e+80, L-BFGS-B ``ABNORMAL``, and trust-constr returning a point where the next
#: evaluation read **-8.97e+69**.
#:
#: The covariates are divided by their own standard deviation before the fit, so a coefficient
#: of 100 means a scale factor of e^100 per standard deviation. The bound is three orders of
#: magnitude outside anything a credit model can mean and ten orders inside the region where
#: the objective stops being one: it cannot bind at an optimum, and `_check_interior` refuses
#: the fit if it ever does.
_PARAMETER_BOUND: Final = 100.0

#: How far the **shape** parameter may go, on the log scale it is estimated on.
#:
#: This is the bound that matters, and the coefficient bound above was aimed at the wrong
#: parameter. The cumulative hazard is ``exp(rho * (log t - log lambda))``: the shape sits in
#: an exponent, so it needs only reach exp(5) for the hazard to overflow and take lifelines'
#: objective with it -- while a scale coefficient of the same size does nothing of the kind.
#:
#: Three on the log scale means a shape between 0.05 and 20. Measured across **125 converged
#: fits** on this book -- default and prepayment, Weibull and log-logistic -- the log shape lies
#: between +0.070 and +0.484, a shape of 1.07 to 1.62. The bound is six times outside the widest
#: of those on the log scale, and a mortgage whose hazard bends twenty times faster than its age
#: is not a mortgage. It cannot bind on this book; if it ever does, the fit is refused as not
#: identified rather than published.
_SHAPE_BOUND: Final = 3.0

#: Consecutive refused points after which the fit is given up.
#:
#: An optimiser turned back this many times in a row is **pinned against the floor**: the only
#: direction it can find an improvement in is the impossible one, so the maximum of the
#: likelihood as lifelines computes it lies in the region lifelines cannot compute. That is a
#: conclusion about the specification, and waiting for the iteration cap to confirm it costs
#: hours -- the prepayment model's step 8 spent 85 minutes on 17 straight refusals without one
#: accepted point, with four more hours to go.
#:
#: It cannot change which model is chosen: a fit that ends this way is refused either way, and
#: step 8 keeps the covariate under rule 11. It decides only how long the run waits to say what
#: the log already shows.
_PINNED_REFUSALS: Final = 25

#: Evaluations looked at when deciding whether the optimiser is pinned, and the share of them
#: refused that says it is.
#:
#: `_PINNED_REFUSALS` counts refusals **in a row**, and the prepayment model's step 8 showed the
#: shape it cannot see: a cycle of six or seven refusals with one accepted point among them, which
#: resets the count and never reaches twenty-five. That fit ran to 211 evaluations, 4.3 hours, and
#: was refused at the end as it would have been at the start.
#:
#: Over a window the two states separate cleanly. On that run the share refused was 73% across the
#: whole of SLSQP and rose to 88% once the cycle set in, while **no** window of forty evaluations
#: fell below 35% after it began -- and the early, productive phase ran at 0%. Forty at
#: three-quarters therefore fires at evaluation 82 rather than 211, a cut of 2.6 times, with the
#: threshold twice the worst the productive phase produced.
#:
#: It is paired with a second condition -- the best accepted objective has not improved inside the
#: window -- which can only hold the guard back, never trip it. A fit still finding better points
#: is not pinned however much of its search is refused.
_PINNED_WINDOW: Final = 40

_PINNED_SHARE: Final = 0.75

#: The polish stops once no Newton step larger than this, in standard errors, remains.
POLISH_TOLERANCE_SE: Final = 1e-3

#: How often a long optimisation reports that it is still moving, in seconds.
_PROGRESS_SECONDS: Final = 60.0
