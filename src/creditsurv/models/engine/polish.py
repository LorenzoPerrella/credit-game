"""Finishing the job the optimiser stops short of, and deciding whether it worked.

SLSQP stops on a change of 1e-10 in the *mean* log-likelihood, a tolerance that takes no account
of how precisely the data pin a coefficient down: on four quarters of this book it stopped up to
**5.9 standard errors** from the maximum, and finishing the job gained 20 log-likelihood units.
So every fit is polished with damped Newton steps until less than a thousandth of a standard
error remains, and **a fit the polish cannot move is refused rather than published** -- without
that, one was cached 6,850 standard errors out while the log said the polish had finished it.

The damping is not decoration. From a warm start a full Newton step once went 8.31e5 standard
errors out, to an objective of -4604, and was taken **because it was lower**: the step is now
taken only to a value a likelihood can have, damped until it lowers the objective.

And since a Hessian costs about twice a value-and-gradient rather than forty-nine times a value,
**Newton goes first from wherever a fit starts**, with the method chain behind it as the
fallback: 10.18 minutes and 16 evaluations against 43.79 and 142 on the production table, to the
same log-likelihood.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from lifelines import exceptions

from creditsurv.models.engine.contract import (
    POLISH_TOLERANCE_SE,
    Pinned,
    _Evaluator,
    _possible,
)

if TYPE_CHECKING:
    import pandas as pd
    from scipy.optimize import OptimizeResult

    #: What the parent asks a worker for, and what a worker sends back.
    Command = tuple[str, np.ndarray | None]
    Prepared = tuple[pd.MultiIndex, np.ndarray, float, dict[str, np.ndarray]]
    Answer = dict[str, Any] | tuple[float, np.ndarray] | np.ndarray
    #: A worker's answer with the part it read, so the parent can add them in one order.
    Tagged = tuple[int, Answer]

log = logging.getLogger(__name__)


def _methods(first: str, prefer: str | None) -> tuple[str, ...]:
    """The optimisers to try, in order, starting with the one that last worked.

    A fit whose design defeats SLSQP is usually beside another one just like it -- the
    backward elimination refits nearly the same model at every step -- and walking the whole
    chain each time is expensive: on the prepayment model, SLSQP spent 25 minutes failing,
    L-BFGS-B 40 more, and trust-constr then took an hour to answer. Starting from what worked
    last time saves the first two on every fit after the first.
    """
    ordered = (first, *_FALLBACK_METHODS)
    if prefer is None or prefer.lower() not in {name.lower() for name in ordered}:
        return ordered
    rest = [name for name in ordered if name.lower() != prefer.lower()]
    return (next(name for name in ordered if name.lower() == prefer.lower()), *rest)


#: Optimisers tried when lifelines' own stops without converging, in order.
#:
#: SLSQP is lifelines' choice and is the fastest here when it works. It solves a quadratic
#: subproblem at each step, and on an ill-conditioned design it reports **"Rank-deficient
#: equality constraint subproblem"** and gives up -- which is what the prepayment model did at
#: step 8 of its selection, from a cold start, at a perfectly finite objective of 56.58 with
#: 23 iterations behind it. The design is the default model's, which converges; what differs is
#: the curvature of a likelihood whose event rate is twenty times higher.
#:
#: L-BFGS-B builds no subproblem and no explicit curvature, so ill-conditioning costs it
#: iterations rather than stopping it; trust-constr is slower again and handles worse. **The
#: estimator is unchanged**: the same likelihood on the same rows has the same optimum, and the
#: damped Newton polish then certifies the answer is at it to under a thousandth of a standard
#: error -- which is what makes trying another path safe rather than a different model.
_FALLBACK_METHODS: Final[tuple[str, ...]] = ("L-BFGS-B", "trust-constr")


def _newton_step(
    curvature: np.ndarray, gradient: np.ndarray, total_weight: float
) -> tuple[np.ndarray, float]:
    """The Newton step in the optimiser's scaled space, and its size in standard errors.

    Measured coefficient by coefficient against that coefficient's own standard error, and
    the largest reported. The ratio does not depend on the scaling: step and error carry
    the same factor of the column's standard deviation.
    """
    step, *_ = np.linalg.lstsq(curvature, gradient, rcond=None)
    variance = np.diag(np.linalg.pinv(total_weight * curvature))
    usable = variance > 0
    if not usable.any():
        return step, 0.0
    return step, float(np.max(np.abs(step[usable]) / np.sqrt(variance[usable])))


def _damped_step(x: np.ndarray, gradient: np.ndarray, damped: np.ndarray) -> np.ndarray | None:
    """``x`` less the damped Newton step, or ``None`` where that matrix is not positive definite."""
    try:
        np.linalg.cholesky(damped)
    except np.linalg.LinAlgError:
        return None
    return np.asarray(x - np.linalg.solve(damped, gradient), dtype=float)


def _polish(
    objective: _Evaluator,
    x: np.ndarray,
    value: float,
    gradient: np.ndarray,
    curvature: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, int, float, float]:
    """Newton steps from where SLSQP stopped, until no step worth taking is left.

    SLSQP stops on a change of 1e-10 in the *mean* log-likelihood, a tolerance that takes no
    account of how precisely the data pin a coefficient down. Measured against the standard
    errors, it stopped up to 5.9 short of the optimum on four quarters of the book and 2.0
    on sixteen: the distance depends on where the optimiser happens to stop, and no sample
    size makes it negligible. The gradient and the Hessian are lifelines' own, added up over
    the blocks, so the point reached is the maximum of the same likelihood, reached more
    exactly.

    Each step solves ``(H + mu D) d = g``, with ``D`` the diagonal of the Hessian: plain
    Newton at ``mu = 0``, a short step along the scaled gradient as ``mu`` grows. A step is
    taken only if it lowers the objective *to a value a likelihood can have*. The objective
    is a mean negative log-likelihood and cannot be negative, but lifelines clips the
    interval probability at 1e-25 and adds the truncation term unclipped, so far enough
    from the data it goes negative -- lower than any real fit, and flat. From a warm start
    on the training half the full step went 8.31e5 standard errors, to -4604, and was taken
    because it was lower; a flat enough cliff would have been reported as the optimum. A
    refused step raises ``mu`` tenfold and a taken one lowers it tenfold, back to plain
    Newton near the optimum.

    Returns the point, its objective and Hessian, the steps taken, and the distance from
    the optimum before and after, in standard errors.
    """
    _, stopped = _newton_step(curvature, gradient, objective.total_weight)
    remaining, steps, damping = stopped, 0, 0.0
    stalled = 0
    slack = 4 * np.finfo(float).eps
    while remaining > POLISH_TOLERANCE_SE and steps < _POLISH_STEPS:
        diagonal = np.diag(curvature)
        floor = 1e-12 * max(float(diagonal.max()), float(np.finfo(float).tiny))
        scale = np.diag(np.maximum(diagonal, floor))
        candidate, candidate_value, candidate_gradient = x, value, gradient
        while True:
            stepped = _damped_step(x, gradient, curvature + damping * scale)
            if stepped is not None:
                candidate = stepped
                candidate_value, candidate_gradient = objective(candidate)
                lower = candidate_value <= value + slack * abs(value)
                if _possible(candidate_value) and lower:
                    break
            damping = max(10.0 * damping, _DAMPING_FLOOR)
            if damping > _DAMPING_CEILING:
                log.warning("no Newton step lowers the objective; polish stopped")
                return x, value, curvature, steps, stopped, remaining
        x, value, gradient = candidate, candidate_value, candidate_gradient
        curvature = _symmetric(objective.hessian(x))
        previous, (_, remaining) = (
            remaining,
            _newton_step(curvature, gradient, objective.total_weight),
        )
        steps += 1
        log.info(
            "Newton step %d, damping %.0e: %.3g standard errors from the optimum",
            steps,
            damping,
            remaining,
        )
        stalled = stalled + 1 if _too_slow(remaining, previous, steps) else 0
        if stalled >= _STALL_STEPS:
            if objective.pinned.against_the_floor:
                # What stopped it is the parent's optimum, not this optimiser or this start:
                # SLSQP gave up 557 standard errors out and L-BFGS-B, from its own path, 762.
                message = (
                    f"The polish stopped {remaining:.3g} standard errors out, every step it "
                    "wanted refused below the parent's optimum. The maximum of the likelihood "
                    "as lifelines computes it lies in the region it cannot compute, so this "
                    "specification cannot be fitted."
                )
                raise Pinned(message)
            log.warning(
                "the polish has closed by %.3g a step for %d steps and needs %.3g to finish "
                "in the %d it has left; it is %.3g standard errors out and stopped",
                remaining / previous,
                stalled,
                _required_ratio(remaining, steps),
                _POLISH_STEPS - steps,
                remaining,
            )
            return x, value, curvature, steps, stopped, remaining
        damping = damping / 10.0 if damping >= 10.0 * _DAMPING_FLOOR else 0.0
    if remaining > POLISH_TOLERANCE_SE:
        log.warning("polish stopped after %d steps, %.3g standard errors out", steps, remaining)
    return x, value, curvature, steps, stopped, remaining


def _required_ratio(remaining: float, steps: int) -> float:
    """The factor a step must close by to finish inside the steps the polish has left."""
    budget = _POLISH_STEPS - steps
    if budget <= 0:
        return 0.0
    return float((POLISH_TOLERANCE_SE / remaining) ** (1.0 / budget))


def _too_slow(remaining: float, previous: float, steps: int) -> bool:
    """Whether this step closed by less than finishing inside the remaining budget asks.

    Compared against the budget rather than a fixed factor, because the two are the same
    question: a polish 975 standard errors out with 30 steps left has to close by 0.631 a step,
    and one 1.37e3 out with 34 left by 0.660. Closing by 0.925 fails both, and by the same
    margin it will fail every step after -- which is what makes the projection worth acting on
    rather than waiting for the cap.
    """
    if previous <= 0.0 or remaining <= POLISH_TOLERANCE_SE:
        return False
    return remaining / previous > _required_ratio(remaining, steps)


def _from_optimiser(
    objective: _Evaluator, results: OptimizeResult, *, polish: bool
) -> tuple[np.ndarray, float, np.ndarray, int, float, float]:
    """Where the optimiser stopped, polished to the optimum unless ``polish`` is off.

    The value and the gradient are **recomputed here** rather than read off the result. Each
    optimiser reports them its own way -- SLSQP's ``jac`` is the gradient, trust-constr's is
    shaped for its constraint machinery -- and reading them cost a run: trust-constr solved a
    prepayment fit SLSQP had given up on, and the polish then died on `LinAlgError:
    Incompatible dimensions`, an hour of fitting thrown away for a field's shape. One
    value-and-gradient is 2% of the Hessian this function computes anyway.
    """
    started = time.perf_counter()
    x = np.asarray(results.x, dtype=float)
    curvature = _symmetric(objective.hessian(x))
    log.info("hessian in %.0fs", time.perf_counter() - started)
    value, gradient = objective(x)
    if polish:
        solution = _polish(objective, x, value, gradient, curvature)
        remaining = solution[5]
        if remaining > POLISH_TOLERANCE_SE:
            # **The polish decides, and this is the deciding.** Without it a fit that the polish
            # could not move was accepted and cached: the prepayment model produced one sitting
            # 6,850 standard errors from the optimum, at an objective seventy times below any
            # real fit, and the log cheerfully said the polish had finished it.
            message = (
                f"The polish stopped {remaining:.3g} standard errors from the optimum, past the "
                f"{POLISH_TOLERANCE_SE:g} this engine promises, after {solution[3]} step(s). "
                "The point the optimiser reached is not a maximum of the likelihood."
            )
            raise exceptions.ConvergenceError(message)
        return solution
    _, stopped = _newton_step(curvature, gradient, objective.total_weight)
    return x, value, curvature, 0, stopped, stopped


def _newton_from(
    objective: _Evaluator, start: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, int, float, float] | None:
    """Damped Newton steps from a warm start, or ``None`` if they do not reach the optimum.

    The start need not be in a concave region -- a damped step is only taken where the
    damped Hessian is positive definite -- but the end must be: a point where the gradient
    vanishes and the curvature does not bend upwards is not a maximum of the likelihood.
    Otherwise the caller falls back on SLSQP, from the same start.
    """
    value, gradient = objective(start)
    if not _possible(value):
        log.info("the warm start's objective is not one a likelihood can take; using SLSQP")
        return None
    curvature = _symmetric(objective.hessian(start))
    solution = _polish(objective, start, value, gradient, curvature)
    if solution[5] > POLISH_TOLERANCE_SE:
        log.info("Newton steps did not converge from the warm start; falling back on SLSQP")
        return None
    if not bool(np.all(np.linalg.eigvalsh(solution[2]) > 0)):
        log.info("Newton steps ended where the likelihood is not concave; falling back on SLSQP")
        return None
    return solution


def _newton_steps_first(
    objective: _Evaluator, start: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, int, float, float] | None:
    """Damped Newton from the starting point, or ``None`` if it could not finish.

    Separated from :func:`_newton_from` so that a failure here is a fallback rather than a
    verdict. What the fallback costs is the attempt -- sixteen evaluations and eight Hessians
    on the production table -- and what it buys is that the optimiser's path stops being 96%
    of a cold fit. `Pinned` goes the same way as the rest: against a cold seed it says the
    damped step met the clipped region, which is what the ladder exists to climb out of, and
    the method chain is entitled to its own opinion.
    """
    try:
        solution = _newton_from(objective, start)
    except exceptions.ConvergenceError as error:
        log.info("Newton from the start did not finish (%s); trying the optimiser", error)
        return None
    if solution is None:
        return None
    *_, remaining = solution
    if remaining > POLISH_TOLERANCE_SE:
        log.info(
            "Newton from the start left %.3g standard errors, past the %.0e this engine "
            "promises; trying the optimiser",
            remaining,
            POLISH_TOLERANCE_SE,
        )
        return None
    return solution


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    """A Hessian symmetrised, as lifelines symmetrises it (lifelines issue 801)."""
    symmetric: np.ndarray = (matrix + matrix.T) / 2
    return symmetric


#: The most Newton steps a polish takes. From where SLSQP stops two or three do; from a warm
#: start that adds a covariate the curvature at the start is a poor guide, and the steps stay
#: damped until it catches up.
_POLISH_STEPS: Final = 40

#: Damping starts here when a step is refused and rises tenfold each time; past the ceiling
#: no step is left that lowers the objective.
_DAMPING_FLOOR: Final = 1e-6

_DAMPING_CEILING: Final = 1e12

#: Steps closing too slowly to reach the tolerance, after which the polish is given up.
#:
#: A damping that cannot come down turns Newton into a short gradient step, and the distance to
#: the optimum then falls by a **constant factor** a step instead of squaring. On the prepayment
#: model's first backward-elimination candidate it settled at 0.925 with the damping stuck at
#: 1e+02 -- 1.24e3 standard errors out, then 1.13e3, 1.05e3, 975 -- which needs **177 steps** to
#: reach 1e-3 against a cap of 40. The cap does end it, three hours later, with the same verdict
#: the fourth step already implied.
#:
#: The damping cannot come down because every step long enough to make progress lands under the
#: floor and is refused, so the damping rises to meet it. The refusals are real, but they are
#: **interleaved with accepted points**, one of each a step, which resets `_PINNED_REFUSALS` and
#: leaves it silent, so this pathology needs its own guard.
#:
#: What was refused there was not lifelines' unbounded region, as this comment first said. The
#: optimiser was converging, to a point 11.8 log-likelihood units better than its parent's
#: certified optimum on exactly the same rows, and the floor's allowance was one unit -- see
#: `creditsurv.models.procedure._NESTED_TOLERANCE`, which carries that measurement. The guard is
#: right either way: a polish closing by a constant factor cannot finish, whatever is holding it.
#:
#: It shortens a phase; it does not decide a fit. A warm start the Newton steps cannot finish
#: falls back on SLSQP from the same point, as it always has, and what happens to the fit is
#: settled there -- by `_check_pinned`, by the polish's verdict, or by converging after all.
#: Five is what it costs to sit out the wild early steps: on that run the count reached two by
#: step 4 and reset at step 5, where two halvings in a row were real progress.
_STALL_STEPS: Final = 5
