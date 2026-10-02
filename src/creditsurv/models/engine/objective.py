"""The objective over the rows this process holds, however the arithmetic is done.

lifelines divides each call's log-likelihood by the weight it saw, so a block's value is its own
mean; weighting each by its share of the **global** weight turns the sum of block means back
into the mean over every row. In a pool that share is a quarter with four processes, which is
why anything compared against the whole objective -- the floor, the progress line -- belongs to
the pooled evaluator and not here.

Either autograd over a stored design, or :mod:`creditsurv.models.kernel` over two tables. That
is the only difference between the two paths, and everything else in this class is the contract
the optimiser relies on.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from autograd import hessian, value_and_grad

from creditsurv.models.engine.contract import (
    _PROGRESS_SECONDS,
    _check_pinned,
    _outside_the_domain,
    _Pinned,
    _possible,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from lifelines.fitters import ParametericAFTRegressionFitter

    from creditsurv.models.engine.storage import (
        _Block,
    )
    from creditsurv.models.kernel import Kernel

    #: What the parent asks a worker for, and what a worker sends back.
    Command = tuple[str, np.ndarray | None]
    Prepared = tuple[pd.MultiIndex, np.ndarray, float, dict[str, np.ndarray]]
    Answer = dict[str, Any] | tuple[float, np.ndarray] | np.ndarray
    #: A worker's answer with the part it read, so the parent can add them in one order.
    Tagged = tuple[int, Answer]

log = logging.getLogger(__name__)


class _Objective:
    """The negative mean log-likelihood and its derivatives, added up block by block.

    lifelines divides each call's log-likelihood by the weight it saw, so a block's value
    is its own mean. Weighting each by its share of the total weight turns the sum of
    block means back into the mean over every row -- the number lifelines optimises.
    """

    def __init__(
        self,
        fitter: ParametericAFTRegressionFitter,
        blocks: list[_Block],
        columns: pd.MultiIndex,
        scale: np.ndarray,
        unflatten: Callable[[np.ndarray], dict[str, np.ndarray]],
        *,
        total_weight: float | None = None,
        with_penalty: bool = True,
        floor: float | None = None,
        pooled: bool = False,
        kernel: Kernel | None = None,
    ) -> None:
        # Inside a pool this objective sees **a share of the rows**, so its value is a share of
        # the objective: with four processes, a quarter. Neither the floor nor the progress line
        # belongs here then -- the floor is a bound on the whole objective, and comparing it with
        # a quarter of one refused every nested fit that was perfectly good. The pooled evaluator
        # owns both.
        self._pooled = pooled
        self.floor = None if pooled else floor
        self._blocks = blocks
        # The arithmetic, and the only thing that differs between the two paths. Everything
        # from here down -- the penalty, the count, the wall, the floor, the progress line and
        # the pinned guards -- is algebra on a handful of numbers and belongs to neither.
        self._kernel = kernel
        self._columns = columns
        self._scale = scale
        # Given when the rows are split across processes: a block's share is of every row
        # fitted, not of the rows this process happens to hold.
        self.total_weight = total_weight or float(sum(block.weight_sum for block in blocks))
        negative = partial(
            fitter._create_neg_likelihood_with_penalty_function,
            likelihood=fitter._log_likelihood_interval_censoring,
        )
        self._value_and_gradient = value_and_grad(negative)
        self._hessian = hessian(negative)

        penalizer = fitter.penalizer
        self._penalty: Callable[[np.ndarray], Any] | None = None
        if with_penalty and (isinstance(penalizer, np.ndarray) or penalizer > 0):

            def penalty(x: np.ndarray) -> float:
                # A cast, not float(): while autograd traces this the value is a box, and
                # converting it would cut the gradient off.
                return cast("float", fitter._add_penalty(unflatten(x), 0.0))

            self._penalty = penalty

        self.evaluations = 0
        self.pinned = _Pinned()
        self._started = time.perf_counter()
        self._reported = -np.inf

    def _terms(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """The unpenalised objective and its gradient over this process's rows.

        Either autograd over a stored design, or the written-out kernel over two tables. A
        block's own value is its mean, so each is weighted by its share of the **global**
        weight -- a quarter of it in a pool of four -- and the kernel is handed that same total,
        so both paths return a share of one objective rather than the whole of a smaller one.
        """
        if self._kernel is not None:
            totals = self._kernel(x)
            return totals.value, totals.gradient
        value = 0.0
        gradient = np.zeros_like(x)
        for block in self._blocks:
            share = block.weight_sum / self.total_weight
            block_value, block_gradient = self._value_and_gradient(
                x, *block.arguments(self._columns, self._scale)
            )
            value += share * float(block_value)
            gradient += share * block_gradient
        return value, gradient

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = self._terms(x)
        if self._penalty is not None:
            penalty_value, penalty_gradient = value_and_grad(self._penalty)(x)
            value += float(penalty_value)
            gradient += penalty_gradient

        self.evaluations += 1
        elapsed = time.perf_counter() - self._started
        refused = _outside_the_domain(value, x, self.floor)
        self.pinned.saw(refused=refused is not None, value=value)
        if not self._pooled and (
            refused is not None or elapsed - self._reported >= _PROGRESS_SECONDS
        ):
            # A refused point is always logged, whatever the interval: it is the surface falling
            # away, and reading a run without seeing that happen is misleading. The two reasons
            # are named apart, because they say different things -- a negative value is
            # lifelines' clipped likelihood breaking, and a value under the floor is a nested
            # model claiming to beat its parent.
            self._reported = elapsed
            why = ""
            if refused is not None:
                why = (
                    " (refused: below the parent's optimum, reported as infinite)"
                    if _possible(value)
                    else " (refused: not a likelihood, reported as infinite)"
                )
            log.info(
                "evaluation %d: objective %.12f%s at %.0fs",
                self.evaluations,
                value,
                why,
                elapsed,
            )
        if not self._pooled:
            _check_pinned(self.pinned, self.floor)
        return refused or (value, gradient)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        if self._kernel is not None:
            curvature = self._kernel(x, curvature=True).curvature
            assert curvature is not None
            total = curvature
        else:
            total = np.zeros((len(x), len(x)))
            for block in self._blocks:
                share = block.weight_sum / self.total_weight
                total = total + share * self._hessian(
                    x, *block.arguments(self._columns, self._scale)
                )
        if self._penalty is not None:
            total = total + hessian(self._penalty)(x)
        return total
