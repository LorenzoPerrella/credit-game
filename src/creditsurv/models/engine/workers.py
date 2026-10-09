"""The objective split across processes, and the pool that holds them.

Each worker reads **its own share** of the rows rather than being sent them: sending the blocks
would cost their memory twice, and autograd traces the likelihood in Python, so threads would
share one core. What crosses the pipe is a point going out and a value, a gradient or a
curvature coming back -- four hundred bytes and twenty kilobytes at the production width.

Two things here were learned the hard way and are the reason this is a module rather than a
loop. The answers are added **in the parts' own order**, never in the order they arrive:
floating-point addition is not associative, so the same point summed two ways differs in its
last digit, and an optimiser turns that into a different search -- two runs of one fit agreed to
every printed digit for eighty evaluations and were five significant figures apart forty later.
And the parent **evaluates its own share while the workers evaluate theirs**: it used to put the
commands and block, so the wall clock was a share plus a share and `of` processes were worth
`of/2`. Measured, 1,747 ms against 803 at four workers.
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import pandas as pd
from autograd import hessian, value_and_grad
from autograd.misc import flatten

from creditsurv.models.engine.contract import (
    _PROGRESS_SECONDS,
    _check_pinned,
    _outside_the_domain,
    _Pinned,
    _possible,
)
from creditsurv.models.engine.lifelines_glue import (
    _seed_regressors,
    _set_censoring,
)
from creditsurv.models.engine.objective import (
    _Objective,
)
from creditsurv.models.engine.scan import (
    _Scan,
    _scan,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from multiprocessing.queues import Queue

    from lifelines.fitters import ParametericAFTRegressionFitter

    #: What the parent asks a worker for, and what a worker sends back.
    Command = tuple[str, np.ndarray | None]
    Prepared = tuple[pd.MultiIndex, np.ndarray, float, dict[str, np.ndarray]]
    Answer = dict[str, Any] | tuple[float, np.ndarray] | np.ndarray
    #: A worker's answer with the part it read, so the parent can add them in one order.
    Tagged = tuple[int, Answer]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Setup:
    """What a worker needs to build the same design from its own share of the rows."""

    fitter_class: type[ParametericAFTRegressionFitter]
    penalizer: float
    formula: str
    ancillary: str | bool | None
    names: tuple[str, str, str, str | None, str | None]


def _summary(scan: _Scan) -> dict[str, Any]:
    """What a worker's scan tells the parent. Not the blocks: those stay where they are."""
    return {
        "rows": scan.rows,
        "first": scan.first,
        "second": scan.second,
        "low": scan.low,
        "high": scan.high,
        "events": scan.events,
        "weight": scan.weight,
        "bounds": pd.concat(scan.bounds).groupby(level=[0, 1, 2]).sum() if scan.bounds else None,
        "columns": scan.columns,
        "blocks": len(scan.blocks),
        "bytes": sum(block.nbytes for block in scan.blocks),
    }


def _combine(scan: _Scan, summaries: Iterable[dict[str, Any]]) -> None:
    """Add the workers' sums to the parent's, so the fit sees every row."""
    for summary in summaries:
        if summary["rows"] == 0:
            continue
        if scan.columns is not None and not summary["columns"].equals(scan.columns):
            message = "A worker's design columns differ from the parent's."
            raise ValueError(message)
        scan.rows += summary["rows"]
        scan.events += summary["events"]
        scan.weight += summary["weight"]
        scan.other_blocks += summary["blocks"]
        scan.other_bytes += summary["bytes"]
        if scan.first is None:
            scan.first, scan.second = summary["first"], summary["second"]
            scan.low, scan.high = summary["low"], summary["high"]
        else:
            assert scan.second is not None and scan.low is not None and scan.high is not None
            scan.first = scan.first + summary["first"]
            scan.second = scan.second + summary["second"]
            scan.low = np.minimum(scan.low, summary["low"])
            scan.high = np.maximum(scan.high, summary["high"])
        if summary["bounds"] is not None:
            scan.bounds.append(summary["bounds"])


def _serve(
    setup: _Setup,
    source: Callable[[int, int], Iterable[pd.DataFrame]],
    part: int,
    of: int,
    commands: Queue[Command | Prepared],
    results: Queue[Tagged],
) -> None:
    """A worker: read a share of the rows, then answer with its part of the objective.

    Runs in its own process. It reads its blocks itself rather than being sent them, keeps
    them for the whole fit, and adds no penalty: the parent adds that once.
    """
    fitter = setup.fitter_class(penalizer=setup.penalizer)
    _set_censoring(fitter, setup.names)
    scan = _scan(
        fitter,
        source(part, of),
        seed=_seed_regressors(fitter, setup.formula, setup.ancillary),
        names=setup.names,
    )
    results.put((part, _summary(scan)))

    columns, scale, total_weight, seeded = cast("Prepared", commands.get())
    raw_std = pd.Series(scale, index=columns)
    fitter.regressors = scan.regressors
    fitter._n_examples = scan.rows
    fitter._cols_to_not_penalize = fitter._find_cols_to_not_penalize(raw_std)
    fitter._norm_std = raw_std
    fitter._initial_point_dicts = [seeded]
    likelihood = fitter._log_likelihood_interval_censoring
    fitter._neg_likelihood_with_penalty_function = partial(
        fitter._create_neg_likelihood_with_penalty_function,
        likelihood=likelihood,
        penalty=fitter._add_penalty,
    )
    fitter._neg_likelihood = partial(
        fitter._create_neg_likelihood_with_penalty_function, likelihood=likelihood
    )
    objective = _Objective(
        fitter,
        scan.blocks,
        columns,
        scale,
        flatten(seeded)[1],
        total_weight=total_weight,
        with_penalty=False,
    )
    while True:
        command, payload = cast("Command", commands.get())
        if command == "stop" or payload is None:
            return
        if command == "value":
            results.put((part, objective(payload)))
        else:
            results.put((part, objective.hessian(payload)))


class _Workers:
    """The worker processes, each holding its own share of the rows for the whole fit."""

    def __init__(
        self,
        setup: _Setup,
        source: Callable[[int, int], Iterable[pd.DataFrame]],
        of: int,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self._results: Queue[Tagged] = context.Queue()
        self._commands: list[Queue[Command | Prepared]] = []
        self._processes: list[multiprocessing.process.BaseProcess] = []
        for part in range(1, of):
            commands: Queue[Command | Prepared] = context.Queue()
            process = context.Process(
                target=_serve,
                args=(setup, source, part, of, commands, self._results),
                daemon=True,
            )
            process.start()
            self._commands.append(commands)
            self._processes.append(process)
        log.info("%d worker process(es) reading their share of the rows", len(self._processes))

    def summaries(self) -> list[dict[str, Any]]:
        return [cast("dict[str, Any]", answer) for answer in self._collect()]

    def prepare(
        self,
        columns: pd.MultiIndex,
        scale: np.ndarray,
        total_weight: float,
        seeded: dict[str, np.ndarray],
    ) -> None:
        for commands in self._commands:
            commands.put((columns, scale, total_weight, seeded))

    def send(self, command: str, x: np.ndarray) -> None:
        """Hand every worker the point and **return**, so the parent can evaluate its own.

        The parent holds a share of the rows like every worker -- part 0 of `of` -- and this
        used to put the commands and then block on the answers, so the workers computed while
        the parent waited and then the parent computed while the workers waited. The wall clock
        was a share plus a share, which makes `of` processes worth `of/2`: measured 1,747 ms
        against 3,053 at four workers on 1.95 million rows of the production table, a 1.75x
        where the arithmetic is split four ways. Collecting is a separate call for that reason,
        and the collectors take no point: a caller cannot collect what it has not sent.
        """
        for commands in self._commands:
            commands.put((command, x))

    def _collect(self) -> list[Answer]:
        """The answers **in the order the rows were split**, never in the order they arrive.

        One queue serves every worker, so `get` returns whichever finished first, and the parent
        used to add them in that order. Floating-point addition is not associative, so the same
        point summed in two arrival orders differs in its last digit -- and an optimiser turns
        that into a different search: two runs of the identical fit agreed to every printed digit
        for eighty evaluations, then diverged at 0.065288491918 against 0.065288491919 and were
        five significant figures apart forty evaluations later. A fit of one specification on one
        cell file has to give one answer, so the answers are added in the parts' own order.

        **A worker that dies is a fit that cannot be completed, not a fit that waits.** The
        `get` here had no timeout, so a worker killed by the memory pressure this engine exists
        to manage left the parent blocked for ever, with nothing in the log after the last
        evaluation. There is no honest fixed deadline -- a first scan is minutes and one logged
        fit spent 457 of them on four evaluations -- so the wait polls instead, and only an
        answer that is missing *while the process that owed it has exited* ends the fit. Its
        share of the rows is gone, so a sum without it would be a different estimator.
        """
        answers: dict[int, Answer] = {}
        while len(answers) < len(self._processes):
            try:
                part, answer = self._results.get(timeout=_WORKER_POLL_SECONDS)
            except queue.Empty:
                gone = [
                    (number, process.exitcode)
                    for number, process in enumerate(self._processes, start=1)
                    if process.exitcode is not None
                ]
                if not gone:
                    continue
                reported = ", ".join(f"part {number} exited {code}" for number, code in gone)
                message = (
                    f"{len(gone)} of {len(self._processes)} worker process(es) stopped before "
                    f"answering ({reported}). Their share of the rows is gone, so the fit "
                    "cannot be finished: a sum over the parts that remain is a different "
                    "likelihood. A worker killed with no traceback of its own is usually out "
                    "of memory -- lower --workers or --block-rows."
                )
                raise RuntimeError(message) from None
            answers[part] = answer
        return [answers[part] for part in sorted(answers)]

    def value_and_gradient(self) -> list[tuple[float, np.ndarray]]:
        """The answers to the ``value`` already sent, in the parts' own order."""
        return [cast("tuple[float, np.ndarray]", answer) for answer in self._collect()]

    def curvature(self) -> list[np.ndarray]:
        """The answers to the ``hessian`` already sent, in the parts' own order."""
        return [cast("np.ndarray", answer) for answer in self._collect()]

    def discard(self) -> None:
        """Wait for the answers to a point nobody will use, so none is read at the next.

        One queue serves the whole pool, so an answer left in it after the parent's own share
        raised would be collected at the *next* evaluation -- a sum of two different points,
        which is worse than the failure that caused it.
        """
        self._collect()

    def close(self) -> None:
        for commands in self._commands:
            commands.put(("stop", None))
        for process in self._processes:
            process.join(timeout=30)


#: How long the parent waits for an answer before it looks at whether the workers are alive.
#: Not a deadline on the answer: an evaluation of the training half is minutes, and the
#: first collection waits for every worker's whole scan.
_WORKER_POLL_SECONDS: Final = 30.0


class _Pooled:
    """The objective over every row: this process's blocks plus the workers'."""

    def __init__(
        self,
        fitter: ParametericAFTRegressionFitter,
        local: _Objective,
        workers: _Workers,
        unflatten: Callable[[np.ndarray], dict[str, np.ndarray]],
        floor: float | None = None,
    ) -> None:
        self._local = local
        self._workers = workers
        self.floor = floor
        self.pinned = _Pinned()
        self._started = time.perf_counter()
        self._reported = -np.inf
        self.total_weight = local.total_weight
        self.evaluations = 0
        penalizer = fitter.penalizer
        self._penalty: Callable[[np.ndarray], Any] | None = None
        if isinstance(penalizer, np.ndarray) or penalizer > 0:

            def penalty(x: np.ndarray) -> float:
                return cast("float", fitter._add_penalty(unflatten(x), 0.0))

            self._penalty = penalty

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        # Sent first, so the workers evaluate their shares while this process evaluates its
        # own; `discard` is there because an answer left in the queue by a failed local share
        # would be collected at the next point.
        self._workers.send("value", x)
        try:
            value, gradient = self._local(x)
        except BaseException:
            self._workers.discard()
            raise
        remote = self._workers.value_and_gradient()
        for block_value, block_gradient in remote:
            value += float(block_value)
            gradient = gradient + block_gradient
        if self._penalty is not None:
            penalty_value, penalty_gradient = value_and_grad(self._penalty)(x)
            value += float(penalty_value)
            gradient = gradient + penalty_gradient
        self.evaluations += 1
        refused = _outside_the_domain(value, x, self.floor)
        self.pinned.saw(refused=refused is not None, value=value)
        elapsed = time.perf_counter() - self._started
        if refused is not None or elapsed - self._reported >= _PROGRESS_SECONDS:
            self._reported = elapsed
            why = ""
            if refused is not None:
                why = (
                    " (refused: below the parent's optimum, reported as infinite)"
                    if _possible(value)
                    else " (refused: not a likelihood, reported as infinite)"
                )
            log.info(
                "evaluation %d: objective %.12f%s at %.0fs", self.evaluations, value, why, elapsed
            )
        _check_pinned(self.pinned, self.floor)
        return refused or (value, gradient)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        self._workers.send("hessian", x)
        try:
            total = self._local.hessian(x)
        except BaseException:
            self._workers.discard()
            raise
        for block in self._workers.curvature():
            total = total + block
        if self._penalty is not None:
            total = total + hessian(self._penalty)(x)
        return total
