"""The fit itself: read the rows, seed, optimise, polish, certify, store.

Two entry points, and the difference is only where the rows come from.
:func:`fit_interval_censoring_in_blocks` reads them; :func:`fit_encoded` takes a reading that
has already been made and builds this model's design from its keys. Both end in the same place:
a fitted lifelines model whose distance to the optimum the polish has certified, and a record of
what the fit saw and what it cost.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from autograd.misc import flatten
from lifelines import exceptions, utils
from scipy.optimize import minimize

from creditsurv.models.engine.contract import (
    POLISH_TOLERANCE_SE,
    Pinned,
    _check_interior,
    _Evaluator,
    _limits,
    _possible,
)
from creditsurv.models.engine.lifelines_glue import (
    _initial_point,
    _seed_regressors,
    _set_censoring,
    _store,
    _warm_start,
)
from creditsurv.models.engine.objective import (
    _Objective,
)
from creditsurv.models.engine.polish import (
    _from_optimiser,
    _methods,
    _newton_steps_first,
)
from creditsurv.models.engine.scan import (
    Encoding,
    _kernel,
    _Scan,
    _scan,
    _standard_deviation,
)
from creditsurv.models.engine.storage import (
    _CONSTANT,
)
from creditsurv.models.engine.workers import (
    _combine,
    _Pooled,
    _Setup,
    _Workers,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from lifelines.fitters import ParametericAFTRegressionFitter
    from scipy.optimize import OptimizeResult

    #: What the parent asks a worker for, and what a worker sends back.
    Command = tuple[str, np.ndarray | None]
    Prepared = tuple[pd.MultiIndex, np.ndarray, float, dict[str, np.ndarray]]
    Answer = dict[str, Any] | tuple[float, np.ndarray] | np.ndarray
    #: A worker's answer with the part it read, so the parent can add them in one order.
    Tagged = tuple[int, Answer]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlockFit:
    """What a block fit saw and what it cost, for the caller's record."""

    rows: int
    blocks: int
    loan_months: float
    events: float
    stored_bytes: int
    evaluations: int
    seconds: float
    #: ``slsqp`` for lifelines' optimiser, ``newton`` when Newton steps ran from a warm start.
    method: str
    #: How far from the optimum the Newton steps began -- where SLSQP stopped, or the warm
    #: start -- as the largest step available, in standard errors of the coefficient it
    #: would move. See :func:`_polish`.
    stopping_error_se: float
    #: Newton steps taken after SLSQP stopped, and what they left, in the same units.
    polish_steps: int
    residual_error_se: float
    #: How many contiguous parts the compiled kernel summed the rows in, or 1 for the NumPy
    #: path. Recorded because it is **part of what determines the answer**: two runs at the
    #: same count agree bit for bit, and two counts agree only to the last digits of a sum
    #: over 72.7 million terms. A fit that cannot say what summed it cannot be reproduced.
    threads: int = 1


def fit_interval_censoring_in_blocks(
    fitter: ParametericAFTRegressionFitter,
    blocks: Iterable[pd.DataFrame] | Callable[[int, int], Iterable[pd.DataFrame]],
    *,
    formula: str,
    lower_bound_col: str,
    upper_bound_col: str,
    event_col: str,
    entry_col: str | None = None,
    weights_col: str | None = None,
    ancillary: str | bool | pd.DataFrame | None = None,
    initial_point: np.ndarray | dict[str, np.ndarray] | pd.Series | None = None,
    fit_options: dict[str, Any] | None = None,
    show_progress: bool = False,
    polish: bool = True,
    workers: int = 1,
    prefer: str | None = None,
    floor: float | None = None,
    calendar: Sequence[str] | None = None,
) -> BlockFit:
    """``fitter.fit_interval_censoring``, reading the rows a block at a time.

    ``blocks`` is any iterable of frames with the covariates and the bound, event, entry
    and weight columns -- a generator is fine, and is the point: the rows need never all
    exist as one frame. Text covariates must already be categorical *across all blocks*,
    because a block's design takes its dummy columns from the levels its column declares.

    The fitter is left fitted, as lifelines leaves it. What is returned is the record of
    the fit: rows, loan-months, events, the memory the stored rows took and the time.

    ``initial_point`` takes what lifelines takes, and also a ``params_`` series from
    another fit -- coefficients on their natural scale, keyed by parameter and covariate
    -- which seeds the columns the two models share. See :func:`_warm_start`.

    ``polish`` carries the fit from where SLSQP stops to the optimum -- see :func:`_polish`.
    Off, the result is the one lifelines itself returns, which is what the equivalence
    tests compare.

    ``workers`` above one evaluates the blocks in that many processes. ``blocks`` must then
    be a **description** of where the rows come from -- callable as ``blocks(part, of)`` and
    picklable, such as :class:`creditsurv.data.panel.CellBlocks` -- because each worker reads
    its own share rather than being sent one: sending the blocks would cost their memory
    twice, and autograd traces the likelihood in Python, so threads would share one core.
    """
    if isinstance(ancillary, pd.DataFrame):
        message = "An ancillary DataFrame cannot be read block by block; pass a formula or True."
        raise TypeError(message)

    started = time.perf_counter()
    names = (lower_bound_col, upper_bound_col, event_col, entry_col, weights_col)
    _set_censoring(fitter, names)

    if calendar is not None and workers > 1:
        # The partition of the design's columns is discovered from the data, so two processes
        # scanning different shares could classify a column differently and index into the
        # parameter vector in two different ways -- silently. Agreeing it across processes is
        # work the kernel makes unnecessary rather than work worth doing: it holds the whole
        # training half in 0.81 GB and does a Hessian over it in 30 s, which is what the four
        # processes were for.
        message = (
            "The written-out kernel runs in one process. It needs no more: the rows it holds "
            "are fifteen bytes each, so the whole training half is under a gigabyte, where "
            "the stored design was 1.79 GB across four workers. Pass workers=1."
        )
        raise ValueError(message)

    if workers > 1 and not callable(blocks):
        message = (
            "Fitting in several processes needs a description of where the rows come from, "
            "callable as blocks(part, of), not an iterator of them."
        )
        raise TypeError(message)
    source = cast("Callable[[int, int], Iterable[pd.DataFrame]]", blocks)
    setup = _Setup(type(fitter), fitter.penalizer, formula, ancillary, names)
    pool = _Workers(setup, source, workers) if workers > 1 else None
    # **The pool is closed on every path out of here, not only on the one that works.**
    # `close()` used to sit just before `_store`, after every `raise` in the body: a fit
    # refused by the domain check, by `_check_interior`, by the polish or by the method
    # chain left `workers - 1` processes alive, each parked in `commands.get()` still
    # holding its share of the stored blocks -- and `procedure._estimate` then started a
    # cold retry beside them. They are daemons, so they die with the parent; a selection
    # makes twenty-odd fits before the parent exits.
    try:
        scan = _scan(
            fitter,
            source(0, workers) if callable(blocks) else blocks,
            seed=_seed_regressors(fitter, formula, ancillary),
            names=names,
            calendar=calendar,
        )
        if pool is not None:
            _combine(scan, pool.summaries())
        return _fit_from_scan(
            fitter,
            scan,
            pool=pool,
            initial_point=initial_point,
            fit_options=fit_options,
            show_progress=show_progress,
            polish=polish,
            prefer=prefer,
            floor=floor,
            started=started,
        )
    finally:
        if pool is not None:
            pool.close()


def _fit_from_scan(
    fitter: ParametericAFTRegressionFitter,
    scan: _Scan,
    *,
    pool: _Workers | None = None,
    initial_point: np.ndarray | dict[str, np.ndarray] | pd.Series | None = None,
    fit_options: dict[str, Any] | None = None,
    show_progress: bool = False,
    polish: bool = True,
    prefer: str | None = None,
    floor: float | None = None,
    started: float | None = None,
    threads: int = 1,
) -> BlockFit:
    """Everything a fit does once the rows have been read: seed, optimise, polish, store.

    Separated from the reading because the reading is the expensive half -- 12.1 minutes
    against 53 seconds of arithmetic on the production table -- and a selection does it
    once per candidate. A scan carrying an encoding rather than stored blocks can be built
    once and handed here for every model whose covariates it covers.
    """
    started = time.perf_counter() if started is None else started
    if scan.rows < 2 or scan.columns is None:
        message = "A fit needs at least two rows."
        raise ValueError(message)
    columns = scan.columns

    raw = _standard_deviation(scan)
    raw_std = pd.Series(raw, index=columns)
    fitter.regressors = scan.regressors
    fitter._n_examples = scan.rows
    fitter._cols_to_not_penalize = fitter._find_cols_to_not_penalize(raw_std)
    norm_std = raw_std.copy()
    norm_std[norm_std < _CONSTANT] = 1.0
    fitter._norm_std = norm_std

    bounds = pd.concat(scan.bounds).groupby(level=[0, 1, 2]).sum().reset_index()
    fitter.timeline = np.unique(bounds["lower"].to_numpy(dtype=float))

    seeded = _initial_point(fitter, columns, raw, bounds)
    fitter._initial_point_dicts = [seeded]
    start, unflatten = flatten(seeded)
    if isinstance(initial_point, pd.Series):
        start = flatten(_warm_start(seeded, columns, norm_std, initial_point))[0]
    elif isinstance(initial_point, dict):
        start = flatten(initial_point)[0]
    elif initial_point is not None:
        start = np.asarray(initial_point, dtype=float)
    if start.shape[0] != columns.size:
        message = "initial_point is not the correct shape."
        raise ValueError(message)

    likelihood = fitter._log_likelihood_interval_censoring
    fitter._neg_likelihood_with_penalty_function = partial(
        fitter._create_neg_likelihood_with_penalty_function,
        likelihood=likelihood,
        penalty=fitter._add_penalty,
    )
    fitter._neg_likelihood = partial(
        fitter._create_neg_likelihood_with_penalty_function, likelihood=likelihood
    )
    total_weight = float(scan.weight)
    kernel = (
        None
        if scan.factorisation is None
        else _kernel(fitter, scan, norm_std.to_numpy(), total_weight, threads)
    )
    local = _Objective(
        fitter,
        scan.blocks,
        columns,
        norm_std.to_numpy(),
        unflatten,
        total_weight=total_weight,
        with_penalty=pool is None,
        floor=floor,
        pooled=pool is not None,
        kernel=kernel,
    )
    objective: _Evaluator = local
    if pool is not None:
        pool.prepare(columns, norm_std.to_numpy(), total_weight, seeded)
        objective = _Pooled(fitter, local, pool, unflatten, floor=floor)

    # **Newton first, from wherever the fit starts, with the method chain behind it.**
    #
    # From a warm start that was always right: SLSQP rebuilds its curvature estimate from
    # nothing and takes as many evaluations as from a cold start, 27 against 27 on the fixture.
    # From a **cold** start it used to be unaffordable, because a Hessian cost 49 times a value
    # and a hundred of them was out of the question. The written-out kernel puts a Hessian at
    # about twice a value-and-gradient, and the arithmetic changes sides. Measured on the
    # production table, the same specification from lifelines' own seed:
    #
    #     SLSQP then the polish    43.79 min   142 evaluations   2 Newton steps
    #     Newton from the seed     10.18 min    16 evaluations   8 Newton steps
    #
    # to the same log-likelihood, -10,691,177.6879, certified 3.31e-05 standard errors from the
    # optimum. It begins 1.89e+03 standard errors out and the first six evaluations are refused
    # as not a likelihood -- the damped step probing lifelines' clipped region -- and then the
    # damping ladder walks it in: 1.26e3, 706, 423, 207, 58.4, 6.61, 0.107, 3.31e-05.
    #
    # The chain stays as the fallback, and anything the Newton attempt raises goes to it rather
    # than to the caller: this replaces the optimiser's path, so no fit that used to succeed may
    # fail. `polish=False` skips it, because that mode exists to reproduce lifelines exactly.
    solution = _newton_steps_first(objective, start) if polish else None
    if solution is None:
        limits = _limits(columns, fitter._primary_parameter_name)
        attempts: list[OptimizeResult] = []
        for method in _methods(fitter._scipy_fit_method, prefer):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = minimize(
                    objective,
                    start,
                    method=method,
                    jac=True,
                    bounds=limits,
                    options={
                        "disp": show_progress,
                        **(fitter._scipy_fit_options if method == fitter._scipy_fit_method else {}),
                        **(fit_options or {}),
                    },
                    callback=fitter._scipy_fit_callback,
                )
            attempts.append(results)
            if show_progress:
                # lifelines prints the optimiser's result under the same flag.
                print(results)
            # **The polish decides whether a method worked, not the method's own flag.** Two
            # cases the flag gets wrong, both met on the production table. lifelines caps SLSQP
            # at 200 iterations, and a fit that reaches the cap comes back `success=False`
            # although it is *at* the answer -- this one had been stable to nine significant
            # figures for twenty evaluations. And trust-constr came back `success=True` at a
            # point whose next evaluation read -8.97e+69. What settles it is the distance to
            # the optimum, which only the polish measures.
            if not _possible(float(results.fun)):
                # **A statement about the surface, not about the method.** Ending outside the
                # likelihood means the optimiser found nothing better than the wall: the
                # specification has the spurious minimum lifelines' unclipped truncation term
                # creates, and every other method finds it too -- measured six times out of six
                # on the prepayment model, warm and cold, with both bounds in place. Trying the
                # rest costs hours and tells us what we already know, so this stops here and the
                # caller decides (step 8 keeps the covariate: rule 11 of docs/rules.md).
                message = (
                    f"{method} ended outside the likelihood ({results.message}) after "
                    f"{objective.evaluations} evaluations of {scan.rows:,} rows. The objective "
                    "is unbounded below on this specification -- lifelines clips the interval "
                    "probability and adds the truncation term unclipped -- so no optimiser can "
                    "maximise it and another method would find the same region."
                )
                raise exceptions.ConvergenceError(message)
            try:
                _check_interior(results.x, limits)
                solution = _from_optimiser(objective, results, polish=polish)
            except Pinned:
                # The floor stops every optimiser in the same place: see Pinned.
                raise
            except exceptions.ConvergenceError as error:
                log.warning("%s could not be polished (%s); trying the next", method, error)
                continue
            if not results.success:
                log.info(
                    "%s stopped short (%s) and the polish finished it", method, results.message
                )
            method_used = method
            break
        else:
            reports = "\n\n".join(f"minimum_results={attempt}" for attempt in attempts)
            message = (
                f"Fitting did not converge after {objective.evaluations} evaluations of "
                f"{scan.rows:,} rows in {len(scan.blocks)} blocks, under "
                f"{len(attempts)} method(s).\n\n{reports}"
            )
            raise exceptions.ConvergenceError(message)
        if method_used != fitter._scipy_fit_method:
            log.info("optimised with %s where %s failed", method_used, fitter._scipy_fit_method)
        method = str(method_used).lower()
    else:
        method = "newton"

    x, value, curvature, steps, stopped, remaining = solution
    if polish and remaining > POLISH_TOLERANCE_SE:
        message = (
            f"The fit ended {remaining:.3g} standard errors from the optimum, past the "
            f"{POLISH_TOLERANCE_SE:g} this engine promises."
        )
        raise exceptions.ConvergenceError(message)
    if not _possible(value):
        message = (
            f"The fit ended at an objective of {value:.6g}, which no likelihood can take: the "
            "optimiser left the region where lifelines computes the likelihood exactly."
        )
        raise exceptions.ConvergenceError(message)
    log.info(
        "%s: Newton steps began %.3g standard errors from the optimum; %d left %.3g",
        method,
        stopped,
        steps,
        remaining,
    )
    _store(fitter, columns, x, value, curvature, objective, unflatten)
    return BlockFit(
        rows=scan.rows,
        blocks=len(scan.blocks) + len(scan.encoded) + scan.other_blocks,
        loan_months=objective.total_weight,
        events=scan.events,
        stored_bytes=(
            sum(block.nbytes for block in scan.blocks)
            + sum(block.nbytes for block in scan.encoded)
            + scan.other_bytes
        ),
        evaluations=objective.evaluations,
        seconds=time.perf_counter() - started,
        method=method,
        stopping_error_se=stopped,
        polish_steps=steps,
        residual_error_se=remaining,
        threads=threads if kernel is not None else 1,
    )


def fit_encoded(
    fitter: ParametericAFTRegressionFitter,
    encoding: Encoding,
    *,
    formula: str,
    ancillary: str | bool | None = None,
    initial_point: np.ndarray | dict[str, np.ndarray] | pd.Series | None = None,
    fit_options: dict[str, Any] | None = None,
    show_progress: bool = False,
    polish: bool = True,
    prefer: str | None = None,
    floor: float | None = None,
    threads: int = 1,
) -> BlockFit:
    """Fit one model from rows that were read once, without reading them again.

    The model's design is two tables, built by putting its formula through the key frames --
    3,001 loan combinations and 153,309 calendar keys on the production table -- and its
    column moments come from the counts the encoding kept, because a column is a function of
    one side and its sum over every row is the sum over combinations of its value times its
    count.
    """
    started = time.perf_counter()
    _set_censoring(fitter, encoding.names)
    regressors = utils.CovariateParameterMappings(
        _seed_regressors(fitter, formula, ancillary),
        encoding.factorisation.probe(),
        force_intercept=fitter.fit_intercept,
        force_no_intercept=fitter.force_no_intercept,
    )
    expanded = encoding.factorisation.expand(
        regressors.transform_df, primary=fitter._primary_parameter_name
    )
    # The design checks lifelines makes, on the two tables rather than on 72 million rows:
    # a design column is a function of one side, so a NaN in it is a NaN in a key.
    utils.check_nans_or_infs(expanded.loan)
    utils.check_nans_or_infs(expanded.calendar)
    scan = _Scan(
        regressors=regressors,
        columns=expanded.columns,
        categories=dict(encoding.categories),
        encoded=list(encoding.rows),
        factorisation=encoding.factorisation,
        expanded=expanded,
        rows=encoding.episodes,
        first=expanded.first,
        second=expanded.second,
        low=expanded.low,
        high=expanded.high,
        bounds=[encoding.bounds],
        events=encoding.events,
        weight=encoding.weight,
    )
    return _fit_from_scan(
        fitter,
        scan,
        initial_point=initial_point,
        fit_options=fit_options,
        show_progress=show_progress,
        polish=polish,
        prefer=prefer,
        floor=floor,
        started=started,
        threads=threads,
    )
