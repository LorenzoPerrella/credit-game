"""Fit a lifelines AFT model block by block, so memory stops growing with the population.

lifelines evaluates the likelihood of every row at once and differentiates it with
autograd, which keeps every intermediate array for the backward pass. On this project's
panel a Weibull fit was measured at about **680 bytes per training row** above the data,
in the optimiser and again in the Hessian: three filtered copies of the design matrix
live on the tape, on top of the two full copies ``_fit`` makes before optimising. At
9.5 million rows the process reached a 13.45 GB footprint and swapped. The exact
calendar key puts the training half at 62 million rows, which is 45-50 GB.

The log-likelihood is a sum over rows, and so are its gradient and its Hessian. Added up
block by block they are the same numbers -- up to the order in which floating-point
additions happen -- so the optimiser takes the same steps to the same optimum while only
one block's tape exists at a time. Nothing is approximated and nothing is sampled.

What is mirrored, from lifelines 0.30:

* ``ParametericAFTRegressionFitter.fit_interval_censoring`` -- the regressors, the checks
  on the bounds, the stand-in for an infinite upper bound;
* ``ParametricRegressionFitter._fit`` -- the design, the column scaling, and the
  attributes ``summary``, ``AIC_`` and ``predict_cumulative_hazard`` read;
* ``ParametricRegressionFitter._fit_model`` -- the objective, the optimiser and its
  options, the Hessian;
* ``ParametericAFTRegressionFitter._create_initial_point`` -- the univariate fit that
  seeds the intercept. It reads the rows only through their bounds and weights, so it
  runs on the distinct bounds with summed weights: a few hundred rows in place of
  millions, and the same likelihood.

All four are private, which is why ``tests/test_blocks.py`` fits the same rows both ways
and compares coefficients, standard errors, log-likelihood and predictions. Re-run it
before trusting this module with a different lifelines.

A model fitted here does not carry the full-length copies of its training data lifelines
attaches -- ``lower_bound``, ``upper_bound``, ``event_observed``, ``entry``, ``weights``
-- nor the predicted medians behind ``concordance_index_`` or the central values behind
the partial-effects plots. Nothing in the project reads them, the cached model is smaller
for it, and anything that does try fails with an ``AttributeError`` rather than a wrong
number.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any, Final, cast

import lifelines
import numpy as np
import pandas as pd
from autograd import hessian, value_and_grad
from autograd.misc import flatten
from lifelines import exceptions, utils
from scipy.optimize import minimize

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from lifelines.fitters import ParametericAFTRegressionFitter
    from scipy.optimize import OptimizeResult

log = logging.getLogger(__name__)

#: Rows evaluated at once. One block's autograd tape costs about 700 bytes a row, so a
#: million rows is 0.7 GB above the stored data: small against the machine, and large
#: enough that the Python overhead per block is noise against the arithmetic.
DEFAULT_BLOCK_ROWS: Final = 1_000_000

#: What lifelines substitutes for an infinite upper bound before evaluating anything.
INFINITY_STAND_IN: Final = 1e25

#: Below this a column's standard deviation counts as zero, as it does in lifelines.
_CONSTANT: Final = 1e-8

#: How often a long optimisation reports that it is still moving, in seconds.
_PROGRESS_SECONDS: Final = 60.0


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
        # Column-major, so each column is contiguous to write into and the frame below
        # can wrap the array without copying it.
        design = np.empty((self.rows, len(self.design)), dtype=np.float64, order="F")
        for position, column in enumerate(self.design):
            column.expand_into(design[:, position])
        design /= scale
        frame = pd.DataFrame(design, columns=columns, copy=False)
        bounds = (self.lower.expand(), self.upper.expand())
        return (
            bounds,
            self.exact,
            self.weight.expand(),
            self.entry.expand(),
            utils.DataframeSlicer(frame),
        )


@dataclass
class _Scan:
    """What one pass over the blocks learns."""

    regressors: Any = None
    columns: pd.MultiIndex | None = None
    categories: dict[str, pd.Index] = field(default_factory=dict)
    blocks: list[_Block] = field(default_factory=list)
    rows: int = 0
    first: np.ndarray | None = None
    second: np.ndarray | None = None
    low: np.ndarray | None = None
    high: np.ndarray | None = None
    bounds: list[pd.Series] = field(default_factory=list)
    events: float = 0.0


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


def fit_interval_censoring_in_blocks(
    fitter: ParametericAFTRegressionFitter,
    blocks: Iterable[pd.DataFrame],
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
    """
    if isinstance(ancillary, pd.DataFrame):
        message = "An ancillary DataFrame cannot be read block by block; pass a formula or True."
        raise TypeError(message)

    started = time.perf_counter()
    utils.CensoringType.set_censoring_type(fitter, utils.CensoringType.INTERVAL)
    fitter._time_fit_was_called = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S") + " UTC"
    fitter.lower_bound_col = lower_bound_col
    fitter.upper_bound_col = upper_bound_col
    fitter.event_col = event_col
    fitter.entry_col = entry_col
    fitter.weights_col = weights_col
    fitter.robust = False

    scan = _scan(
        fitter,
        blocks,
        seed=_seed_regressors(fitter, formula, ancillary),
        names=(lower_bound_col, upper_bound_col, event_col, entry_col, weights_col),
    )
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
    objective = _Objective(fitter, scan.blocks, columns, norm_std.to_numpy(), unflatten)

    # From a warm start Newton goes straight to the optimum. SLSQP would rebuild its
    # curvature estimate from nothing and take as many evaluations as from a cold start:
    # 27 against 27 on the test fixture.
    solution = (
        _newton_from(objective, start) if polish and isinstance(initial_point, pd.Series) else None
    )
    if solution is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = minimize(
                objective,
                start,
                method=fitter._scipy_fit_method,
                jac=True,
                options={
                    "disp": show_progress,
                    **fitter._scipy_fit_options,
                    **(fit_options or {}),
                },
                callback=fitter._scipy_fit_callback,
            )
        if show_progress:
            # lifelines prints the optimiser's result under the same flag.
            print(results)
        if not (results.fun < np.inf and results.success):
            message = (
                f"Fitting did not converge after {objective.evaluations} evaluations of "
                f"{scan.rows:,} rows in {len(scan.blocks)} blocks.\n\nminimum_results={results}"
            )
            raise exceptions.ConvergenceError(message)
        solution = _from_slsqp(objective, results, polish=polish)
        method = "slsqp"
    else:
        method = "newton"

    x, value, curvature, steps, stopped, remaining = solution
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
        blocks=len(scan.blocks),
        loan_months=objective.total_weight,
        events=scan.events,
        stored_bytes=sum(block.nbytes for block in scan.blocks),
        evaluations=objective.evaluations,
        seconds=time.perf_counter() - started,
        method=method,
        stopping_error_se=stopped,
        polish_steps=steps,
        residual_error_se=remaining,
    )


def _seed_regressors(
    fitter: ParametericAFTRegressionFitter, formula: str, ancillary: str | bool | None
) -> dict[str, str]:
    """The parameter-to-formula mapping ``fit_interval_censoring`` builds for a formula."""
    regressors = {fitter._primary_parameter_name: formula}
    if isinstance(ancillary, str):
        fitter.model_ancillary = True
        regressors[fitter._ancillary_parameter_name] = ancillary
    elif ancillary is True or fitter.model_ancillary:
        fitter.model_ancillary = True
        regressors[fitter._ancillary_parameter_name] = formula
    else:
        regressors[fitter._ancillary_parameter_name] = "1"
    return regressors


def _scan(
    fitter: ParametericAFTRegressionFitter,
    blocks: Iterable[pd.DataFrame],
    *,
    seed: dict[str, str],
    names: tuple[str, str, str, str | None, str | None],
) -> _Scan:
    """Build the design block by block, check it as lifelines would, and store it."""
    lower_bound_col, upper_bound_col, event_col, entry_col, weights_col = names
    special = [name for name in names if name is not None]
    scan = _Scan()

    for number, frame in enumerate(blocks):
        if frame.empty:
            continue
        lower = frame[lower_bound_col].to_numpy(dtype=np.float64)
        upper = frame[upper_bound_col].to_numpy(dtype=np.float64)
        exact = frame[event_col].to_numpy(dtype=bool)
        if ((lower == upper) != exact).any():
            message = (
                "For all rows, lower_bound == upper_bound if and only if event observed = 1 "
                "(uncensored). Likewise, lower_bound < upper_bound if and only if event "
                "observed = 0 (censored)"
            )
            raise ValueError(message)
        if (lower > upper).any():
            message = "All upper bound measurements must be >= lower bound measurements."
            raise ValueError(message)
        weights = (
            frame[weights_col].to_numpy(dtype=np.float64)
            if weights_col is not None
            else np.ones(len(frame))
        )
        entries = (
            frame[entry_col].to_numpy(dtype=np.float64)
            if entry_col is not None
            else np.zeros(len(frame))
        )

        covariates = frame.drop(columns=special)
        _check_categories(scan, covariates, number)
        if scan.regressors is None:
            scan.regressors = utils.CovariateParameterMappings(
                seed,
                covariates,
                force_intercept=fitter.fit_intercept,
                force_no_intercept=fitter.force_no_intercept,
            )
        design = scan.regressors.transform_df(covariates)
        if len(design) != len(frame):
            message = (
                f"The design of block {number} has {len(design):,} rows where the block has "
                f"{len(frame):,}: the formula dropped rows with missing values, which would "
                "misalign every bound after them."
            )
            raise ValueError(message)
        if scan.columns is None:
            scan.columns = design.columns
        elif not design.columns.equals(scan.columns):
            message = f"Block {number} produced design columns different from the first block's."
            raise ValueError(message)

        used_upper = np.clip(upper, 0, INFINITY_STAND_IN)
        _check_like_lifelines(fitter, design, used_upper, exact, weights, entries, names)
        values = design.to_numpy(dtype=np.float64)
        _accumulate(scan, values)

        scan.blocks.append(
            _Block(
                design=tuple(StoredColumn.of(values[:, j]) for j in range(values.shape[1])),
                lower=StoredColumn.of(lower),
                upper=StoredColumn.of(used_upper),
                entry=StoredColumn.of(entries),
                weight=StoredColumn.of(weights),
                exact=exact.copy(),
                weight_sum=float(weights.sum()),
            )
        )
        distinct = pd.DataFrame(
            {"lower": lower, "upper": used_upper, "entry": entries, "weight": weights}
        )
        scan.bounds.append(
            distinct.groupby(["lower", "upper", "entry"], sort=False)["weight"].sum()
        )
        scan.events += float(weights[np.isfinite(upper)].sum())
        log.debug("block %d: %s rows", number, f"{len(frame):,}")

    stored = sum(block.nbytes for block in scan.blocks)
    log.info(
        "stored %s rows in %d blocks: %.2f GB, %.0f bytes a row",
        f"{scan.rows:,}",
        len(scan.blocks),
        stored / 1e9,
        stored / max(scan.rows, 1),
    )
    return scan


def _check_categories(scan: _Scan, covariates: pd.DataFrame, number: int) -> None:
    """Refuse a block whose categorical levels differ from the first block's.

    formulaic takes the dummy columns from the levels a categorical column declares. A
    column categorised over the whole table declares every level in every block, even one
    a block lacks; a text column, or one categorised block by block, would give each
    block its own columns -- and a missing level would be encoded as all zeros, silently.
    """
    for name, dtype in covariates.dtypes.items():
        if isinstance(dtype, pd.CategoricalDtype):
            known = scan.categories.setdefault(str(name), dtype.categories)
            if not dtype.categories.equals(known):
                message = (
                    f"Column {name!r} declares levels {list(dtype.categories)} in block "
                    f"{number} and {list(known)} in the first block. Categorise it over the "
                    "whole table before splitting it into blocks."
                )
                raise ValueError(message)
        elif not (pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)):
            message = (
                f"Column {name!r} is {dtype}. Convert text covariates to category over the "
                "whole table first: a block would otherwise see only its own levels."
            )
            raise TypeError(message)


def _check_like_lifelines(
    fitter: ParametericAFTRegressionFitter,
    design: pd.DataFrame,
    upper: np.ndarray,
    exact: np.ndarray,
    weights: np.ndarray,
    entries: np.ndarray,
    names: tuple[str, str, str, str | None, str | None],
) -> None:
    """``ParametricRegressionFitter._check_values_pre_fitting``, on one block."""
    *_, entry_col, weights_col = names
    utils.check_for_numeric_dtypes_or_raise(design)
    utils.check_nans_or_infs(design)
    utils.check_nans_or_infs(upper)
    utils.check_nans_or_infs(exact)
    utils.check_positivity(upper)
    if weights_col is not None:
        if (weights.astype(int) != weights).any() and not fitter.robust:
            warnings.warn(
                "Non-integer weights bias the naive variance estimates.",
                exceptions.StatisticalWarning,
                stacklevel=3,
            )
        if (weights <= 0).any():
            message = f"values in weight column {weights_col} must be positive."
            raise ValueError(message)
    if entry_col is not None:
        utils.check_entry_times(upper, entries)


def _accumulate(scan: _Scan, values: np.ndarray) -> None:
    """Running sums for the column standard deviations."""
    if scan.first is None or scan.second is None or scan.low is None or scan.high is None:
        width = values.shape[1]
        scan.first, scan.second = np.zeros(width), np.zeros(width)
        scan.low, scan.high = np.full(width, np.inf), np.full(width, -np.inf)
    scan.rows += len(values)
    scan.first += values.sum(axis=0)
    scan.second += np.einsum("ij,ij->j", values, values)
    scan.low = np.minimum(scan.low, values.min(axis=0))
    scan.high = np.maximum(scan.high, values.max(axis=0))


def _standard_deviation(scan: _Scan) -> np.ndarray:
    """Column standard deviations with ``ddof=1``, as ``DataFrame.std`` gives them.

    From running sums, because the rows are never all in memory. A column that never
    varies is set to exactly zero, as pandas makes it, rather than left at the rounding
    residue of ``sum(x**2) - sum(x)**2 / n``. That matters: lifelines decides which
    columns are constant -- the intercept among them -- by comparing with 1e-8.
    """
    assert scan.first is not None and scan.second is not None
    assert scan.low is not None and scan.high is not None
    mean = scan.first / scan.rows
    variance = np.maximum(scan.second - scan.first * mean, 0.0) / (scan.rows - 1)
    deviation: np.ndarray = np.sqrt(variance)
    deviation[scan.low == scan.high] = 0.0
    return deviation


def _initial_point(
    fitter: ParametericAFTRegressionFitter,
    columns: pd.MultiIndex,
    raw_std: np.ndarray,
    bounds: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """``ParametericAFTRegressionFitter._create_initial_point``, on the distinct bounds.

    lifelines fits the matching univariate model to every row's bounds and weight, and
    seeds the first constant column -- the scale intercept -- with the log of its
    parameter. Rows sharing their bounds contribute identically, so fitting the distinct
    bounds with their weights summed is the same likelihood on a few hundred rows.
    """
    constant_col = pd.Series(raw_std < _CONSTANT, index=columns).idxmax()
    univariate_class = getattr(lifelines, fitter._class_name.replace("AFT", ""), None)
    if univariate_class is None:
        return {
            name: np.zeros(int((columns.get_level_values(0) == name).sum()))
            for name in fitter._fitted_parameter_names
        }

    univariate = univariate_class()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        univariate.fit_interval_censoring(
            bounds["lower"].to_numpy(dtype=float),
            bounds["upper"].to_numpy(dtype=float),
            entry=bounds["entry"].to_numpy(dtype=float),
            weights=bounds["weight"].to_numpy(dtype=float),
        )
    fitter._ll_null_ = univariate.log_likelihood_

    # Parameter blocks in sorted order, which is how ``Index.groupby`` hands them over.
    parameters = columns.get_level_values(0)
    seeded: dict[str, np.ndarray] = {}
    for name in sorted(set(parameters)):
        covariates = columns[parameters == name].tolist()
        seeded[name] = np.zeros(len(covariates))
        if constant_col in covariates:
            value = getattr(univariate, name)
            seeded[name][covariates.index(constant_col)] = value if value <= 0 else np.log(value)
    return seeded


def _warm_start(
    seeded: dict[str, np.ndarray],
    columns: pd.MultiIndex,
    norm_std: pd.Series,
    params: pd.Series,
) -> dict[str, np.ndarray]:
    """A starting point from coefficients on their natural scale, a nested model's say.

    lifelines optimises each coefficient multiplied by its column's standard deviation,
    so a fitted ``params_`` is scaled back into that space before it can seed another
    fit. Columns the other model did not have keep lifelines' own seed.

    Backward elimination refits a model one covariate smaller at every step. On the whole
    population each fit is well over an hour, and the model it removes a covariate from is
    the best available guess at where the smaller one ends.
    """
    started = {name: values.copy() for name, values in seeded.items()}
    parameters = columns.get_level_values(0)
    for name, values in started.items():
        for position, key in enumerate(columns[parameters == name].tolist()):
            if key in params.index:
                values[position] = float(params[key]) * float(norm_std[key])
    return started


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
    ) -> None:
        self._blocks = blocks
        self._columns = columns
        self._scale = scale
        self.total_weight = float(sum(block.weight_sum for block in blocks))
        negative = partial(
            fitter._create_neg_likelihood_with_penalty_function,
            likelihood=fitter._log_likelihood_interval_censoring,
        )
        self._value_and_gradient = value_and_grad(negative)
        self._hessian = hessian(negative)

        penalizer = fitter.penalizer
        self._penalty: Callable[[np.ndarray], Any] | None = None
        if isinstance(penalizer, np.ndarray) or penalizer > 0:

            def penalty(x: np.ndarray) -> float:
                # A cast, not float(): while autograd traces this the value is a box, and
                # converting it would cut the gradient off.
                return cast("float", fitter._add_penalty(unflatten(x), 0.0))

            self._penalty = penalty

        self.evaluations = 0
        self._started = time.perf_counter()
        self._reported = -np.inf

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        value = 0.0
        gradient = np.zeros_like(x)
        for block in self._blocks:
            share = block.weight_sum / self.total_weight
            block_value, block_gradient = self._value_and_gradient(
                x, *block.arguments(self._columns, self._scale)
            )
            value += share * float(block_value)
            gradient += share * block_gradient
        if self._penalty is not None:
            penalty_value, penalty_gradient = value_and_grad(self._penalty)(x)
            value += float(penalty_value)
            gradient += penalty_gradient

        self.evaluations += 1
        elapsed = time.perf_counter() - self._started
        if elapsed - self._reported >= _PROGRESS_SECONDS:
            self._reported = elapsed
            log.info("evaluation %d: objective %.12f at %.0fs", self.evaluations, value, elapsed)
        return value, gradient

    def hessian(self, x: np.ndarray) -> np.ndarray:
        total = np.zeros((len(x), len(x)))
        for block in self._blocks:
            share = block.weight_sum / self.total_weight
            total += share * self._hessian(x, *block.arguments(self._columns, self._scale))
        if self._penalty is not None:
            total += hessian(self._penalty)(x)
        return total


#: The polish stops once no Newton step larger than this, in standard errors, remains.
POLISH_TOLERANCE_SE: Final = 1e-3

#: The most Newton steps a polish takes. From where SLSQP stops two or three do; from a warm
#: start that adds a covariate the curvature at the start is a poor guide, and the steps stay
#: damped until it catches up.
_POLISH_STEPS: Final = 40

#: Damping starts here when a step is refused and rises tenfold each time; past the ceiling
#: no step is left that lowers the objective.
_DAMPING_FLOOR: Final = 1e-6
_DAMPING_CEILING: Final = 1e12


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


def _polish(
    objective: _Objective,
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
        _, remaining = _newton_step(curvature, gradient, objective.total_weight)
        steps += 1
        log.info(
            "Newton step %d, damping %.0e: %.3g standard errors from the optimum",
            steps,
            damping,
            remaining,
        )
        damping = damping / 10.0 if damping >= 10.0 * _DAMPING_FLOOR else 0.0
    if remaining > POLISH_TOLERANCE_SE:
        log.warning("polish stopped after %d steps, %.3g standard errors out", steps, remaining)
    return x, value, curvature, steps, stopped, remaining


def _possible(value: float) -> bool:
    """Whether a mean negative log-likelihood could take this value: finite and not negative."""
    return bool(np.isfinite(value)) and value >= 0.0


def _damped_step(x: np.ndarray, gradient: np.ndarray, damped: np.ndarray) -> np.ndarray | None:
    """``x`` less the damped Newton step, or ``None`` where that matrix is not positive definite."""
    try:
        np.linalg.cholesky(damped)
    except np.linalg.LinAlgError:
        return None
    return np.asarray(x - np.linalg.solve(damped, gradient), dtype=float)


def _from_slsqp(
    objective: _Objective, results: OptimizeResult, *, polish: bool
) -> tuple[np.ndarray, float, np.ndarray, int, float, float]:
    """Where SLSQP stopped, polished to the optimum unless ``polish`` is off."""
    started = time.perf_counter()
    x = np.asarray(results.x, dtype=float)
    curvature = _symmetric(objective.hessian(x))
    log.info(
        "hessian over %d blocks in %.0fs", len(objective._blocks), time.perf_counter() - started
    )
    value = float(results.fun)
    gradient = np.asarray(results.jac, dtype=float)
    if polish:
        return _polish(objective, x, value, gradient, curvature)
    _, stopped = _newton_step(curvature, gradient, objective.total_weight)
    return x, value, curvature, 0, stopped, stopped


def _newton_from(
    objective: _Objective, start: np.ndarray
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


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    """A Hessian symmetrised, as lifelines symmetrises it (lifelines issue 801)."""
    symmetric: np.ndarray = (matrix + matrix.T) / 2
    return symmetric


def _store(
    fitter: ParametericAFTRegressionFitter,
    columns: pd.MultiIndex,
    x: np.ndarray,
    value: float,
    curvature: np.ndarray,
    objective: _Objective,
    unflatten: Callable[[np.ndarray], dict[str, np.ndarray]],
) -> None:
    """What ``_fit_model`` and ``_fit`` set once the optimum is found."""
    params = unflatten(x)
    fitter.log_likelihood_ = -objective.total_weight * value
    fitter._hessian_ = objective.total_weight * curvature

    keys = list(fitter.regressors.keys())
    if keys != list(fitter._norm_std.index.get_level_values(0).unique()):
        message = "Parameter blocks and design columns are out of order."
        raise AssertionError(message)
    fitter.params_ = np.concatenate([params[key] for key in keys]) / fitter._norm_std
    fitter._compare_to_values = np.zeros_like(fitter.params_)
    fitter.variance_matrix_ = pd.DataFrame(
        fitter._compute_variance_matrix(), index=columns, columns=columns
    )
    # Without robust errors lifelines reads only the variance matrix here.
    fitter.standard_errors_ = fitter._compute_standard_errors(None, None, None, None, None)
    fitter.confidence_intervals_ = fitter._compute_confidence_intervals()
