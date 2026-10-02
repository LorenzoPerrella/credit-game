"""One pass over the rows: the design checked as lifelines checks it, then kept or encoded.

Two ways of keeping what a pass found, and the difference between them is the difference between
a fit that re-reads and a selection that does not.

:func:`_scan` builds the design block by block and stores it compactly. It is what a single fit
needs, and what the autograd path has to have.

:func:`encode_blocks` reads the rows with **no formula involved** -- no design to expand, no
moments to accumulate over 26 columns, nothing put through formulaic -- and keeps fifteen bytes
a row plus the key of every combination. A fit through the written-out likelihood on the
production table is 53 seconds of arithmetic behind 10.9 minutes of reading, and a selection
used to pay that reading once per candidate: about thirty times, with the fifteen step-7 fits
each beginning by recomputing the identical base objective to twelve digits. An
:class:`Encoding` is that reading, done once, and every model's design is then two tables built
by putting its formula through 3,001 loan combinations and 152,565 calendar keys.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from lifelines import exceptions, utils

from creditsurv.models.engine.lifelines_glue import (
    _family,
)
from creditsurv.models.engine.storage import (
    INFINITY_STAND_IN,
    StoredColumn,
    _Block,
    _column_slices,
)
from creditsurv.models.kernel import Expanded, Factorisation, Kernel, Rows

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lifelines.fitters import ParametericAFTRegressionFitter

    #: What the parent asks a worker for, and what a worker sends back.
    Command = tuple[str, np.ndarray | None]
    Prepared = tuple[pd.MultiIndex, np.ndarray, float, dict[str, np.ndarray]]
    Answer = dict[str, Any] | tuple[float, np.ndarray] | np.ndarray
    #: A worker's answer with the part it read, so the parent can add them in one order.
    Tagged = tuple[int, Answer]

log = logging.getLogger(__name__)


@dataclass
class _Scan:
    """What one pass over the blocks learns."""

    regressors: Any = None
    columns: pd.MultiIndex | None = None
    categories: dict[str, pd.Index] = field(default_factory=dict)
    blocks: list[_Block] = field(default_factory=list)
    #: The same rows factorised, when a partition was declared: two indices, an age, an exit
    #: and a weight, fifteen bytes each, with the design never stored at all.
    encoded: list[Rows] = field(default_factory=list)
    factorisation: Factorisation | None = None
    #: This model's two design tables, when they came from the key frames rather than from the
    #: rows -- which is how a fit from an :class:`Encoding` gets them.
    expanded: Expanded | None = None
    rows: int = 0
    first: np.ndarray | None = None
    second: np.ndarray | None = None
    low: np.ndarray | None = None
    high: np.ndarray | None = None
    bounds: list[pd.Series] = field(default_factory=list)
    events: float = 0.0
    weight: float = 0.0
    #: Blocks and bytes held by the worker processes, when there are any.
    other_blocks: int = 0
    other_bytes: int = 0


def _scan(
    fitter: ParametericAFTRegressionFitter,
    blocks: Iterable[pd.DataFrame],
    *,
    seed: dict[str, str],
    names: tuple[str, str, str, str | None, str | None],
    calendar: Sequence[str] | None = None,
) -> _Scan:
    """Build the design block by block, check it as lifelines would, and store it.

    Given ``calendar`` -- the covariates that are functions of the calendar rather than of the
    loan -- the design is still built, because lifelines' own checks read it and so do the
    moments that give every coefficient its scale, but it is **not stored**. What is kept
    instead is two indices and three narrow columns a row: fifteen bytes against the thirty a
    compacted design costs and the two hundred and eight an expanded one does.
    """
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

        if calendar is not None:
            primary = _primary_columns(fitter, scan.columns)
            if scan.factorisation is None:
                held = [name for name in covariates.columns if name not in set(calendar)]
                scan.factorisation = Factorisation(
                    loan=held,
                    calendar=[name for name in covariates.columns if name in set(calendar)],
                    age_column=entry_col if entry_col is not None else lower_bound_col,
                    columns=[str(name) for _, name in scan.columns[primary]],
                )
            scan.encoded.append(
                scan.factorisation.add(
                    frame,
                    values[:, primary],
                    # The exit is whether the interval closes, never `exact`: on this panel no
                    # observation is exact, and `blocks` hands lifelines a flag that is always
                    # False with the event carried by a finite upper bound.
                    event=np.isfinite(upper),
                    weight=weights,
                )
            )
        else:
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
        scan.weight += float(weights.sum())
        log.debug("block %d: %s rows", number, f"{len(frame):,}")

    stored = sum(block.nbytes for block in scan.blocks) + sum(
        block.nbytes for block in scan.encoded
    )
    log.info(
        "stored %s rows in %d blocks: %.2f GB, %.0f bytes a row",
        f"{scan.rows:,}",
        len(scan.blocks) + len(scan.encoded),
        stored / 1e9,
        stored / max(scan.rows, 1),
    )
    return scan


@dataclass(frozen=True)
class Encoding:
    """The rows read **once**, reusable by every model whose covariates it covers.

    A fit through the written-out kernel on the production table is 53 seconds of arithmetic
    behind 12.1 minutes of reading, and a selection pays that reading once per candidate --
    about thirty times, plus the fifteen step-7 fits that each begin by recomputing the
    identical base objective to twelve digits. None of it is necessary: the rows carry two
    indices into the combinations of a widest key, and every design column is a function of
    one side, so a candidate's two tables are built by putting its formula through 3,001 and
    153,309 rows.

    So this pass involves no formula at all. There is no design to expand, no moments to
    accumulate over 26 columns and nothing to put through formulaic -- only the keys, the
    counts, and fifteen bytes a row.
    """

    factorisation: Factorisation
    rows: tuple[Rows, ...]
    #: The distinct ``(lower, upper, entry)`` with their weights summed, which is all the
    #: univariate seed needs: lifelines fits the matching one-parameter model to the bounds,
    #: and rows sharing them contribute identically.
    bounds: pd.Series
    events: float
    weight: float
    categories: dict[str, pd.Index]
    names: tuple[str, str, str, str | None, str | None]

    @property
    def episodes(self) -> int:
        return self.factorisation.rows

    @property
    def nbytes(self) -> int:
        return sum(block.nbytes for block in self.rows)


def encode_blocks(
    blocks: Iterable[pd.DataFrame],
    *,
    loan: Sequence[str],
    calendar: Sequence[str],
    lower_bound_col: str,
    upper_bound_col: str,
    event_col: str,
    entry_col: str | None = None,
    weights_col: str | None = None,
) -> Encoding:
    """Read the rows once, keeping fifteen bytes of each and the key of every combination.

    ``loan`` names the covariates the cell key carries and ``calendar`` those that are
    functions of the calendar; between them they must cover every covariate column the blocks
    carry, because a covariate in neither has no index to be looked up by. The lists are the
    project's own declaration -- ``config.MACRO_CANDIDATES`` is the calendar side, and its
    comment says why: "a macro covariate is a function of the vintage quarter and the loan
    age, both already in the aggregation key".
    """
    names = (lower_bound_col, upper_bound_col, event_col, entry_col, weights_col)
    special = [name for name in names if name is not None]
    wanted = {*loan, *calendar}
    scan = _Scan()
    factorisation: Factorisation | None = None
    tallies: list[pd.Series] = []

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
        unclassified = [name for name in covariates.columns if name not in wanted]
        if unclassified:
            message = (
                f"The covariate(s) {', '.join(map(str, unclassified))} are neither in the loan "
                "key nor in the calendar key, so an encoded row has no index to look them up "
                "by. Name them on one side, or drop them from the block's columns."
            )
            raise ValueError(message)
        used_upper = np.clip(upper, 0, INFINITY_STAND_IN)
        _check_rows_like_lifelines(used_upper, exact, weights, entries, names)
        if factorisation is None:
            factorisation = Factorisation(
                loan=[name for name in covariates.columns if name in set(loan)],
                calendar=[name for name in covariates.columns if name in set(calendar)],
                age_column=entry_col if entry_col is not None else lower_bound_col,
            )
        scan.encoded.append(
            # The exit is whether the interval closes, never `exact`: on this panel no
            # observation is exact, and the event is carried by a finite upper bound.
            factorisation.add(frame, event=np.isfinite(upper), weight=weights)
        )
        distinct = pd.DataFrame(
            {"lower": lower, "upper": used_upper, "entry": entries, "weight": weights}
        )
        tallies.append(distinct.groupby(["lower", "upper", "entry"], sort=False)["weight"].sum())
        scan.rows += len(frame)
        scan.events += float(weights[np.isfinite(upper)].sum())
        scan.weight += float(weights.sum())
        log.debug("block %d: %s rows", number, f"{len(frame):,}")

    if factorisation is None:
        message = "Nothing was encoded: every block was empty."
        raise ValueError(message)
    held = sum(block.nbytes for block in scan.encoded)
    log.info(
        "encoded %s rows in %d blocks: %.2f GB, %.0f bytes a row; %s loan and %s calendar keys",
        f"{scan.rows:,}",
        len(scan.encoded),
        held / 1e9,
        held / max(scan.rows, 1),
        f"{len(factorisation.keys()[0]):,}",
        f"{len(factorisation.keys()[1]):,}",
    )
    return Encoding(
        factorisation=factorisation,
        rows=tuple(scan.encoded),
        bounds=pd.concat(tallies).groupby(level=[0, 1, 2]).sum(),
        events=scan.events,
        weight=scan.weight,
        categories=dict(scan.categories),
        names=names,
    )


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
    utils.check_for_numeric_dtypes_or_raise(design)
    utils.check_nans_or_infs(design)
    _check_rows_like_lifelines(upper, exact, weights, entries, names, robust=fitter.robust)


def _check_rows_like_lifelines(
    upper: np.ndarray,
    exact: np.ndarray,
    weights: np.ndarray,
    entries: np.ndarray,
    names: tuple[str, str, str, str | None, str | None],
    *,
    robust: bool = False,
) -> None:
    """The half of lifelines' pre-fitting checks that is about the rows, not the design.

    Separated because an encoding pass has no design to check: it reads the rows once and
    without a formula, and each model's own columns are checked when its tables are built.
    """
    *_, entry_col, weights_col = names
    utils.check_nans_or_infs(upper)
    utils.check_nans_or_infs(exact)
    utils.check_positivity(upper)
    if weights_col is not None:
        if (weights.astype(int) != weights).any() and not robust:
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


def _primary_columns(
    fitter: ParametericAFTRegressionFitter, columns: pd.MultiIndex | None
) -> slice:
    """Where the scale's own coefficients sit in the design, as a slice.

    The design is built in the order of its MultiIndex, so each parameter's columns are
    adjacent; `_column_slices` is the same analysis the slicer does. Only the scale's block is
    factorised -- the shape's is one constant column, and a shape with covariates is refused
    before any of this.
    """
    if columns is None:
        message = "The design has no columns yet."
        raise ValueError(message)
    found = _column_slices(columns)[fitter._primary_parameter_name]
    if not isinstance(found, slice):
        message = (
            "The scale's coefficients are not adjacent in the design, so the factorisation "
            "cannot name them by a slice."
        )
        raise ValueError(message)
    return found


def _kernel(
    fitter: ParametericAFTRegressionFitter,
    scan: _Scan,
    scale: np.ndarray,
    total_weight: float,
) -> Kernel:
    """The written-out objective over the rows this scan encoded.

    Two things are required of the shape and refused rather than worked around. It must have
    **one** column and that column must be the constant one: a shape with covariates is a
    different model -- the `occupancy` test in `docs/decisions.md` is exactly that fit -- and
    the kernel's whole economy comes from a row depending on two scalars rather than many. And
    that column must be 1.0 with a scale of 1.0, which is what lifelines produces for an
    intercept, so the shape's coefficient reaches the likelihood unchanged.
    """
    columns = scan.columns
    if columns is None or scan.factorisation is None or scan.low is None or scan.high is None:
        message = "The scan did not encode anything to fit."
        raise ValueError(message)
    primary = _primary_columns(fitter, columns)
    ancillary = _column_slices(columns)[fitter._ancillary_parameter_name]
    if not isinstance(ancillary, slice) or ancillary.stop - ancillary.start != 1:
        message = (
            "The shape has more than one coefficient, so a row does not depend on two "
            "scalars and this kernel cannot fit it. Fit a shape with covariates on the "
            "autograd evaluator."
        )
        raise ValueError(message)
    position = int(ancillary.start)
    if not (scan.low[position] == scan.high[position] == 1.0 and scale[position] == 1.0):
        message = (
            f"The shape's column is not the constant 1.0 that lifelines builds for an "
            f"intercept: it runs {scan.low[position]} to {scan.high[position]} at a scale of "
            f"{scale[position]}. The kernel reads the shape's coefficient directly, so it "
            "would be reading a different number."
        )
        raise ValueError(message)
    expanded = scan.expanded
    if expanded is None:
        loan, calendar, loan_positions, calendar_positions = scan.factorisation.tables()
    else:
        loan, calendar = expanded.loan, expanded.calendar
        loan_positions, calendar_positions = expanded.loan_positions, expanded.calendar_positions
    loan_index = primary.start + loan_positions
    calendar_index = primary.start + calendar_positions
    return Kernel(
        distribution=_family(fitter),
        # lifelines optimises each coefficient multiplied by its column's standard deviation,
        # so the tables are divided by it once here instead of the design being divided by it
        # on every evaluation.
        loan=loan / scale[loan_index],
        calendar=calendar / scale[calendar_index],
        loan_index=loan_index,
        calendar_index=calendar_index,
        shape_index=position,
        blocks=tuple(scan.encoded),
        total_weight=total_weight,
    )
