"""Every fit the selection makes: cached, warm-started, and bounded by its parent.

This is where all the I/O of the procedure lives, and that is the point of it being its own
module. The rules that decide an elimination are pure functions of a coefficient table and
live in :mod:`creditsurv.models.rules`; the sequence of steps lives in
:mod:`creditsurv.models.procedure`; and everything that reads a file, writes one, or reads 72
million rows is here.

Three things it owns that a caller should not have to think about:

* **a fit is saved the moment it succeeds**, with a readable description beside the bytes,
  because a run that completes a 154-minute fit and is then killed writing its reports has
  thrown away the expensive part and kept nothing;
* **a fit starts from the nearest model that has one** -- the parent at the step before, or
  the same specification fitted on a cell table since rebuilt, which is a starting point and
  never a result;
* **a nested fit is bounded by its parent's optimum**, which makes the region where lifelines'
  clipped likelihood is unbounded below unreachable while the fit runs.

Beside :class:`Fits`, which is the selection's own cache, are the fits every *other* command
makes -- one model on the training half, the family comparisons, the window cuts. They used to
live at the bottom of `cli.py`, 288 lines of it, and nothing about them was a command: a
description of the cell file for the window a command reads it in, the dict a fit is cached
under, a fit saved the moment it lands, and the caching wrapper the extra fits of a report go
through. Their progress goes to the log now rather than to a command's stdout, which is where
the rest of the package reports it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final

from lifelines import exceptions

from creditsurv.config import (
    DISTRIBUTION,
    SELECTION_RECORD,
    default_covariates,
    default_formula,
    reports_dir,
)
from creditsurv.data.panel import (
    AGE_START,
    DEFAULT_CAUSE,
    EXACT_OBSERVATION,
    LOWER_BOUND,
    UPPER_BOUND,
    WEIGHT,
    CellBlocks,
)
from creditsurv.data.store import find_fits, fit_fingerprint, load_fit, save_fit
from creditsurv.models.aft import FitResult, Likelihood, fit_aft, fit_encoding, fit_streamed
from creditsurv.models.engine import DEFAULT_BLOCK_ROWS, Encoding, Pinned, encode_blocks

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import numpy as np
    import pandas as pd

    from creditsurv.models.specification import Specification

log = logging.getLogger(__name__)


@dataclass
class Fits:
    """Every fit the selection makes: saved as it lands, started from the nearest model.

    ``identity`` names the cell table the rows came from -- its file, size and time of
    writing -- so a cached fit is never reused for a table that has since been rebuilt.

    **Either the rows or a description of them.** ``train`` is an expanded episode frame, as
    the selection used to hold: 91.6 million cells expand to a frame of many gigabytes, and
    a selection makes twenty-odd fits beside it. ``blocks`` is the alternative -- the cell
    file, the window and the covariates -- and each fit then reads its own share of the
    parquet in ``workers`` processes and keeps it compactly. The same specification fitted
    the two ways agrees to 9.4e-07 standard errors, measured; the difference is 68.1 minutes
    at 4.7 GB against 91 at 15.
    """

    train: pd.DataFrame | None
    identity: str
    as_of: str
    moratorium: str
    #: The distribution family every fit of this run uses. An input, not a constant: rule 2
    #: of docs/rules.md takes both families through the whole selection and compares the two
    #: *selected* models, which cannot be done while the family is written into the procedure.
    distribution: str = DISTRIBUTION
    #: Which exit these fits are of. It is in every fingerprint, because the prepayment
    #: model reads the same cells, the same formulas and the same window as the default one.
    cause: str = DEFAULT_CAUSE
    block_rows: int = DEFAULT_BLOCK_ROWS
    #: Where the rows come from when they are not held: the cell file and its window.
    blocks: CellBlocks | None = None
    #: Processes the likelihood is evaluated in, when streaming. Each holds one batch at a
    #: time plus its share of the stored blocks.
    workers: int = 1
    #: The optimiser that last answered, tried first on the next fit. A design that defeats
    #: SLSQP is usually beside another one just like it -- the elimination refits nearly the
    #: same model at every step -- and walking the chain each time cost over an hour a fit on
    #: the prepayment model.
    preferred: str | None = None
    #: The covariates that are functions of the calendar rather than of the loan. Given, the
    #: rows are **read once** and every fit of the run is made from that reading, through
    #: `models.kernel`: a fit on the production table is 53 seconds of arithmetic behind 12.1
    #: minutes of reading, and a selection used to pay the reading once per candidate. The
    #: list is `config.MACRO_CANDIDATES`, declared before any fit with its own argument -- "a
    #: macro covariate is a function of the vintage quarter and the loan age, both already in
    #: the aggregation key".
    calendar: Sequence[str] | None = None
    record: list[dict[str, object]] = field(default_factory=list)
    #: One encoding per sample, kept for the life of the run. Three at most: the training half
    #: and the two origination-year halves step 9 compares, against thirty readings.
    held: dict[int | None, Encoding] = field(default_factory=dict, repr=False)
    #: Every covariate the run will ever read, which a reading has to cover because it is made
    #: once and before the first specification exists. `run_selection` sets it from its own
    #: candidate lists; without it the description's own covariates are all that is assumed,
    #: and a formula naming anything else fails where it is expanded rather than silently.
    covering: tuple[str, ...] | None = None

    def fit(
        self,
        spec: Specification,
        *,
        sample: str = "training half",
        where: np.ndarray | None = None,
        parity: int | None = None,
        start: FitResult | None = None,
        parent: Specification | None = None,
    ) -> FitResult:
        # The floor and the nested check both compare this fit's log-likelihood against the
        # parent's, which says nothing unless the two saw the same rows: the floor is the
        # parent's optimum divided by the parent's exposure. Step 9 fits the same
        # specification on two halves of the book, warm-started from the whole, and a parent
        # passed there would hand a fit on half the rows a bound computed on all of them.
        if parent is not None and (where is not None or parity is not None):
            message = (
                "A nested parent bounds a fit only on the rows the parent itself saw; "
                f"{sample!r} is a different sample, so it cannot be given one."
            )
            raise ValueError(message)
        described = selection_description(
            identity=self.identity,
            as_of=self.as_of,
            moratorium=self.moratorium,
            formula=spec.formula,
            sample=sample,
            distribution=self.distribution,
            cause=self.cause,
        )
        fingerprint = fit_fingerprint(**described)
        cached = load_fit(fingerprint)
        # A log-likelihood of zero or more is no likelihood at all: a fit saved before the
        # polish refused such values may have stopped at one.
        if isinstance(cached, FitResult) and cached.log_likelihood < 0:
            log.info("cached: %s on the %s", spec.formula, sample)
            self.record.append({**described, "minutes": 0.0, "evaluations": None, "cached": True})
            return cached

        log.info("fitting: %s on the %s", spec.formula, sample)
        started = time.perf_counter()
        # `_floor` and `_check_nested` keep reading the caller's own start, never the
        # borrowed one: a bound taken from a fit on another cell table bounds nothing here.
        elsewhere = self._elsewhere(described) if start is None else None
        result = self._estimate(
            spec,
            where=where,
            parity=parity,
            start=start if elsewhere is None else elsewhere,
            floor=_floor(spec, parent, start),
            speculative=elsewhere is not None,
        )
        _check_nested(spec, result, parent=parent, parent_fit=start)
        minutes = (time.perf_counter() - started) / 60
        save_fit(result, fingerprint, {**described, "minutes": minutes})
        evaluations = None if result.blocks is None else result.blocks.evaluations
        self.record.append(
            {**described, "minutes": minutes, "evaluations": evaluations, "cached": False}
        )
        return result

    def _encoding(self, parity: int | None) -> Encoding:
        """The rows of one sample, read once and kept for every fit that wants them.

        A reading involves no formula -- no design to expand, no moments over 26 columns,
        nothing through formulaic -- and keeps fifteen bytes a row plus the key of every
        combination, 1.09 GB for the production table's training half. Each candidate's design
        is then two tables built from those keys.

        The two halves of step 9 are different rows, so they are different readings; nothing
        else in a selection is.
        """
        held = self.held.get(parity)
        if held is not None:
            return held
        assert self.blocks is not None
        assert self.calendar is not None
        calendar = set(self.calendar)
        covering = self.covering or self.blocks.covariates
        source = replace(self.blocks, covariates=tuple(covering), vintage_parity=parity).prepared()
        held = encode_blocks(
            source(),
            loan=[name for name in covering if name not in calendar],
            calendar=[name for name in covering if name in calendar],
            lower_bound_col=LOWER_BOUND,
            upper_bound_col=UPPER_BOUND,
            event_col=EXACT_OBSERVATION,
            entry_col=AGE_START,
            weights_col=WEIGHT,
        )
        self.held[parity] = held
        return held

    def _elsewhere(self, described: dict[str, object]) -> FitResult | None:
        """The same specification fitted on **another** cell table, as a starting point only.

        A fit is cached under the table's name, size and time of writing, so rebuilding the
        table invalidates every fit made before it -- 36 of the 175 on disk are a selection run
        on a table that has since been replaced, 17.5 hours of them. They are not useless. The
        specification is the same and the book is mostly the same book, so the old optimum is a
        far better guess at the new one than lifelines' seed: a cold fit of the training half is
        45 to 91 minutes and a warm one 4.7 to 13.

        **It can only ever be a starting point.** Where a fit ends is settled by the polish,
        which measures the distance to the optimum on the gradient and the curvature of *these*
        rows and refuses a fit it cannot drive under a thousandth of a standard error. A start
        changes how long that takes, not where it arrives -- and because this start is a guess
        about a different table, a failure from it is retried cold rather than reported.
        """
        wanted = {key: value for key, value in described.items() if key != "cells"}
        # `find_fits` reads `None` as "this key must be absent", which is how a default-cause
        # fit is told from a prepayment one: `selection_description` omits the key for the
        # default, so every fit made before the prepayment model existed keeps its name.
        wanted.setdefault("cause", None)
        for fingerprint, found in find_fits(**wanted):
            if found.get("cells") == described["cells"]:
                continue
            cached = load_fit(fingerprint)
            if isinstance(cached, FitResult) and cached.log_likelihood < 0:
                log.info("starting from the same model fitted on %s", found.get("cells"))
                return cached
        return None

    def _estimate(
        self,
        spec: Specification,
        *,
        where: np.ndarray | None,
        parity: int | None,
        start: FitResult | None,
        floor: float | None = None,
        speculative: bool = False,
    ) -> FitResult:
        """One fit, from the rows in hand or from the cell file.

        A warm start is an optimisation, not part of any rule here: it is what turns a
        45-minute cold fit into 13 minutes by beginning at the nearest model's coefficients.
        When it fails, it usually fails as a *starting point* -- the prepayment model's backward
        elimination walked six damped Newton steps from 144 standard errors out to 3.86e+03,
        with damping at 1e+12, and then SLSQP diverged -- so the answer is to start where
        lifelines would have started and pay for it, not to give up on the model.

        :class:`~creditsurv.models.engine.Pinned` is the exception, and it is not about the
        start. An optimiser held against the parent's optimum has found where lifelines' clipped
        region begins; a cold fit walks back to the same maximum and meets the same edge, for
        another hour. It is raised through.
        """
        try:
            return self._once(spec, where=where, parity=parity, start=start, floor=floor)
        except Pinned:
            # Not a bad starting point but the shape of the surface: see blocks.Pinned. The
            # exception is a start borrowed from another cell table, which is a guess about
            # these rows and must never be able to fail a fit that would otherwise succeed.
            if not speculative:
                raise
            log.warning("a start from another table was pinned; refitting cold: %s", spec.formula)
            return self._once(spec, where=where, parity=parity, start=None, floor=floor)
        except exceptions.ConvergenceError:
            if start is None:
                raise
            log.warning("warm start did not converge; refitting cold: %s", spec.formula)
            return self._once(spec, where=where, parity=parity, start=None, floor=floor)

    def _once(
        self,
        spec: Specification,
        *,
        where: np.ndarray | None,
        parity: int | None,
        start: FitResult | None,
        floor: float | None = None,
    ) -> FitResult:
        """One attempt, from the rows in hand, from an encoding of them, or from the file."""
        initial_point = None if start is None else start.fitter.params_
        if self.blocks is not None and self.calendar is not None:
            if where is not None:
                # A mask selects rows of a frame this path never builds. The sample is a
                # property of the reading -- `vintage_parity` while the cells are read -- and
                # a mask silently ignored here would fit the whole half and call it a half.
                message = (
                    "A row mask has no meaning when the rows are read from the cell file: "
                    "the sample is taken while reading, through the parity."
                )
                raise ValueError(message)
            fitted = fit_encoding(
                self._encoding(parity),
                spec.covariates,
                spec.formula,
                distribution=self.distribution,
                initial_point=initial_point,
                prefer=self.preferred,
                floor=floor,
            )
            if fitted.blocks is not None and fitted.blocks.method not in {"newton", "warm"}:
                self.preferred = fitted.blocks.method
            return fitted
        if self.blocks is None:
            if self.train is None:
                message = "Fits needs either an episode frame or a CellBlocks to read."
                raise ValueError(message)
            return fit_aft(
                self.train,
                spec.covariates,
                spec.formula,
                distribution=self.distribution,
                weights_col=WEIGHT,
                block_rows=self.block_rows,
                initial_point=initial_point,
                where=where,
            )
        # The sample is a property of the source, not a mask over rows the caller holds: a
        # half of the book taken while reading is the same half, and it never exists twice.
        source = replace(
            self.blocks, covariates=tuple(spec.covariates), vintage_parity=parity
        ).prepared()
        fitted = fit_streamed(
            source,
            spec.covariates,
            spec.formula,
            distribution=self.distribution,
            weights_col=WEIGHT,
            initial_point=initial_point,
            workers=self.workers,
            prefer=self.preferred,
            floor=floor,
        )
        if fitted.blocks is not None and fitted.blocks.method not in {"newton", "warm"}:
            self.preferred = fitted.blocks.method
        return fitted


def _floor(
    spec: Specification, parent: Specification | None, parent_fit: FitResult | None
) -> float | None:
    """The objective's floor for a nested model: the parent's own optimum.

    The child's parameters are the parent's with a coefficient held at zero, so every point of
    the child is a point of the parent and the parent's maximum bounds all of them. Handing that
    bound to the objective makes lifelines' unbounded region **unreachable while the fit runs**,
    which is the difference between converging and running for three hours to be thrown away.

    The objective is a mean, so the total log-likelihood is divided by the exposure, with
    :data:`_NESTED_TOLERANCE` allowed back -- which is as much as there is to allow, because
    lifelines' clipped region begins directly below the maximum rather than far under it.
    """
    if parent is None or parent_fit is None or parent_fit.blocks is None:
        return None
    if not set(spec.covariates) <= set(parent.covariates):
        return None
    weight = parent_fit.blocks.loan_months
    if weight <= 0:
        return None
    return float(-(parent_fit.log_likelihood + _NESTED_TOLERANCE) / weight)


def _check_nested(
    spec: Specification,
    result: FitResult,
    *,
    parent: Specification | None,
    parent_fit: FitResult | None,
) -> None:
    """Refuse a nested model that fits *better* than the model it is nested in.

    Mathematics, not a threshold: dropping a covariate cannot raise the maximised
    log-likelihood, because the parent could always have set that coefficient to zero. A fit
    that reports an improvement has not found a maximum -- it has found the region where
    lifelines clips the interval probability and adds the truncation term unclipped, where the
    objective is unbounded below.

    This is the cheapest guard available and the one that should have been written first. The
    prepayment model's step 8 spent **two hours and forty minutes** reaching a "solution" with a
    log-likelihood of -3.06e+06 against its parent's, and a comparison that costs nothing would
    have refused it in the first evaluation.
    """
    if parent is None or parent_fit is None:
        return
    if not set(spec.covariates) <= set(parent.covariates):
        return
    if result.log_likelihood <= parent_fit.log_likelihood + _NESTED_TOLERANCE:
        return
    message = (
        f"The nested model reports a log-likelihood of {result.log_likelihood:,.3f} against its "
        f"parent's {parent_fit.log_likelihood:,.3f}. Dropping a covariate cannot fit better, so "
        "this is not a maximum: it is the region where lifelines' clipped likelihood is "
        "unbounded below."
    )
    raise exceptions.ConvergenceError(message)


#: How much better a nested model may look before it is refused, in log-likelihood units.
#:
#: One unit, and the reason it is one unit is not the reason first written here. That said "the
#: last digits of a sum over 72 million terms", and the arithmetic was then measured: the same
#: specification evaluated at the same coefficients over 800 blocks instead of 443 reproduces the
#: log-likelihood to **1.97e-16** relative, three hundredths of a millionth of a unit. Summation
#: needs no allowance at all.
#:
#: What settles the size is where lifelines' clipped region begins, and the answer is **directly
#: below the optimum**, not far under it. On the prepayment model's step 8 the optimiser probed
#: points reading 11.8, then 158, 183 and 213 units better than the parent's optimum, in the same
#: line searches that also produced 10,699 units and, further along, 4.6 million. The parent is
#: not the one in the wrong: re-polished from its own coefficients with the tolerance driven from
#: 1e-03 to 1e-09 standard errors, it takes two more Newton steps and gains **-0.000 units** --
#: the same log-likelihood to six decimals. So those probes are not a better fit but the shallow
#: edge of the region where the interval probability is clipped and the truncation term is not,
#: and that edge abuts the maximum.
#:
#: There is therefore **no slack to be had**: any allowance wide enough to admit a probe 213 units
#: better is an allowance that lets a fit converge onto clipped ground and be cached as an
#: optimum. This was raised to 1e-06 of the log-likelihood -- 151 units -- on a first reading that
#: took the 11.8 for the anomaly's scale; the same cycle reached 213 twenty minutes later. One
#: unit is thirty million times the reproducibility measured above and far inside the edge, which
#: is the whole of what it has to be.
_NESTED_TOLERANCE: Final = 1.0


def selection_description(
    *,
    identity: str,
    as_of: str,
    moratorium: str,
    formula: str,
    sample: str = "training half",
    distribution: str = DISTRIBUTION,
    cause: str = DEFAULT_CAUSE,
) -> dict[str, object]:
    """What a selection fit is saved under, and so how it is found again.

    The cause enters only when it is not ``default``, so every fit made before the
    prepayment model existed keeps the name it has.
    """
    described: dict[str, object] = {
        "purpose": "selection",
        "cells": identity,
        "as_of": as_of,
        "moratorium": moratorium,
        "sample": sample,
        "formula": formula,
        "distribution": distribution,
        "likelihood": "interval_censored",
    }
    if cause != DEFAULT_CAUSE:
        described["cause"] = cause
    return described


def selected_fit(
    *,
    identity: str,
    as_of: str,
    moratorium: str,
    formula: str,
    distribution: str = DISTRIBUTION,
    cause: str = DEFAULT_CAUSE,
) -> FitResult | None:
    """The selection's own fit of ``formula`` on the training half, if it made one.

    The model ``creditsurv report`` fits is the specification the selection ended on, on
    the same rows, so the selection has already found its optimum. Started there, the fit
    takes Newton steps to the same point instead of SLSQP's whole path from lifelines' seed.
    """
    cached = load_fit(
        fit_fingerprint(
            **selection_description(
                identity=identity,
                as_of=as_of,
                moratorium=moratorium,
                formula=formula,
                distribution=distribution,
                cause=cause,
            )
        )
    )
    return cached if isinstance(cached, FitResult) and cached.log_likelihood < 0 else None


def _terms(result: FitResult) -> pd.DataFrame:
    """The coefficient table of the scale parameter, where every covariate enters."""
    summary: pd.DataFrame = result.fitter.summary.loc[result.fitter._primary_parameter_name]
    return summary


def cell_source(
    moratorium: str,
    macro: pd.DataFrame,
    covariates: Iterable[str],
    *,
    block_rows: int,
    until: int | None,
    cause: str = DEFAULT_CAUSE,
    model_only: bool = True,
) -> CellBlocks:
    """The cell file described for the window a command reads it in, ready to be handed out.

    Five commands said the same nine arguments, and what they had in common was the whole
    convention: the table for this moratorium policy, the macro panel beside it, the
    covariates as a tuple, batches of `block_rows`, and **every month up to the cut and none
    after it** -- which is what keeps the test window out of an estimation sample. `until=None`
    is the one caller that wants the whole table, `fit` without an `--as-of`, and it has to say
    so rather than pass a cut it made up.

    `prepared()` is part of the convention too: it reads the file's episode width and
    categorical levels once here rather than in each of four worker processes, which was
    3 GB of a peak when they each read them.
    """
    from creditsurv.data.panel import CellBlocks as Blocks
    from creditsurv.data.store import cells_path

    return Blocks(
        str(cells_path(moratorium)),
        macro,
        tuple(covariates),
        rows=block_rows,
        months=(None, until),
        cause=cause,
        model_only=model_only,
    ).prepared()


def fit_from_cells(
    *,
    as_of: str,
    moratorium: str,
    distribution: str,
    workers: int,
    block_rows: int,
    cause: str = DEFAULT_CAUSE,
    traced: bool = False,
) -> FitResult:
    """Fit from the cell file and save it under its fingerprint.

    The rows are never held. By default the likelihood is the one `models.kernel` writes out,
    over two tables rather than a stored design: 11.8x on a value-and-gradient-with-Hessian,
    and a fit of the training half is 10.2 minutes against 43.8. ``traced`` goes back to
    tracing lifelines' likelihood with autograd in ``workers`` processes, which is what the
    equivalence tests hold the other to. The fit is cached under the same description either
    way, so `report` and `views` find it by the specification rather than by how it was made.
    """
    import pandas as pd

    from creditsurv.config import MACRO_CANDIDATES
    from creditsurv.data.fred import load_macro_panel
    from creditsurv.data.panel import WEIGHT, month_ordinal
    from creditsurv.data.store import fit_fingerprint, save_fit
    from creditsurv.models.aft import fit_streamed

    formula = default_formula()
    cut = None
    if as_of:
        reporting_date = pd.Period(as_of, freq="M")
        cut = month_ordinal(reporting_date)
    source = cell_source(
        moratorium,
        load_macro_panel(),
        default_covariates(),
        block_rows=block_rows,
        until=cut,
        cause=cause,
    )
    written = None if traced else [n for n in default_covariates() if n in MACRO_CANDIDATES]
    log.info(
        "Reading %s%s",
        source.source,
        f" in {workers} process(es)" if traced else ", writing the likelihood out",
    )
    result = fit_streamed(
        source,
        default_covariates(),
        formula,
        distribution=distribution,
        weights_col=WEIGHT,
        workers=workers if traced else 1,
        calendar=written,
    )
    record = result.blocks
    assert record is not None
    described = fit_description(
        (result.n_episodes, int(record.loan_months)),
        formula,
        as_of=as_of,
        moratorium=moratorium,
        distribution=distribution,
        cause=cause,
    )
    fingerprint = fit_fingerprint(**described)
    path = save_fit(result, fingerprint, {**described, "minutes": result.elapsed_seconds / 60})
    log.info(
        "%s cells in %d blocks, %.2f GB stored, %d evaluations, %.1f minutes",
        f"{record.rows:,}",
        record.blocks,
        record.stored_bytes / 1e9,
        record.evaluations,
        result.elapsed_seconds / 60,
    )
    log.info("saved as %s to %s", fingerprint, path)
    return result


def fit_description(
    counted: pd.DataFrame | tuple[int, int],
    formula: str,
    *,
    as_of: str,
    moratorium: str,
    distribution: str = DISTRIBUTION,
    weights_col: str | None = "loan_months",
    ancillary: str | None = None,
    likelihood: Likelihood | None = None,
    cause: str = DEFAULT_CAUSE,
) -> dict[str, object]:
    """What a report's fit is cached under, and so how any command finds it again.

    ``counted`` is the panel, or its row and loan-month counts when the rows were never held
    as a frame -- a fit streamed from the cell file counts them as it reads.

    The moratorium policy is in it because the two treatments can produce panels of similar
    size, and a censor fit silently reused for exclude would compare a model with itself.
    The ancillary formula, the likelihood and the cause enter only when they are not the
    defaults, so the report's own Weibull keeps the name it has always had -- and a
    prepayment fit, which reads the same cells and the same formula and would otherwise
    collide with the default fit's fingerprint, does not.
    """
    from creditsurv.models.aft import Likelihood as Likelihoods

    if isinstance(counted, tuple):
        rows, loan_months = counted
    else:
        rows = len(counted)
        loan_months = int(counted[weights_col].sum()) if weights_col else rows
    described: dict[str, object] = {
        "as_of": as_of,
        "moratorium": moratorium,
        "formula": formula,
        "distribution": distribution,
        "weights_col": weights_col,
        "rows": rows,
        "loan_months": loan_months,
    }
    if cause != DEFAULT_CAUSE:
        described["cause"] = cause
    if ancillary is not None:
        described["ancillary"] = ancillary
    if likelihood is not None and likelihood is not Likelihoods.INTERVAL_CENSORED:
        described["likelihood"] = likelihood.value
    return described


def fit_once(
    train: pd.DataFrame,
    covariates: list[str],
    formula: str,
    *,
    as_of: str,
    reuse: bool,
    moratorium: str = "exclude",
    start: pd.Series | None = None,
) -> object:
    """Fit the training half, reusing a cached model when one matches exactly.

    ``start`` is where the optimiser begins -- the selection's own fit of the specification,
    say -- and changes how long the fit takes, not where it ends.

    A fit on this population is two and a half hours, and the run that discovered that
    completed one and was then killed while writing its reports -- throwing away the
    expensive part and keeping nothing. The fit is therefore saved the moment it
    succeeds, before anything downstream can fail.

    Reuse is **opt-in**. The fingerprint covers the specification and the panel's size,
    which does not catch a re-aggregation that happens to leave the row count alone, so
    silently reusing would eventually mean reporting on a model built from data that no
    longer exists.
    """
    from creditsurv.data.panel import WEIGHT
    from creditsurv.data.store import fit_fingerprint, load_fit, save_fit
    from creditsurv.models.aft import fit_aft

    described = fit_description(train, formula, as_of=as_of, moratorium=moratorium)
    fingerprint = fit_fingerprint(**described)

    if reuse:
        cached = load_fit(fingerprint)
        if cached is not None:
            log.info("reusing the cached fit %s", fingerprint)
            return cached
        log.info("no cached fit %s; fitting", fingerprint)

    fitted = fit_aft(train, covariates, formula, weights_col=WEIGHT, initial_point=start)
    log.info("%.1f minutes", fitted.elapsed_seconds / 60)
    path = save_fit(fitted, fingerprint, {**described, "minutes": fitted.elapsed_seconds / 60})
    log.info("saved to %s", path)
    return fitted


def cached_fit(*, as_of: str, moratorium: str) -> Callable[..., FitResult]:
    """``fit_aft``, saved the moment each fit lands and read back when asked for again.

    For the extra fits a report makes -- the other distribution families and the shape
    test -- which ran 78 and 25 minutes on the training half and used to be thrown away, so
    that regenerating a report's prose cost them again. The fingerprint has the fields
    ``fit_once`` uses, plus the ancillary formula and the likelihood when they are not the
    defaults, so the Weibull the report already holds is found under its own name.
    """

    def fit(
        encoded: pd.DataFrame,
        covariates: Sequence[str],
        formula: str,
        *,
        distribution: str = DISTRIBUTION,
        likelihood: Likelihood | None = None,
        weights_col: str | None = None,
        ancillary: str | None = None,
        initial_point: pd.Series | None = None,
    ) -> FitResult:
        from creditsurv.data.store import fit_fingerprint, load_fit, save_fit
        from creditsurv.models import aft

        likelihood = likelihood or aft.Likelihood.INTERVAL_CENSORED
        described = fit_description(
            encoded,
            formula,
            as_of=as_of,
            moratorium=moratorium,
            distribution=distribution,
            weights_col=weights_col,
            ancillary=ancillary,
            likelihood=likelihood,
        )
        fingerprint = fit_fingerprint(**described)
        cached = load_fit(fingerprint)
        if isinstance(cached, aft.FitResult) and cached.log_likelihood < 0:
            log.info("reusing the cached %s fit %s", distribution, fingerprint)
            return cached
        result = aft.fit_aft(
            encoded,
            covariates,
            formula,
            distribution=distribution,
            likelihood=likelihood,
            weights_col=weights_col,
            ancillary=ancillary,
            initial_point=initial_point,
        )
        save_fit(result, fingerprint, {**described, "minutes": result.elapsed_seconds / 60})
        return result

    return fit


def selection_start(as_of: str, moratorium: str) -> pd.Series | None:
    """The coefficients the selection ended on, when it chose on the same half and table.

    ``None`` when no selection has been recorded, when it selected for another date or
    moratorium policy, or when its fit is no longer in the cache: the fit then starts from
    lifelines' own seed, as it always did. A record older than the cell table cannot match,
    because the fit is cached under the table's size and time of writing.
    """
    import json

    from creditsurv.data.store import cells_identity
    from creditsurv.models.fits import selected_fit

    path = reports_dir() / SELECTION_RECORD
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    if summary.get("as_of") != as_of or summary.get("moratorium") != moratorium:
        return None
    fitted = selected_fit(
        identity=cells_identity(moratorium),
        as_of=as_of,
        moratorium=moratorium,
        formula=str(summary["formula"]),
    )
    if fitted is None:
        return None
    log.info("starting from the selection's fit of the specification it chose")
    params: pd.Series = fitted.fitter.params_
    return params
