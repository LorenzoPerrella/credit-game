"""The backtest over the declared cuts: fit, score, anchor, grade.

`backtest/runner.py` owns the pieces -- the cuts, the window length, the acceptance criteria,
one window's score and the predicted hazard. What was missing was the campaign over them, and
it was written inline in the command that calls it, three closures and five numbered steps deep.

Five things happen here, in this order, and the order is the point:

1. **each declared cut, judged on the months after it.** One fit per cut, warm-started from the
   cut before and estimated on everything up to its own cut, so no window is scored by a model
   that saw it;
2. **the development model**, on everything up to the reporting date, which the anchoring and
   the grades are of;
3. **the level**, anchored on a window of its own -- not the development window, whose months
   the coefficients have already seen, and not the test window, which would be marking its own
   homework;
4. **the test window, scored twice**: the same ranking at two levels, because one multiplier on
   every hazard cannot change an order;
5. **the cycle in sample, a calendar year at a time** -- the dispersion a single out-of-time
   ratio hides, and the reason `docs/rules.md` says it cannot be read alone.

Nothing large is ever held. A window is read out of the parquet by its observation months,
which is about 3% of the table, and each is released before the next is read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pandas as pd

from creditsurv.backtest.metrics import cycle_in_band, grade_backtest
from creditsurv.backtest.runner import ACCEPTANCE, predicted_hazard, score
from creditsurv.config import DISTRIBUTION, MACRO_CANDIDATES
from creditsurv.data.panel import WEIGHT, cells_to_episodes, ended_in, month_ordinal
from creditsurv.data.store import fit_fingerprint, load_cells_window, save_fit
from creditsurv.models.aft import fit_streamed
from creditsurv.models.anchoring import anchor_on_window
from creditsurv.models.fits import cell_source, fit_description

if TYPE_CHECKING:
    from collections.abc import Sequence

    from creditsurv.models.aft import FitResult
    from creditsurv.models.anchoring import Anchor

log: Final = logging.getLogger(__name__)


class NoExposure(ValueError):
    """A window the cell table has no rows in. Nothing to score is not a result.

    A domain fact rather than a bad option, which is what it used to be raised as: the command
    line translates it, and the thing that knows a window is empty does not have to know it is
    being run from a command line.
    """


@dataclass(frozen=True)
class Campaign:
    """Everything the windows report shows, from one pass over the declared cuts."""

    #: One row per declared cut: what it saw, what it predicted, and what it cost.
    windows: pd.DataFrame
    #: Actual over expected by calendar year, in sample, with the band around it.
    cycle: pd.DataFrame
    #: The multiplier the level is anchored by, and the window it was measured on.
    anchor: Anchor
    #: The test window scored at both levels, unanchored and anchored.
    level: pd.DataFrame
    #: The master scale's grades, with the interval around what each one's loans did.
    grades: pd.DataFrame
    #: Every criterion assessed, one frame per window.
    acceptance: list[pd.DataFrame]
    #: The model the anchoring and the grades are of, so a caller can report on it.
    development: FitResult


def run_campaign(
    *,
    moratorium: str,
    covariates: Sequence[str],
    formula: str,
    macro: pd.DataFrame,
    declared: Sequence[tuple[pd.Period, pd.Period]],
    development_cut: pd.Period,
    anchor_window: tuple[str, str],
    block_rows: int,
) -> Campaign:
    """Fit at each declared cut, score the months after it, then anchor and grade."""
    first, last = anchor_window
    rows: list[dict[str, object]] = []
    criteria: list[pd.DataFrame] = []
    previous: FitResult | None = None

    for cut, until in declared:
        log.info("window %s to %s: fitting on everything up to the cut", cut, until)
        model = _fitted_to(
            cut,
            previous,
            moratorium=moratorium,
            covariates=covariates,
            formula=formula,
            macro=macro,
            block_rows=block_rows,
        )
        previous = model
        window = _scored(
            month_ordinal(cut) + 1,
            month_ordinal(until),
            f"{cut} to {until}",
            moratorium=moratorium,
            covariates=covariates,
            macro=macro,
        )
        result = score(model, window, covariates, as_of=cut)
        summary = result.summary()
        rows.append(
            {
                "cut": str(cut),
                "until": str(until),
                "loan_months": summary["loan_months"],
                "actual_defaults": summary["actual_defaults"],
                "expected_defaults": summary["expected_defaults"],
                "actual_over_expected": summary["actual_over_expected"],
                "gini": summary["gini"],
                "minutes": round(model.elapsed_seconds / 60, 1),
            }
        )
        criteria.append(ACCEPTANCE.assess(result).assign(window=f"{cut} to {until}"))
        log.info("actual over expected %.4f, Gini %.4f", result.actual_over_expected, result.gini)
        del window

    log.info("the development model, on everything up to %s", development_cut)
    development = _fitted_to(
        development_cut,
        previous,
        moratorium=moratorium,
        covariates=covariates,
        formula=formula,
        macro=macro,
        block_rows=block_rows,
    )

    anchoring = _scored(
        month_ordinal(pd.Period(first, freq="M")),
        month_ordinal(pd.Period(last, freq="M")),
        f"the anchoring window {first} to {last}",
        moratorium=moratorium,
        covariates=covariates,
        macro=macro,
    )
    anchor = anchor_on_window(
        predicted_hazard(development, anchoring, covariates),
        ended_in(anchoring),
        anchoring[WEIGHT],
        anchoring["period"],
        window=(first, last),
    )
    log.info("multiplier %.4f on %s to %s", anchor.multiplier, *anchor.window)
    del anchoring

    test = _scored(
        month_ordinal(pd.Period(last, freq="M")) + 1,
        None,
        f"the months after {last}",
        moratorium=moratorium,
        covariates=covariates,
        macro=macro,
    )
    unanchored = score(development, test, covariates, as_of=development_cut)
    exposure = test[WEIGHT].astype(float)
    events = exposure * ended_in(test)
    scaled = pd.Series(
        anchor.apply(predicted_hazard(development, test, covariates)), index=test.index
    )
    anchored = score(development, test, covariates, as_of=development_cut, hazard=scaled)
    level = pd.DataFrame(
        [
            {"model": "unanchored", **unanchored.summary()},
            {"model": "anchored", **anchored.summary()},
        ]
    ).drop(columns="as_of")
    grades = grade_backtest(scaled, pd.Series(events.to_numpy()), exposure)
    # The criteria of the published model are the criteria of the model **as published**: the
    # multiplier is part of it. The Gini is the same on both rows because one multiplier on
    # every hazard cannot change an order, which is the point of anchoring that way.
    criteria.append(ACCEPTANCE.assess(unanchored).assign(window=f"after {last}, unanchored"))
    criteria.append(ACCEPTANCE.assess(anchored).assign(window=f"after {last}, anchored"))
    del test

    log.info("the cycle, year by year in sample")
    by_year = _by_year(
        development,
        first_year=int(macro.index.min().year),
        last_cut=development_cut,
        moratorium=moratorium,
        covariates=covariates,
        macro=macro,
    )
    cycle = pd.concat(
        [cycle_in_band(by_year), by_year.assign(year=by_year["year"].astype(str))],
        ignore_index=True,
    )
    return Campaign(
        windows=pd.DataFrame(rows),
        cycle=cycle,
        anchor=anchor,
        level=level,
        grades=grades,
        acceptance=criteria,
        development=development,
    )


def _fitted_to(
    cut: pd.Period,
    start: FitResult | None,
    *,
    moratorium: str,
    covariates: Sequence[str],
    formula: str,
    macro: pd.DataFrame,
    block_rows: int,
) -> FitResult:
    """The model estimated on everything up to ``cut``, read from the cell file and saved."""
    source = cell_source(
        moratorium, macro, covariates, block_rows=block_rows, until=month_ordinal(cut)
    )
    result = fit_streamed(
        source,
        covariates,
        formula,
        distribution=DISTRIBUTION,
        weights_col=WEIGHT,
        initial_point=None if start is None else start.fitter.params_,
        # One process, because the written-out likelihood needs no more: fifteen bytes a row
        # puts the whole window under a gigabyte. Each cut is warm-started from the one
        # before, so this is a few Newton steps at 39 seconds a Hessian.
        workers=1,
        calendar=[name for name in covariates if name in MACRO_CANDIDATES],
    )
    record = result.blocks
    assert record is not None
    described = fit_description(
        (result.n_episodes, int(record.loan_months)),
        formula,
        as_of=str(cut),
        moratorium=moratorium,
        distribution=DISTRIBUTION,
    )
    save_fit(
        result, fit_fingerprint(**described), {**described, "minutes": result.elapsed_seconds / 60}
    )
    return result


def _episodes(
    opens: int | None,
    closes: int | None,
    *,
    moratorium: str,
    covariates: Sequence[str],
    macro: pd.DataFrame,
) -> pd.DataFrame | None:
    """The episodes of one window, and nothing else from the table.

    ``None`` where the window holds no cells, which for the in-sample cycle is every calendar
    year before the book opens.
    """
    cells = load_cells_window(moratorium, first=opens, last=closes)
    if cells.empty:
        return None
    return cells_to_episodes(cells, macro, covariates=list(covariates))


def _scored(
    opens: int | None,
    closes: int | None,
    what: str,
    *,
    moratorium: str,
    covariates: Sequence[str],
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """The same, where an empty window is an error: nothing to score is not a result."""
    frame = _episodes(opens, closes, moratorium=moratorium, covariates=covariates, macro=macro)
    if frame is None:
        message = f"No exposure in {what}; there is nothing to score there."
        raise NoExposure(message)
    return frame


def _by_year(
    development: FitResult,
    *,
    first_year: int,
    last_cut: pd.Period,
    moratorium: str,
    covariates: Sequence[str],
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """Actual over expected by calendar year, in sample, a year at a time."""
    years: list[dict[str, object]] = []
    for year in range(first_year, last_cut.year + 1):
        opens = month_ordinal(pd.Period(f"{year}-01", freq="M"))
        closes = min(month_ordinal(pd.Period(f"{year}-12", freq="M")), month_ordinal(last_cut))
        rows = _episodes(opens, closes, moratorium=moratorium, covariates=covariates, macro=macro)
        if rows is None:
            continue
        weight = rows[WEIGHT].astype(float)
        actual = float((weight * ended_in(rows)).sum())
        predicted = float((predicted_hazard(development, rows, covariates) * weight).sum())
        years.append(
            {
                "year": year,
                "loan_months": int(weight.sum()),
                "actual_defaults": round(actual, 1),
                "expected_defaults": round(predicted, 1),
                "actual_over_expected": actual / predicted if predicted > 0 else float("nan"),
            }
        )
        del rows
    return pd.DataFrame(years)
