"""Canonical panel schema, episode splitting and the interval-censoring encoding.

The Freddie Mac loader and the aggregated cells both resolve to the same canonical
episode schema, so everything downstream is written once.

The encoding in :func:`to_interval_censored` is the heart of the project. lifelines
documents interval censoring as a one-row-per-subject method and steers
time-varying covariates towards the semi-parametric ``CoxTimeVaryingFitter``. Both
can be had at once, because on an episode-split panel the interval-censored
likelihood with left truncation telescopes into exactly the discrete-time
likelihood with time-varying covariates.

For an episode covering loan age ``(a, a+1]``:

===================  =======  ===========  ===========  =======================================
Case                 entry    lower_bound  upper_bound  Likelihood contribution
===================  =======  ===========  ===========  =======================================
Survived the month   ``a``    ``a+1``      ``inf``      ``log S(a+1) + H(a) = log S(a+1)/S(a)``
Defaulted the month  ``a``    ``a``        ``a+1``      ``log[S(a) - S(a+1)] + H(a)``
===================  =======  ===========  ===========  =======================================

Both rows carry ``exact_observation = False``. That looks wrong at first sight and
is not: lifelines enforces ``lower == upper`` if and only if the event flag is
true, and uses that flag to mean "the event time is known *exactly*". Genuine
interval knowledge -- we know the month, never the day -- is represented as a
censored row with finite bounds. A right-censored row is the same construct with
an infinite upper bound.
"""

from __future__ import annotations

import warnings
from itertools import pairwise
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from creditsurv.features import MACRO_DERIVED, add_macro_family

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

LOAN_ID: Final = "loan_id"
AGE: Final = "age"
EVENT: Final = "event"
AGE_START: Final = "age_start"
AGE_STOP: Final = "age_stop"
LOWER_BOUND: Final = "lower_bound"
UPPER_BOUND: Final = "upper_bound"
EXACT_OBSERVATION: Final = "exact_observation"
WEIGHT: Final = "n"

#: Columns every canonical loan-month panel must carry.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (LOAN_ID, AGE, EVENT)


class PanelValidationError(ValueError):
    """Raised when a loan-month panel violates a structural invariant."""


def validate_episodes(panel: pd.DataFrame) -> None:
    """Check the structural invariants an episode panel must satisfy.

    Time-varying pipelines fail quietly rather than loudly: a duplicated
    loan-month double-counts its likelihood contribution, a gap silently drops
    exposure, and an event on a non-terminal row implies the loan kept paying
    after defaulting. None of these raise on their own, and all of them bias the
    fit. They are checked explicitly instead.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in panel.columns]
    if missing:
        message = f"Panel is missing required column(s): {missing}"
        raise PanelValidationError(message)

    if panel.empty:
        message = "Panel is empty."
        raise PanelValidationError(message)

    if (panel[AGE] < 0).any():
        message = "Panel contains negative loan ages."
        raise PanelValidationError(message)

    duplicated = panel.duplicated(subset=[LOAN_ID, AGE]).sum()
    if duplicated:
        message = f"Panel contains {duplicated} duplicated loan-month row(s)."
        raise PanelValidationError(message)

    ordered = panel.sort_values([LOAN_ID, AGE], kind="stable")
    grouped = ordered.groupby(LOAN_ID, observed=True)

    # Ages must tile each loan's history in unit steps, with no gaps.
    steps = grouped[AGE].diff()
    if not bool(((steps.isna()) | (steps == 1)).all()):
        message = "Panel contains gaps or non-unit steps in loan age."
        raise PanelValidationError(message)

    events_per_loan = grouped[EVENT].sum()
    if bool((events_per_loan > 1).any()):
        offenders = int((events_per_loan > 1).sum())
        message = f"{offenders} loan(s) carry more than one event."
        raise PanelValidationError(message)

    # An event may only fall on a loan's final observed month.
    event_age = ordered.loc[ordered[EVENT].astype(bool)].groupby(LOAN_ID, observed=True)[AGE].max()
    final_age = grouped[AGE].max()
    misplaced = int((event_age != final_age.reindex(event_age.index)).sum())
    if misplaced:
        message = f"{misplaced} loan(s) record an event before their final month."
        raise PanelValidationError(message)


def to_counting_process(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the half-open episode bounds ``(age_start, age_stop]``."""
    validate_episodes(panel)
    episodes = panel.copy()
    episodes[AGE_START] = episodes[AGE].astype(float)
    episodes[AGE_STOP] = episodes[AGE_START] + 1.0
    return episodes


def to_interval_censored(panel: pd.DataFrame) -> pd.DataFrame:
    """Encode an episode panel for ``fit_interval_censoring`` with left truncation.

    See the module docstring for the encoding and why the event flag is always
    ``False``.
    """
    episodes = to_counting_process(panel)
    defaulted = episodes[EVENT].to_numpy(dtype=bool)
    start = episodes[AGE_START].to_numpy(dtype=float)
    stop = episodes[AGE_STOP].to_numpy(dtype=float)

    episodes[LOWER_BOUND] = np.where(defaulted, start, stop)
    episodes[UPPER_BOUND] = np.where(defaulted, stop, np.inf)
    # Never True: the reporting interval tells us the month, never the day.
    episodes[EXACT_OBSERVATION] = False
    return episodes


def model_frame(panel: pd.DataFrame, covariates: Sequence[str]) -> pd.DataFrame:
    """Reduce an encoded panel to the columns lifelines should see.

    Passing the full panel would let identifiers and raw macro columns into the
    design matrix through a formula wildcard, so the frame is narrowed explicitly.
    """
    columns = [*covariates, AGE_START, LOWER_BOUND, UPPER_BOUND, EXACT_OBSERVATION]
    missing = [column for column in columns if column not in panel.columns]
    if missing:
        message = f"Encoded panel is missing column(s): {missing}"
        raise PanelValidationError(message)
    return panel.loc[:, columns].copy()


def model_blocks(
    panel: pd.DataFrame,
    covariates: Sequence[str],
    *,
    rows: int,
    weights_col: str | None = None,
) -> Iterator[pd.DataFrame]:
    """:func:`model_frame` a block of ``rows`` at a time, with the weight alongside.

    For :func:`creditsurv.models.blocks.fit_interval_censoring_in_blocks`. The whole
    narrowed frame is never built: on the production panel it would be a second copy of
    the largest object the pipeline holds.
    """
    if rows <= 0:
        message = "rows must be positive."
        raise ValueError(message)
    for start in range(0, len(panel), rows):
        block = panel.iloc[start : start + rows]
        frame = model_frame(block, covariates)
        if weights_col is not None:
            frame[weights_col] = block[weights_col].to_numpy()
        yield frame


def aggregate_episodes(
    panel: pd.DataFrame,
    covariates: Sequence[str],
    *,
    weight_col: str = WEIGHT,
) -> pd.DataFrame:
    """Collapse identical episodes into weighted cells.

    Episodes agreeing on every covariate *and* on their interval bounds are
    exchangeable, so they can be replaced by one row carrying a count. The
    likelihood treats that count as a frequency weight and is unchanged, which
    :mod:`tests.test_aggregation` asserts rather than assumes.

    The bounds must be part of the key. They are functions of loan age, so
    grouping on covariates alone would merge different points of the hazard curve
    into one cell and destroy the time structure entirely.

    The weight is a **count of loan-months, never an exposure amount**. Weighting
    by outstanding balance estimates a value-weighted default rate rather than a
    borrower probability of default, which is not what Basel or IFRS 9 define. It
    also breaks inference: lifelines warns that non-integer weights bias the
    variance estimates, and integer counts keep the standard errors valid.

    Note this compresses very little at realistic covariate counts -- see
    ``docs/methodology.md``. It is implemented because being able to show the
    aggregation is lossless is worth more than the speed-up it rarely delivers.
    """
    key = [*covariates, AGE_START, LOWER_BOUND, UPPER_BOUND, EXACT_OBSERVATION]
    missing = [column for column in key if column not in panel.columns]
    if missing:
        message = f"Encoded panel is missing column(s): {missing}"
        raise PanelValidationError(message)

    grouped = (
        panel.loc[:, key]
        .groupby(key, observed=True, dropna=False, sort=False)
        .size()
        .reset_index(name=weight_col)
    )
    grouped[weight_col] = grouped[weight_col].astype("int64")
    return grouped


def right_censored_frame(panel: pd.DataFrame, covariates: Sequence[str]) -> pd.DataFrame:
    """Reduce an encoded panel to the columns for a right-censored fit.

    The same episodes, read the other way: the duration is the episode end and
    the event flag is taken at face value, so a default is treated as having
    happened exactly at month end. That is the approximation the interval form
    exists to avoid, and fitting both on identical episodes is what makes the
    cost of the approximation measurable.
    """
    columns = [*covariates, AGE_START, AGE_STOP, EVENT]
    missing = [column for column in columns if column not in panel.columns]
    if missing:
        message = f"Encoded panel is missing column(s): {missing}"
        raise PanelValidationError(message)
    frame = panel.loc[:, columns].copy()
    frame[EVENT] = frame[EVENT].astype(bool)
    return frame


def at_origination(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per loan, holding its covariates as at ``age = 0``.

    This is the state a lifetime PD is quoted from: what was known when the loan
    was written. :func:`to_loan_level` takes the *last* row instead, which is what
    is needed to score a loan already on the books.
    """
    validate_episodes(panel)
    ordered = panel.sort_values([LOAN_ID, AGE], kind="stable")
    return ordered.groupby(LOAN_ID, observed=True).first().reset_index()


def to_loan_level(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per loan: observed duration and terminal status.

    Needed wherever the unit of analysis is the loan rather than the loan-month --
    Kaplan-Meier curves, discrimination metrics and calibration by vintage. Metrics
    computed on the episode panel would weight long-lived loans by their length.
    """
    validate_episodes(panel)
    ordered = panel.sort_values([LOAN_ID, AGE], kind="stable")
    grouped = ordered.groupby(LOAN_ID, observed=True)

    loans = grouped.last()
    loans["duration"] = grouped[AGE].max().to_numpy(dtype=float) + 1.0
    loans[EVENT] = grouped[EVENT].max().to_numpy(dtype=bool)
    return loans.reset_index()


def to_loan_level_weighted(
    episodes: pd.DataFrame, *, weight: str = WEIGHT, age: str = AGE, event: str = EVENT
) -> pd.DataFrame:
    """Recover the loan-level duration distribution from a weighted episode panel.

    Aggregation destroys the loan id -- a cell is a count of loan-months, not a
    subject -- so :func:`to_loan_level` cannot run. The duration distribution is
    still recoverable, and exactly rather than approximately, from the counts at
    risk at each age.

    Write ``R(a)`` for the loan-months observed at age ``a`` and ``d(a)`` for those
    ending in default. A loan at risk at ``a`` either defaults, leaves the window
    (prepayment or the end of the observation period, both censoring), or is at risk
    again at ``a + 1``. So::

        censored(a) = R(a) - d(a) - R(a + 1)

    which is the identity Kaplan-Meier is built on, read backwards. The result is
    one row per (age, outcome) carrying a count, and feeding it to a fitter with
    ``weights`` gives the same curve the loan-level panel would have, on the whole
    population rather than a sample of it.

    Durations follow :func:`to_loan_level`: a loan last observed at age ``a`` has
    duration ``a + 1``.
    """
    exposure = episodes[weight].to_numpy(dtype=float)
    working = pd.DataFrame(
        {
            age: episodes[age].to_numpy(dtype=int),
            "at_risk": exposure,
            "defaults": exposure * episodes[event].to_numpy(dtype=bool),
        }
    )
    counts = working.groupby(age, observed=True)[["at_risk", "defaults"]].sum()
    counts = counts.reindex(range(int(counts.index.max()) + 1), fill_value=0.0)

    at_risk = counts["at_risk"].to_numpy(dtype=float)
    defaults = counts["defaults"].to_numpy(dtype=float)
    # The last age has no successor: everything still at risk there is censored.
    survivors = np.append(at_risk[1:], 0.0)
    # A panel with gaps -- a loan absent for a month and back the next -- would make
    # this negative. Freddie Mac reports contiguously, so clipping is a guard rather
    # than a correction, but a silent negative weight would poison the fit.
    censored = np.maximum(at_risk - defaults - survivors, 0.0)

    duration = counts.index.to_numpy(dtype=float) + 1.0
    frame = pd.DataFrame(
        {
            "duration": np.concatenate([duration, duration]),
            event: np.concatenate([np.ones(len(duration), bool), np.zeros(len(duration), bool)]),
            weight: np.concatenate([defaults, censored]),
        }
    )
    return frame[frame[weight] > 0].sort_values("duration").reset_index(drop=True)


def duration_view(panel: pd.DataFrame, *, weights_col: str | None = None) -> pd.DataFrame:
    """One row per subject, or per (duration, outcome) when subjects are weighted.

    The single entry point for every estimator whose unit is the loan rather than
    the loan-month. Passing ``weights_col`` says the panel is aggregated, and the
    returned frame carries that column for the fitter's ``weights`` argument;
    omitting it takes the loan-level path. Callers then differ by one keyword
    instead of by a branch each.
    """
    if weights_col is None:
        loans = to_loan_level(panel)
        return loans.loc[:, ["duration", EVENT]]
    return to_loan_level_weighted(panel, weight=weights_col)


#: Longest loan age an episode can run to, in months. A thirty-year mortgage is 360,
#: and the final open-ended age band has to close somewhere for the likelihood to
#: evaluate.
MAX_AGE_MONTHS: Final = 360


#: pandas numbers monthly periods from January 1970; this module counts from year zero.
_EPOCH_MONTHS: Final = 1970 * 12


def _months_to_periods(months: pd.Series) -> pd.PeriodIndex:
    """Month ordinals since year zero, back to a monthly PeriodIndex.

    From the ordinals directly. The first version formatted ``"YYYY-MM"`` labels and
    parsed them back: the same periods, measured at 38 seconds for five million rows
    against 0.02 -- eight minutes a column at the exact key's 66 million cells, with a
    Python string per row while it ran.
    """
    ordinals = months.to_numpy(dtype=np.int64) - _EPOCH_MONTHS
    return pd.PeriodIndex(pd.arrays.PeriodArray(ordinals, dtype=pd.PeriodDtype("M")))


def cells_to_episodes(
    cells: pd.DataFrame,
    macro: pd.DataFrame,
    *,
    lag_months: int = 3,
    covariates: Sequence[str] | None = None,
    where: np.ndarray | pd.Series | None = None,
    step: int | None = None,
) -> pd.DataFrame:
    """Turn aggregated cells into weighted episodes the fitter can read.

    Cells carry the start of a fixed-width episode rather than a single month, so an
    episode spans ``(start, start + step]``. The encoding is unchanged — that is the
    point of stating it in terms of bounds rather than months: a survivor contributes
    ``log[S(stop)/S(start)]`` and a default ``log[1 - S(stop)/S(start)]`` whatever the
    width of the interval, which is the same conditional-survival construction a
    monthly panel uses.

    The width is read off the data rather than passed in, so a cell table can never
    disagree with the width it was built with.

    Macro covariates are recomputed here from the origination month and the age, which
    is why they were kept out of the grouping key: ``period = orig_month + age``, so
    nothing was lost by leaving them out and the cardinality was spared. See
    :func:`creditsurv.features.add_macro_family` for the family and why it is free.

    ``covariates`` narrows the macro family to what the caller will actually read.
    Building all thirteen costs nine unused ``float64`` columns, and this frame is the
    largest object the pipeline holds: on the production table it reached 5.9 GB, of
    which **44% was three categorical columns stored as Python strings**. At the cell
    counts an exact calendar key implies that is the difference between fitting and
    not, so the frame is built narrow rather than trimmed afterwards.

    ``where`` expands only the cells it selects, which is how the training and test
    halves are built without the whole panel existing first -- see
    :func:`creditsurv.backtest.splits.split_cells`. The caller's table is never written
    to and never copied whole: the first version copied it outright, then copied the
    result again to drop incomplete rows.
    """
    if cells.empty:
        message = "No cells to expand."
        raise PanelValidationError(message)

    # Episodes are fixed width, so the stop is the start plus the step. The step is
    # read off the data -- the spacing of the distinct ages -- so a cell table can
    # never disagree with the width it was built with. Off the whole table, not the
    # selection: a selection holding ages 0 and 12 is not a table of year-long episodes.
    if step is None:
        ages = sorted(int(age) for age in cells[AGE].unique())
        step = min((b - a) for a, b in pairwise(ages)) if len(ages) > 1 else 1

    if where is None:
        episodes = cells.copy(deep=False)
    else:
        selected = np.asarray(where, dtype=bool)
        if len(selected) != len(cells):
            message = f"where selects from {len(selected):,} rows, the table has {len(cells):,}."
            raise PanelValidationError(message)
        episodes = cells.iloc[np.flatnonzero(selected)].copy(deep=False)
        if episodes.empty:
            message = "No cells to expand."
            raise PanelValidationError(message)
    episodes.index = pd.RangeIndex(len(episodes))
    for column in episodes.columns:
        if episodes[column].dtype == object:
            episodes[column] = episodes[column].astype("category")

    start_ages = episodes[AGE].to_numpy(dtype=np.float32)
    episodes[AGE_START] = start_ages
    episodes[AGE_STOP] = start_ages + np.float32(step)

    orig_month = origination_months(episodes)
    # A period ordinal in months since year zero, so age can simply be added.
    observation = orig_month + episodes[AGE].astype(int)

    add_macro_family(episodes, macro, orig_month, observation, lag_months, names=covariates)

    # Calendar columns, so a split can be taken on time without recomputing them.
    # The episode is dated at its start: a band spans several months and has to be
    # attributed to one of them, and the start is the only choice that cannot place
    # an episode after a reporting date its loan was still performing at.
    episodes["orig_period"] = _months_to_periods(orig_month)
    episodes["period"] = _months_to_periods(observation)

    defaulted = episodes[EVENT].to_numpy(dtype=bool)
    start = episodes[AGE_START].to_numpy(dtype=np.float32)
    stop = episodes[AGE_STOP].to_numpy(dtype=np.float32)
    episodes[LOWER_BOUND] = np.where(defaulted, start, stop)
    # Infinity has no float32 hazard: the upper bound stays float64 because lifelines
    # compares it against one, and an overflow here would silently become a finite
    # bound, turning every censored row into an observed default.
    episodes[UPPER_BOUND] = np.where(defaulted, stop.astype(float), np.inf)
    episodes[EXACT_OBSERVATION] = False

    required = [name for name in MACRO_DERIVED if name in episodes.columns]
    complete = episodes[required].notna().all(axis=1).to_numpy()
    if complete.all():
        return episodes
    kept = episodes.loc[complete].copy(deep=False)
    kept.index = pd.RangeIndex(len(kept))
    return kept


def origination_months(cells: pd.DataFrame) -> pd.Series:
    """The origination month of each cell, as an ordinal in months since year zero.

    Two key shapes are accepted, and which one a table carries is the difference
    between a correct calendar and one two months early:

    * ``orig_month`` -- the month itself, which is what the aggregation now emits;
    * ``vintage`` -- the origination *quarter*, which it used to. Reconstructing the
      month from it takes the quarter's first month, and loans are not all written in
      it: the mean offset is **+2.15 months**, so every macro covariate is read that
      much late and the backtest boundary sits two months inside the training half.

    The quarterly branch is kept so an older cell table can still be read, and it warns
    rather than pretending the two are equivalent.
    """
    if "orig_month" in cells.columns:
        return cells["orig_month"].astype(int)

    warnings.warn(
        "This cell table is keyed by origination quarter, so the observation month is "
        "reconstructed from the quarter's first month and runs about two months early. "
        "Re-run `creditsurv aggregate` to key it by origination month.",
        stacklevel=2,
    )
    quarter = cells["vintage"].astype(str).str.extract(r"(\d{4})Q(\d)")
    return quarter[0].astype(int) * 12 + (quarter[1].astype(int) - 1) * 3


def observation_months(cells: pd.DataFrame) -> pd.Series:
    """The month each cell observes, as an ordinal in months since year zero.

    Origination month plus age, so a cell table can be divided in calendar time before
    it is expanded.
    """
    return origination_months(cells) + cells[AGE].astype(int)
