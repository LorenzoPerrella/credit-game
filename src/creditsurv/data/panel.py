"""Canonical panel schema, episode splitting and the interval-censoring encoding.

Both data sources -- the synthetic generator and the Freddie Mac loader -- produce
the same canonical loan-month panel, so everything downstream is written once.

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

from itertools import pairwise
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

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


#: Longest loan age an episode can run to, in months. A thirty-year mortgage is 360,
#: and the final open-ended age band has to close somewhere for the likelihood to
#: evaluate.
MAX_AGE_MONTHS: Final = 360


def _months_to_periods(months: pd.Series) -> pd.PeriodIndex:
    """Month ordinals since year zero, back to a monthly PeriodIndex.

    Built from labels rather than through ``PeriodIndex(year=..., month=...)``, whose
    keyword form is deprecated in pandas and absent from its type stubs.
    """
    labels = (months // 12).astype(str) + "-" + (months % 12 + 1).astype(str).str.zfill(2)
    return pd.PeriodIndex(labels, freq="M")


def cells_to_episodes(
    cells: pd.DataFrame,
    macro: pd.DataFrame,
    *,
    lag_months: int = 3,
    max_age: int = MAX_AGE_MONTHS,
) -> pd.DataFrame:
    """Turn aggregated cells into weighted episodes the fitter can read.

    Cells carry a loan age *band* rather than a month, so episodes are the intervals
    between band edges rather than single months. The encoding is unchanged — that is
    the point of stating it in terms of bounds rather than months in the first place:
    a survivor contributes ``log[S(stop)/S(start)]`` and a default
    ``log[1 - S(stop)/S(start)]`` whatever the width of the interval.

    Band edges are read off the data rather than passed in. The ``age`` column holds
    each band's lower edge, so the distinct values *are* the edges, and deriving them
    means a cell table can never disagree with the bands it was built with.

    Macro covariates are recomputed here from vintage and age, which is why they were
    kept out of the grouping key: ``period = vintage + age``, so nothing was lost by
    leaving them out and the cardinality was spared. Only the two that depend on
    nothing else are rebuilt -- unemployment gap and financial conditions. Anything
    needing a loan-level quantity that is not in the key, such as the refinancing
    incentive, would have to have that quantity added to the specification first.
    """
    if cells.empty:
        message = "No cells to expand."
        raise PanelValidationError(message)

    episodes = cells.copy()
    edges = sorted(int(edge) for edge in episodes[AGE].unique())
    # Each band runs to the next edge; the last one runs to the horizon.
    upper_of = dict(pairwise(edges))
    upper_of[edges[-1]] = max_age

    episodes[AGE_START] = episodes[AGE].astype(float)
    episodes[AGE_STOP] = episodes[AGE].map(upper_of).astype(float)

    quarter = episodes["vintage"].str.extract(r"(\d{4})Q(\d)")
    orig_month = quarter[0].astype(int) * 12 + (quarter[1].astype(int) - 1) * 3
    # A period ordinal in months since year zero, so age can simply be added.
    observation = orig_month + episodes[AGE_START].astype(int)

    lagged = macro.shift(lag_months)
    macro_index = pd.PeriodIndex(macro.index)
    macro_month = macro_index.year * 12 + (macro_index.month - 1)
    unemployment = pd.Series(lagged["unemployment_rate"].to_numpy(), index=macro_month)
    conditions = pd.Series(lagged["nfci"].to_numpy(), index=macro_month)

    episodes["unemp_gap"] = (
        observation.map(unemployment).to_numpy() - orig_month.map(unemployment).to_numpy()
    )
    episodes["nfci_lagged"] = observation.map(conditions).to_numpy()

    # Mark-to-market leverage, from the national house price index. Derived here
    # rather than carried in the grouping key because it is a function of orig_ltv
    # and the macro path, both of which the key already holds -- carrying it was
    # doubling the cell count for information already there.
    #
    # The index rather than Freddie's own per-loan ELTV, which would be better if it
    # were usable: its coverage runs from 0.8% of the 1999 vintage to 94% of 2021, so
    # a model built on it would estimate a different quantity in every decade.
    if "orig_ltv" in episodes.columns:
        prices = pd.Series(macro["hpi"].to_numpy(), index=macro_month)
        at_origination = orig_month.map(prices).to_numpy()
        now = observation.map(prices).to_numpy()
        original = episodes["orig_ltv"].to_numpy(dtype=float)
        episodes["cltv_drift"] = original * at_origination / now - original

    # Calendar columns, so a split can be taken on time without recomputing them.
    # The episode is dated at its start: a band spans several months and has to be
    # attributed to one of them, and the start is the only choice that cannot place
    # an episode after a reporting date its loan was still performing at.
    episodes["orig_period"] = _months_to_periods(orig_month)
    episodes["period"] = _months_to_periods(observation)

    defaulted = episodes[EVENT].to_numpy(dtype=bool)
    start = episodes[AGE_START].to_numpy(dtype=float)
    stop = episodes[AGE_STOP].to_numpy(dtype=float)
    episodes[LOWER_BOUND] = np.where(defaulted, start, stop)
    episodes[UPPER_BOUND] = np.where(defaulted, stop, np.inf)
    episodes[EXACT_OBSERVATION] = False

    return episodes.dropna(subset=["unemp_gap", "nfci_lagged"]).reset_index(drop=True)
