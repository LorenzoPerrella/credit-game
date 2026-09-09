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

from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

LOAN_ID: Final = "loan_id"
AGE: Final = "age"
EVENT: Final = "event"
AGE_START: Final = "age_start"
AGE_STOP: Final = "age_stop"
LOWER_BOUND: Final = "lower_bound"
UPPER_BOUND: Final = "upper_bound"
EXACT_OBSERVATION: Final = "exact_observation"

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
