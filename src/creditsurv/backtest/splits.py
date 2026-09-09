"""Train and test splits for credit models.

A random split is the wrong default here. Credit models are used to write business
that does not exist yet, in an economy that has not happened yet, so the question
is never "can it predict a held-out row" but "does it hold up on a later cohort,
or in a later year". Those are different questions and get different splits.

**Every split is truncated in calendar time**, which is the point most easily
missed. Training on all loan-months of an early vintage means training on calendar
periods that overlap the test window, so the model has seen the macro conditions it
is about to be judged on. That is look-ahead even though no individual loan appears
in both halves, and it flatters the result exactly where the model is weakest --
during the downturns nobody had seen yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from creditsurv.data.panel import LOAN_ID, validate_episodes

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

PERIOD = "period"
ORIGINATION = "orig_period"


@dataclass(frozen=True)
class Split:
    """One train/test division, anchored at a reporting date.

    ``as_of`` is the last period the model is allowed to have seen. Everything in
    ``train`` falls at or before it; everything in ``test`` falls after.
    """

    name: str
    as_of: pd.Period
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def n_train_loans(self) -> int:
        return int(self.train[LOAN_ID].nunique())

    @property
    def n_test_loans(self) -> int:
        return int(self.test[LOAN_ID].nunique())

    def describe(self) -> dict[str, object]:
        return {
            "split": self.name,
            "as_of": str(self.as_of),
            "train_loans": self.n_train_loans,
            "train_episodes": len(self.train),
            "test_loans": self.n_test_loans,
            "test_episodes": len(self.test),
        }


def _truncate(panel: pd.DataFrame, as_of: pd.Period) -> pd.DataFrame:
    return panel[panel[PERIOD] <= as_of].reset_index(drop=True)


def as_of_split(panel: pd.DataFrame, as_of: pd.Period, *, name: str = "as_of") -> Split:
    """Train on everything known by ``as_of``; test the loans still performing then.

    The existing-book question: given the portfolio on the books at the reporting
    date, how well does the model predict what happens next? This is the split a
    walk-forward backtest repeats.
    """
    train = _truncate(panel, as_of)
    if train.empty:
        message = f"No episodes at or before {as_of}."
        raise ValueError(message)

    # On the books at as_of means written by then *and* not yet terminated. The
    # origination condition is easy to omit and lets later vintages in, whose test
    # window would then open years after the reporting date -- so their covariates
    # would describe a different economy from the one being scored.
    by_loan = panel.groupby(LOAN_ID, observed=True)
    last_seen = by_loan[PERIOD].max()
    written = by_loan[ORIGINATION].first()
    still_open = last_seen[(last_seen >= as_of) & (written <= as_of)].index
    future = panel[(panel[PERIOD] > as_of) & panel[LOAN_ID].isin(still_open)]

    return Split(name=name, as_of=as_of, train=train, test=future.reset_index(drop=True))


def out_of_time(panel: pd.DataFrame, as_of: pd.Period) -> Split:
    """Train on everything known by ``as_of``; test on cohorts written after it.

    The new-business question, and the harder one: the test loans share no history
    with the training set at all, so nothing about them was available when the model
    was fitted.
    """
    train = _truncate(panel, as_of)
    later_vintages = panel[panel[ORIGINATION] > as_of]
    return Split(
        name="out_of_time", as_of=as_of, train=train, test=later_vintages.reset_index(drop=True)
    )


def out_of_sample(
    panel: pd.DataFrame,
    as_of: pd.Period,
    *,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> Split:
    """Hold out a random set of loans from the same period.

    Deliberately the easy split. It isolates estimation noise from cohort and
    regime change, so the gap between this and :func:`out_of_time` says how much of
    any degradation is the economy moving rather than the sample being small. Read
    on its own it flatters the model.
    """
    truncated = _truncate(panel, as_of)
    loans = truncated[LOAN_ID].unique()
    rng = np.random.default_rng(seed)
    held_out = set(rng.choice(loans, size=int(len(loans) * test_fraction), replace=False))

    mask = truncated[LOAN_ID].isin(held_out)
    return Split(
        name="out_of_sample",
        as_of=as_of,
        train=truncated[~mask].reset_index(drop=True),
        test=truncated[mask].reset_index(drop=True),
    )


def walk_forward(
    panel: pd.DataFrame,
    as_of_dates: Sequence[pd.Period],
) -> list[Split]:
    """Repeat the as-of split at successive reporting dates, expanding the window.

    A single holdout says how the model did in one regime. Refitting at successive
    dates says whether it holds up across several, which is the difference between
    a result and a fluke.
    """
    return [as_of_split(panel, as_of, name=f"walk_forward_{as_of}") for as_of in as_of_dates]


def assert_no_lookahead(split: Split) -> None:
    """Raise if the training half contains anything from after the reporting date."""
    validate_episodes(split.train)
    latest = split.train[PERIOD].max()
    if latest > split.as_of:
        message = (
            f"Training data reaches {latest}, beyond the reporting date {split.as_of}. "
            "The model would be fitted on conditions it is about to be judged on."
        )
        raise ValueError(message)
    if not split.test.empty:
        earliest = split.test[PERIOD].min()
        if split.name != "out_of_sample" and earliest <= split.as_of:
            message = f"Test data starts {earliest}, at or before the reporting date."
            raise ValueError(message)
