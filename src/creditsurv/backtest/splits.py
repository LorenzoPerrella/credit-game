"""Splitting the panel in calendar time, at a single reporting date.

One cut, and the reason it is one is worth stating. A walk-forward backtest refits
the model at four or five dates to ask whether a result held across regimes; that is
a good question and an expensive one, and on this panel a fit is hours rather than
seconds. The cut here is late on purpose -- everything up to it trains the model,
because a credit model wants every loan-month it can get.

**The cut is in calendar time, not by loan**, which is the point most easily missed.
Holding out random loans still trains the model on the calendar periods it is about
to be judged on, so it has already seen the macro conditions of the test window. That
is look-ahead even though no loan appears on both sides, and it flatters the result
exactly where the model is weakest -- during the downturns nobody had seen yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from creditsurv.data.panel import (
    WEIGHT,
    cells_to_episodes,
    episode_step,
    observation_months,
    validate_episodes,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

PERIOD = "period"
ORIGINATION = "origination_period"


@dataclass(frozen=True)
class Split:
    """One train/test division, anchored at a reporting date.

    ``as_of`` is the last period the model is allowed to have seen. Everything in
    ``train`` falls at or before it; everything in ``test`` falls after.
    """

    as_of: pd.Period
    train: pd.DataFrame
    test: pd.DataFrame
    #: The last period the test window covers, when it has an end. A window with no end
    #: runs to wherever the data stops, which is the right thing for the one reporting date
    #: the model is published at and the wrong thing for a cut in the middle of the history:
    #: the 2018 cut judged on everything after it would be judged on the pandemic as well.
    until: pd.Period | None = None

    @staticmethod
    def _exposure(frame: pd.DataFrame) -> int:
        """Loan-months on either side, however the panel is shaped.

        An aggregated panel has a weight and no loan identifier; a loan-level one has
        the identifier and no weight. Exposure is the quantity both can report, and
        the one a backtest is actually sized by.
        """
        if WEIGHT in frame.columns:
            return int(frame[WEIGHT].sum())
        return len(frame)

    @staticmethod
    def _defaults(frame: pd.DataFrame) -> int:
        if WEIGHT in frame.columns:
            return int(frame.loc[frame["event"].astype(bool), WEIGHT].sum())
        return int(frame["event"].sum())

    def describe(self) -> dict[str, object]:
        return {
            "as_of": str(self.as_of),
            "until": None if self.until is None else str(self.until),
            "train_loan_months": self._exposure(self.train),
            "train_defaults": self._defaults(self.train),
            "train_rows": len(self.train),
            "test_loan_months": self._exposure(self.test),
            "test_defaults": self._defaults(self.test),
            "test_rows": len(self.test),
        }


def cell_split(cells: pd.DataFrame, as_of: pd.Period, *, until: pd.Period | None = None) -> Split:
    """Everything observed by ``as_of`` trains; everything after it tests.

    Aggregated cells have no loan identifier -- that is what aggregating means -- so
    a split that follows named loans across the boundary is not available. Splitting
    on the observation period asks the same question of the same data: everything the
    model could have seen by the reporting date fits it, everything after judges it.

    What is lost is any metric defined per loan. What is kept is the one a backtest is
    actually about: how much exposure sat on either side of the date, and how much of
    it defaulted.
    """
    train = cells[cells[PERIOD] <= as_of].reset_index(drop=True)
    after = cells[PERIOD] > as_of
    if until is not None:
        after &= cells[PERIOD] <= until
    test = cells[after].reset_index(drop=True)
    if train.empty:
        message = f"No exposure at or before {as_of}."
        raise ValueError(message)
    return Split(as_of=as_of, train=train, test=test, until=until)


def split_cells(
    cells: pd.DataFrame,
    macro: pd.DataFrame,
    as_of: pd.Period,
    *,
    covariates: Sequence[str] | None = None,
    until: pd.Period | None = None,
) -> Split:
    """:func:`cell_split` of the expanded panel, without the whole panel ever existing.

    Expanding every cell and then splitting holds the panel and both of its halves at
    once: on the exact calendar key, about 6 GB for the panel and 6 GB more for the
    halves, on a 16 GB machine, before a single fit. The month a cell observes is fixed
    by its origination month and its age, so dividing the cells first gives the same
    halves -- which a test holds to the row -- and each half is expanded on its own.
    """
    cut = as_of.year * 12 + as_of.month - 1
    months = observation_months(cells).to_numpy()
    before = months <= cut
    after = ~before
    if until is not None:
        after &= months <= until.year * 12 + until.month - 1
    if not before.any():
        message = f"No exposure at or before {as_of}."
        raise ValueError(message)

    # Each half is taken out first and the table let go before either is expanded. Expanding
    # from the whole table held it beside the growing half, and the comparison of moratorium
    # policies that did so peaked at a 17.3 GB footprint on a 16 GB machine. The caller that
    # passes a table it does not keep frees it here.
    step = episode_step(cells)
    train_cells = cells.iloc[np.flatnonzero(before)]
    test_cells = cells.iloc[np.flatnonzero(after)]
    del cells

    train = cells_to_episodes(train_cells, macro, covariates=covariates, step=step)
    del train_cells
    if train.empty:
        message = f"No exposure at or before {as_of}."
        raise ValueError(message)
    test = (
        cells_to_episodes(test_cells, macro, covariates=covariates, step=step)
        if len(test_cells)
        else train.iloc[:0].copy()
    )
    return Split(as_of=as_of, train=train, test=test, until=until)


def assert_no_lookahead(split: Split) -> None:
    """Raise if the training half contains anything from after the reporting date."""
    if "loan_id" in split.train.columns:
        validate_episodes(split.train)
    latest = split.train[PERIOD].max()
    if latest > split.as_of:
        message = (
            f"Training data reaches {latest}, beyond the reporting date {split.as_of}. "
            "The model would be fitted on conditions it is about to be judged on."
        )
        raise ValueError(message)
