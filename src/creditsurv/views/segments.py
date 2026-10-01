"""The sub-items a view can be opened by, each defined once.

Every view that is split by loan purpose, credit score band or vintage era takes its labels
from here, so the same loan-month carries the same label on the portfolio page and on the
calibration page. The bands follow ``PRODUCTION_EDGES``, the grid the cells are built on:
a view cut on a finer grid than the key would be inventing distinctions the model never saw.

Labels stay categorical. On the training half a column of strings is sixty million Python
objects; a categorical is sixty million bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from creditsurv.data.aggregate import PRODUCTION_EDGES
from creditsurv.data.panel import AGE

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Segment:
    """A way of splitting loan-months into named groups."""

    name: str
    title: str
    columns: tuple[str, ...]
    label: Callable[[pd.DataFrame], pd.Series]

    def available(self, frame: pd.DataFrame) -> bool:
        return all(column in frame.columns for column in self.columns)


def _categorical(column: str) -> Callable[..., pd.Series]:
    """The column's own codes. A page shows their labels, from :mod:`creditsurv.names`."""

    def label(frame: pd.DataFrame) -> pd.Series:
        values = frame[column]
        if not isinstance(values.dtype, pd.CategoricalDtype):
            values = values.astype("category")
        return values.rename(column)

    return label


def _banded(
    column: str, edges: tuple[float, ...], shown: Callable[[float], str]
) -> Callable[..., pd.Series]:
    # The outer edges are opened, so a value past the grid -- an LTV above 100 -- falls in the
    # outermost band rather than out of the view.
    bins = [-np.inf, *edges[1:-1], np.inf]
    labels = [f"{shown(low)} to {shown(high)}" for low, high in pairwise(edges)]

    def label(frame: pd.DataFrame) -> pd.Series:
        values = frame[column].to_numpy(dtype=float)
        banded = pd.cut(values, bins=bins, labels=labels, right=False)
        return pd.Series(banded, index=frame.index, name=column)

    return label


def _years(column: str, edges: tuple[int, ...]) -> Callable[..., pd.Series]:
    # Open at both ends, as the numeric bands are: a vintage before the first edge or after
    # the last belongs to the outermost era rather than to no era at all.
    inner = [f"{low} to {high - 1}" for low, high in pairwise(edges[1:-1])]
    labels = [f"up to {edges[1] - 1}", *inner, f"{edges[-2]} on"]
    bins = [-np.inf, *edges[1:-1], np.inf]

    def label(frame: pd.DataFrame) -> pd.Series:
        years = pd.PeriodIndex(frame[column]).year.to_numpy()
        banded = pd.cut(years, bins=bins, labels=labels, right=False)
        return pd.Series(banded, index=frame.index, name=column)

    return label


def _term(frame: pd.DataFrame) -> pd.Series:
    years = frame["term_years"].to_numpy()
    codes, levels = pd.factorize(years, sort=True)
    labelled = pd.Categorical.from_codes(
        codes, categories=pd.Index([f"{int(v)} years" for v in levels])
    )
    return pd.Series(labelled, index=frame.index, name="term_years")


def _number(value: float) -> str:
    return f"{value:.0f}"


SEGMENTS: Final[dict[str, Segment]] = {
    segment.name: segment
    for segment in (
        Segment("purpose", "Loan purpose", ("purpose",), _categorical("purpose")),
        Segment("occupancy", "Occupancy", ("occupancy",), _categorical("occupancy")),
        Segment(
            "mortgage_insurance",
            "Mortgage insurance",
            ("mortgage_insurance",),
            _categorical("mortgage_insurance"),
        ),
        Segment("buyer_type", "Buyer type", ("buyer_type",), _categorical("buyer_type")),
        Segment("term", "Original term", ("term_years",), _term),
        Segment(
            "fico",
            "Credit score band",
            ("credit_score",),
            _banded("credit_score", PRODUCTION_EDGES["credit_score"], _number),
        ),
        Segment(
            "ltv",
            "Loan-to-value band",
            ("original_ltv",),
            _banded("original_ltv", PRODUCTION_EDGES["original_ltv"], _number),
        ),
        Segment(
            "dti",
            "Debt-to-income band",
            ("debt_to_income",),
            _banded("debt_to_income", PRODUCTION_EDGES["debt_to_income"], _number),
        ),
        Segment(
            "vintage_era",
            "Vintage era",
            ("origination_period",),
            _years("origination_period", (1999, 2004, 2009, 2015, 2020, 2027)),
        ),
    )
}

#: Loan age bands, for calibration by seasoning.
AGE_BANDS: Final[tuple[int, ...]] = (0, 12, 24, 36, 60, 120, 240, 400)


def age_bands(frame: pd.DataFrame) -> pd.Series:
    """The seasoning band of each loan-month, in months of loan age."""
    labels = [f"{low} to {high - 1}" for low, high in pairwise(AGE_BANDS)]
    banded = pd.cut(frame[AGE].to_numpy(), bins=list(AGE_BANDS), labels=labels, right=False)
    return pd.Series(banded, index=frame.index, name="age_band")


def calendar_years(frame: pd.DataFrame, column: str = "period") -> pd.Series:
    """The calendar year each loan-month is observed in."""
    return pd.Series(pd.PeriodIndex(frame[column]).year, index=frame.index, name="year")


def available(frame: pd.DataFrame) -> list[Segment]:
    """The segments a frame carries the columns for."""
    return [segment for segment in SEGMENTS.values() if segment.available(frame)]
