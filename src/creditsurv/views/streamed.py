"""The site's tables, accumulated over the cell file instead of over an expanded panel.

Every table behind the calibration pages is made of sums -- loan-months at risk, the defaults
among them, the defaults the model expected -- grouped by an age, a year, a segment or a
decile. Sums add over batches, so the table can be built a batch at a time and the ratios,
the product limits and the bands taken once at the end.

That is the whole of this module, and the reason for it is a number: scoring the training half
as a frame took the footprint to 15 GB on 59.7 million cells, and the table is now 72.7
million. The fits were moved off the panel first (see `creditsurv.models.blocks`); this moves
the views, which were the last thing holding it.

One quantity is not a sum: a decile of predicted risk needs the whole distribution before any
row can be assigned. It is taken in two passes -- a weighted histogram of the log hazard, then
the boundaries it implies -- which costs one more read and is exact to the width of a bin.

Two consequences worth knowing. The boundary lands within a bin of the true quantile, and a
histogram **cannot split a tie**: cells sharing a key share a hazard, so a boundary falling
inside such a cluster puts all of it on one side. The deciles therefore hold a tenth of the
exposure each to a fraction of a percent on the production table, rather than exactly, and
the difference is in the membership of the boundary cluster and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from creditsurv.backtest.runner import predicted_hazard
from creditsurv.data.panel import DEFAULT_CAUSE, WEIGHT
from creditsurv.views.calibration import curves_from, exposure_totals, rates_from, risk_sets

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from creditsurv.models.aft import FitResult

#: Bins of the weighted hazard histogram a decile boundary is read off. Ten thousand over the
#: log of a hazard that spans four orders of magnitude puts a boundary within 0.1% of its true
#: quantile, which is finer than the deciles themselves can distinguish.
HISTOGRAM_BINS: int = 10_000


def _hazard(fitted: FitResult, frame: pd.DataFrame, covariates: Sequence[str]) -> np.ndarray:
    return predicted_hazard(fitted, frame, covariates).to_numpy(dtype=float)


@dataclass(frozen=True)
class Recipe:
    """One additive table to accumulate, and the keys it is grouped by.

    ``build`` takes a batch of episodes and its hazards and returns a table whose every
    column is a sum. ``finish`` turns the accumulated sums into the table a view publishes --
    the curves, the rates -- and runs once, at the end, on a few hundred rows.
    """

    name: str
    keys: tuple[str, ...]
    build: Callable[[pd.DataFrame, Mapping[str, np.ndarray]], pd.DataFrame]
    finish: Callable[[pd.DataFrame], pd.DataFrame] = lambda table: table


def accumulate(
    blocks: Iterable[pd.DataFrame],
    models: Mapping[str, tuple[FitResult, Sequence[str]]],
    recipes: Sequence[Recipe],
) -> dict[str, pd.DataFrame]:
    """Every table in one pass over the cell file.

    A pass is the expensive thing here -- reading the parquet, rebuilding the macro family and
    scoring -- and there are a dozen tables, most of them the same three sums cut a different
    way. One pass scores each batch once per model and adds that batch's contribution to every
    table, which is the difference between one read of the training half and a dozen.

    ``models`` maps a name to a fit and the covariates it reads, so the comparison of
    distribution families costs no extra read either: both hazards of a batch are computed
    while the batch is in hand.
    """
    pieces: dict[str, list[pd.DataFrame]] = {recipe.name: [] for recipe in recipes}
    for frame in blocks:
        hazards = {
            name: _hazard(fitted, frame, covariates)
            for name, (fitted, covariates) in models.items()
        }
        for recipe in recipes:
            pieces[recipe.name].append(recipe.build(frame, hazards))
    return {
        recipe.name: recipe.finish(_summed(pieces[recipe.name], recipe.keys)) for recipe in recipes
    }


def risk_sets_over(
    blocks: Iterable[pd.DataFrame],
    fitted: FitResult,
    covariates: Sequence[str],
    *,
    groups: Callable[[pd.DataFrame], pd.Series | None] | None = None,
    cause: str = DEFAULT_CAUSE,
) -> pd.DataFrame:
    """Survival by age and group, summed over the batches and turned into curves once."""
    pieces = [
        risk_sets(
            frame,
            _hazard(fitted, frame, covariates),
            groups=None if groups is None else groups(frame),
            cause=cause,
        )
        for frame in blocks
    ]
    summed = _summed(pieces, ["group", "age"])
    return curves_from(summed)


def totals_over(
    blocks: Iterable[pd.DataFrame],
    fitted: FitResult,
    covariates: Sequence[str],
    dimensions: Callable[[pd.DataFrame], Mapping[str, pd.Series]],
    *,
    cause: str = DEFAULT_CAUSE,
) -> pd.DataFrame:
    """Defaults against expectation by any dimensions, summed over the batches."""
    pieces = []
    keys: list[str] = []
    for frame in blocks:
        dimension = dimensions(frame)
        keys = list(dimension)
        pieces.append(exposure_totals(frame, _hazard(fitted, frame, covariates), dimension, cause))
    return rates_from(_summed(pieces, keys))


def _summed(pieces: Sequence[pd.DataFrame], keys: Sequence[str]) -> pd.DataFrame:
    """Add the batches' tables together on their keys.

    Every column of these tables is a sum, so addition is the whole of the merge. The keys
    are read back as columns rather than an index because that is how the views and the site
    hand them on.
    """
    if not pieces:
        message = "No batches to accumulate; the window selected nothing."
        raise ValueError(message)
    joined = pd.concat(pieces, ignore_index=True)
    columns = [name for name in joined.columns if name not in set(keys)]
    return joined.groupby(list(keys), observed=True, as_index=False)[columns].sum()


def decile_boundaries(
    blocks: Iterable[pd.DataFrame],
    fitted: FitResult,
    covariates: Sequence[str],
    *,
    buckets: int = 10,
    weights_col: str = WEIGHT,
    bins: int = HISTOGRAM_BINS,
) -> np.ndarray:
    """The hazards that divide the book into ``buckets`` of equal exposure.

    A weighted histogram of the **log** hazard, because the hazard spans four orders of
    magnitude and equal-width bins on its own scale would put nine tenths of the book in the
    first bin. The boundaries come back on the hazard's own scale, so a later pass can assign
    a row by comparison without repeating the histogram.
    """
    edges: np.ndarray | None = None
    counts = np.zeros(bins)
    for frame in blocks:
        hazard = _hazard(fitted, frame, covariates)
        weight = frame[weights_col].to_numpy(dtype=float)
        logged = np.log(np.clip(hazard, 1e-12, None))
        if edges is None:
            # The first batch fixes the range. A later batch outside it is clipped into the
            # end bins, which moves a boundary by at most one bin's width.
            edges = np.linspace(logged.min() - 1.0, logged.max() + 1.0, bins + 1)
        counts += np.histogram(np.clip(logged, edges[0], edges[-1]), bins=edges, weights=weight)[0]
    if edges is None:
        message = "No batches to take a histogram over; the window selected nothing."
        raise ValueError(message)

    cumulative = np.cumsum(counts)
    total = cumulative[-1]
    wanted = [total * (index + 1) / buckets for index in range(buckets - 1)]
    positions = np.searchsorted(cumulative, wanted)
    return np.exp(edges[1:][np.minimum(positions, bins - 1)])


def deciles_of(hazard: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Which bucket each row falls in, given boundaries taken over the whole book."""
    return np.searchsorted(boundaries, np.asarray(hazard, dtype=float), side="left")
