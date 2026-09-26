"""Non-parametric against parametric, opened by any segment.

Two tables carry every calibration view on the site.

:func:`survival_by_age` sets the Kaplan-Meier curve beside the model's, age by age, for the
whole book or for each group of a segment. Both are built from the same three sums over the
episodes at each age -- the loan-months at risk, the defaults among them and the defaults the
model expected -- so the two curves differ only in the hazard they chain:

    Kaplan-Meier  S(t) = prod over a < t of (1 - defaults(a) / at_risk(a))
    model         S(t) = prod over a < t of (1 - expected(a) / at_risk(a))

The second is :func:`creditsurv.models.nonparametric.predicted_survival_curve` split by group,
and the first is the product-limit estimator on the same risk sets. Taken from the episodes
rather than from reconstructed loan durations, loans that entered the panel late sit in the
risk sets of the ages they were observed at, which a duration view has to absorb. Greenwood's
formula gives the band.

:func:`actual_expected` compares defaults with expectations across any combination of
dimensions -- calendar year by segment, decile by segment -- with the totals added up by
``bincount`` rather than a group-by, which on the training half would first build a frame of
several gigabytes to fill a table of a few hundred rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from creditsurv.data.panel import AGE, DEFAULT_CAUSE, WEIGHT, ended_in

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The group label of a table computed over the whole book.
WHOLE_BOOK = "all loans"


def _codes(groups: pd.Series | None, rows: int) -> tuple[np.ndarray, list[str]]:
    if groups is None:
        return np.zeros(rows, dtype=np.int64), [WHOLE_BOOK]
    codes, levels = pd.factorize(groups, sort=True)
    return codes.astype(np.int64), [str(level) for level in levels]


def risk_sets(
    episodes: pd.DataFrame,
    hazard: np.ndarray | None = None,
    *,
    groups: pd.Series | None = None,
    cause: str = DEFAULT_CAUSE,
) -> pd.DataFrame:
    """The three sums a survival curve is made of, by group and loan age.

    Loan-months at risk, the defaults among them, and the defaults the model expected. All
    three are **additive over rows**, which is what lets the table be accumulated over the
    batches of the cell file and the curves derived once at the end -- see
    :func:`creditsurv.views.streamed.risk_sets_over`. Every ratio, product-limit and band in
    :func:`curves_from` is a function of these and of nothing else.
    """
    ages = episodes[AGE].to_numpy(dtype=np.int64)
    weight = episodes[WEIGHT].to_numpy(dtype=float)
    # Episodes carry the flag of the cause they were expanded for; cells carry the outcome.
    defaulted = weight * ended_in(episodes, cause)
    codes, labels = _codes(groups, len(episodes))
    present = codes >= 0

    width = int(ages.max()) + 1 if len(ages) else 1
    key = codes[present] * width + ages[present]
    size = len(labels) * width

    def total(values: np.ndarray) -> np.ndarray:
        summed: np.ndarray = np.bincount(key, weights=values[present], minlength=size)
        return summed.reshape(len(labels), width)

    at_risk = total(weight)
    defaults = total(defaulted)
    expected = total(weight * hazard) if hazard is not None else None

    frames = []
    for position, label in enumerate(labels):
        exposed = np.flatnonzero(at_risk[position] > 0)
        if exposed.size == 0:
            continue
        frame = pd.DataFrame(
            {
                "group": label,
                "age": exposed,
                "at_risk": at_risk[position, exposed],
                "defaults": defaults[position, exposed],
            }
        )
        if expected is not None:
            frame["expected"] = expected[position, exposed]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def curves_from(sums: pd.DataFrame, *, z: float = 1.96) -> pd.DataFrame:
    """Kaplan-Meier, the model's survival and the Greenwood band, from the risk sets.

    Survival is at the *end* of the month of age, as in ``predicted_survival_curve``. Every
    age with loan-months at risk is kept, thin tails included; ``at_risk`` is in the table so
    a figure can apply an exposure floor rather than the table deciding one.
    """
    frames = []
    for label, rows in sums.groupby("group", observed=True, sort=False):
        ordered = rows.sort_values("age")
        exposed = ordered["age"].to_numpy(dtype=np.int64)
        risk = ordered["at_risk"].to_numpy(dtype=float)
        died = ordered["defaults"].to_numpy(dtype=float)
        observed = died / risk
        survival = np.cumprod(1.0 - observed)
        # Greenwood. A risk set that loses every loan to default has no variance term.
        surviving = risk - died
        terms = np.divide(died, risk * surviving, out=np.zeros_like(died), where=surviving > 0)
        spread = z * survival * np.sqrt(np.cumsum(terms))
        frame = pd.DataFrame(
            {
                "group": label,
                "age": exposed,
                "at_risk": risk,
                "defaults": died,
                "observed_hazard": observed,
                "km_survival": survival,
                "km_lower": np.clip(survival - spread, 0.0, 1.0),
                "km_upper": np.clip(survival + spread, 0.0, 1.0),
            }
        )
        if "expected" in ordered.columns:
            anticipated = ordered["expected"].to_numpy(dtype=float)
            frame["expected"] = anticipated
            frame["predicted_hazard"] = anticipated / risk
            frame["predicted_survival"] = np.cumprod(1.0 - anticipated / risk)
            frame["deviation"] = frame["predicted_survival"] - frame["km_survival"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def survival_by_age(
    episodes: pd.DataFrame,
    hazard: np.ndarray | None = None,
    *,
    groups: pd.Series | None = None,
    z: float = 1.96,
    cause: str = DEFAULT_CAUSE,
) -> pd.DataFrame:
    """Kaplan-Meier and the model's survival, age by age, for each group.

    ``hazard`` is the model's monthly default probability for each episode, from
    :func:`creditsurv.backtest.runner.predicted_hazard`; without it only the non-parametric
    columns are filled. ``groups`` labels each episode; without it the table covers the whole
    book.

    The two halves are separate because the sums are additive and the curves are not: a table
    of 72 million episodes is accumulated as risk sets and turned into curves once.
    """
    return curves_from(risk_sets(episodes, hazard, groups=groups, cause=cause), z=z)


def exposure_totals(
    episodes: pd.DataFrame,
    hazard: np.ndarray,
    dimensions: Mapping[str, pd.Series | pd.Categorical | np.ndarray],
    cause: str = DEFAULT_CAUSE,
) -> pd.DataFrame:
    """The three sums, for every combination of ``dimensions`` with loan-months in it.

    Exposure, defaults and expected defaults -- **additive over rows**, so the table can be
    accumulated over the batches of the cell file and the rates taken once at the end. See
    :func:`rates_from` and :func:`creditsurv.views.streamed.totals_over`.
    """
    weight = episodes[WEIGHT].to_numpy(dtype=float)
    defaulted = weight * ended_in(episodes, cause)

    combined = np.zeros(len(episodes), dtype=np.int64)
    present = np.ones(len(episodes), dtype=bool)
    levels: list[list[object]] = []
    for values in dimensions.values():
        # Factorised as given: a categorical keeps its codes, where np.asarray would turn sixty
        # million labels into sixty million Python objects first.
        codes, found = pd.factorize(values, sort=True)
        present &= codes >= 0
        combined = combined * len(found) + np.maximum(codes, 0)
        levels.append(list(found))
    sizes = [len(found) for found in levels]
    size = int(np.prod(sizes)) if sizes else 1

    def total(values: np.ndarray) -> np.ndarray:
        summed: np.ndarray = np.bincount(combined[present], weights=values[present], minlength=size)
        return summed

    exposure = total(weight)
    kept = np.flatnonzero(exposure > 0)
    positions = np.unravel_index(kept, sizes) if sizes else ()
    table = pd.DataFrame(
        {
            name: np.asarray(found, dtype=object)[index]
            for (name, found), index in zip(
                zip(dimensions, levels, strict=True), positions, strict=True
            )
        }
    )
    table["exposure"] = exposure[kept]
    table["events"] = total(defaulted)[kept]
    table["expected"] = total(weight * hazard)[kept]
    return table


def rates_from(totals: pd.DataFrame) -> pd.DataFrame:
    """Both rates and their ratio, from the sums: above one the model under-predicts."""
    table = totals.copy()
    table["actual_rate"] = table["events"] / table["exposure"]
    table["expected_rate"] = table["expected"] / table["exposure"]
    table["actual_over_expected"] = np.where(
        table["expected"] > 0, table["events"] / table["expected"], np.nan
    )
    return table


def actual_expected(
    episodes: pd.DataFrame,
    hazard: np.ndarray,
    dimensions: Mapping[str, pd.Series | pd.Categorical | np.ndarray],
    cause: str = DEFAULT_CAUSE,
) -> pd.DataFrame:
    """Defaults against the model's expectation, for every combination of ``dimensions``.

    One row per combination with loan-months in it: the exposure, the defaults, the defaults
    expected, both rates, and their ratio.
    """
    return rates_from(exposure_totals(episodes, hazard, dimensions, cause))
