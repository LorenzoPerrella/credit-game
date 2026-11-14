"""A lifetime PD judged the way it is used: each vintage projected from origination.

Section 5 of `docs/rules.md` declares a cumulative PD by vintage among its criteria, and nothing in
the package computed it. The twelve-month PD is judged loan-month by loan-month at the covariates
the loans actually had; a lifetime PD is a different object -- a curve that starts at
origination and has to survive the competing exit -- so it is judged as one.

For one origination year:

- **predicted:** every origination profile of the year, at the weight it carries, projected from
  its own origination month along the **realised** macro path, with the default and the
  prepayment hazards chained against each other into a cumulative incidence
  (`models.lifetime_pd.competing_paths`), then averaged over the book by weight. No survivor is
  chosen by the data: the model decides who is still there;
- **observed:** the Aalen-Johansen incidence of the same year's loans, read off its cells at
  every age (`models.nonparametric.cumulative_incidence`), with prepayment as the competing exit
  and the end of the table as censoring.

The two are compared at declared horizons. What this deliberately grants the model is the
economy that happened, exactly as the twelve-month backtest does, so a miss here is the model's
and not a forecast's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from creditsurv.config import CAUSES
from creditsurv.data.panel import AGE, LOAN_ID, OUTCOME, WEIGHT, cells_to_episodes
from creditsurv.data.store import load_originations, load_vintage
from creditsurv.models.lifetime_pd import competing_paths, hazard_paths, project_panel
from creditsurv.models.nonparametric import cumulative_incidence

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from creditsurv.models.aft import FitResult

log = logging.getLogger(__name__)

#: The horizons a vintage's cumulative incidence is read at, in months from origination.
HORIZONS: Final[tuple[int, ...]] = (12, 36, 60)

#: Origination profiles projected at a time. The paths of two profiles never meet, so the book
#: can be cut anywhere; a block of this size times sixty months is about a quarter of a million
#: loan-months, which is what the macro family is rebuilt over in one go.
PROFILES_PER_BLOCK: Final = 4_000


def _number(value: object) -> float:
    return float(value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class VintageCurve:
    """One origination year: the model's cumulative incidence and the book's own."""

    vintage: int
    loans: float
    #: By month from origination: survival and the incidence of each cause, weighted by the book.
    predicted: pd.DataFrame
    #: The Aalen-Johansen table of the same loans, by age.
    observed: pd.DataFrame

    def at(self, horizons: Sequence[int] = HORIZONS) -> pd.DataFrame:
        """Both curves at each horizon the projection and the data reach.

        Month ``m`` of the projection is the incidence by the end of the ``m``-th month at risk,
        which is the Aalen-Johansen value at age ``m - 1``: ages count from zero.
        """
        rows = []
        observed = self.observed.set_index("age")
        predicted = self.predicted.set_index("month")
        for horizon in horizons:
            if horizon not in predicted.index or horizon - 1 not in observed.index:
                continue
            row: dict[str, object] = {
                "vintage": self.vintage,
                "horizon": horizon,
                "at_risk": _number(observed.at[horizon - 1, "at_risk"]),
            }
            for cause in CAUSES:
                model = _number(predicted.at[horizon, cause])
                book = _number(observed.at[horizon - 1, cause])
                row[f"{cause}_predicted"] = model
                row[f"{cause}_observed"] = book
                row[f"{cause}_ratio"] = book / model if model > 0 else float("nan")
            rows.append(row)
        return pd.DataFrame(rows)


def projected_incidence(
    models: Mapping[str, tuple[FitResult, Sequence[str]]],
    book: pd.DataFrame,
    macro: pd.DataFrame,
    *,
    horizon: int,
    multipliers: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """The book's cumulative incidence by month, each profile projected from its own start.

    ``models`` maps each cause to its fit and the covariates it reads; both causes are needed,
    because the incidence of one is what the other leaves. ``book`` is one row a profile with its
    ``age``, its ``period`` and its weight. ``multipliers`` scales a cause's hazard -- rule 4's
    anchoring, which applies to the default hazard only -- and is clipped below one.
    """
    if set(models) != set(CAUSES):
        message = f"A cumulative incidence needs both causes, {CAUSES}; got {sorted(models)}."
        raise ValueError(message)
    scale = {cause: 1.0 for cause in CAUSES} | dict(multipliers or {})
    book = book.reset_index(drop=True)
    book = book.assign(**{LOAN_ID: [f"row_{index:09d}" for index in range(len(book))]})

    loans = float(book[WEIGHT].sum())
    survival = np.zeros(horizon)
    incidence = {cause: np.zeros(horizon) for cause in CAUSES}
    for start in range(0, len(book), PROFILES_PER_BLOCK):
        part = book.iloc[start : start + PROFILES_PER_BLOCK]
        panel = project_panel(part, macro, horizon_months=horizon)
        hazards = {
            cause: (hazard_paths(fit, panel, covariates) * scale[cause]).clip(upper=1.0)
            for cause, (fit, covariates) in models.items()
        }
        complete = pd.concat(hazards.values(), axis=1).notna().all(axis=1)
        if not complete.all():
            message = (
                f"{int((~complete).sum())} profiles have no covariates for part of the horizon, "
                "so their paths are ragged; the macro panel does not reach back far enough."
            )
            raise ValueError(message)
        paths = competing_paths(hazards)
        weight = part.set_index(LOAN_ID)[WEIGHT].astype(float).reindex(paths.survival.index)
        survival += paths.survival.mul(weight, axis=0).sum(axis=0).to_numpy()
        for cause in CAUSES:
            incidence[cause] += paths.incidence[cause].mul(weight, axis=0).sum(axis=0).to_numpy()

    table = pd.DataFrame({"month": np.arange(1, horizon + 1), "survival": survival / loans})
    for cause in CAUSES:
        table[cause] = incidence[cause] / loans
    return table


def vintage_curve(
    models: Mapping[str, tuple[FitResult, Sequence[str]]],
    vintage: int,
    *,
    moratorium: str,
    macro: pd.DataFrame,
    horizon: int,
    multipliers: Mapping[str, float] | None = None,
) -> VintageCurve:
    """One origination year, projected and observed.

    The projection runs to ``horizon`` months or to the end of the macro panel for the year's
    last origination month, whichever is sooner, so every profile has the same number of months.
    """
    first, last = vintage * 12, vintage * 12 + 11
    covariates = list(dict.fromkeys(name for _, names in models.values() for name in names))
    book = cells_to_episodes(
        load_originations(moratorium, first=first, last=last), macro, covariates=covariates
    )
    reach = macro.index.max() - pd.Period(year=vintage, month=12, freq="M")
    months = int(min(horizon, reach.n + 1))
    predicted = projected_incidence(models, book, macro, horizon=months, multipliers=multipliers)
    cells = load_vintage(
        moratorium, first=first, last=last, columns=["origination_month", AGE, WEIGHT, OUTCOME]
    )
    observed = cumulative_incidence(cells, causes=CAUSES)
    log.info(
        "vintage %d: %s loans, %d months projected", vintage, f"{book[WEIGHT].sum():,.0f}", months
    )
    return VintageCurve(
        vintage=vintage, loans=float(book[WEIGHT].sum()), predicted=predicted, observed=observed
    )
