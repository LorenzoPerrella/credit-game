"""Every table the site shows of the fitted model, from two passes over the cell file.

`views/model.py` owns the recipes and the view builders, `views/streamed.py` the accumulation.
What was missing was the sequence: find the fit, collect whatever other families are cached
beside it, take the decile boundaries, accumulate everything else, then read the test window
and the origination profiles. It was written inline in the command that calls it.

**The training half is never an object.** These tables used to be computed from the expanded
split, which put the footprint near 15 GB on 59.7 million cells, and the table is now 72.7
million. Two passes over the cell file do it instead: the first takes the decile boundaries,
which need the whole distribution and cannot be accumulated, and the second accumulates every
table -- each segment, each family -- from one read.

**And nothing here fits.** A view describes a model that already exists; computing one is the
cheap half and the expensive half is the fit. If the fit is not in the cache this raises, and
says which command makes it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pandas as pd

from creditsurv.backtest.runner import predicted_hazard
from creditsurv.config import (
    DISTRIBUTION,
    ORDINAL,
    STATIC_CONTINUOUS,
    TIME_VARYING_CONTINUOUS,
)
from creditsurv.data.panel import WEIGHT, cells_to_episodes, month_ordinal
from creditsurv.data.store import find_fits, load_cells_window, load_fit, load_largest_cells
from creditsurv.models.aft import CONVERGENT_DISTRIBUTIONS, FitResult
from creditsurv.models.fits import cell_source
from creditsurv.models.lifetime_pd import origination_book
from creditsurv.models.selection import weighted_moments
from creditsurv.views.model import (
    backtest_views,
    coefficient_view,
    covariate_means_recipe,
    in_sample_recipes,
    in_sample_views,
    projection_views,
)
from creditsurv.views.streamed import accumulate, decile_boundaries
from creditsurv.views.tables import View

if TYPE_CHECKING:
    from collections.abc import Sequence

log: Final = logging.getLogger(__name__)


class NoCachedFit(LookupError):
    """The model a view describes has to exist already, because views never fit."""


@dataclass(frozen=True)
class ModelViews:
    """The tables, and the fingerprint of the fit they are all of.

    One fingerprint, deliberately: `write_views` refuses a manifest naming more than one fit,
    because a page showing two models' numbers side by side without saying so is worse than a
    page showing none.
    """

    tables: list[View]
    fingerprint: str


def model_views(
    *,
    as_of: str,
    moratorium: str,
    covariates: Sequence[str],
    formula: str,
    macro: pd.DataFrame,
    block_rows: int,
    loans: int,
    horizon: int,
) -> ModelViews:
    """Find the cached fit of this specification and build every view of it."""
    found = find_fits(
        as_of=as_of,
        moratorium=moratorium,
        formula=formula,
        distribution=DISTRIBUTION,
        purpose=None,
    )
    if not found:
        message = (
            f"No cached fit of this specification at {as_of}. Run `creditsurv report` or "
            "`creditsurv fit --streamed` first: views never fit."
        )
        raise NoCachedFit(message)
    fingerprint, described = found[0]
    fitted = load_fit(fingerprint)
    if not isinstance(fitted, FitResult):
        message = f"The cached fit {fingerprint} cannot be read; views never fit."
        raise NoCachedFit(message)
    log.info("scoring with fit %s (%s cells)", fingerprint, f"{described.get('rows'):,}")

    cut = month_ordinal(pd.Period(as_of, freq="M"))
    source = cell_source(
        moratorium, macro, covariates, block_rows=block_rows, until=cut, model_only=False
    )
    models = _families(as_of, moratorium, fitted, covariates)

    log.info("pass one: the decile boundaries")
    boundaries = decile_boundaries(source(), fitted, covariates)
    log.info("pass two: survival, calibration and the covariate means by segment")
    accumulated = accumulate(
        source(),
        models,
        [
            *in_sample_recipes(
                primary=DISTRIBUTION,
                families=[name for name in models if name != DISTRIBUTION],
                boundaries=boundaries,
            ),
            covariate_means_recipe(TIME_VARYING_CONTINUOUS),
        ],
    )
    continuous = [*STATIC_CONTINUOUS, *TIME_VARYING_CONTINUOUS, *ORDINAL]
    tables = [
        *in_sample_views(accumulated, as_of=as_of),
        View(
            "covariates_over_time",
            "Macro covariates over time",
            "The exposure-weighted mean of each time-varying covariate across the loans "
            "observed in each month, up to the reporting date.",
            accumulated["covariates_over_time"],
            source="fit",
        ),
        coefficient_view(
            fitted,
            None,
            continuous,
            deviations=weighted_moments(source(), continuous, weight=WEIGHT).deviations,
        ),
    ]

    # The test window and the projections read only what they need: the months after the
    # reporting date, and the origination profiles at age zero.
    log.info("the test window, and the projections from the book written today")
    test = cells_to_episodes(
        load_cells_window(moratorium, first=cut + 1), macro, covariates=list(covariates)
    )
    test_hazard = predicted_hazard(fitted, test, covariates).to_numpy()
    tables += [
        *backtest_views(test, test_hazard, as_of=as_of),
        *projection_views(
            fitted,
            origination_book(
                cells_to_episodes(
                    load_largest_cells(moratorium, age=0, limit=loans),
                    macro,
                    covariates=list(covariates),
                ),
                macro,
                loans,
            ),
            macro,
            covariates,
            horizon_months=horizon,
        ),
    ]
    return ModelViews(tables=tables, fingerprint=fingerprint)


def _families(
    as_of: str, moratorium: str, fitted: FitResult, covariates: Sequence[str]
) -> dict[str, tuple[FitResult, list[str]]]:
    """The published model, and whatever other families are cached on the same rows.

    Rule 2 compares the *selected* models of two families, so the comparison only exists where
    both have been fitted. A family with no cached fit is left out rather than fitted here.
    """
    models: dict[str, tuple[FitResult, list[str]]] = {DISTRIBUTION: (fitted, list(covariates))}
    for distribution in CONVERGENT_DISTRIBUTIONS:
        if distribution == DISTRIBUTION:
            continue
        other = find_fits(
            as_of=as_of, moratorium=moratorium, distribution=distribution, purpose=None
        )
        if not other:
            continue
        candidate = load_fit(other[0][0])
        if isinstance(candidate, FitResult):
            log.info("and the cached %s fit %s", distribution, other[0][0])
            models[distribution] = (candidate, list(covariates))
    return models
