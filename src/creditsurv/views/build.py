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
    BOOK_WINDOW_MONTHS,
    DISTRIBUTION,
    ORDINAL,
    STATIC_CONTINUOUS,
    TIME_VARYING_CONTINUOUS,
)
from creditsurv.data.panel import WEIGHT, cells_to_episodes, ended_in, month_ordinal
from creditsurv.data.store import find_fits, load_cells_window, load_fit, load_originations
from creditsurv.models.aft import CONVERGENT_DISTRIBUTIONS, FitResult
from creditsurv.models.anchoring import ANCHOR_WINDOW, anchor_on_window
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
    ancillary: str | None = None,
    macro: pd.DataFrame,
    block_rows: int,
    book_months: int = BOOK_WINDOW_MONTHS,
    anchor_window: tuple[str, str] = ANCHOR_WINDOW,
    horizon: int,
) -> ModelViews:
    """Find the cached fit of this specification and build every view of it.

    ``book_months`` is how many months of originations the projection book is, ending at the
    reporting date. It replaces a ``loans`` count, which was a truncation rather than a window:
    `config.BOOK_WINDOW_MONTHS` says why twelve, and says it before the book was rebuilt.

    ``anchor_window`` is rule 1's, and is an argument for the reason `windows` makes it one:
    the criteria are measured on the model **as published**, multiplier included, and a
    fixture book does not run to 2024.
    """
    # `ancillary` is named rather than left out, and a criterion of `None` means the key must be
    # **absent** -- so this asks for a fit of this scale formula with exactly this shape, and a
    # scale-only fit of the same formula is a miss rather than a near-enough hit. With rule 14's
    # shape part of the specification the two are different models, and the site publishing the
    # wrong one would say nothing about it.
    found = find_fits(
        as_of=as_of,
        moratorium=moratorium,
        formula=formula,
        distribution=DISTRIBUTION,
        ancillary=ancillary,
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

    # The test window and the projections read only what they need: the months rule 1 declares
    # the test window to be, and the origination profiles at age zero.
    #
    # **Both the window and the level were wrong, and they are the same correction.** This read
    # from `as_of + 1`, which is 2022-01 -- the anchoring window *and* the test window together
    # -- and scored it on an unanchored hazard. Rule 1 reserves the anchoring window for the
    # level adjustment "and nothing else" and begins the test window at 2025-01, and rule 5's
    # criteria are of the model as published, multiplier included. So `acceptance_by_segment`
    # was published with a `deciles_passed` column that did not evaluate the criteria it is
    # named after: it read False at a worst decile of 1.456 where the declared criterion reads
    # True at 1.150, and three separate readings of that one decile were taken off it in a day.
    #
    # It costs nothing: the anchoring window is three years of observation and the test window
    # about one and a half, against the four and a half this used to read in one go.
    first_anchor, last_anchor = anchor_window
    log.info("the anchoring window %s to %s, for the level", first_anchor, last_anchor)
    anchoring = cells_to_episodes(
        load_cells_window(
            moratorium,
            first=month_ordinal(pd.Period(first_anchor, freq="M")),
            last=month_ordinal(pd.Period(last_anchor, freq="M")),
        ),
        macro,
        covariates=list(covariates),
    )
    if anchoring.empty:
        message = (
            f"No cells observed between {first_anchor} and {last_anchor}, so the level cannot be "
            "anchored and the criteria cannot be measured on the model as published. Pass an "
            "anchor_window the table has rows in."
        )
        raise NoCachedFit(message)
    anchor = anchor_on_window(
        predicted_hazard(fitted, anchoring, covariates),
        ended_in(anchoring),
        anchoring[WEIGHT],
        anchoring["period"],
        window=(first_anchor, last_anchor),
    )
    log.info("multiplier %.4f", anchor.multiplier)
    del anchoring

    log.info("the test window, and the projections from the book written today")
    test = cells_to_episodes(
        load_cells_window(moratorium, first=month_ordinal(pd.Period(last_anchor, freq="M")) + 1),
        macro,
        covariates=list(covariates),
    )
    test_hazard = anchor.apply(predicted_hazard(fitted, test, covariates))
    tables += [
        *backtest_views(
            test,
            test_hazard,
            out_of_time=f"On the test window rule 1 declares -- the months after "
            f"{last_anchor}, which neither the fit nor the anchoring saw -- at the "
            f"anchored level, multiplier {anchor.multiplier:.4f}.",
        ),
        *projection_views(
            fitted,
            # The declared book: every origination of the `BOOK_WINDOW_MONTHS` ending at the
            # reporting date, at the weights it carries. No ordering and no truncation, which
            # is what `load_largest_cells(limit=loans)` used to impose -- and what made the
            # published book 0.6% purchase against the window's own 32.2%.
            origination_book(
                cells_to_episodes(
                    load_originations(moratorium, first=cut - (book_months - 1), last=cut),
                    macro,
                    covariates=list(covariates),
                ),
                macro,
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
