"""Rule 2 applied: how far each family's *selected* model sits from what the book did.

The rule is in `docs/rules.md` and it is not the likelihood. Both families go through the whole
selection, and the winner is the one whose selected model sits closest to the Aalen-Johansen
cumulative incidence of default, in mean absolute percentage points, over the loan ages carrying
enough exposure to measure. A family whose selected model turns a declared sign is excluded
whatever its fit: a model saying tighter financial conditions lengthen survival is not a better
model, it is a broken one.

The likelihood is the reason the rule is not the likelihood. On the specification before the
validation the Weibull led by 623,126 AIC points; on the one after it the log-logistic led by
83,961. A criterion that changes its mind with the specification is not a criterion, and AIC on
72 million episodes measures fit where the data is dense rather than where a lifetime PD spends
its time.

**And a cumulative incidence needs both hazards.** Default and prepayment are competing risks,
so neither curve exists without the other: the incidence of default is the survival of *both*
times the default hazard, accumulated. That is why this reads the prepayment model's record as
well, and why it refuses rather than guesses when that selection has not been run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Final, cast

import pandas as pd

from creditsurv.config import (
    DEFAULT_CAUSE,
    DISTRIBUTION,
    PREPAYMENT_CAUSE,
    record_name,
    reports_dir,
)
from creditsurv.data.panel import WEIGHT, month_ordinal
from creditsurv.data.store import cells_identity, outcomes_by_age
from creditsurv.models.aft import CONVERGENT_DISTRIBUTIONS, FitResult
from creditsurv.models.fits import cell_source, selected_fit
from creditsurv.models.nonparametric import (
    cumulative_incidence,
    hazard_by_age,
    incidence_from_hazards,
    incidence_gap,
)
from creditsurv.models.selection import PREPAYMENT_SIGNS, signs_against_prior

log: Final = logging.getLogger(__name__)


class NotSelected(LookupError):
    """A family or a cause the rule needs, whose selection has not been run on these cells.

    Named rather than guessed around. Comparing a family that was not taken through the whole
    procedure would be comparing a fit to a selected model, which is the one thing rule 2 says
    it does not do.
    """


@dataclass(frozen=True)
class Comparison:
    """What the rule reads, for every family that has a record on these cells."""

    #: Each family's distance from the observed incidence, age by age.
    gaps: dict[str, pd.DataFrame]
    #: Each family's covariates that turn a declared sign, which exclude it whatever its fit.
    signs: dict[str, list[str]]
    #: The formula each family's own selection ended on.
    formulas: dict[str, str]
    #: Covariates of the **prepayment** model that turn rule 6's prior. Reported rather than
    #: acted on: rule 11 names the ones kept because their removal left an unfittable model.
    prepayment_against_prior: list[str]


def compare_families(
    *, as_of: str, moratorium: str, macro: pd.DataFrame, block_rows: int
) -> Comparison:
    """Read every selection record on these cells and measure each family against the book."""
    identity = cells_identity(moratorium)
    cut = month_ordinal(pd.Period(as_of, freq="M"))

    # The observed side is a statement about sums, so it is taken inside the parquet reader:
    # a few hundred rows out of 91.6 million cells, with nothing expanded.
    log.info("the observed cumulative incidence, from the cells")
    observed = cumulative_incidence(outcomes_by_age(moratorium, last=cut))

    prepayment = _record(PREPAYMENT_CAUSE, distribution=DISTRIBUTION)
    if prepayment is None:
        message = (
            "The prepayment model has not been selected on these cells, and a cumulative "
            "incidence needs both hazards. Run `creditsurv select --cause prepayment`."
        )
        raise NotSelected(message)
    prepayment_fit, prepaid = _fit_and_hazards(
        prepayment,
        PREPAYMENT_CAUSE,
        identity=identity,
        as_of=as_of,
        moratorium=moratorium,
        macro=macro,
        block_rows=block_rows,
    )
    against_prior = signs_against_prior(prepayment_fit, PREPAYMENT_SIGNS)
    if against_prior:
        log.info(
            "the prepayment model turns %s against rule 6's prior; rule 11 names the "
            "covariates it kept because their removal left an unfittable model",
            ", ".join(against_prior),
        )

    gaps: dict[str, pd.DataFrame] = {}
    signs: dict[str, list[str]] = {}
    formulas: dict[str, str] = {}
    for distribution in CONVERGENT_DISTRIBUTIONS:
        described = _record(DEFAULT_CAUSE, distribution=distribution)
        if described is None:
            log.info("%s: no selection record on these cells; skipped", distribution)
            continue
        formulas[distribution] = str(described["formula"])
        fitted, defaults = _fit_and_hazards(
            described,
            DEFAULT_CAUSE,
            identity=identity,
            as_of=as_of,
            moratorium=moratorium,
            macro=macro,
            block_rows=block_rows,
        )
        predicted = incidence_from_hazards({DEFAULT_CAUSE: defaults, PREPAYMENT_CAUSE: prepaid})
        gaps[distribution] = incidence_gap(predicted, observed)
        # The default model's priors, which are not the prepayment model's: rule 6 declares its
        # own, and reading the wrong map would exclude a family under rule 2 for a sign nobody
        # ever expected of it.
        signs[distribution] = signs_against_prior(fitted)

    if not gaps:
        message = "No selection has been run on these cells; there is nothing to compare."
        raise NotSelected(message)
    return Comparison(
        gaps=gaps, signs=signs, formulas=formulas, prepayment_against_prior=against_prior
    )


def _record(cause: str, *, distribution: str) -> dict[str, object] | None:
    """One selection's record, read from the JSON it wrote beside its report.

    The reporting date is not checked here, and deliberately: a record written for another
    `--as-of` names a formula whose fit is cached under that date, so `_fit_and_hazards` finds
    nothing and says which command to run. Two guards for one condition would mean two
    messages for it.
    """
    name = record_name(distribution=distribution, cause=cause, published=DISTRIBUTION)
    path = reports_dir() / f"{name}.json"
    if not path.exists():
        return None
    return cast("dict[str, object]", json.loads(path.read_text()))


def _covariates_of(described: dict[str, object]) -> list[str]:
    """What the selected model reads, from the record rather than from its formula.

    The inverse of what `SelectionRecord.summary()` writes, which is why the two are tested
    against each other: a record whose covariates cannot be read back is a record of nothing.
    """
    return [
        *cast("list[str]", described["static_continuous"]),
        *cast("list[str]", described["ordinal"]),
        *cast("list[str]", described["time_varying_continuous"]),
        *cast("dict[str, str]", described["categorical"]),
    ]


def _fit_and_hazards(
    described: dict[str, object],
    cause: str,
    *,
    identity: str,
    as_of: str,
    moratorium: str,
    macro: pd.DataFrame,
    block_rows: int,
) -> tuple[FitResult, pd.DataFrame]:
    """The selected model and its mean hazard by age, from **one** read of the pickle.

    The two used to be fetched separately -- the hazards through one `selected_fit` and the
    signs through a second call with the same arguments -- which unpickled a fitted lifelines
    model twice per family and twice again for the prepayment model.
    """
    distribution = str(described["distribution"])
    fitted = selected_fit(
        identity=identity,
        as_of=as_of,
        moratorium=moratorium,
        formula=str(described["formula"]),
        distribution=distribution,
        cause=cause,
    )
    if fitted is None:
        flags = f"--dist {distribution}" + (f" --cause {cause}" if cause != DEFAULT_CAUSE else "")
        message = (
            f"No cached fit of the {distribution} {cause} model on these cells. "
            f"Run `creditsurv select {flags}`."
        )
        raise NotSelected(message)
    covariates = _covariates_of(described)
    source = cell_source(
        moratorium,
        macro,
        covariates,
        block_rows=block_rows,
        until=month_ordinal(pd.Period(as_of, freq="M")),
        cause=cause,
    )
    log.info("reading the %s %s hazards by age", distribution, cause)
    return fitted, hazard_by_age(source(), fitted, covariates, weights_col=WEIGHT)
