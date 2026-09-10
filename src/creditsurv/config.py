"""Central configuration: filesystem layout, macro series and the model formula.

Keeping these in one place means the data dictionary, the data preparation and
the model all describe the same variables, rather than drifting apart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------------------

_DATA_DIR_ENV: Final = "CREDITSURV_DATA_DIR"
_REPORTS_DIR_ENV: Final = "CREDITSURV_REPORTS_DIR"


#: Files that mark the root of a source checkout.
_ROOT_MARKERS: Final = ("pyproject.toml", ".git")


def project_root() -> Path:
    """The checkout root, or the working directory when there is not one.

    Walks up from the working directory looking for a marker. Without this, running
    from a subdirectory silently resolves ``data/`` relative to wherever the process
    happened to start -- a notebook in ``notebooks/`` looks for
    ``notebooks/data/processed`` and reports the panel as missing when it is sitting
    one level up.
    """
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    return current


def data_dir() -> Path:
    """Root for cached downloads and generated panels.

    Resolved at call time rather than at import, so an environment variable set by a
    test or a notebook takes effect without reimporting the package.
    """
    override = os.environ.get(_DATA_DIR_ENV)
    return Path(override) if override else project_root() / "data"


def raw_dir() -> Path:
    """Unmodified third-party downloads, one subdirectory per source."""
    return data_dir() / "raw"


def processed_dir() -> Path:
    """Panels derived from raw inputs and ready for modelling."""
    return data_dir() / "processed"


def reports_dir() -> Path:
    """Generated figures and numeric tables.

    Under ``docs/`` alongside the hand-written documentation. Splitting narrative
    from generated output across two top-level directories cost more than it
    explained: a reader looking for "the documentation" should find one place.
    """
    override = os.environ.get(_REPORTS_DIR_ENV)
    return Path(override) if override else project_root() / "docs" / "reports"


# --------------------------------------------------------------------------------------
# Macroeconomic series
# --------------------------------------------------------------------------------------


class Frequency(StrEnum):
    """Native publication frequency of a FRED series.

    Declared rather than inferred. Inference from observation spacing is fragile
    around missing observations, and the aggregation rule depends on it.
    """

    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    """A FRED series and the column it becomes in the monthly macro panel."""

    series_id: str
    column: str
    frequency: Frequency
    description: str


#: Series that feed model covariates.
MACRO_SERIES: Final[tuple[SeriesSpec, ...]] = (
    SeriesSpec(
        series_id="UNRATE",
        column="unemployment_rate",
        frequency=Frequency.MONTHLY,
        description="Civilian unemployment rate, percent, seasonally adjusted.",
    ),
    SeriesSpec(
        series_id="CSUSHPINSA",
        column="hpi",
        frequency=Frequency.MONTHLY,
        description="S&P CoreLogic Case-Shiller U.S. National Home Price Index.",
    ),
    SeriesSpec(
        series_id="MORTGAGE30US",
        column="mortgage_rate_30y",
        frequency=Frequency.WEEKLY,
        description="30-year fixed rate mortgage average, percent.",
    ),
    SeriesSpec(
        series_id="NFCI",
        column="nfci",
        frequency=Frequency.WEEKLY,
        description="Chicago Fed National Financial Conditions Index.",
    ),
)

#: Series used only to sanity-check observed default rates against a published aggregate.
#: Never a model covariate: it is an outcome, not a driver.
REFERENCE_SERIES: Final[tuple[SeriesSpec, ...]] = (
    SeriesSpec(
        series_id="DRSFRMACBS",
        column="observed_mortgage_delinquency_rate",
        frequency=Frequency.QUARTERLY,
        description=(
            "Delinquency rate on single-family residential mortgages, all commercial banks."
        ),
    ),
)

#: Earliest observation to request. 1999 matches the Freddie Mac dataset start.
MACRO_START: Final = "1999-01-01"

#: Months by which every macro covariate is lagged before entering the model.
#: Guards against using information that was not yet published at the time.
MACRO_LAG_MONTHS: Final = 3

# --------------------------------------------------------------------------------------
# Covariates
# --------------------------------------------------------------------------------------

#: Measured once at origination and constant thereafter.
#:
#: These are exactly the continuous covariates the aggregation coarse-classes into
#: the cell key, and the correspondence is not optional: a covariate the key does not
#: carry is not recoverable from a cell, so a formula naming one cannot be fitted at
#: all. ``tests/test_aggregate.py`` holds the two in step.
#:
#: ``log_orig_upb`` and ``orig_spread`` are absent for that reason rather than on
#: their merits -- the cell count is the product of the band counts, so every
#: covariate added to the key multiplies it. What each would cost is measured in
#: docs/variable_selection.md.
STATIC_CONTINUOUS: Final[tuple[str, ...]] = (
    "fico_s",
    "orig_ltv",
    "dti",
)

#: Takes two values, 15 and 30, so a linear term and a dummy are the same model.
#: Kept numeric because the aggregation emits it as an integer, and a treatment
#: reference would then have to agree with that dtype across the SQL and the formula.
ORDINAL: Final[tuple[str, ...]] = ("term_years",)

#: Recomputed every loan-month. Loan age is the time scale, not a covariate.
#:
#: ``cltv_drift`` rather than ``indexed_cltv``: the indexed ratio equals
#: ``orig_ltv`` at origination and stays strongly correlated with it, so fitting
#: both gives unstable coefficients. The pair is decomposed into a level
#: (``orig_ltv``, underwriting at origination) and a movement (``cltv_drift``,
#: how far house prices have carried the position since, zero at origination).
#: ``refi_incentive`` is absent: it needs the note rate, which the key does not
#: carry. It is recoverable the same way ``cltv_drift`` is -- from a banded
#: ``orig_spread`` plus the mortgage-rate path, both functions of the key -- which is
#: the cheapest of the candidate additions and the one to weigh first.
TIME_VARYING_CONTINUOUS: Final[tuple[str, ...]] = (
    "cltv_drift",
    "unemp_gap",
    "nfci_lagged",
)

#: Categorical covariates mapped to their treatment-coding reference level.
#: ``channel``, ``region`` and ``first_time_buyer`` are screened and mapped but not
#: in the key: together they would multiply the cell count by sixteen.
CATEGORICAL_REFERENCE: Final[dict[str, str]] = {
    "purpose": "purchase",
    "occupancy": "owner_occupied",
}


def default_formula() -> str:
    """Formulaic specification for the full model.

    Categorical reference levels are stated explicitly so coefficients remain
    comparable across refits even if a level goes missing from a training slice.
    """
    continuous = " + ".join(STATIC_CONTINUOUS + TIME_VARYING_CONTINUOUS + ORDINAL)
    categorical = " + ".join(
        f"C({name}, Treatment('{reference}'))" for name, reference in CATEGORICAL_REFERENCE.items()
    )
    return f"{continuous} + {categorical}"
