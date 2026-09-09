"""Central configuration: filesystem layout, macro series and the model formula.

Keeping these in one place means the data dictionary, the synthetic generator and
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


def data_dir() -> Path:
    """Root for cached downloads and generated panels.

    Resolved at call time from the working directory so the package behaves the same
    whether it is run from a source checkout or installed as a wheel.
    """
    return Path(os.environ.get(_DATA_DIR_ENV, Path.cwd() / "data"))


def raw_dir() -> Path:
    """Unmodified third-party downloads, one subdirectory per source."""
    return data_dir() / "raw"


def processed_dir() -> Path:
    """Panels derived from raw inputs and ready for modelling."""
    return data_dir() / "processed"


def reports_dir() -> Path:
    """Generated figures and numeric tables."""
    return Path(os.environ.get(_REPORTS_DIR_ENV, Path.cwd() / "reports"))


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

#: Series used only to sanity-check simulated default rates against reality.
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
STATIC_CONTINUOUS: Final[tuple[str, ...]] = (
    "fico_s",
    "orig_ltv",
    "dti",
    "log_orig_upb",
    "orig_spread",
)

#: Recomputed every loan-month. Loan age is the time scale, not a covariate.
#:
#: ``cltv_drift`` rather than ``indexed_cltv``: the indexed ratio equals
#: ``orig_ltv`` at origination and stays strongly correlated with it, so fitting
#: both gives unstable coefficients. The pair is decomposed into a level
#: (``orig_ltv``, underwriting at origination) and a movement (``cltv_drift``,
#: how far house prices have carried the position since, zero at origination).
TIME_VARYING_CONTINUOUS: Final[tuple[str, ...]] = (
    "cltv_drift",
    "unemp_gap",
    "refi_incentive",
    "nfci_lagged",
)

#: Categorical covariates mapped to their treatment-coding reference level.
CATEGORICAL_REFERENCE: Final[dict[str, str]] = {
    "purpose": "purchase",
    "occupancy": "owner_occupied",
    "channel": "retail",
    "region": "South",
    "first_time_buyer": "N",
}


def default_formula() -> str:
    """Formulaic specification for the full model.

    Categorical reference levels are stated explicitly so coefficients remain
    comparable across refits even if a level goes missing from a training slice.
    """
    continuous = " + ".join(STATIC_CONTINUOUS + TIME_VARYING_CONTINUOUS)
    categorical = " + ".join(
        f"C({name}, Treatment('{reference}'))" for name, reference in CATEGORICAL_REFERENCE.items()
    )
    return f"{continuous} + {categorical}"
