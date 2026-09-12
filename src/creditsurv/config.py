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

    DAILY = "daily"
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
    # Everything below is free, in the sense that matters here. The cell count is the
    # product of the band counts of whatever enters the aggregation key -- and a macro
    # series never does. Every covariate derived from one is a function of the
    # vintage quarter and the loan age, both of which the key already holds, so it is
    # recomputed after the collapse at no cost in cardinality at all.
    #
    # That asymmetry is worth stating plainly: adding a loan characteristic to the
    # model can multiply the table by five, and adding a macro series cannot change
    # its size by a single row. Which is why the macro side is generous and the loan
    # side is not.
    SeriesSpec(
        series_id="MORTGAGE15US",
        column="mortgage_rate_15y",
        frequency=Frequency.WEEKLY,
        description="15-year fixed rate mortgage average, percent.",
    ),
    SeriesSpec(
        series_id="FEDFUNDS",
        column="policy_rate",
        frequency=Frequency.MONTHLY,
        description="Effective federal funds rate, percent.",
    ),
    SeriesSpec(
        series_id="DGS10",
        column="treasury_10y",
        frequency=Frequency.DAILY,
        description="10-year Treasury constant maturity yield, percent.",
    ),
    SeriesSpec(
        series_id="T10Y2Y",
        column="term_spread",
        frequency=Frequency.DAILY,
        description="10-year minus 2-year Treasury spread, percentage points.",
    ),
    SeriesSpec(
        series_id="BAA10Y",
        column="credit_spread",
        frequency=Frequency.DAILY,
        description="Moody's Baa corporate yield over the 10-year Treasury.",
    ),
    SeriesSpec(
        series_id="CPIAUCSL",
        column="cpi",
        frequency=Frequency.MONTHLY,
        description="Consumer price index, all urban consumers, seasonally adjusted.",
    ),
    SeriesSpec(
        series_id="NASDAQCOM",
        column="equity_index",
        frequency=Frequency.DAILY,
        description="Nasdaq Composite index.",
    ),
    SeriesSpec(
        series_id="VIXCLS",
        column="vix",
        frequency=Frequency.DAILY,
        description="CBOE volatility index.",
    ),
    SeriesSpec(
        series_id="UMCSENT",
        column="sentiment",
        frequency=Frequency.MONTHLY,
        description="University of Michigan consumer sentiment index.",
    ),
    SeriesSpec(
        series_id="HOUST",
        column="housing_starts",
        frequency=Frequency.MONTHLY,
        description="Privately owned housing units started, thousands, annual rate.",
    ),
)

#: ``SP500`` and ``DJIA`` are unusable here: FRED keeps only ten years of them, so
#: both start in 2016 and would leave seventeen vintages with no equity covariate at
#: all. ``NASDAQCOM`` reaches back to 1999 and is the substitute. ``TDSP``, the
#: household debt service ratio, is the most directly relevant series of the lot and
#: is excluded for the same reason: it begins in 2005.
EXCLUDED_SERIES: Final[dict[str, str]] = {
    "SP500": "FRED retains ten years only; starts 2016",
    "DJIA": "FRED retains ten years only; starts 2016",
    "TDSP": "starts 2005, leaving six vintages uncovered",
}

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

#: Earliest observation to request.
#:
#: Two years before the loan data, not alongside it. A covariate lagged three months
#: and measured as a year-on-year change reaches fifteen months back, so a panel
#: starting in January 1999 would have no such covariate until April 2000 -- and the
#: rows would be *dropped*, silently removing the first fifteen months of every loan
#: in the earliest vintages while keeping the rest of them.
#:
#: That is left truncation the likelihood is never told about, which is precisely
#: what ``validate_episodes`` refuses on a loan-level panel. Reaching further back
#: for the macro series costs nothing and removes the problem at its source.
MACRO_START: Final = "1997-01-01"

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
#: **One covariate per economic dimension**, and that rule is the outcome of getting
#: it wrong twice. Housing, labour, financial stress, prices -- four dimensions, four
#: covariates. Everything else tried duplicated one of them and was not identified
#: beside it. See ``ELIMINATED`` for the measurements.
TIME_VARYING_CONTINUOUS: Final[tuple[str, ...]] = (
    "cltv_drift",
    "unemp_gap",
    "vix",
    "inflation",
)

#: Every macro-derived covariate available, including the ones the default model does
#: not use. This is the set variable selection runs on; ``TIME_VARYING_CONTINUOUS``
#: holds what survived it.
#:
#: They are deliberately collinear -- five interest-rate series will not all live
#: through a VIF pass -- and that is the point of having a selection procedure rather
#: than a list of opinions. Adding them costs nothing: a macro covariate is a function
#: of the vintage quarter and the loan age, both already in the aggregation key, so
#: none of them adds a single cell. The elimination priority below decides who goes
#: first, and it is fixed here, before any result is looked at.
MACRO_CANDIDATES: Final[tuple[str, ...]] = (
    "cltv_drift",
    "unemp_gap",
    "nfci_lagged",
    "rate_gap",
    "hpi_growth",
    "policy_rate_gap",
    "term_spread",
    "credit_spread",
    "inflation",
    "equity_return",
    "vix",
    "sentiment",
    "starts_growth",
)

#: Order in which collinear macro covariates are given up, most expendable first.
#:
#: Fixed in advance, the way `nmds` fixes its own scale before running the selection,
#: because a priority chosen after seeing the VIF table is not a priority -- it is a
#: preference for whichever answer came out. The principle: keep what is specific to
#: mortgage credit, give up what is a general business-cycle proxy, and among
#: equivalents keep the series with the longest clean history.
#: Macro candidates removed by the selection, each with the rule that removed it.
#: Recorded here rather than only in the documentation, so the list and the model
#: cannot drift apart -- and so the omissions read as decisions.
#:
#: See docs/variable_selection.md for the measured tables behind each.
ELIMINATED: Final[dict[str, str]] = {
    # --- not identified beside a larger covariate carrying the same information ---
    #
    # These four passed the sign and collinearity screens and were eliminated by a rule
    # those screens do not contain: a covariate whose standardised effect is small
    # **and** which sits beside a larger correlated one carrying the same economic
    # information is not identified, and its sign is noise.
    #
    # The evidence is that the sign moves when the sample does. Restricting the panel
    # to everything before 2024-12 -- dropping 6.3% of exposure -- flipped all three of
    # these that had a sign at all, while every covariate with a standardised effect
    # above 0.1 held to within a few percent. In log survival time per standard
    # deviation:
    #
    #     vix          -0.32  stable      hpi_growth       -0.09  flipped
    #     cltv_drift   -0.29  stable      nfci_lagged      +0.05  flipped
    #     inflation    +0.13  stable      policy_rate_gap  -0.04  flipped
    #     unemp_gap    -0.09  stable      rate_gap         -0.01  ~zero
    #
    "hpi_growth": (
        "not identified beside cltv_drift, which is built from the *same* house price "
        "index: the information enters twice and the second time is residual. 1sd "
        "effect -0.09 against -0.29, and the sign flips with the sample"
    ),
    "nfci_lagged": (
        "not identified beside vix (rho +0.55), which carries financial stress with "
        "six times the effect and holds its sign"
    ),
    "policy_rate_gap": (
        "not identified beside rate_gap (rho -0.79); both effects are inside the noise"
    ),
    "rate_gap": (
        "conditional effect -0.0066 per standard deviation, which is zero. Its 4.8x "
        "marginal ordering is the macro cycle, not the refinancing incentive, and the "
        "argument made earlier for revising its expected sign on the strength of that "
        "ordering is RETRACTED"
    ),
    # --- eliminated earlier, by the sign and content screens ---
    "credit_spread": (
        "marginal/conditional sign reversal: alone it orders default 4.1x in the "
        "right direction, and flips once nfci_lagged is present -- which is built "
        "from 105 indicators including this very spread"
    ),
    "term_spread": (
        "marginal/conditional sign reversal: 2.9x alone in the right direction, "
        "flips against policy_rate_gap (rho -0.69), its other view of the same cycle"
    ),
    "equity_return": (
        "no marginal signal: 1.4x across its whole range and unordered, against "
        "8.7x for fico_s -- yet p = 0.0000, which is what 2.5 billion loan-months "
        "does to a p-value"
    ),
    "sentiment": "no marginal signal: 2.0x and unordered, same story",
    "starts_growth": (
        "U-shaped: 16.8 bp when construction collapses, 5.3 bp in the middle, "
        "15.1 bp when it booms. The information is real and a linear term cannot "
        "carry it; re-entering it banded would"
    ),
}

MACRO_ELIMINATION_PRIORITY: Final[tuple[str, ...]] = (
    "equity_return",
    "vix",
    "sentiment",
    "term_spread",
    "inflation",
    "starts_growth",
    "policy_rate_gap",
    "credit_spread",
    "hpi_growth",
    "rate_gap",
    "nfci_lagged",
    "unemp_gap",
    "cltv_drift",
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
