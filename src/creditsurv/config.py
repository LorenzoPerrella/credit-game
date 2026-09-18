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
_TABLES_DIR_ENV: Final = "CREDITSURV_TABLES_DIR"


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


def tables_dir() -> Path:
    """The aggregate tables behind the documentation site's views.

    Committed, because the data they come from cannot reach the CI that builds the site: a
    view is computed locally by ``creditsurv views`` and turned into a figure at build time.
    """
    override = os.environ.get(_TABLES_DIR_ENV)
    return Path(override) if override else project_root() / "docs" / "tables"


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
        column="house_price_index",
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
        column="financial_conditions_index",
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
        column="fed_funds_rate",
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
        column="treasury_10y_2y_spread",
        frequency=Frequency.DAILY,
        description="10-year minus 2-year Treasury spread, percentage points.",
    ),
    SeriesSpec(
        series_id="BAA10Y",
        column="baa_treasury_spread",
        frequency=Frequency.DAILY,
        description="Moody's Baa corporate yield over the 10-year Treasury.",
    ),
    SeriesSpec(
        series_id="CPIAUCSL",
        column="consumer_price_index",
        frequency=Frequency.MONTHLY,
        description="Consumer price index, all urban consumers, seasonally adjusted.",
    ),
    SeriesSpec(
        series_id="NASDAQCOM",
        column="nasdaq_composite",
        frequency=Frequency.DAILY,
        description="Nasdaq Composite index.",
    ),
    SeriesSpec(
        series_id="VIXCLS",
        column="vix_index",
        frequency=Frequency.DAILY,
        description="CBOE volatility index.",
    ),
    SeriesSpec(
        series_id="UMCSENT",
        column="consumer_sentiment_index",
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
#: ``log_original_balance`` and ``origination_spread`` are absent for that reason rather than on
#: their merits -- the cell count is the product of the band counts, so every
#: covariate added to the key multiplies it. What each would cost is measured in
#: docs/variable_selection.md.
STATIC_CONTINUOUS: Final[tuple[str, ...]] = (
    "credit_score",
    "original_ltv",
    "debt_to_income",
)

#: Takes two values, 15 and 30, so a linear term and a dummy are the same model.
#: Kept numeric because the aggregation emits it as an integer, and a treatment
#: reference would then have to agree with that dtype across the SQL and the formula.
ORDINAL: Final[tuple[str, ...]] = ("term_years",)

#: Recomputed every loan-month. Loan age is the time scale, not a covariate.
#:
#: ``ltv_change`` rather than ``indexed_cltv``: the indexed ratio equals ``original_ltv`` at
#: origination and stays strongly correlated with it, so fitting both gives unstable coefficients.
#: The pair is decomposed into a level (``original_ltv``, underwriting at origination) and a
#: movement (``ltv_change``, how far house prices have carried the position since, zero at
#: origination). ``refinance_incentive`` is absent: it needs the note rate, which the key does not
#: carry. It is recoverable the same way ``ltv_change`` is -- from a banded ``origination_spread``
#: plus the mortgage-rate path, both functions of the key -- which is the cheapest of the candidate
#: additions and the one to weigh first. **The output of ``creditsurv select``, not a choice made
#: here.** Steps 5 to 9 on the training half kept seven of the fifteen macro candidates, and
#: ``tests/test_procedure.py`` fails if this tuple and ``docs/reports/selection.json`` part. One per
#: economic dimension except housing, where ``housing_starts_growth``, the construction cycle, held
#: its sign beside ``ltv_change`` on both halves of the book (1 sd effect +0.068 and +0.088 against
#: -0.168 and -0.186). The first run, done by hand before the validation, kept ``equity_volatility``
#: and ``inflation_rate``; the selection removed both. See ``ELIMINATED``.
TIME_VARYING_CONTINUOUS: Final[tuple[str, ...]] = (
    "ltv_change",
    "unemployment_change",
    "financial_conditions",
    "policy_rate_change",
    "consumer_sentiment",
    "housing_starts_growth",
    "inflation_change",
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
    "ltv_change",
    "unemployment_change",
    "financial_conditions",
    "mortgage_rate_decline",
    "house_price_growth",
    "policy_rate_change",
    "yield_curve_slope",
    "corporate_bond_spread",
    "inflation_rate",
    "equity_return",
    "equity_volatility",
    "consumer_sentiment",
    "housing_starts_growth",
    # Added for the validation's S5: ``equity_volatility`` and ``inflation_rate`` enter as levels at
    # the observation date, identical for every loan in a month, so their coefficients are calendar
    # effects by construction. Their moves since origination vary across loans in the same month.
    # Both forms are candidates and the selection decides.
    "volatility_change",
    "inflation_change",
)

#: The economic dimension each candidate measures, fixed before any selection result.
#:
#: The stability rule of step 9 removes a covariate that is small, changes sign when the
#: sample does, and sits beside a larger covariate of the **same** dimension. Which
#: covariates share a dimension therefore has to be decided in advance: decided after the
#: fits, the rule would be a way of dropping whatever came out inconvenient.
ECONOMIC_DIMENSION: Final[dict[str, str]] = {
    "credit_score": "credit quality",
    "original_ltv": "leverage at origination",
    "debt_to_income": "debt burden",
    "term_years": "term",
    "ltv_change": "housing",
    "house_price_growth": "housing",
    "housing_starts_growth": "housing",
    "unemployment_change": "labour",
    "equity_volatility": "financial stress",
    "volatility_change": "financial stress",
    "financial_conditions": "financial stress",
    "corporate_bond_spread": "financial stress",
    "mortgage_rate_decline": "interest rates",
    "policy_rate_change": "interest rates",
    "yield_curve_slope": "interest rates",
    "inflation_rate": "prices",
    "inflation_change": "prices",
    "equity_return": "asset prices",
    "consumer_sentiment": "confidence",
}

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
    # The keys are the selection record's and are tested against it; the reasons carry its
    # numbers. The first run's arguments for eliminating by hand are in
    # docs/variable_selection.md, kept as history.
    #
    # --- step 6: collinearity, in the priority fixed before any fit ---
    "corporate_bond_spread": (
        "variance inflation 11.9, above 10 -- the first candidate over the threshold. The "
        "first run had it at 8.41 and removed it by hand for a sign reversal"
    ),
    # --- step 8: backwards against a declared prior ---
    "volatility_change": (
        "wrong sign: +0.00254 in the full model, where stress should shorten survival"
    ),
    "equity_volatility": (
        "wrong sign: +0.00935 in the full model once volatility_change is gone, against "
        "-0.0198 beside the loan block alone. The first run's largest effect: part of it was "
        "the 2020 moratoria, as the validation suspected (S5), and the rest is shared with "
        "financial_conditions"
    ),
    # --- step 8: its sign reversed against its own ---
    "mortgage_rate_decline": (
        "reversed: -0.190 beside the loan block alone, +0.018 in the full model"
    ),
    "equity_return": "reversed: +0.431 beside the loan block alone, -0.086 in the full model",
    "inflation_rate": (
        "reversed: +10.04 beside the loan block alone, -8.33 in the full model, against "
        "inflation_change's +6.9 there -- together the pair was reading inflation at origination"
    ),
    # --- step 9: not identified beside a larger covariate of the same dimension ---
    "yield_curve_slope": (
        "1 sd effect -0.058 on even and +0.013 on odd origination years, beside "
        "policy_rate_change at +0.119"
    ),
    "house_price_growth": (
        "1 sd effect -0.004 on even and +0.015 on odd origination years, beside ltv_change "
        "at -0.179, which is built from the same house price index"
    ),
}

MACRO_ELIMINATION_PRIORITY: Final[tuple[str, ...]] = (
    "equity_return",
    "equity_volatility",
    # A level gives way before its own gap form, for the reason at MACRO_CANDIDATES.
    "volatility_change",
    "consumer_sentiment",
    "yield_curve_slope",
    "inflation_rate",
    "inflation_change",
    "housing_starts_growth",
    "policy_rate_change",
    "corporate_bond_spread",
    "house_price_growth",
    "mortgage_rate_decline",
    "financial_conditions",
    "unemployment_change",
    "ltv_change",
)

#: The distribution family the published model uses.
#:
#: **An output, like the specification.** Rule 2 of `docs/rules.md` takes both families
#: through the whole selection and keeps the selected model closest to the Aalen-Johansen
#: cumulative incidence of default, excluding any family that turns a declared sign and
#: falling back on the Weibull inside a tenth of a percentage point.
#: ``tests/test_procedure.py`` fails when this and the selection record part, so the family
#: cannot be changed here without the run that justifies it.
DISTRIBUTION: Final = "weibull"

#: Categorical covariates mapped to their treatment-coding reference level.
#: ``mortgage_insurance`` and ``buyer_type`` entered the key with the validation (M3), at 1.19x
#: the cells where an unmeasured sixteenfold had kept them out, and the model with the
#: selection, whose screen beside the loan block put them at z = -46 and +26. ``channel``
#: and ``region`` are mapped but not in the key.
CATEGORICAL_REFERENCE: Final[dict[str, str]] = {
    "purpose": "purchase",
    "occupancy": "owner_occupied",
    "mortgage_insurance": "uninsured",
    "buyer_type": "repeat",
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
