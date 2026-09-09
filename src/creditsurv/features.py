"""Covariates derived from loan attributes and the macro panel.

Shared by the synthetic generator and the Freddie Mac loader so both produce
identical model inputs, and so the data-generating process cannot accidentally
use a covariate the model never sees.

Two decisions are encoded here and are worth stating explicitly.

**Publication lag.** Series that are published in arrears and later revised are
lagged before use: unemployment, the house price index and the financial
conditions index. The 30-year mortgage rate is not, because it is a market
quote published weekly and never revised, so a borrower and a lender both know
it in real time. Applying a blanket lag to everything would be simpler but
would misstate what was actually knowable.

**Loan-to-value is decomposed rather than replaced.** ``orig_ltv`` and
``indexed_cltv`` are equal at origination and remain strongly correlated
afterwards, so putting both in a model invites unstable coefficients. They are
split into a level and a movement instead: ``orig_ltv`` measures underwriting
at origination, and ``cltv_drift`` measures how far house prices have carried
the position since. ``cltv_drift`` is zero at origination by construction, so
the two carry almost independent information.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from creditsurv.config import MACRO_LAG_MONTHS

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Published in arrears and revised, so lagged before entering any covariate.
LAGGED_SERIES: tuple[str, ...] = ("unemployment_rate", "hpi", "nfci")

#: Published in real time and never revised, so used contemporaneously.
CONTEMPORANEOUS_SERIES: tuple[str, ...] = ("mortgage_rate_30y",)

#: Covariates this module adds to a loan-month panel.
DERIVED_COLUMNS: tuple[str, ...] = (
    "indexed_cltv",
    "cltv_drift",
    "unemp_gap",
    "refi_incentive",
    "nfci_lagged",
)


def lag_macro(macro: pd.DataFrame, *, lag_months: int = MACRO_LAG_MONTHS) -> pd.DataFrame:
    """Shift revised series forward so only published information is used.

    Returns a frame on the same index, with the lagged series overwritten and the
    contemporaneous ones untouched. The leading ``lag_months`` rows of the lagged
    columns become missing, which is correct: that information did not exist yet.
    """
    lagged = macro.copy()
    for column in LAGGED_SERIES:
        if column in lagged.columns:
            lagged[column] = lagged[column].shift(lag_months)
    return lagged


def _lookup(periods: pd.Series, values: pd.Series) -> pd.Series:
    """Map a column of periods onto a period-indexed macro series."""
    return periods.map(values)


def add_macro_covariates(
    panel: pd.DataFrame,
    macro: pd.DataFrame,
    *,
    lag_months: int = MACRO_LAG_MONTHS,
) -> pd.DataFrame:
    """Attach macro-derived covariates to a loan-month panel.

    ``panel`` must carry ``period``, ``orig_period``, ``orig_ltv`` and
    ``note_rate``. Rows whose covariates cannot be built -- typically the
    earliest vintages, where the lagged macro history does not reach back far
    enough -- are dropped rather than imputed.
    """
    lagged = lag_macro(macro, lag_months=lag_months)

    hpi_at_origination = _lookup(panel["orig_period"], lagged["hpi"])
    hpi_now = _lookup(panel["period"], lagged["hpi"])
    unemployment_at_origination = _lookup(panel["orig_period"], lagged["unemployment_rate"])
    unemployment_now = _lookup(panel["period"], lagged["unemployment_rate"])

    enriched = panel.copy()
    # Mark-to-market loan-to-value: the original ratio re-expressed at today's
    # house prices. Prices up, the ratio falls and the borrower has more equity.
    enriched["indexed_cltv"] = panel["orig_ltv"] * hpi_at_origination / hpi_now
    enriched["cltv_drift"] = enriched["indexed_cltv"] - panel["orig_ltv"]
    enriched["unemp_gap"] = unemployment_now - unemployment_at_origination
    enriched["refi_incentive"] = panel["note_rate"] - _lookup(
        panel["period"], macro["mortgage_rate_30y"]
    )
    enriched["nfci_lagged"] = _lookup(panel["period"], lagged["nfci"])

    return enriched.dropna(subset=list(DERIVED_COLUMNS)).reset_index(drop=True)


def assert_no_lookahead(
    panel: pd.DataFrame,
    macro: pd.DataFrame,
    columns: Sequence[str] = LAGGED_SERIES,
    *,
    lag_months: int = MACRO_LAG_MONTHS,
) -> None:
    """Raise if any lagged covariate could only be known after its own period.

    A cheap structural check: every value used at period ``p`` must be traceable
    to an observation at or before ``p - lag_months``.
    """
    lagged = lag_macro(macro, lag_months=lag_months)
    for column in columns:
        if column not in macro.columns:
            continue
        # An all-missing column has no first valid index to compare.
        if bool(macro[column].isna().all()) or bool(lagged[column].isna().all()):
            continue
        source_available_from = cast("pd.Period", macro[column].first_valid_index())
        usable_from = cast("pd.Period", lagged[column].first_valid_index())
        gap = (usable_from - source_available_from).n
        if gap < lag_months:
            message = (
                f"{column} becomes usable only {gap} month(s) after its first observation, "
                f"which is less than the required lag of {lag_months}."
            )
            raise ValueError(message)


# --------------------------------------------------------------------------------------
# Coarse classing
# --------------------------------------------------------------------------------------

#: Cut points for each continuous covariate, in the covariate's own units.
#:
#: Chosen from credit conventions rather than from the data: the loan-to-value
#: breaks sit at 80, 85, 90 and 95 because that is where mortgage insurance and
#: pricing tiers actually change, and the debt-to-income breaks sit at 36 and 43
#: because those are long-standing underwriting thresholds. Data-driven cuts
#: would fit this sample better and would have to be refitted, and re-justified,
#: for every new one.
#:
#: Bands are closed on the right, matching ``pd.cut`` defaults. Values outside the
#: outer edges are clipped into the end bands rather than dropped.
BIN_EDGES: dict[str, tuple[float, ...]] = {
    # (score - 700) / 50, so -2.4 is a score of 580 and +2.0 is 800.
    "fico_s": (-2.4, -1.6, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 2.4),
    "orig_ltv": (30.0, 60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0),
    "dti": (10.0, 20.0, 28.0, 36.0, 43.0, 50.0, 55.0),
    "log_orig_upb": (10.0, 11.3, 11.8, 12.1, 12.4, 12.7, 13.2, 14.5),
    "orig_spread": (-2.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 4.0),
    # Percentage points of loan-to-value gained or lost since origination.
    "cltv_drift": (-60.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 80.0),
    "unemp_gap": (-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 12.0),
    "refi_incentive": (-6.0, -1.0, 0.0, 0.5, 1.0, 2.0, 6.0),
    "nfci_lagged": (-2.0, -0.6, -0.4, -0.2, 0.0, 0.5, 1.0, 5.0),
}

#: Suffix for the representative value of a covariate's band.
BINNED_SUFFIX: str = "_binned"

#: Suffix for the band label, kept for reporting rather than modelling.
BAND_SUFFIX: str = "_band"


def bin_covariates(
    panel: pd.DataFrame,
    edges: dict[str, tuple[float, ...]] | None = None,
) -> pd.DataFrame:
    """Replace continuous covariates by the midpoint of their band.

    Coarse classing is a modelling decision, not a performance trick, and it is
    worth being explicit about the trade. It buys robustness to outliers,
    non-linearity without splines, and an explainable model of the kind credit
    committees and regulators expect. It costs within-bin information and turns a
    smooth covariate effect into a step function.

    It also makes grouped estimation worthwhile. Continuous, loan-specific
    covariates leave almost every episode unique, so aggregating identical rows
    saves nothing until the values are coarsened.

    The representative value is the band midpoint, so the binned covariate stays
    on the original scale and its coefficient remains directly comparable with the
    unbinned fit. Values beyond the outer edges are clipped into the end bands.
    """
    cut_points = BIN_EDGES if edges is None else edges
    binned = panel.copy()

    for column, breaks in cut_points.items():
        if column not in binned.columns:
            continue
        values = binned[column].clip(lower=breaks[0], upper=breaks[-1])
        bands = pd.cut(values, bins=list(breaks), include_lowest=True)
        midpoints = np.array([(breaks[i] + breaks[i + 1]) / 2.0 for i in range(len(breaks) - 1)])
        codes = bands.cat.codes.to_numpy()
        binned[column + BINNED_SUFFIX] = midpoints[codes]
        binned[column + BAND_SUFFIX] = bands.astype(str)

    return binned


def binned_formula(formula: str, edges: dict[str, tuple[float, ...]] | None = None) -> str:
    """Rewrite a formula to use the binned form of every coarse-classed covariate.

    Keeps a single source of truth for the model specification: the binned and
    unbinned fits differ only in which column each term reads.
    """
    cut_points = BIN_EDGES if edges is None else edges
    rewritten = formula
    # Longest names first so a shorter name cannot match inside a longer one.
    for column in sorted(cut_points, key=len, reverse=True):
        rewritten = re.sub(
            rf"(?<![\w.]){re.escape(column)}(?![\w.])",
            column + BINNED_SUFFIX,
            rewritten,
        )
    return rewritten
