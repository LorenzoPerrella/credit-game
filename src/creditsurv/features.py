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

from typing import TYPE_CHECKING, cast

from creditsurv.config import MACRO_LAG_MONTHS

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

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
