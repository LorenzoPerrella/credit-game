"""Covariates derived from loan attributes and the macro panel.

Applied identically wherever a panel comes from, so a difference in results can
never come from a difference in feature construction.

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
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pandas as pd

from creditsurv.config import MACRO_LAG_MONTHS

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Published in arrears and later revised. The **publication lag**: a value for month t
#: is not known in month t.
PUBLICATION_LAGGED: tuple[str, ...] = (
    "unemployment_rate",
    "hpi",
    "nfci",
    "cpi",
    "sentiment",
    "housing_starts",
)

#: Quoted in real time and never revised, so no publication lag applies -- but a
#: **transmission lag** does, and it applies to every series rather than to the one that
#: was noticed.
#:
#: A loan is 90+ days delinquent in month t because payments were missed in t-3, t-2 and
#: t-1. A market reading from month t cannot be what caused that. ``vix`` used to be read
#: contemporaneously, justified by the data being available in real time -- an argument
#: about *availability*, not *transmission* -- and the backtest showed the result:
#: predicted default spiked in April 2025 and March 2026, the two VIX peaks of the test
#: window, at actual-over-expected 0.47 and 0.59, while realised default did not move.
#:
#: Correcting only ``vix`` would repeat the pattern the validation criticised in the
#: marginal-effects table, a fix applied to the covariate that happened to be found, so
#: the lag reaches every market series the macro family can read.
TRANSMISSION_LAGGED: tuple[str, ...] = (
    "mortgage_rate_30y",
    "mortgage_rate_15y",
    "treasury_10y",
    "term_spread",
    "credit_spread",
    "equity_index",
    "vix",
    "policy_rate",
)

#: Every series is lagged. The two lists above record why, and it is not the same reason.
LAGGED_SERIES: tuple[str, ...] = PUBLICATION_LAGGED + TRANSMISSION_LAGGED

#: Empty on purpose. Kept so the rule it used to encode is visibly retired rather than
#: silently absent.
CONTEMPORANEOUS_SERIES: tuple[str, ...] = ()

#: Covariates this module adds to a loan-month panel, and which must all be present
#: for a row to be usable. ``refi_incentive`` and ``indexed_cltv`` are built only where
#: the columns they read exist, so they are checked by presence rather than listed.
DERIVED_COLUMNS: tuple[str, ...] = (
    "cltv_drift",
    "unemp_gap",
    "nfci_lagged",
    "rate_gap",
    "hpi_growth",
)

#: The macro series each derived covariate reads, stated once.
#:
#: It used to exist only implicitly, inside the body of ``add_macro_family``, and that
#: is how a stress scenario could shock two series no fitted covariate read and leave two
#: fitted covariates without a path, with nothing noticing. Stated, it lets a test tie
#: the scenario to the formula; and a second test holds this map to what the builder
#: actually does, so it cannot drift from the code it describes.
MACRO_SOURCES: Final[dict[str, tuple[str, ...]]] = {
    "cltv_drift": ("hpi",),
    "hpi_growth": ("hpi",),
    "unemp_gap": ("unemployment_rate",),
    "policy_rate_gap": ("policy_rate",),
    "rate_gap": ("mortgage_rate_30y", "mortgage_rate_15y"),
    "nfci_lagged": ("nfci",),
    "term_spread": ("term_spread",),
    "credit_spread": ("credit_spread",),
    "vix": ("vix",),
    "sentiment": ("sentiment",),
    "inflation": ("cpi",),
    "equity_return": ("equity_index",),
    "starts_growth": ("housing_starts",),
}


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

    ``panel`` must carry ``period`` and ``orig_period``; each derived covariate is
    built only if the columns it reads are there. That matters because a book
    reconstructed from aggregated cells carries the covariates in the cell key and
    nothing else -- ``note_rate`` is per loan, so it is not among them, and demanding
    it would make the whole projection unavailable for the sake of one covariate the
    model is not fitting.

    Rows whose covariates cannot be built -- typically the earliest vintages, where
    the lagged macro history does not reach back far enough -- are dropped rather
    than imputed.

    The whole macro family comes from :func:`add_macro_family`, which is also what
    the aggregated path calls. One definition, one implementation: a covariate built
    two ways is a covariate that will eventually be built two *different* ways, and
    the backtest is exactly where that would go unnoticed.
    """
    enriched = panel.copy()
    orig_month = _month_ordinal(panel["orig_period"])
    observation = _month_ordinal(panel["period"])
    add_macro_family(enriched, macro, orig_month, observation, lag_months)

    if "orig_ltv" in panel.columns:
        # Kept for inspection rather than modelling: the level and the movement are
        # what the model reads, and this is their sum.
        enriched["indexed_cltv"] = panel["orig_ltv"] + enriched["cltv_drift"]
    if "note_rate" in panel.columns:
        # The full refinancing incentive, available only where the note rate is --
        # which is the loan-level path. The aggregated one carries rate_gap, the part
        # of this that varies over the life of the loan.
        enriched["refi_incentive"] = panel["note_rate"] - _lookup(
            panel["period"], macro["mortgage_rate_30y"]
        )

    built = [name for name in DERIVED_COLUMNS if name in enriched.columns]
    return enriched.dropna(subset=built).reset_index(drop=True)


def _month_ordinal(periods: pd.Series) -> pd.Series:
    """Months since year zero, so a loan age can simply be added to a calendar month."""
    index = pd.PeriodIndex(periods)
    return pd.Series(index.year * 12 + (index.month - 1), index=periods.index)


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
    # Macro candidates. Edges are on the units the series is quoted in -- percentage
    # points for rates and spreads, fractions for year-on-year changes -- so a band
    # can be read without a conversion table. They are starting points: `profile
    # --covariate <name>` proposes quantile cut points from the data, and where the
    # two disagree the data wins.
    "rate_gap": (-6.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 6.0),
    "hpi_growth": (-0.5, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.5),
    "policy_rate_gap": (-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0),
    "term_spread": (-3.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0),
    "credit_spread": (0.0, 1.5, 2.0, 2.5, 3.0, 4.0, 7.0),
    "inflation": (-0.05, 0.0, 0.02, 0.03, 0.05, 0.10),
    "equity_return": (-0.7, -0.2, 0.0, 0.1, 0.2, 0.4, 1.5),
    "vix": (8.0, 14.0, 18.0, 22.0, 28.0, 40.0, 90.0),
    "sentiment": (50.0, 65.0, 75.0, 85.0, 95.0, 115.0),
    "starts_growth": (-0.7, -0.3, -0.1, 0.0, 0.1, 0.3, 1.5),
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


#: Macro series published in real time and never revised, so read contemporaneously
#: rather than lagged. Market quotes: a borrower comparing their note rate with
#: today's does not wait three months to do it, and a Treasury yield is not restated.
CONTEMPORANEOUS_MACRO: Final[tuple[str, ...]] = CONTEMPORANEOUS_SERIES


#: Covariates :func:`add_macro_family` builds. Every one is a function of the
#: vintage quarter and the loan age, so none of them enters the aggregation key and
#: none of them costs a single cell.
MACRO_DERIVED: Final[tuple[str, ...]] = (
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


def add_macro_family(
    episodes: pd.DataFrame,
    macro: pd.DataFrame,
    orig_month: pd.Series,
    observation: pd.Series,
    lag_months: int,
    names: Sequence[str] | None = None,
) -> None:
    """Rebuild every macro-derived covariate, in place.

    Each is a function of the vintage quarter and the loan age, which the cell key
    already carries, so the whole family is free: adding a macro series cannot change
    the size of the cell table by one row, while adding a loan characteristic to the
    key can multiply it. That asymmetry is why the macro side of the specification is
    generous and the loan side is not.

    Three shapes appear, and the distinction is not cosmetic:

    * **a gap since origination** -- unemployment, the policy rate, the market
      mortgage rate. Zero at origination by construction, so it carries the
      *movement* and leaves the level to the origination covariates, which is what
      keeps the pair from being collinear.
    * **a level at the observation date** -- financial conditions, the term and
      credit spreads, volatility, sentiment. The state of the world the loan is
      living in, regardless of what it was written into.
    * **a year-on-year change** -- house prices, inflation, equities, housing
      starts. A twelve-month window rather than a one-month one because the monthly
      change in these is mostly noise.

    Nothing is selected here. These are candidates, and they are deliberately
    collinear -- five interest-rate series will not survive together. Pruning is the
    job of the correlation, VIF and backward-elimination passes in
    :mod:`creditsurv.models.selection`, which is the place the decision is recorded.

    ``names`` restricts the family to what the caller will read. Building all of it
    costs nine unused ``float64`` columns on the largest frame the pipeline holds, and
    at the cell counts an exact calendar key implies that is the difference between
    fitting and not.
    """
    wanted = None if names is None else set(names)

    def requested(name: str) -> bool:
        return wanted is None or name in wanted

    lagged = macro.shift(lag_months)
    for column in CONTEMPORANEOUS_MACRO:
        if column in macro.columns:
            lagged[column] = macro[column]

    index = pd.PeriodIndex(macro.index)
    month = index.year * 12 + (index.month - 1)
    available = set(macro.columns)

    def at(column: str, when: pd.Series, *, offset: int = 0) -> np.ndarray:
        """A macro column read at ``when``, optionally shifted back ``offset`` months."""
        series = pd.Series(lagged[column].to_numpy(), index=month)
        return np.asarray((when - offset).map(series).to_numpy(), dtype=float)

    def store(name: str, values: np.ndarray) -> None:
        """Keep a covariate at single precision.

        These are rebuilt from macro indices published to three or four significant
        figures, so the seventh decimal of a float64 carries no information -- only
        bytes, on the frame where bytes decide whether the fit runs.
        """
        episodes[name] = values.astype(np.float32)

    def level(column: str) -> np.ndarray:
        """The series as it stands at the observation date."""
        return at(column, observation)

    def gap(column: str) -> np.ndarray:
        """How far a series has moved since the loan was written."""
        moved: np.ndarray = at(column, observation) - at(column, orig_month)
        return moved

    def growth(column: str) -> np.ndarray:
        """Year-on-year change, as a fraction."""
        change: np.ndarray = at(column, observation) / at(column, observation, offset=12) - 1.0
        return change

    # Built only where the series exists, so a panel assembled from a partial cache
    # yields the covariates it can rather than raising on the first one it cannot.
    for name, source, build in (
        ("unemp_gap", "unemployment_rate", gap),
        ("policy_rate_gap", "policy_rate", gap),
        ("nfci_lagged", "nfci", level),
        ("term_spread", "term_spread", level),
        ("credit_spread", "credit_spread", level),
        ("vix", "vix", level),
        ("sentiment", "sentiment", level),
        ("hpi_growth", "hpi", growth),
        ("inflation", "cpi", growth),
        ("equity_return", "equity_index", growth),
        ("starts_growth", "housing_starts", growth),
    ):
        if source in available and requested(name):
            store(name, build(source))

    # The refinancing benchmark is switched by term, repeating what the source data
    # says rather than what is convenient: a fifteen-year loan is refinanced against
    # the fifteen-year rate. That is possible only because term_years is in the cell
    # key -- and until it was, every loan was compared with the thirty-year rate,
    # which is simply the wrong benchmark for a quarter of the book.
    #
    # This is the movement in the market rate since origination, not the full
    # refinancing incentive: the latter needs the loan's own note rate, which the key
    # does not carry. The constant part is missing; the part that varies is here.
    if "mortgage_rate_30y" in available and requested("rate_gap"):
        thirty = -gap("mortgage_rate_30y")
        if "term_years" in episodes.columns and "mortgage_rate_15y" in available:
            short = episodes["term_years"].to_numpy(dtype=float) <= 20.0
            store("rate_gap", np.where(short, -gap("mortgage_rate_15y"), thirty))
        else:
            store("rate_gap", thirty)

    # Mark-to-market leverage, from the national house price index. Derived here
    # rather than carried in the grouping key because it is a function of orig_ltv
    # and the macro path, both of which the key already holds -- carrying it was
    # doubling the cell count for information already there.
    #
    # The index rather than Freddie's own per-loan ELTV, which would be better if it
    # were usable: its coverage runs from 0.8% of the 1999 vintage to 94% of 2021, so
    # a model built on it would estimate a different quantity in every decade.
    if "orig_ltv" in episodes.columns and "hpi" in available and requested("cltv_drift"):
        original = episodes["orig_ltv"].to_numpy(dtype=float)
        store("cltv_drift", original * at("hpi", orig_month) / at("hpi", observation) - original)
