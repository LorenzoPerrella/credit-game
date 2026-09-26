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

**Loan-to-value is decomposed rather than replaced.** ``original_ltv`` and
``indexed_cltv`` are equal at origination and remain strongly correlated
afterwards, so putting both in a model invites unstable coefficients. They are
split into a level and a movement instead: ``original_ltv`` measures underwriting
at origination, and ``ltv_change`` measures how far house prices have carried
the position since. ``ltv_change`` is zero at origination by construction, so
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
    "house_price_index",
    "financial_conditions_index",
    "consumer_price_index",
    "consumer_sentiment_index",
    "housing_starts",
)

#: Quoted in real time and never revised, so no publication lag applies -- but a
#: **transmission lag** does, and it applies to every series rather than to the one that
#: was noticed.
#:
#: A loan is 90+ days delinquent in month t because payments were missed in t-3, t-2 and t-1. A
#: market reading from month t cannot be what caused that. ``equity_volatility`` used to be read
#: contemporaneously, justified by the data being available in real time -- an argument about
#: *availability*, not *transmission* -- and the backtest showed the result: predicted default
#: spiked in April 2025 and March 2026, the two VIX peaks of the test window, at
#: actual-over-expected 0.47 and 0.59, while realised default did not move.
#:
#: Correcting only ``equity_volatility`` would repeat the pattern the validation criticised in the
#: marginal-effects table, a fix applied to the covariate that happened to be found, so
#: the lag reaches every market series the macro family can read.
TRANSMISSION_LAGGED: tuple[str, ...] = (
    "mortgage_rate_30y",
    "mortgage_rate_15y",
    "treasury_10y",
    "treasury_10y_2y_spread",
    "baa_treasury_spread",
    "nasdaq_composite",
    "vix_index",
    "fed_funds_rate",
)

#: Every series is lagged. The two lists above record why, and it is not the same reason.
LAGGED_SERIES: tuple[str, ...] = PUBLICATION_LAGGED + TRANSMISSION_LAGGED

#: Empty on purpose. Kept so the rule it used to encode is visibly retired rather than
#: silently absent.
CONTEMPORANEOUS_SERIES: tuple[str, ...] = ()

#: The macro series each derived covariate reads, stated once.
#:
#: It used to exist only implicitly, inside the body of ``add_macro_family``, and that
#: is how a stress scenario could shock two series no fitted covariate read and leave two
#: fitted covariates without a path, with nothing noticing. Stated, it lets a test tie
#: the scenario to the formula; and a second test holds this map to what the builder
#: actually does, so it cannot drift from the code it describes.
MACRO_SOURCES: Final[dict[str, tuple[str, ...]]] = {
    "ltv_change": ("house_price_index",),
    "house_price_growth": ("house_price_index",),
    "unemployment_change": ("unemployment_rate",),
    "policy_rate_change": ("fed_funds_rate",),
    "mortgage_rate_decline": ("mortgage_rate_30y", "mortgage_rate_15y"),
    "financial_conditions": ("financial_conditions_index",),
    "yield_curve_slope": ("treasury_10y_2y_spread",),
    "corporate_bond_spread": ("baa_treasury_spread",),
    "equity_volatility": ("vix_index",),
    "volatility_change": ("vix_index",),
    "consumer_sentiment": ("consumer_sentiment_index",),
    "inflation_rate": ("consumer_price_index",),
    "inflation_change": ("consumer_price_index",),
    "equity_return": ("nasdaq_composite",),
    "housing_starts_growth": ("housing_starts",),
    "origination_spread": ("mortgage_rate_30y", "mortgage_rate_15y"),
    "refinance_incentive": ("mortgage_rate_30y", "mortgage_rate_15y"),
}

#: Exact linear identities among the derived covariates. A design holding every member of
#: one of these is singular by construction, not nearly so, and the fit either fails or
#: returns whatever the pseudo-inverse chose.
#:
#: The note rate against the market rate is the only one: the incentive to refinance now is
#: the spread the loan was written at plus the fall in the market rate since. Which two of the
#: three a specification carries is a modelling decision, declared in docs/rules.md; the
#: correlation pass would not catch it, since two of the three are already in the model
#: before the third arrives.
MACRO_IDENTITIES: Final[tuple[tuple[str, ...], ...]] = (
    ("refinance_incentive", "origination_spread", "mortgage_rate_decline"),
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

    ``panel`` must carry ``period`` and ``origination_period``; each derived covariate is
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
    origination_month = _month_ordinal(panel["origination_period"])
    observation = _month_ordinal(panel["period"])
    add_macro_family(enriched, macro, origination_month, observation, lag_months)

    if "original_ltv" in panel.columns:
        # Kept for inspection rather than modelling: the level and the movement are
        # what the model reads, and this is their sum.
        enriched["indexed_cltv"] = panel["original_ltv"] + enriched["ltv_change"]
    if "note_rate" in panel.columns:
        # The full refinancing incentive, available only where the note rate is --
        # which is the loan-level path. The aggregated one carries mortgage_rate_decline, the part
        # of this that varies over the life of the loan.
        enriched["refinance_incentive"] = panel["note_rate"] - _lookup(
            panel["period"], macro["mortgage_rate_30y"]
        )

    # Every macro covariate built, as the aggregated path checks. The list used here had
    # been frozen at the specification before the selection, so a missing sentiment or
    # inflation_change passed through a projection to become a missing PD.
    built = [name for name in MACRO_DERIVED if name in enriched.columns]
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
    # Credit score points.
    "credit_score": (580.0, 620.0, 660.0, 680.0, 700.0, 720.0, 740.0, 760.0, 820.0),
    "original_ltv": (30.0, 60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0),
    "debt_to_income": (10.0, 20.0, 28.0, 36.0, 43.0, 50.0, 55.0),
    "log_original_balance": (10.0, 11.3, 11.8, 12.1, 12.4, 12.7, 13.2, 14.5),
    "origination_spread": (-2.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 4.0),
    # The note rate itself is banded only because it is in the cell key, where it buys the
    # spread and the refinancing incentive, and a band of the rate is a band of both. Half a
    # point, which is roughly the width of the pricing grid the loans were written on and is
    # coarse enough that the origination month -- already in the key -- carries most of the
    # variation. The outer bands hold the 1999 and 2000 book above 8% and the 2021 book below 3.
    "note_rate": (2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 12.0),
    # Percentage points of loan-to-value gained or lost since origination.
    "ltv_change": (-60.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 80.0),
    "unemployment_change": (-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 12.0),
    "refinance_incentive": (-6.0, -1.0, 0.0, 0.5, 1.0, 2.0, 6.0),
    "financial_conditions": (-2.0, -0.6, -0.4, -0.2, 0.0, 0.5, 1.0, 5.0),
    # Macro candidates. Edges are on the units the series is quoted in -- percentage
    # points for rates and spreads, fractions for year-on-year changes -- so a band
    # can be read without a conversion table. They are starting points: `profile
    # --covariate <name>` proposes quantile cut points from the data, and where the
    # two disagree the data wins.
    "mortgage_rate_decline": (-6.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 6.0),
    "house_price_growth": (-0.5, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.5),
    "policy_rate_change": (-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0),
    "yield_curve_slope": (-3.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0),
    "corporate_bond_spread": (0.0, 1.5, 2.0, 2.5, 3.0, 4.0, 7.0),
    "inflation_rate": (-0.05, 0.0, 0.02, 0.03, 0.05, 0.10),
    # Year-on-year CPI inflation now, less at origination. Whole points either side of zero;
    # the outer bands reach past the exposure's 1st and 99th percentiles, -3.7 and +6.0.
    "inflation_change": (-0.10, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.12),
    "equity_return": (-0.7, -0.2, 0.0, 0.1, 0.2, 0.4, 1.5),
    "equity_volatility": (8.0, 14.0, 18.0, 22.0, 28.0, 40.0, 90.0),
    "consumer_sentiment": (50.0, 65.0, 75.0, 85.0, 95.0, 115.0),
    "housing_starts_growth": (-0.7, -0.3, -0.1, 0.0, 0.1, 0.3, 1.5),
}

#: Covariates a loan may legitimately not report: the value standing in for the missing
#: one, and the level that has to be in the model for it to stand in harmlessly.
#:
#: A HARP refinance reports no debt-to-income, and it is the *only* reason one is missing
#: in the cells: every other loan without the ratio is dropped by the complete-case rule.
#: So the missing indicator and the HARP level are the same column, and filling those rows
#: with a constant while the model holds that level is the **dummy-variable adjustment**: the
#: slope is estimated on the loans that report the ratio, and the constant is absorbed whole
#: by the level. It is exactly a "not reported" band of its own, written on the scale the
#: covariate is already on, and the constant is arbitrary for the same reason -- any value
#: gives the same likelihood and the same slope, moving only the level's own coefficient.
#:
#: Nothing is imputed. No HARP loan is given a debt-to-income anyone could read as its own:
#: the cell table keeps the missing value missing, and this is the model's side of it.
NOT_REPORTED: Final[dict[str, tuple[str, str, float]]] = {
    "debt_to_income": ("harp", "harp", 32.0),
}


def absorb_not_reported(episodes: pd.DataFrame, requested: Sequence[str] | None = None) -> None:
    """Fill what a loan does not report, where a level of the model absorbs the fill.

    In place, and refusing rather than guessing: a missing value whose indicator is not in
    the frame, or which the indicator does not explain, is a gap nobody has looked at.

    ``requested`` is the covariate list the caller will fit on, and the fill is only harmless
    while the indicator is in it. A model reading a filled debt-to-income without the HARP
    level would read a constant of 32 as a real ratio for 18% of a decade of vintages, which
    is the imputation this whole arrangement exists to avoid, so it raises instead. ``None``
    means the caller has not decided yet and the check belongs to whoever does.
    """
    for name, (indicator, level, fill) in NOT_REPORTED.items():
        if name not in episodes.columns:
            continue
        missing = episodes[name].isna().to_numpy()
        if not missing.any():
            continue
        if requested is not None and name in requested and indicator not in requested:
            message = (
                f"{name} is fitted without {indicator}, and is missing on "
                f"{missing.sum():,} rows. The level that absorbs the fill has to be in the model."
            )
            raise ValueError(message)
        if indicator not in episodes.columns:
            message = (
                f"{name} is missing on {missing.sum():,} rows and the frame does not carry "
                f"{indicator}, the level that would absorb the fill."
            )
            raise ValueError(message)
        unexplained = missing & (episodes[indicator].astype(str) != level).to_numpy()
        if unexplained.any():
            message = (
                f"{unexplained.sum():,} rows do not report {name} and are not {indicator} "
                f"= {level}. Only a {level} loan is allowed to be missing it."
            )
            raise ValueError(message)
        episodes[name] = episodes[name].fillna(fill)


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
    "volatility_change",
    "inflation_change",
    # The two that need the loan's own note rate, which is why the cell key carries it:
    # with the rate in the key they are functions of the key like every other member here,
    # and cost no cells of their own.
    "origination_spread",
    "refinance_incentive",
)


def add_macro_family(
    episodes: pd.DataFrame,
    macro: pd.DataFrame,
    origination_month: pd.Series,
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
        moved: np.ndarray = at(column, observation) - at(column, origination_month)
        return moved

    def growth_at(column: str, when: pd.Series) -> np.ndarray:
        """Year-on-year change to ``when``, as a fraction."""
        change: np.ndarray = at(column, when) / at(column, when, offset=12) - 1.0
        return change

    def growth(column: str) -> np.ndarray:
        """Year-on-year change at the observation date."""
        return growth_at(column, observation)

    def growth_gap(column: str) -> np.ndarray:
        """How far the year-on-year change has moved since the loan was written."""
        moved: np.ndarray = growth_at(column, observation) - growth_at(column, origination_month)
        return moved

    # Built only where the series exists, so a panel assembled from a partial cache
    # yields the covariates it can rather than raising on the first one it cannot.
    for name, source, build in (
        ("unemployment_change", "unemployment_rate", gap),
        ("policy_rate_change", "fed_funds_rate", gap),
        ("financial_conditions", "financial_conditions_index", level),
        ("yield_curve_slope", "treasury_10y_2y_spread", level),
        ("corporate_bond_spread", "baa_treasury_spread", level),
        ("equity_volatility", "vix_index", level),
        # The gap forms of the two levels the validation called calendar effects (S5).
        # A level is the same number for every loan in a month; its move since
        # origination is not, so it is identified from loans living through the same
        # month on different terms as well as from the calendar.
        ("volatility_change", "vix_index", gap),
        ("consumer_sentiment", "consumer_sentiment_index", level),
        ("house_price_growth", "house_price_index", growth),
        ("inflation_rate", "consumer_price_index", growth),
        ("inflation_change", "consumer_price_index", growth_gap),
        ("equity_return", "nasdaq_composite", growth),
        ("housing_starts_growth", "housing_starts", growth),
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
    if "mortgage_rate_30y" in available and requested("mortgage_rate_decline"):
        thirty = -gap("mortgage_rate_30y")
        if "term_years" in episodes.columns and "mortgage_rate_15y" in available:
            short = episodes["term_years"].to_numpy(dtype=float) <= 20.0
            store("mortgage_rate_decline", np.where(short, -gap("mortgage_rate_15y"), thirty))
        else:
            store("mortgage_rate_decline", thirty)

    # The loan's own rate against the market's. Until the note rate entered the cell key
    # this was out of reach after the collapse, and mortgage_rate_decline -- the movement
    # without the constant -- was as close as the panel could get.
    #
    # Same benchmark switch by term, for the same reason: a fifteen-year loan is refinanced
    # against the fifteen-year rate.
    #
    # Only two of the three may enter a model: see MACRO_IDENTITIES.
    if "note_rate" in episodes.columns and "mortgage_rate_30y" in available:
        note = episodes["note_rate"].to_numpy(dtype=float)

        def market(when: pd.Series) -> np.ndarray:
            """The market mortgage rate the loan would be refinanced at."""
            thirty = at("mortgage_rate_30y", when)
            if "term_years" not in episodes.columns or "mortgage_rate_15y" not in available:
                return thirty
            short = episodes["term_years"].to_numpy(dtype=float) <= 20.0
            switched: np.ndarray = np.where(short, at("mortgage_rate_15y", when), thirty)
            return switched

        if requested("origination_spread"):
            store("origination_spread", note - market(origination_month))
        if requested("refinance_incentive"):
            store("refinance_incentive", note - market(observation))

    # Mark-to-market leverage, from the national house price index. Derived here
    # rather than carried in the grouping key because it is a function of original_ltv
    # and the macro path, both of which the key already holds -- carrying it was
    # doubling the cell count for information already there.
    #
    # The index rather than Freddie's own per-loan ELTV, which would be better if it
    # were usable: its coverage runs from 0.8% of the 1999 vintage to 94% of 2021, so
    # a model built on it would estimate a different quantity in every decade.
    if (
        "original_ltv" in episodes.columns
        and "house_price_index" in available
        and requested("ltv_change")
    ):
        original = episodes["original_ltv"].to_numpy(dtype=float)
        store(
            "ltv_change",
            original
            * at("house_price_index", origination_month)
            / at("house_price_index", observation)
            - original,
        )
