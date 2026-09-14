"""Collapsing 1.75 billion loan-months into a table a model can be fitted to.

Episodes that agree on every covariate and on their position in time are
exchangeable, so they can be replaced by one row carrying a count. That is the whole
trick, and at this scale it is not an optimisation but the only thing that makes the
problem tractable: a fit over two billion rows is out of reach, a fit over a million
weighted cells is a minute.

The work is done in DuckDB rather than pandas because the join, the truncation and
the group-by all have to happen out of core — the inputs are far larger than memory
and never need to be resident.

**The macro series are deliberately absent from the grouping key.** Since
``period = orig_period + age``, unemployment, house prices and financial conditions
are a deterministic function of two columns that are already in the key, so they can
be recomputed on the aggregate at no cost in cardinality. Putting them in the key
instead would multiply it by the number of distinct months and destroy the collapse.

An earlier measurement in this project found the same technique compressing
1.00x and concluded it was not worth having. That was true at 215,000 rows, where
the possible cells vastly outnumbered the rows. At 1.75 billion the ratio inverts.
The measurement was right for the regime it was taken in.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Final, TypeAlias

import duckdb
import pandas as pd

from creditsurv.data.ingest import completed_files

_LOGGER: Final = logging.getLogger(__name__)

#: What the readers accept: nothing (use the manifest), one path, or many.
PathSpec: TypeAlias = str | Path | Sequence[str] | None

#: Delinquency at which a loan counts as defaulted: three missed payments.
DEFAULT_DELINQUENCY: Final = 3

#: Zero-balance codes that end a loan through credit loss rather than repayment.
#:
#: ``16`` is a reperforming loan sale -- a loan that went bad, was cured, and was then
#: sold. It is creditworthy information but not a loss at the point of sale, and by the
#: time it appears the 90+ event has almost always already fired; it is classified as
#: repayment rather than left to fall through. ``96`` is an administrative removal.
#: Both were previously unclassified and so treated as ordinary censoring, which
#: violated this module's own rule that every CASE lists its branches.
DEFAULT_ZERO_BALANCE: Final = ("02", "03", "09", "15")
PREPAYMENT_ZERO_BALANCE: Final = ("01", "16", "96")


class MoratoriumPolicy(StrEnum):
    """What to do with a delinquency the borrower was not required to cure.

    The event definition is "90+ days delinquent, or a loss zero-balance code", and
    for two years that was not a statement about credit. **The CARES Act required
    loans in forbearance to be reported as delinquent**, so a payment holiday granted
    by statute reads identically to a borrower who has stopped paying. Natural-disaster
    forbearance does the same on a smaller scale.

    The scale is not marginal. On 2019Q3, **87% of all 90+ rows carry an accommodation
    marker** and only 13% are clean credit delinquency. Across the whole book it is 17%
    of events, peaking at 90.4 bp a month in May 2020 against a 2019 baseline of 3.07 --
    a factor of 29, where the 2008 crisis managed 24.5 bp. Of the loans first reaching
    90+ in 2020, **99.5% returned to performing**.

    Two defensible treatments, and they are not equivalent, so both are available and
    the choice is settled by measuring the difference rather than by argument:

    ``EXCLUDE``
        An accommodated month is **not an event**, and the loan stays under
        observation. Keeps the exposure at risk, and a loan that later defaults for
        real -- without a marker -- is still caught. Assumes the accommodation itself
        carries no information about credit risk.

    ``CENSOR``
        Observation **ends** at the accommodation, as it does at a prepayment or a
        modification. Assumes nothing about why it was granted, and pays for that by
        losing the subsequent exposure and any genuine default that follows.

    ``IGNORE``
        The behaviour before this was found: an accommodation is a default. Kept only
        so the contaminated model can be reproduced for comparison.
    """

    EXCLUDE = "exclude"
    CENSOR = "censor"
    IGNORE = "ignore"


#: The markers that say a delinquency was not a failure to pay.
#:
#: Read from the data rather than the layout: on 2019Q3 the fields take
#: ``delinquency_due_to_disaster`` Y, ``borrower_assistance_plan`` F/T/R and
#: ``payment_deferral_flag`` P/C.
#:
#: ``F`` is forbearance, a payment holiday. ``P`` and ``C`` are deferrals, where the
#: missed payments move to the end of the loan. Those are accommodations.
#:
#: ``T`` and ``R`` are **not** on this list and the distinction matters: a trial period
#: plan and a repayment plan are loss mitigation, which *follows* genuine distress
#: rather than substituting for it. Treating them as accommodations would excuse real
#: defaults.
MORATORIUM_MARKERS: Final[str] = (
    "COALESCE(delinquency_due_to_disaster = 'Y', FALSE) "
    "OR COALESCE(borrower_assistance_plan = 'F', FALSE) "
    "OR COALESCE(payment_deferral_flag IN ('P', 'C'), FALSE)"
)

#: Modification flags. ``Y`` is the month the loan was modified, ``P`` every month
#: after it -- a prior modification.
#:
#: Modification ends observation of the contract, the way prepayment does. It is not
#: a nicety: **the dataset restarts ``loan_age`` at the modification**, because the
#: field counts scheduled payments since the loan was originated *or modified*. One
#: loan in the 2006 vintage runs to age 192 at twenty months delinquent, is modified,
#: and reappears at age 3 with a clean delinquency status, then climbs again.
#:
#: Left alone that does three things, none of them visible in a coefficient:
#:
#: * the same loan contributes two episodes at the same age, double-counting its
#:   likelihood contribution and breaking one-event-per-loan;
#: * seasoned, previously-distressed months are re-attributed to young ages, where
#:   they arrive *performing* -- so they dilute exactly the part of the hazard curve
#:   the model is most sensitive to;
#: * the affected population is not random. It is 0.4% of the 1999 vintage's loans,
#:   5.2% of 2006's and 1.9% of 2021's, and every one of them is a loan that got
#:   into trouble -- which is where nearly all the events are.
MODIFICATION_FLAGS: Final = ("Y", "P")

#: Months per episode. Loan age is collapsed to multiples of this, so an episode
#: spans ``(k*step, (k+1)*step]``.
#:
#: **Set by how often the covariates move, not by how much it compresses.** The
#: time-varying covariates come from monthly series -- unemployment, the house price
#: index, financial conditions -- so an episode that spans more than a month asks the
#: model to hold constant something the data says changed. Monthly is what the data
#: supports, so monthly is what this is.
#:
#: Measured on 1999Q1, 27.7 million loan-months, with the current specification:
#:
#: ===============  ==========  =============  ==================
#: Episode          Cells       Compression    Whole dataset
#: ===============  ==========  =============  ==================
#: **Monthly**        226,229           122x               ~24 M
#: Quarterly            79,930           347x              ~8.6 M
#: Half-yearly          42,343           654x              ~4.5 M
#: ===============  ==========  =============  ==================
#:
#: Two earlier versions got this wrong in the same way, by choosing on compression
#: rather than on the data. The first used bands widening to four years because they
#: compressed 2,600x -- which asks a model to treat unemployment as constant across
#: a presidency. The second rested on a measurement taken while a monthly-varying
#: covariate was still in the grouping key, which made quarterly episodes look no
#: better than monthly: nothing can collapse on age while a covariate moves
#: underneath it. With that covariate derived instead of carried, the comparison is
#: honest, and monthly costs a factor of three against quarterly for a panel that is
#: tractable either way.
EPISODE_MONTHS: Final = 1

_STATE_TO_REGION: Final[dict[str, str]] = {}
for _region, _states in {
    "Northeast": ("CT", "ME", "MA", "NH", "NJ", "NY", "PA", "RI", "VT"),
    "Midwest": ("IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"),
    "South": (
        "AL",
        "AR",
        "DE",
        "DC",
        "FL",
        "GA",
        "KY",
        "LA",
        "MD",
        "MS",
        "NC",
        "OK",
        "SC",
        "TN",
        "TX",
        "VA",
        "WV",
    ),
    "West": ("AK", "AZ", "CA", "CO", "HI", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY"),
}.items():
    for _state in _states:
        _STATE_TO_REGION[_state] = _region


def _case_expression(column: str, edges: Sequence[float], alias: str) -> str:
    """Render a coarse-classing rule as SQL.

    The band's midpoint is used as its value, so a binned covariate keeps the scale
    of the one it replaces and its coefficient stays comparable with an unbinned fit.
    """
    clauses = []
    for lower, upper in pairwise(edges):
        midpoint = (lower + upper) / 2.0
        clauses.append(f"WHEN {column} <= {upper} THEN {midpoint}")
    outer = (edges[0] + edges[1]) / 2.0
    last = (edges[-2] + edges[-1]) / 2.0
    return (
        f"CASE WHEN {column} IS NULL THEN NULL "
        f"WHEN {column} <= {edges[0]} THEN {outer} "
        + " ".join(clauses)
        + f" ELSE {last} END AS {alias}"
    )


def _region_case() -> str:
    whens = " ".join(
        f"WHEN property_state = '{state}' THEN '{region}'"
        for state, region in _STATE_TO_REGION.items()
    )
    return f"CASE {whens} ELSE 'Other' END AS region"


def _state_of_the_book_sql(
    policy: MoratoriumPolicy = MoratoriumPolicy.EXCLUDE, *, complete_only: bool = True
) -> str:
    """The loan-month panel, cleaned and truncated, before any aggregation.

    ``complete_only`` drops loans missing a credit score, loan-to-value or debt-to-income,
    as the cells do. Off, they are kept, so :func:`incomplete_cases` can say what is lost.
    """
    complete = (
        "WHERE o.credit_score IS NOT NULL AND o.orig_ltv IS NOT NULL AND o.dti IS NOT NULL"
        if complete_only
        else ""
    )
    default_codes = ", ".join(f"'{code}'" for code in DEFAULT_ZERO_BALANCE)
    prepayment_codes = ", ".join(f"'{code}'" for code in PREPAYMENT_ZERO_BALANCE)
    modification_flags = ", ".join(f"'{flag}'" for flag in MODIFICATION_FLAGS)
    # Under EXCLUDE an accommodated month is not an event but the loan stays at risk;
    # under CENSOR it ends observation the way a modification does.
    accommodated = "FALSE" if policy is MoratoriumPolicy.IGNORE else f"({MORATORIUM_MARKERS})"
    excluded = accommodated if policy is MoratoriumPolicy.EXCLUDE else "FALSE"
    censoring = accommodated if policy is MoratoriumPolicy.CENSOR else "FALSE"
    return f"""
    WITH perf AS (
        SELECT
            loan_identifier,
            CAST(loan_age AS INTEGER)                                   AS age,
            CAST(period AS INTEGER)                                     AS period_key,
            -- 999 marks "not available" here exactly as it does for LTV and DTI in
            -- the origination file. Untreated it is an ordinary number: the median
            -- ELTV of the 2006 vintage is literally 999.
            NULLIF(TRY_CAST(estimated_loan_to_value AS DOUBLE), 999)     AS eltv,
            -- COALESCE wraps the whole expression, not just the cast. Most rows
            -- have no zero-balance code, so `IN (...)` is NULL there, and
            -- `FALSE OR NULL` is NULL rather than FALSE -- which propagates a
            -- nullable event flag all the way to the fitter. The fixtures did not
            -- catch it because they write an empty string where the real files
            -- leave the field absent.
            COALESCE(
                (
                    COALESCE(TRY_CAST(current_loan_delinquency_status AS INTEGER)
                             >= {DEFAULT_DELINQUENCY}, FALSE)
                    OR COALESCE(zero_balance_code IN ({default_codes}), FALSE)
                )
                -- A delinquency the borrower was not required to cure is not a
                -- failure to pay. See MoratoriumPolicy.
                AND NOT ({excluded}),
                FALSE
            )                                                           AS defaulted,
            COALESCE(zero_balance_code IN ({prepayment_codes}), FALSE)   AS prepaid,
            COALESCE(modification_flag IN ({modification_flags}), FALSE)
                OR ({censoring})                                    AS ends_observation
        FROM read_parquet(?)
        WHERE TRY_CAST(loan_age AS INTEGER) >= 0
    ),
    -- Servicing files keep reporting through foreclosure and loss settlement, so a
    -- defaulted loan carries several flagged rows. Cutting at the first terminating
    -- month is what keeps one event per loan.
    --
    -- Ordered by **calendar period**, not by age. Age is not monotone within a loan:
    -- a modification restarts it, so MIN(age) over the terminating rows can land on a
    -- post-modification row and the cut then keeps an arbitrary mixture of months
    -- from before and after. Calendar time is monotone by construction.
    terminal AS (
        SELECT
            loan_identifier,
            MIN(CASE WHEN defaulted OR prepaid THEN period_key END) AS terminal_period,
            MIN(CASE WHEN ends_observation THEN period_key END)     AS ended_period
        FROM perf GROUP BY loan_identifier
    ),
    -- Default and prepayment happen *during* their month, so that month is kept and
    -- carries the flag. The two censoring conditions are different and end observation
    -- the month *before*: a modification's flagged row already reports the restarted
    -- age, so keeping it would file a distressed month at age zero, and an
    -- accommodation's flagged month is the one the delinquency counter is already
    -- misreporting.
    truncated AS (
        SELECT p.*, t.terminal_period
        FROM perf p LEFT JOIN terminal t USING (loan_identifier)
        WHERE (t.terminal_period IS NULL OR p.period_key <= t.terminal_period)
          AND (t.ended_period IS NULL OR p.period_key < t.ended_period)
    ),
    orig AS (
        SELECT
            loan_identifier,
            -- 9999 and 999 are the dataset's own "not available" markers. They are
            -- ordinary numbers, and left in place they produce a portfolio whose
            -- average credit score is several thousand.
            NULLIF(TRY_CAST(classic_fico AS DOUBLE), 9999)              AS credit_score,
            NULLIF(TRY_CAST(original_ltv AS DOUBLE), 999)               AS orig_ltv,
            NULLIF(TRY_CAST(original_cltv AS DOUBLE), 999)              AS orig_cltv,
            NULLIF(TRY_CAST(original_dti AS DOUBLE), 999)               AS dti,
            TRY_CAST(original_upb AS DOUBLE)                            AS orig_upb,
            TRY_CAST(original_interest_rate AS DOUBLE)                  AS note_rate,
            TRY_CAST(original_loan_term AS INTEGER)                     AS orig_term,
            -- 999 is "not available" for MI exactly as for LTV and DTI. Untreated, has_mi
            -- read a missing percentage as an insured loan.
            NULLIF(TRY_CAST(mortgage_insurance_percentage AS DOUBLE), 999) AS mi_percent,
            -- Raw codes, deliberately. The mapping lives in _CATEGORICAL, where the
            -- choices are documented next to the frequencies that justify them, and
            -- an ELSE branch here would silently fold a "not available" code into a
            -- real level before anyone could see it.
            loan_purpose,
            occupancy_status,
            channel,
            first_time_homebuyer_indicator,
            super_conforming_flag,
            property_type,
            number_of_units,
            number_of_borrowers,
            {_region_case()}
        FROM read_parquet(?)
    )
    SELECT
        t.age,
        t.period_key,
        t.eltv,
        COALESCE(t.defaulted AND t.period_key = t.terminal_period, FALSE) AS event,
        COALESCE(t.prepaid AND t.period_key = t.terminal_period, FALSE)  AS prepaid,
        o.*
    FROM truncated t JOIN orig o USING (loan_identifier)
    {complete}
    """


#: Source expression for each continuous covariate, keyed by the name it takes.
_SOURCE: Final[dict[str, str]] = {
    "fico_s": "(credit_score - 700.0) / 50.0",
    "orig_ltv": "orig_ltv",
    "orig_cltv": "orig_cltv",
    "dti": "dti",
    "log_orig_upb": "ln(orig_upb)",
    # Freddie's own mark-to-market valuation. Available as a covariate, but not in
    # the default specification: coverage runs from 0.8% of the 1999 vintage to 94%
    # of 2021, so a model using it would be estimating a different quantity in every
    # decade. The house-price-indexed drift computed in cells_to_episodes covers
    # every vintage evenly instead.
    "eltv_drift": "COALESCE(eltv, orig_ltv) - orig_ltv",
    "mi_percent": "mi_percent",
}

#: Categorical covariates and the SQL that produces them.
#:
#: Every mapping here was decided from a distinct-and-count over the raw values across
#: seven vintages spanning 1999 to 2024, not from the file layout. Four mistakes came
#: out of doing it that way round rather than assuming:
#:
#: * ``9`` and ``99`` are "not available" codes, not categories. Folding them into a
#:   real level -- which an ``ELSE`` branch does silently -- invents data.
#: * ``channel`` cannot be used at four levels at all. Until 2008 about half of
#:   originations are coded ``T`` and broker and correspondent are near zero; from
#:   2009 ``T`` vanishes and those two absorb it. That is a coding change, not a
#:   market one, and a model given the four levels reads it as a risk effect. It is
#:   collapsed to retail against third-party, which is stable across the history.
#: * ``amortization_type`` and ``interest_only_indicator`` each take exactly **one**
#:   value across the whole dataset. Dropped rather than modelled.
#: * ``property_type`` and ``number_of_units`` have long tails below 5%, merged into an
#:   explicit "other" rather than left as levels with nothing to estimate from.
#:
#: Every branch is explicit and there is no ``ELSE``: an unmapped code becomes NULL and
#: the loan is dropped, which is the honest outcome for a value nobody has looked at.
#: See docs/variable_selection.md for the frequencies these rest on.
_CATEGORICAL: Final[dict[str, str]] = {
    "purpose": (
        "CASE loan_purpose WHEN 'P' THEN 'purchase' WHEN 'C' THEN 'refinance_cashout' "
        "WHEN 'N' THEN 'refinance_rate_term' WHEN 'R' THEN 'refinance_rate_term' END"
    ),
    "occupancy": (
        "CASE occupancy_status WHEN 'P' THEN 'owner_occupied' "
        "WHEN 'S' THEN 'second_home' WHEN 'I' THEN 'investor' END"
    ),
    # Retail against everything else, because the finer split is not comparable
    # across the history. Until 2008 roughly half of originations are coded T,
    # third-party not otherwise specified, and broker and correspondent are near
    # zero; from 2009 T vanishes and those two absorb it. That is a change in how
    # Freddie Mac coded the field, not a change in how loans were sold, and a model
    # given the four levels would read the coding change as a risk effect. Retail's
    # own share is stable throughout -- 53.8% in 1999, 57.9% in 2021 -- so the binary
    # split is the part that means the same thing in every vintage.
    "channel": (
        "CASE channel WHEN 'R' THEN 'retail' "
        "WHEN 'B' THEN 'third_party' WHEN 'C' THEN 'third_party' "
        "WHEN 'T' THEN 'third_party' END"
    ),
    "region": "region",
    # In the cell key, so a 9 -- "not available" -- drops the loan. Measured across the
    # whole book that is 19,053 of 49.2 million loans, 0.04%, and at most 0.92% of any
    # vintage (1999): too small for the drop to be the informative loss D4 is about.
    "first_time_buyer": (
        "CASE first_time_homebuyer_indicator WHEN 'Y' THEN 'Y' WHEN 'N' THEN 'N' END"
    ),
    # SF, PU and CO carry 99.3% between them; the rest is a tail of half-percents.
    "property_type": (
        "CASE property_type WHEN 'SF' THEN 'single_family' WHEN 'PU' THEN 'planned_unit' "
        "WHEN 'CO' THEN 'condo' WHEN 'MH' THEN 'other' WHEN 'CP' THEN 'other' END"
    ),
    # 98.1% are single-unit; two, three and four are one category together.
    "units": (
        "CASE WHEN TRY_CAST(number_of_units AS INTEGER) = 1 THEN '1' "
        "WHEN TRY_CAST(number_of_units AS INTEGER) BETWEEN 2 AND 4 THEN '2-4' END"
    ),
    # Both branches explicit. With an ELSE, a term that failed to parse became thirty years.
    "term_years": "CASE WHEN orig_term <= 190 THEN 15 WHEN orig_term > 190 THEN 30 END",
    # Both branches explicit, reading a mi_percent the 999 sentinel has been removed from.
    # With an ELSE and no sentinel treatment, a percentage recorded as "not available"
    # read as *insured*: 735 loans, negligible in number, and precisely the two rules
    # this module states -- sentinels are real numbers, no ELSE -- broken in the covariate
    # the cardinality argument had just been corrected to admit. Its content is real: the
    # insured share runs from 6.8% of the 2010 vintage to 38.9% of 2023's.
    "has_mi": "CASE WHEN mi_percent > 0 THEN 'Y' WHEN mi_percent = 0 THEN 'N' END",
    # Screened like everything else rather than ingested and forgotten. It reached the
    # parquet without appearing in any screening table or in DEGENERATE_FIELDS, which is
    # the gap that let three performance fields disappear silently.
    #
    # Mapped from the field, not from the layout. The layout calls a blank "not super
    # conforming" and the first mapping turned NULL into N, but across all 49.2 million
    # loans the field holds exactly N (48,223,363) and Y (962,808) and never a blank: that
    # mapping would have dropped 98% of the book the day the flag entered a key. It is not
    # in one. No loan before 2008 is Y, because the category did not exist, so its N for
    # those vintages records a date rather than a loan.
    "super_conforming": "CASE super_conforming_flag WHEN 'Y' THEN 'Y' WHEN 'N' THEN 'N' END",
    "n_borrowers": (
        "CASE WHEN TRY_CAST(number_of_borrowers AS INTEGER) = 1 THEN '1' "
        "WHEN TRY_CAST(number_of_borrowers AS INTEGER) BETWEEN 2 AND 5 THEN '2+' END"
    ),
}

#: Fields taking exactly one value across the whole dataset. Recorded rather than
#: quietly omitted, so the next reader does not spend an afternoon adding them back.
DEGENERATE_FIELDS: Final[tuple[str, ...]] = (
    "amortization_type",  # FRM, 100%
    "interest_only_indicator",  # N, 100%
)


@dataclass(frozen=True)
class CellSpec:
    """Which covariates enter the aggregation, and how coarsely.

    Deliberately a parameter rather than a constant. The cell count is the product of
    every covariate's band count, so the specification *is* the cardinality, and it
    cannot be fixed before variable selection has said which covariates earn their
    place. Aggregating on everything available and selecting afterwards is the wrong
    order: it produces a table too large to fit, which is exactly what happened here
    on the first attempt.

    Measured on 1999Q1 (27.7 million loan-months):

    ==========================================  ==========  ============
    Specification                               Cells       Compression
    ==========================================  ==========  ============
    9 continuous + 9 categorical, monthly ages  12,752,331          2.2x
    same, quarterly ages                        12,752,331          2.2x
    same, banded ages                              919,634         30.1x
    4 coarse continuous + 3 categorical, banded     14,221      1,948.0x
    ==========================================  ==========  ============
    """

    continuous: dict[str, tuple[float, ...]]
    categorical: tuple[str, ...]
    episode_months: int = EPISODE_MONTHS

    def validate(self) -> None:
        unknown = set(self.continuous) - set(_SOURCE)
        if unknown:
            message = f"Unknown continuous covariate(s): {sorted(unknown)}"
            raise ValueError(message)
        unknown = set(self.categorical) - set(_CATEGORICAL)
        if unknown:
            message = f"Unknown categorical covariate(s): {sorted(unknown)}"
            raise ValueError(message)


#: Coarse by design: five bands a covariate, three categoricals, banded ages. Chosen
#: from the measurement above as the point where the whole population fits in roughly
#: a million cells. Variable selection may replace it.
#: The classing actually used to build cells, as a **subset of** ``BIN_EDGES``.
#:
#: Two schemes used to coexist and only one was in the production path. The
#: documentation justified a DTI break at 43, "a long-standing underwriting threshold",
#: while the model used 45; it named LTV breaks at 85 and 95 that the model did not
#: have. A reader checking the economic justification of the bands found a
#: justification that did not describe the model.
#:
#: They are reconciled by making this grid a strict subset of the documented one -- a
#: test enforces it -- so every boundary that exists is one the documentation argues
#: for, and the coarsening is the only thing left to explain.
#:
#: **Why coarser rather than unified on the full set.** The cell count is the product
#: of the band counts. BIN_EDGES gives 8 x 8 x 6 = 384 combinations against 80 here, a
#: 4.8x multiplier, and it would land on top of the 3.11x the exact calendar key costs
#: and the 1.23x of the two loan covariates -- seventeen times the table. The
#: thresholds dropped are the finer ones; the MI break at 80 and the underwriting
#: break at 43 survive, and those are the two the economics actually turns on.
PRODUCTION_EDGES: Final[dict[str, tuple[float, ...]]] = {
    "fico_s": (-2.4, -0.8, 0.0, 0.8, 1.2, 2.4),
    "orig_ltv": (30.0, 70.0, 80.0, 90.0, 100.0),
    "dti": (10.0, 28.0, 36.0, 43.0, 55.0),
}

DEFAULT_SPEC: Final = CellSpec(
    continuous=PRODUCTION_EDGES,
    # cltv_drift is absent on purpose: it is a function of orig_ltv and the macro
    # path, both recoverable from the key, so carrying it would multiply the
    # cardinality for information already there.
    #
    # has_mi and first_time_buyer are present because the argument that excluded them
    # was wrong. It asserted a cost of "up to 16x the table" from the product of the
    # level counts; the cell space is sparse and the measured cost of the two together
    # is **1.23x**. Mortgage insurance is a classic credit predictor and already had an
    # expected sign in the code. The claim that the loan side could not afford more
    # covariates did not survive being measured.
    categorical=(
        "purpose",
        "occupancy",
        "term_years",
        "has_mi",
        "first_time_buyer",
    ),
)


def _age_expression(step: int) -> str:
    """Loan age collapsed to the start of its episode, in months.

    The lower edge rather than an index, so the value keeps the units of loan age and
    the episode bounds read straight off it.
    """
    return f"CAST(age / {step} AS INTEGER) * {step} AS age"


def _not_null_filter(spec: CellSpec) -> str:
    """WHERE clause dropping rows whose categorical mapping came back NULL."""
    if not spec.categorical:
        return ""
    conditions = " AND ".join(f"{name} IS NOT NULL" for name in spec.categorical)
    return f"WHERE {conditions}"


#: The loan's origination month, as an ordinal in months since year zero.
#:
#: This replaced the vintage *quarter*, and the difference is two months of calendar
#: on every macro covariate in the model. Loans in a quarter are not all written in its
#: first month -- the mean offset is **+2.15 months** -- so reconstructing the month
#: from the quarter reads every macro series that much late, and pushes the backtest
#: boundary two months inside the training half where ``assert_no_lookahead`` cannot
#: see it, because it checks the distorted quantity.
#:
#: Verified by cross-correlating the true monthly default series against the
#: reconstructed one: the maximum sat at a lag of **+2**, not 0.
#:
#: It costs 3.11x the cells, measured. The convention matches ``_months_to_periods``:
#: ``year * 12 + (month - 1)``.
_ORIGINATION_MONTH: Final = "(period_key // 100) * 12 + (period_key % 100) - 1 - age AS orig_month"


def _select_columns(spec: CellSpec) -> str:
    """Every covariate column of the SELECT, as SQL.

    Assembled from a list rather than interpolated as separate blocks: an empty
    continuous or categorical set would otherwise leave a dangling comma and fail
    with a parser error that says nothing about the specification that caused it.
    """
    columns = [
        _case_expression(_SOURCE[name], edges, name) for name, edges in spec.continuous.items()
    ]
    columns += [f"{_CATEGORICAL[name]} AS {name}" for name in spec.categorical]
    columns.append(_age_expression(spec.episode_months))
    columns.append(_ORIGINATION_MONTH)
    columns.append("event")
    return ",\n            ".join(columns)


def _connect() -> duckdb.DuckDBPyConnection:
    """A connection that will not fill the working tree with spill files.

    DuckDB defaults its temporary directory to the process's working directory, so a
    query that spills leaves gigabytes inside the repository -- 20 GB, the first time
    this ran. Quarters are processed one at a time now and it should not spill at
    all, but the setting is cheap insurance against the next query that does.
    """
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{tempfile.gettempdir()}'")
    con.execute("SET memory_limit = '6GB'")
    return con


def _resolve(spec: PathSpec, kind: str) -> list[str]:
    """Turn a caller's argument into a concrete list of parquet paths."""
    if spec is None:
        return completed_files(kind)
    if isinstance(spec, str | Path):
        return [str(spec)]
    return [str(path) for path in spec]


def _cells_for_quarter(
    connection: duckdb.DuckDBPyConnection,
    perf_path: str,
    orig_path: str,
    vintage: str,
    spec: CellSpec,
    policy: MoratoriumPolicy,
) -> pd.DataFrame:
    """Aggregate one vintage quarter.

    Every loan appears in exactly one quarter's files -- verified, not assumed: the
    identifiers of 1999Q1 and 1999Q2 do not intersect at all. So a quarter can be
    collapsed on its own and the results concatenated, which keeps memory flat.

    The vintage is attached as a constant too, but only as a label: the key carries
    the **origination month**, derived from the data as ``period - age``. With the
    month and the age in the key the observation month follows exactly, which is what
    lets the macro series stay out of the key entirely and be read at the right date.
    """
    query = f"""
    WITH book AS ({_state_of_the_book_sql(policy)}), classed AS (
        SELECT
            {_select_columns(spec)}
        FROM book
    )
    SELECT '{vintage}' AS vintage, *, COUNT(*) AS n
    FROM classed
    -- A categorical that mapped to NULL is a code nobody has looked at. The loan is
    -- dropped rather than aggregated into a NULL level, for the same reason a loan
    -- with no credit score is dropped: it cannot be modelled, and imputing the
    -- category would invent the thing being measured.
    {_not_null_filter(spec)}
    GROUP BY ALL
    """
    frame: pd.DataFrame = connection.execute(query, [perf_path, orig_path]).df()
    return frame


def build_cells(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    spec: CellSpec = DEFAULT_SPEC,
    policy: MoratoriumPolicy = MoratoriumPolicy.EXCLUDE,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Aggregate the parquet panel into weighted cells, one quarter at a time.

    The key is the coarse-classed covariates together with the origination month, the
    loan age and the event flag. The weight is the loan-month count.

    ``policy`` decides what a delinquency the borrower was not required to cure counts
    as. It is a parameter rather than a constant because the two defensible treatments
    are not equivalent and the choice is settled by measuring the difference -- see
    :class:`MoratoriumPolicy`.
    """
    perf = _resolve(perf_source, "perf")
    orig = _resolve(orig_source, "orig")
    if not perf or not orig:
        message = "No ingested quarters found. Run `creditsurv ingest` first."
        raise FileNotFoundError(message)

    spec.validate()
    con = connection or _connect()
    frames = []
    for perf_path, orig_path in zip(sorted(perf), sorted(orig), strict=True):
        vintage = Path(perf_path).stem
        cells = _compact(_cells_for_quarter(con, perf_path, orig_path, vintage, spec, policy))
        frames.append(cells)
        _LOGGER.info("%s: %d cells from %d loan-months", vintage, len(cells), int(cells["n"].sum()))

    # Vintage is in the key and constant within a quarter, so the pieces are already
    # disjoint: concatenating needs no second group-by.
    combined = _concatenate(frames)
    _LOGGER.info("Collapsed to %d cells", len(combined))
    return combined


def _compact(cells: pd.DataFrame) -> pd.DataFrame:
    """One quarter's cells, in the types they should have come back in.

    DuckDB returns text as Python strings, one object per value. On the quarter-keyed
    table three such columns were 44% of the episode frame; the exact key carries five
    over some 66 million cells. So they become categorical as each quarter arrives,
    before a table of strings can exist, and the origination month -- an ordinal near
    24,000 -- is kept as a 32-bit integer.
    """
    for column in cells.columns:
        if cells[column].dtype == object:
            cells[column] = cells[column].astype("category")
    if "orig_month" in cells.columns:
        cells["orig_month"] = cells["orig_month"].astype("int32")
    return cells


def _concatenate(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack the quarters, keeping every categorical column categorical.

    pandas keeps a categorical through ``concat`` only when every piece declares the
    same levels, and otherwise turns the whole column back into strings: one quarter
    without an investor loan would undo :func:`_compact` for the entire table. The
    levels are unified first, and sorted, so they do not depend on reading order.
    """
    if not frames:
        message = "No cells to concatenate."
        raise ValueError(message)
    categorical = [
        column
        for column in frames[0].columns
        if isinstance(frames[0][column].dtype, pd.CategoricalDtype)
    ]
    for column in categorical:
        levels = sorted({level for frame in frames for level in frame[column].cat.categories})
        for frame in frames:
            frame[column] = frame[column].cat.set_categories(levels)
    return pd.concat(frames, ignore_index=True)


def cardinality_report(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    spec: CellSpec = DEFAULT_SPEC,
) -> pd.DataFrame:
    """How far the collapse actually gets, key by key.

    Run before fixing the grain, not after. The plan calls for reducing the vintage
    to a year, or the episode to a quarter, if the cells run past what a fit can
    carry — and that decision needs a number rather than an intuition.
    """
    perf = _resolve(perf_source, "perf")
    orig = _resolve(orig_source, "orig")
    con = _connect()

    counted = con.execute(
        f"SELECT COUNT(*) FROM ({_state_of_the_book_sql()})", [perf, orig]
    ).fetchone()
    rows = int(counted[0]) if counted else 0
    cells = build_cells(perf, orig, spec=spec, connection=con)
    return pd.DataFrame(
        [
            {
                "loan_months": rows,
                "cells": len(cells),
                "compression": rows / max(len(cells), 1),
                "weight_total": int(cells["n"].sum()),
            }
        ]
    )


#: Origination fields whose absence drops a loan before the categorical keys are read.
_COMPLETE_CASE_FIELDS: Final[tuple[str, ...]] = ("credit_score", "orig_ltv", "dti")


def incomplete_cases(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    spec: CellSpec = DEFAULT_SPEC,
    policy: MoratoriumPolicy = MoratoriumPolicy.EXCLUDE,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """The loans the cells leave out, vintage by vintage, and how they default.

    The validation's D4. A loan missing its credit score, loan-to-value or debt-to-income,
    or carrying a categorical code no mapping names, is dropped rather than imputed:
    imputing an underwriting characteristic invents the thing being measured. Dropping is
    harmless only if what goes is small or looks like what stays, and neither can be
    assumed -- the validation found the share varying by two orders of magnitude across
    vintages, almost all of it missing debt-to-income, and the dropped loans riskier.

    One row per vintage: loans, how many are dropped, how many lack each field (a loan can
    lack several), and the ever-default rate of the loans kept and of those dropped, under
    ``policy``'s event definition. A pass over every performance file, quarter by quarter.
    """
    perf = _resolve(perf_source, "perf")
    orig = _resolve(orig_source, "orig")
    if not perf or not orig:
        message = "No ingested quarters found. Run `creditsurv ingest` first."
        raise FileNotFoundError(message)

    missing = {field: f"{field} IS NULL" for field in _COMPLETE_CASE_FIELDS}
    missing.update({name: f"({_CATEGORICAL[name]}) IS NULL" for name in spec.categorical})
    flags = ", ".join(f"BOOL_OR({condition}) AS no_{name}" for name, condition in missing.items())
    dropped = " OR ".join(f"no_{name}" for name in missing)
    counts = ", ".join(f"COUNT(*) FILTER (WHERE no_{name}) AS no_{name}" for name in missing)

    con = connection or _connect()
    frames = []
    for perf_path, orig_path in zip(sorted(perf), sorted(orig), strict=True):
        vintage = Path(perf_path).stem
        query = f"""
        WITH book AS ({_state_of_the_book_sql(policy, complete_only=False)}),
        loans AS (
            SELECT loan_identifier, BOOL_OR(event) AS defaulted, {flags}
            FROM book
            GROUP BY loan_identifier
        ),
        judged AS (SELECT *, ({dropped}) AS dropped FROM loans)
        SELECT
            '{vintage}' AS vintage,
            COUNT(*) AS loans,
            COUNT(*) FILTER (WHERE dropped) AS dropped,
            {counts},
            AVG(CASE WHEN NOT dropped THEN defaulted::INTEGER END) AS default_rate_kept,
            AVG(CASE WHEN dropped THEN defaulted::INTEGER END) AS default_rate_dropped
        FROM judged
        """
        frames.append(con.execute(query, [perf_path, orig_path]).df())
        _LOGGER.info("%s: incomplete cases counted", vintage)

    table = pd.concat(frames, ignore_index=True)
    table["dropped_share"] = table["dropped"] / table["loans"]
    table["relative_risk"] = table["default_rate_dropped"] / table["default_rate_kept"]
    return table


def defaults_by_month(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    spec: CellSpec = DEFAULT_SPEC,
    policy: MoratoriumPolicy = MoratoriumPolicy.EXCLUDE,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.Series:
    """Defaults by calendar month, counted on the loan-months the cells are built from.

    The reference for the validation's M1 test. A cell carries its origination month and
    its age, and the month its defaults happened in is their sum; this counts the same
    defaults without the cells, under the same rules -- the complete-case filter, the
    categorical keys, the moratorium policy -- so the two series have to agree to the unit.
    """
    perf = _resolve(perf_source, "perf")
    orig = _resolve(orig_source, "orig")
    if not perf or not orig:
        message = "No ingested quarters found. Run `creditsurv ingest` first."
        raise FileNotFoundError(message)
    spec.validate()

    con = connection or _connect()
    frames = []
    for perf_path, orig_path in zip(sorted(perf), sorted(orig), strict=True):
        query = f"""
        WITH book AS ({_state_of_the_book_sql(policy)}), classed AS (
            SELECT {_select_columns(spec)}, period_key FROM book
        )
        SELECT
            (period_key // 100) * 12 + (period_key % 100) - 1 AS month,
            SUM(CAST(event AS INTEGER)) AS defaults
        FROM classed
        {_not_null_filter(spec)}
        GROUP BY 1
        """
        frames.append(con.execute(query, [perf_path, orig_path]).df())
    counts = pd.concat(frames, ignore_index=True).groupby("month")["defaults"].sum()
    return counts.astype("int64").rename("defaults")
