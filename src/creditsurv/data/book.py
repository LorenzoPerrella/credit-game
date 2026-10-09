"""The book as SQL, and the connection that reads it.

One view of the Freddie Mac panel -- every loan-month with its covariates coarse-classed, its
event flags and its moratorium treatment -- expressed as the query that produces it, plus the
mappings that query is built from and the DuckDB connection it runs on.

It is a module because four others were reaching into `data.aggregate` for its private names:
the portfolio tables, the covariate profiling, the portfolio views and the CLI each imported
some of `_connect`, `_resolve`, `_CATEGORICAL`, `_SOURCE`, `_case_expression` and
`_state_of_the_book_sql`. The cell table is one consumer of this view among several, and the
aggregation is the only one that was ever allowed to say so.

**The mappings have no `ELSE` branch**, and that is the rule this module exists to keep. An
unmapped code becomes NULL and the loan is dropped, because `9` and `99` are "not available"
codes rather than categories and an `ELSE` folds them into whichever level was written last.
Four mistakes came from doing it the other way. `CATEGORICAL_LEVELS` writes out what each
mapping can produce, so a reader can be given the whole level set before a row arrives.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Final, TypeAlias

import duckdb

from creditsurv.config import CENSORED, DEFAULT_CAUSE, PREPAYMENT_CAUSE
from creditsurv.data.ingest import completed_files

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

#: A voluntary payoff, and the only one of the three that is a prepayment.
#:
#: With prepayment modelled as a competing risk it has to be the borrower's own decision to
#: repay: that is what the refinancing incentive predicts. A reperforming sale (16) and an
#: administrative removal (96) are neither default nor repayment -- they are the loan leaving
#: the dataset -- so they end observation and are censoring, which is what `--report-exits`
#: measured them to be: 90.1% of reperforming sales had already defaulted, and 57.5% of
#: removals were performing when they went.
PREPAYMENT_ZERO_BALANCE: Final = ("01",)
CENSORING_ZERO_BALANCE: Final = ("16", "96")


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


def case_expression(column: str, edges: Sequence[float], alias: str) -> str:
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


def state_of_the_book_sql(
    policy: MoratoriumPolicy = MoratoriumPolicy.EXCLUDE,
    *,
    complete_only: bool = True,
    harp_level: bool = True,
) -> str:
    """The loan-month panel, cleaned and truncated, before any aggregation.

    ``complete_only`` drops loans missing a credit score, loan-to-value or debt-to-income,
    as the cells do. Off, they are kept, so :func:`incomplete_cases` can say what is lost.

    ``harp_level`` says whether the key carries ``harp``, and with it whether a loan may be
    kept without a debt-to-income. The two go together and cannot be chosen separately: the
    ratio is missing for exactly the HARP refinances -- 181,302 of the 181,356 missing in
    2012Q2, measured -- so without the level in the key those loans would collapse into a
    band of NULL that nothing in the table explains. Off, the rule is the one that held
    before September 2026, which is what makes the two comparable when the extension is
    priced.
    """
    # A HARP refinance reports no debt-to-income, and dropping it dropped 18% of the 2009Q2
    # to 2019Q1 vintages, at three times the default rate of the loans kept. It is kept, and
    # the missing ratio stays missing all the way into the cell table: `harp` is a level of
    # the key, so a NULL debt-to-income is a HARP loan and reads as one. Nothing is imputed.
    ratio = (
        "(o.debt_to_income IS NOT NULL OR o.harp_indicator = 'Y')"
        if harp_level
        else "o.debt_to_income IS NOT NULL"
    )
    complete = (
        f"WHERE o.credit_score IS NOT NULL AND o.original_ltv IS NOT NULL AND {ratio}"
        if complete_only
        else ""
    )
    default_codes = ", ".join(f"'{code}'" for code in DEFAULT_ZERO_BALANCE)
    prepayment_codes = ", ".join(f"'{code}'" for code in PREPAYMENT_ZERO_BALANCE)
    exit_codes = ", ".join(f"'{code}'" for code in CENSORING_ZERO_BALANCE)
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
            -- What the borrower's payment history said *before* this month. The state
            -- itself is a mediator -- a loan 90 days late has already defaulted -- but the
            -- month before is what a servicer knows when the month opens, and it is the
            -- strongest thing the file holds that the model never read.
            -- Lagged as the file writes it, not as a number: RA is a real value of this
            -- field (REO acquisition) and casting first would turn it into the same NULL
            -- as "this is the loan's first month", which reads as current.
            LAG(current_loan_delinquency_status)
                OVER (PARTITION BY loan_identifier ORDER BY CAST(period AS INTEGER))
                                                                        AS previous_status,
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
            COALESCE(zero_balance_code IN ({exit_codes}), FALSE)         AS left_the_book,
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
            MIN(CASE WHEN defaulted OR prepaid OR left_the_book THEN period_key END)
                AS terminal_period,
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
            NULLIF(TRY_CAST(original_ltv AS DOUBLE), 999)               AS original_ltv,
            NULLIF(TRY_CAST(original_cltv AS DOUBLE), 999)              AS original_cltv,
            NULLIF(TRY_CAST(original_dti AS DOUBLE), 999)               AS debt_to_income,
            TRY_CAST(original_upb AS DOUBLE)                            AS original_balance,
            TRY_CAST(original_interest_rate AS DOUBLE)                  AS note_rate,
            TRY_CAST(original_loan_term AS INTEGER)                     AS orig_term,
            -- 999 is "not available" for MI exactly as for LTV and DTI. Untreated,
            -- mortgage_insurance read a missing percentage as an insured loan.
            NULLIF(TRY_CAST(mortgage_insurance_percentage AS DOUBLE), 999) AS insurance_coverage,
            -- Raw codes, deliberately. The mapping lives in CATEGORICAL, where the
            -- choices are documented next to the frequencies that justify them, and
            -- an ELSE branch here would silently fold a "not available" code into a
            -- real level before anyone could see it.
            harp_indicator,
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
        COALESCE(t.left_the_book AND t.period_key = t.terminal_period, FALSE)
                                                                         AS left_the_book,
        t.previous_status,
        -- **One spelling of what ended the month.** This used to be three in this one
        -- SELECT -- a boolean `event`, a boolean `prepaid` and this column -- and the two
        -- booleans were not the same fact: `prepaid` is a prepayment code in the terminal
        -- month and takes no view of a default in it, where `outcome` gives default
        -- precedence. Measured over the whole book, 52,357 loan-months are both: 3.07% of
        -- the 1,704,432 defaults and 0.15% of the 34,316,184 prepayments, counted once as a
        -- default by every model and once again as a prepayment by the site's CPR.
        --
        -- The precedence here is the one rule 1 states: at three missed payments the loan has
        -- defaulted by definition, so a payoff after that is a recovery and not a voluntary
        -- prepayment. The cells carry this column and both hazards are fitted on it, so it is
        -- the definition; a caller that wants a boolean writes `outcome = 'default'`.
        CASE
            WHEN COALESCE(t.defaulted AND t.period_key = t.terminal_period, FALSE)
                THEN '{DEFAULT_CAUSE}'
            WHEN COALESCE(t.prepaid AND t.period_key = t.terminal_period, FALSE)
                THEN '{PREPAYMENT_CAUSE}'
            ELSE '{CENSORED}'
        END                                                              AS outcome,
        o.*
    FROM truncated t JOIN orig o USING (loan_identifier)
    {complete}
    """


#: Source expression for each continuous covariate, keyed by the name it takes.
SOURCE: Final[dict[str, str]] = {
    "credit_score": "credit_score",
    "original_ltv": "original_ltv",
    "original_cltv": "original_cltv",
    "debt_to_income": "debt_to_income",
    "log_original_balance": "ln(original_balance)",
    # Freddie's own mark-to-market valuation. Available as a covariate, but not in
    # the default specification: coverage runs from 0.8% of the 1999 vintage to 94%
    # of 2021, so a model using it would be estimating a different quantity in every
    # decade. The house-price-indexed drift computed in cells_to_episodes covers
    # every vintage evenly instead.
    "estimated_ltv_change": "COALESCE(eltv, original_ltv) - original_ltv",
    "insurance_coverage": "insurance_coverage",
    # In the key so that the refinancing incentive and the spread at origination can be
    # rebuilt after the collapse: both are the note rate against a mortgage rate, and the
    # mortgage rate is a function of the origination month, which the key already carries.
    # It is the cheapest way to reach the covariate a prepayment model turns on.
    "note_rate": "note_rate",
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
CATEGORICAL: Final[dict[str, str]] = {
    "purpose": (
        "CASE loan_purpose WHEN 'P' THEN 'purchase' WHEN 'C' THEN 'cash_out_refinance' "
        "WHEN 'N' THEN 'rate_term_refinance' WHEN 'R' THEN 'rate_term_refinance' END"
    ),
    "occupancy": (
        "CASE occupancy_status WHEN 'P' THEN 'owner_occupied' "
        "WHEN 'S' THEN 'second_home' WHEN 'I' THEN 'investment_property' END"
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
        "WHEN 'B' THEN 'broker_or_correspondent' WHEN 'C' THEN 'broker_or_correspondent' "
        "WHEN 'T' THEN 'broker_or_correspondent' END"
    ),
    "region": "region",
    # Both branches explicit, as everywhere here: a blank is neither, and drops the loan.
    # Before September 2026 the flag was not even ingested, and every HARP loan fell out
    # of the model through its missing debt-to-income.
    "harp": "CASE harp_indicator WHEN 'Y' THEN 'harp' WHEN 'N' THEN 'standard' END",
    # What the payment history said when the month opened. The state *in* the month is a
    # mediator -- at three missed payments the loan has defaulted by definition -- but the
    # month before is what a servicer knows in time to act on, and it is the strongest thing
    # the performance file holds that this model has never read.
    #
    # No earlier month means the loan's first observed month, which opens current: that is
    # the origination month, not a missing value. Everything else is read from the code, and
    # a code that is neither a number nor absent -- RA, an REO acquisition -- has no branch
    # and drops the row, as every other mapping here does.
    #
    # ``three_or_more`` exists only under MoratoriumPolicy.EXCLUDE, where an accommodated
    # 90+ month is not an event and the loan stays under observation: the month after it
    # opens at three. Under IGNORE the loan has already defaulted and no such row survives.
    "delinquency_state": (
        "CASE WHEN previous_status IS NULL THEN 'current' "
        "WHEN TRY_CAST(previous_status AS INTEGER) = 0 THEN 'current' "
        "WHEN TRY_CAST(previous_status AS INTEGER) = 1 THEN 'one_month' "
        "WHEN TRY_CAST(previous_status AS INTEGER) = 2 THEN 'two_months' "
        f"WHEN TRY_CAST(previous_status AS INTEGER) >= {DEFAULT_DELINQUENCY} "
        "THEN 'three_or_more' END"
    ),
    # In the cell key, so a 9 -- "not available" -- drops the loan. Measured across the
    # whole book that is 19,053 of 49.2 million loans, 0.04%, and at most 0.92% of any
    # vintage (1999): too small for the drop to be the informative loss D4 is about.
    "buyer_type": (
        "CASE first_time_homebuyer_indicator WHEN 'Y' THEN 'first_time' WHEN 'N' THEN 'repeat' END"
    ),
    # SF, PU and CO carry 99.3% between them; the rest is a tail of half-percents.
    "property_type": (
        "CASE property_type WHEN 'SF' THEN 'single_family' "
        "WHEN 'PU' THEN 'planned_unit_development' WHEN 'CO' THEN 'condominium' "
        "WHEN 'MH' THEN 'manufactured_or_coop' WHEN 'CP' THEN 'manufactured_or_coop' END"
    ),
    # 98.1% are single-unit; two, three and four are one category together.
    "units": (
        "CASE WHEN TRY_CAST(number_of_units AS INTEGER) = 1 THEN 'one_unit' "
        "WHEN TRY_CAST(number_of_units AS INTEGER) BETWEEN 2 AND 4 THEN 'two_to_four_units' END"
    ),
    # Both branches explicit. With an ELSE, a term that failed to parse became thirty years.
    "term_years": "CASE WHEN orig_term <= 190 THEN 15 WHEN orig_term > 190 THEN 30 END",
    # Both branches explicit, reading an insurance coverage the 999 sentinel has been removed from.
    # With an ELSE and no sentinel treatment, a percentage recorded as "not available" read as
    # *insured*: 735 loans, negligible in number, and precisely the two rules this module states --
    # sentinels are real numbers, no ELSE -- broken in the covariate the cardinality argument had
    # just been corrected to admit. Its content is real: the insured share runs from 6.8% of the
    # 2010 vintage to 38.9% of 2023's.
    "mortgage_insurance": (
        "CASE WHEN insurance_coverage > 0 THEN 'insured' "
        "WHEN insurance_coverage = 0 THEN 'uninsured' END"
    ),
    # Screened like everything else rather than ingested and forgotten. It reached the
    # parquet without appearing in any screening table or in `aggregate.DEGENERATE_FIELDS`,
    # which is the gap that let three performance fields disappear silently. That register is
    # now held to its consequence by a test: a field taking one value is in no key and in no
    # formula (`tests/test_aggregate.py`).
    #
    # Mapped from the field, not from the layout. The layout calls a blank "not super
    # conforming" and the first mapping turned NULL into N, but across all 49.2 million
    # loans the field holds exactly N (48,223,363) and Y (962,808) and never a blank: that
    # mapping would have dropped 98% of the book the day the flag entered a key. It is not
    # in one. No loan before 2008 is Y, because the category did not exist, so its N for
    # those vintages records a date rather than a loan.
    "loan_size": (
        "CASE super_conforming_flag WHEN 'Y' THEN 'super_conforming' WHEN 'N' THEN 'conforming' END"
    ),
    "borrower_count": (
        "CASE WHEN TRY_CAST(number_of_borrowers AS INTEGER) = 1 THEN 'one' "
        "WHEN TRY_CAST(number_of_borrowers AS INTEGER) BETWEEN 2 AND 5 THEN 'two_or_more' END"
    ),
}
#: Every level a mapped categorical can take, **declared** rather than discovered.
#:
#: They are string literals inside the `CASE` expressions of `CATEGORICAL`, which carry no
#: `ELSE` branch because an unmapped code has to drop the row. Writing them out is what lets the
#: cell table be **written one quarter at a time**: a quarter can be given the whole level set as
#: it is aggregated, where before the levels were collected from the quarters and unified
#: afterwards -- which needs every quarter resident at once, and cost 11.3 GB of peak at 91.6
#: million cells.
#:
#: Sorted, and sorted for a reason: the unification it replaces sorted too, so the dictionary
#: encoding parquet writes is the one it wrote before and the table is unchanged byte for byte.
#:
#: `term_years` is absent because it is not text -- the mapping returns 15 or 30 and the column is
#: an `int32`. `tests/test_aggregate.py` reads the literals back out of the SQL and holds them to
#: this in both directions, so a mapping cannot gain a level without this gaining it too.
CATEGORICAL_LEVELS: Final[dict[str, tuple[str, ...]]] = {
    "borrower_count": ("one", "two_or_more"),
    "buyer_type": ("first_time", "repeat"),
    "channel": ("broker_or_correspondent", "retail"),
    "delinquency_state": ("current", "one_month", "three_or_more", "two_months"),
    "harp": ("harp", "standard"),
    "loan_size": ("conforming", "super_conforming"),
    "mortgage_insurance": ("insured", "uninsured"),
    "occupancy": ("investment_property", "owner_occupied", "second_home"),
    "property_type": (
        "condominium",
        "manufactured_or_coop",
        "planned_unit_development",
        "single_family",
    ),
    "purpose": ("cash_out_refinance", "purchase", "rate_term_refinance"),
    "region": ("Midwest", "Northeast", "South", "West"),
    "units": ("one_unit", "two_to_four_units"),
}


def connect() -> duckdb.DuckDBPyConnection:
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


def sources(spec: PathSpec, kind: str) -> list[str]:
    """Turn a caller's argument into a concrete list of parquet paths.

    ``None`` means every quarter the manifest records as complete, which is what every command
    that reads the whole book passes. This import used to be taken inside the function, because
    `data.ingest` reads the record layout from `data.freddiemac` and that module read the event
    definition from this one -- a cycle, broken by hand. The loader that closed it now lives in
    `tests/freddiemac_sample.py`, and there is no knot left to break.
    """
    if spec is None:
        return completed_files(kind)
    if isinstance(spec, str | Path):
        return [str(spec)]
    return [str(path) for path in spec]
