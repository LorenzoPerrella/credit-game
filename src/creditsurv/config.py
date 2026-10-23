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
#: origination). ``refinance_incentive`` is absent because the cell key cannot carry the note
#: rate at this cell count: `docs/reports/key_extensions.csv` priced it at 2.13x and the give-up
#: order of rule 7 reached it.
#:
#: **The output of ``creditsurv select``, not a choice made here.** Steps 5 to 10 on the training
#: half of the rebuilt cells -- 72.7 million, up to 2021-12 -- kept six of the fifteen macro
#: candidates, and ``tests/test_procedure.py`` fails if this tuple and
#: ``docs/reports/selection.json`` part.
#:
#: What changed from the run before it, and why, is in ``ELIMINATED``: the new **step 10** removed
#: ``financial_conditions`` at -0.0155 per standard deviation and ``inflation_change`` at +0.0032,
#: both under the 0.02 of rule 3 -- which named financial conditions in advance. Step 9 removed
#: ``mortgage_rate_decline``, whose sign flips between even and odd vintages beside
#: ``policy_rate_change``, and ``yield_curve_slope`` survived in its place.
TIME_VARYING_CONTINUOUS: Final[tuple[str, ...]] = (
    # Rule 1's regime indicator, protected rather than selected: see PROTECTED_MACRO.
    "moratorium_regime",
    "ltv_change",
    "unemployment_change",
    "financial_conditions",
    "consumer_sentiment",
)

#: The macro block of the **prepayment** model, as `creditsurv select --cause prepayment`
#: chose it -- `docs/reports/selection_weibull_prepayment.json`.
#:
#: Only the macro block, and for one consumer: a stress scenario has to move every covariate a
#: published hazard reads, and the lifetime PD now chains two of them. Everything else about the
#: prepayment model is read from the fit's own formula, so copying it here would be duplication
#: with nothing to keep it honest; this much is tied to the record by a test.
#:
#: What it says about a refinancing is what one would hope: `mortgage_rate_decline` at -0.198 per
#: standard deviation and `equity_volatility` at -0.215 are its largest effects, while the
#: general-cycle covariates -- inflation, its change, volatility change, sentiment -- were
#: eliminated for turning their signs.
PREPAYMENT_TIME_VARYING_CONTINUOUS: Final[tuple[str, ...]] = (
    "unemployment_change",
    "financial_conditions",
    "mortgage_rate_decline",
    "house_price_growth",
    "policy_rate_change",
    "equity_return",
    "equity_volatility",
    "housing_starts_growth",
)

#: Every macro covariate a published hazard reads, in one tuple: what a stress scenario has to
#: move, and nothing it moves may fall outside.
#:
#: The validation's finding was a scenario that shocked four series of which two fed no covariate,
#: while two covariates had no shocked path -- the published multiplier understated the model's own
#: sensitivity and nothing failed. With a competing risk there are two ways to make that mistake
#: now, so the set is derived rather than written: `tests/test_scenarios.py` holds the adverse path
#: to it in both directions.
STRESSED_COVARIATES: Final[tuple[str, ...]] = tuple(
    dict.fromkeys((*TIME_VARYING_CONTINUOUS, *PREPAYMENT_TIME_VARYING_CONTINUOUS))
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
    "corporate_bond_spread",
    "equity_volatility",
    "consumer_sentiment",
    "housing_starts_growth",
    # Added for the validation's S5: ``equity_volatility`` entered as a level at the observation
    # date, identical for every loan in a month, so its coefficient is a calendar effect by
    # construction. Its move since origination varies across loans in the same month. Both forms
    # are candidates and the selection decides.
    "volatility_change",
)

#: Five candidates withdrawn on 17 October 2026 under **rule 15**, which asks a declared expected
#: sign of every candidate and says that the honest form of "theory commits to no direction" is
#: **not a candidate** rather than an unsigned one. Kept here rather than deleted, because a
#: covariate removed without its reason on the record is a covariate that comes back.
#:
#: Three of the five the selection had already eliminated for turning their own signs, so
#: withdrawing them changes nothing in fact. **Two were in the published model**, and those two
#: are why the rule exists.
WITHDRAWN_CANDIDATES: Final[dict[str, str]] = {
    # In the published model, and its second-largest macro effect. On a fixed-rate book the
    # borrower's payment does not move with the policy rate, so the only channels are the
    # market's -- lock-in and real-debt relief on one side, affordability and the cost of living
    # on the other -- and which dominates depends on whether rates rose with growth or with
    # inflation, which this model does not observe. Measured by the second validation: the term
    # lowered the predicted hazard 33% in 2023 and 36% in 2024, is about 80% of the level drift
    # from 2021, and on the 2022 regime cut is the difference between 1.420 and 0.959.
    "policy_rate_change": "no committed direction: the sign depends on an unobserved regime",
    # In the published model. A slope is a forward-looking cycle proxy whose entire relation to
    # default runs through a cycle this model already carries in unemployment and house prices,
    # so there is no direction left for theory to commit to. Measured: +0.083 at the 2018 cut and
    # -0.032, -0.039, -0.048 at the three after it, and the September selection had removed it at
    # step 9 for exactly that instability.
    "yield_curve_slope": "no committed direction, and the sign flips across time cuts",
    # Already eliminated at step 8 for turning its sign. Inflation erodes the real value of a
    # fixed nominal debt, which is relief, and erodes real income, which is stress; which
    # dominates depends on whether wages keep up.
    "inflation_rate": "no committed direction: real debt against real income",
    "inflation_change": "no committed direction, as the level",
    # Already eliminated at step 8 for turning its sign. A wealth channel for a minority of
    # these borrowers and a cycle proxy otherwise.
    "equity_return": "no committed direction for a mortgage book",
}

#: The months a payment moratorium suppressed the event, and the indicator that absorbs them.
#:
#: Rule 1 of `docs/rules.md`, amended 17 October 2026 before the selection it governs. Under
#: `exclude` a moratorium month is not an event and the loan stays at risk, which is right for the
#: dependent variable -- but the largest unemployment shock in the series, over eleven points in
#: two months, arrives with defaults suspended by law, and the coefficient records it. Measured
#: across the declared cuts, `unemployment_change` reads **-0.0920 at 2018-12** and **-0.0305 at
#: 2021-12**: four points of unemployment multiply the hazard by 1.67 on the first and 1.18 on the
#: second, a **threefold** attenuation of the one sensitivity a mortgage stress test turns on.
#:
#: CARES Act forbearance ran from March 2020; the window closes at September 2021, when the
#: enrolment and the reporting of it had wound down. It is a function of the **observation month
#: alone**, so it costs zero cells -- the same argument every macro covariate rests on.
MORATORIUM_REGIME: Final[tuple[str, str]] = ("2020-03", "2021-09")

#: Where step 9's time cut falls, by arithmetic rather than by judgement.
#:
#: Rule 9 of `docs/rules.md`, amended 17 October 2026: step 9 compares sign stability across
#: **time cuts** as well as across origination parity, because the second validation found a
#: covariate reading +0.083 at the 2018 cut and -0.032, -0.039 and -0.048 at the three after it
#: while the parity split called it stable.
#:
#: The **calendar midpoint of the development window**: 1999-01 to 2021-12 is 276 months, so the
#: first half ends at month 138, which is 2010-06. Chosen this way on purpose -- a cut placed to
#: put the crisis on one side would be a choice about results, and the midpoint is a number
#: anyone can check. It happens to put the crisis and its immediate aftermath in the first half
#: and the recovery and the pandemic in the second, which is a consequence and not the reason.
STABILITY_CUT: Final = "2010-06"

#: Macro covariates the selection may not remove, because their purpose is to absorb rather than
#: to be tested. `moratorium_regime` is here for the reason `harp` is in
#: `procedure.BASE_CATEGORICAL`: a selection free to drop it would be free to read the macro
#: coefficients off a book in which a policy intervention is still unmarked. Rule 15 asks a
#: declared sign of every *candidate*, and a protected covariate is not one -- it is not screened,
#: not eliminated, and carries no prior.
PROTECTED_MACRO: Final[tuple[str, ...]] = ("moratorium_regime",)

#: The economic dimension each candidate measures, fixed before any selection result.
#:
#: The stability rule of step 9 removes a covariate that is small, changes sign when the
#: sample does, and sits beside a larger covariate of the **same** dimension. Which
#: covariates share a dimension therefore has to be decided in advance: decided after the
#: fits, the rule would be a way of dropping whatever came out inconvenient.
ECONOMIC_DIMENSION: Final[dict[str, str]] = {
    # Its own dimension, with nothing beside it: step 9 removes an unstable covariate only
    # where a *larger* one of the same dimension sits next to it, and a regime indicator shares
    # its information with nothing. Protected anyway by `PROTECTED_MACRO`, so this is the belt
    # rather than the braces.
    "moratorium_regime": "policy regime",
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
    # The payment state is **not** in this dictionary, and the distinction is the point: this
    # records what the procedure removed, and the state was never a candidate for it to
    # remove. Why it is not one is in `procedure.CANDIDATE_CATEGORICAL` and rule 10 of
    # docs/rules.md, with the measurement.
    #
    # **Five of the six removals are wrong signs**, where the run before this had one. That is
    # rule 15 working rather than a worse model: every candidate now carries a declared prior,
    # so the strongest filter in the procedure reaches covariates it could not touch before --
    # and four of these five are a prior declared on 17 October 2026 meeting the data for the
    # first time. Two of those four priors the data contradicted; they are not revised, because
    # revising a prior after the fit is a retraction this project already carries.
    #
    # --- step 8: backwards against a declared prior ---
    "corporate_bond_spread": (
        "wrong sign: +0.1499 where - is expected. A wider spread is the price of credit risk "
        "rising, so a positive coefficient says tighter credit lengthens survival. The run "
        "before this removed it at step 6 for a variance inflation of 10.8, and the first run "
        "by hand for a reversal: three procedures, three reasons, one covariate"
    ),
    "equity_volatility": (
        "wrong sign: +0.001269 where - is expected. Was +0.009578 on the previous macro block, "
        "so the sign is not an artefact of this one: part of it was the 2020 moratoria, as the "
        "validation suspected (S5), and the rest is shared with financial_conditions -- which "
        "survives here where it did not before"
    ),
    "volatility_change": "wrong sign: +0.0004502 where - is expected",
    "house_price_growth": (
        "wrong sign: -0.7808 where + is expected. It is built from the same house price index "
        "as ltv_change, which survives, so conditional on the movement the growth rate reads "
        "backwards. -0.1456 on the previous block, five times larger here"
    ),
    "mortgage_rate_decline": (
        "wrong sign: -0.1571 where + is expected -- a **prior declared on 17 October 2026 and "
        "contradicted on its first fit**. The reasoning was that a market rate below the note "
        "rate makes refinancing into a lower payment available, which is relief; the data says "
        "otherwise conditional on the rest. The prior is not revised, and the covariate goes. "
        "The run before this removed it at step 9 for parity instability instead"
    ),
    # --- step 7: nothing left to carry beside the loan block ---
    #
    # The one covariate the **band form** removed that the linear form kept, and the reason it
    # is worth its own note: at this sample size the p-value arm of the procedure is otherwise
    # inert -- every p-value in the screening table reads 0.0000 -- so a p of 0.177 is not a
    # thin result, it is a covariate with nothing of its own left to say.
    #
    # What changed is the loan block around it. Under the linear form that block was three
    # coefficients; read as rule 12's bands it is **twelve**, and with twelve coefficients
    # describing underwriting risk, being a first-time buyer is no longer distinguishable from
    # the score, the leverage and the debt burden that go with it. Rule 12 warned that "a
    # covariate that survived on a straight line might not on a bend"; this is that, and it
    # landed on a categorical rather than on the macro block, which came through unchanged.
    "buyer_type": "step 7: p = 0.177 beside the loan block, where every other p-value reads 0.0000",
    # --- step 9: not identified beside a larger covariate of the same dimension ---
    #
    # **Rule 9's time cut, firing on its first run and catching what parity could not.** Both
    # origination halves and the first calendar half agree on the sign; the second does not.
    "housing_starts_growth": (
        "1 sd effect +0.040 on even vintages, +0.044 on odd vintages, +0.046 to 2010-06 and "
        "**-0.003 after it**, beside ltv_change (-0.201), both housing. The parity split called "
        "it stable -- which is why rule 9 was amended to compare time cuts, and this is the "
        "first covariate the amendment removed"
    ),
    # --- step 10: nothing. Materiality removed no covariate on this block, where it removed
    # two on the one before. financial_conditions read -0.0155 there, under the 0.02 of rule 3,
    # and is above it here: a threshold does not move, but what a covariate is worth depends on
    # what sits beside it, and five fewer macro series sit beside it now.
}

#: Order in which collinear macro covariates are given up, most expendable first.
#:
#: Fixed in advance, the way `nmds` fixes its own scale before running the selection, because a
#: priority chosen after seeing the VIF table is not a priority -- it is a preference for
#: whichever answer came out. The principle: keep what is specific to mortgage credit, give up
#: what is a general business-cycle proxy, and among equivalents keep the series with the longest
#: clean history.
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
#:
#: ``harp`` is here because the model cannot read the debt-to-income without it: the ratio is
#: missing for exactly the HARP refinances, and the constant that fills it is absorbed by this
#: level. It is in the selection's protected block for that reason rather than on the strength of
#: its coefficient -- rule 8 of `docs/rules.md`, and `features.NOT_REPORTED`.
#: Every categorical **offered** at origination, with its reference level: the input, where
#: `CATEGORICAL_REFERENCE` below is the selection's output.
#:
#: Rule 14's shape reads this one and not the output, and the difference is a defect found on 19
#: October 2026 rather than a distinction drawn in advance. `default_shape` derived the shape
#: from `CATEGORICAL_REFERENCE`, so when the band-form selection removed `buyer_type` from the
#: **scale** it silently left the **shape** too -- and the report then fitted a different model
#: from the one the selection chose, under the same name, with the warm start failing to find it
#: because a fit's fingerprint carries its ancillary formula.
#:
#: Rule 14 defines its set by an identity -- what is fixed at origination -- and says a subset
#: would be a pick. What the scale selected is not that identity. The two blocks are independent:
#: a covariate may be in the shape and not in the scale, and coupling them is the kind of thing
#: that is wrong quietly. `tests/test_procedure.py` holds this equal to `procedure`'s two blocks.
ORIGINATION_CATEGORICAL: Final[dict[str, str]] = {
    "purpose": "purchase",
    "occupancy": "owner_occupied",
    "harp": "standard",
    "mortgage_insurance": "uninsured",
    "buyer_type": "repeat",
}

CATEGORICAL_REFERENCE: Final[dict[str, str]] = {
    "purpose": "purchase",
    "occupancy": "owner_occupied",
    "harp": "standard",
    "mortgage_insurance": "uninsured",
    # `buyer_type` was here until 19 October 2026 and the band-form selection removed it at
    # step 7. It stays in `procedure.CANDIDATE_CATEGORICAL`, which is the **input** and is fixed
    # before any fit; this is the **output**, so it is a subset of that input and not a copy.
}


#: The banded loan covariates, and the band each one's coefficients read against.
#:
#: Rule 12 of `docs/rules.md`: these are bands in the cell key already -- five, four and four --
#: and reading them as a **line through the band midpoints** compressed the model's risk spread by
#: 40%. In sample, actual over expected ran 0.594 in the safest decile to 1.204 in the ninth, and
#: the model predicted 74.4x between the extremes where the book realises 125.3x. One coefficient
#: a band instead, at the cost of seven parameters and no cells.
#:
#: The reference is the band carrying the most loan-months, measured on the table before the rule
#: was written: 45.67%, 43.00% and 35.70% of exposure. `term_years` is not here because two levels
#: make a factor and a line the same model.
#:
#: They were re-measured once on the finer 8 / 8 / 6 grid of 10 October 2026, by the same rule
#: rather than carried over -- 790.0 again at 45.67%, 45.0 at 27.58% where 50.0 had been 43.00%,
#: and 32.0 at 22.18% where 19.0 had been 35.70%, the debt-to-income reference moving to a middle
#: band because the rule asks for exposure and not for safety. That grid was given back when it
#: proved to buy 0.015 on the worst in-sample decile; see `data.aggregate.DEFAULT_SPEC`.
#:
#: The levels are numbers, not strings, and the formula must say so: `Treatment('790.0')` is not a
#: level of a float column and formulaic refuses it.
BANDED_REFERENCE: Final[dict[str, float]] = {
    "credit_score": 790.0,
    "original_ltv": 50.0,
    "debt_to_income": 19.0,
}


#: Where the selection writes what it chose, under :func:`reports_dir`.
#:
#: Here rather than beside the report that writes it, because the *reader* is lower than the
#: writer: a fit starts from the specification the selection ended on, and `models` may not
#: import `reporting`. The name is the record's address, which is configuration.
SELECTION_RECORD: Final = "selection.json"


#: Threads the compiled kernel cuts its row loop into, and **part of what determines the answer**.
#:
#: Declared here rather than read off the machine, because the rows go into this many contiguous
#: parts and the partial sums are added in the parts' own order: two runs at the same count agree
#: bit for bit, and two different counts agree only to the last digits of a sum over 72.7 million
#: terms. A number taken from `os.cpu_count()` would make a fit's last digits a property of the
#: hardware, which is the mistake a pooled fit in this project already made once -- one queue,
#: `get` returning whichever worker finished first, two runs five significant figures apart forty
#: evaluations later.
#:
#: **Four**, measured on the production half against the gate rule 13 declared: a Hessian in
#: 4.21 s where one thread takes 11.39 and NumPy 31.18 -- 2.71x within the run, 7.4x against
#: NumPy. Six and eight reach 3.76 and 3.55 on this machine's four physical cores, and half a
#: second is not worth a count that has stopped meaning anything.
KERNEL_THREADS: Final = 4


#: How a cell's loan-months ended: in default, in a voluntary repayment, or in neither.
#:
#: Declared here rather than in `data.panel`, which is where the columns are, because three
#: things below that module need the names: this file's `record_name`, and the command line,
#: which was keeping a private copy to avoid the **0.85 s** importing `data.panel` puts on
#: every `creditsurv --help`. `data.panel` re-exports them, so the forty-odd call sites that
#: read them from there are unchanged.
DEFAULT_CAUSE: Final = "default"
PREPAYMENT_CAUSE: Final = "prepayment"
CENSORED: Final = "none"
CAUSES: Final[tuple[str, ...]] = (DEFAULT_CAUSE, PREPAYMENT_CAUSE)


def record_name(*, distribution: str, cause: str, published: str = DISTRIBUTION) -> str:
    """What this run's files are called.

    Beside the record's own name, and below both the report that writes these files and
    the fit that reads them: an address is configuration.

    ``selection`` is the **published** model's record, which ``tests/test_procedure.py``
    holds the configuration to. Every other run -- the other family, the prepayment model --
    is a record of a run rather than of the model in the configuration, and would otherwise
    overwrite it: two selections write two records, and the family rule then decides which
    one the configuration should agree with.
    """
    if distribution == published and cause == DEFAULT_CAUSE:
        return "selection"
    parts = ["selection", distribution]
    if cause != DEFAULT_CAUSE:
        parts.append(cause)
    return "_".join(parts)


def default_covariates() -> list[str]:
    """Every column the published model is allowed to read -- the **scale and the shape**.

    Beside the formulas that read the same lists, because the three have to agree and because
    the package needs it: it lived in `cli.py` until the fits moved out of there, which made a
    model module reach up into the command line for the definition of its own design.

    `ORIGINATION_CATEGORICAL` and not `CATEGORICAL_REFERENCE`, which is the same correction as
    in `default_shape` and was found one layer further out: rule 14's shape reads every
    categorical **offered** at origination, so the frame has to carry them whether or not the
    scale's selection kept each one. With the output set here, the shape named
    `C(buyer_type, Treatment('repeat'))` while nothing loaded the column, and the fit died on
    `NameError: name 'buyer_type' is not defined` -- in 138 seconds rather than the ten hours
    the two defects before it threatened, because a missing column fails where it is expanded.

    A superset is safe and a subset is not: a column loaded and unread costs memory, a column
    read and unloaded is a fit that cannot start.
    """
    return [
        *STATIC_CONTINUOUS,
        *TIME_VARYING_CONTINUOUS,
        *ORDINAL,
        *dict.fromkeys([*CATEGORICAL_REFERENCE, *ORIGINATION_CATEGORICAL]),
    ]


def default_shape() -> str:
    """Rule 14's shape: every covariate fixed at origination, in rule 12's forms.

    One source, for the reason `default_formula` is one: the shape is a property of the
    specification rather than a model of its own, so the selection, the three regime cuts, the
    report and the views all have to read the same string or they are fitting different models
    and calling one a refit.

    What is **not** here is any of the six covariates recomputed every loan-month. That is not a
    judgement about them -- `ltv_change` is the strongest covariate in the model -- but an
    identity: `period = cohort + age` holds, so a covariate that moves over a loan's life moves
    with the age, and rule 14 refuses an age-dependent shape because the hazard would stop
    integrating to a survival function. `Factorisation.expand` refuses it in code as well, so the
    rule does not rest on this function being written correctly.
    """
    linear = [name for name in STATIC_CONTINUOUS + ORDINAL if name not in BANDED_REFERENCE]
    banded = [f"C({name}, Treatment({band}))" for name, band in BANDED_REFERENCE.items()]
    # `ORIGINATION_CATEGORICAL`, the set **offered**, not `CATEGORICAL_REFERENCE`, the set the
    # scale's selection kept. Rule 14's admissible side is an identity -- what is fixed at
    # origination -- and does not shrink when the scale's does.
    categorical = [
        f"C({name}, Treatment('{reference}'))"
        for name, reference in ORIGINATION_CATEGORICAL.items()
    ]
    return " + ".join([*linear, *banded, *categorical])


def default_formula() -> str:
    """Formulaic specification for the full model.

    Reference levels are stated explicitly so coefficients remain comparable across refits even
    if a level goes missing from a training slice -- for the bands of rule 12 as well as for the
    categoricals, and there the level is a number.
    """
    linear = [
        name
        for name in STATIC_CONTINUOUS + TIME_VARYING_CONTINUOUS + ORDINAL
        if name not in BANDED_REFERENCE
    ]
    banded = [f"C({name}, Treatment({band}))" for name, band in BANDED_REFERENCE.items()]
    categorical = [
        f"C({name}, Treatment('{reference}'))" for name, reference in CATEGORICAL_REFERENCE.items()
    ]
    return " + ".join([*linear, *banded, *categorical])
