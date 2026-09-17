"""Every variable the data, the model and the reports name, and how a reader should see it.

Three things used to be conflated in a column name: what the code calls a variable, what a
reader should be shown, and what the number means. A reader of the site met ``fico_s``,
``cltv_drift`` and ``nfci_lagged``, and levels such as ``refinance_cashout`` or ``Y``, and had
to go to the source to learn that the first was a credit score recentred on 700 and divided
by 50. This registry holds all three once:

* the **name** the code uses, readable snake case;
* the **label** a figure, a table or a sentence shows;
* the **definition**, **unit** and **source** a reader needs to interpret the number.

It also records each variable's **former name**, so tables, reports and fits written before
the rename can still be read and are still labelled correctly.

A categorical carries its **levels** the same way: the code value, the label, and the value
it had before. ``Y`` and ``N`` are gone from the code: an insured loan is ``insured``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class Kind(StrEnum):
    """Where a variable comes from, which decides what its former name could collide with."""

    LOAN = "loan at origination"
    MACRO = "macro covariate"
    SERIES = "macro series"
    STRUCTURE = "panel structure"


@dataclass(frozen=True)
class Level:
    """A value a categorical variable takes."""

    code: str
    label: str
    former: str | None = None


@dataclass(frozen=True)
class Variable:
    name: str
    label: str
    kind: Kind
    definition: str
    unit: str = ""
    source: str = ""
    former: str | None = None
    levels: tuple[Level, ...] = ()

    def level(self, code: object) -> Level:
        """The level with this code, current or former."""
        text = str(code)
        for level in self.levels:
            if text in (level.code, level.former):
                return level
        message = f"{self.name} has no level {text!r}"
        raise KeyError(message)


def _loan(
    name: str,
    label: str,
    definition: str,
    *,
    unit: str = "",
    source: str = "",
    former: str | None = None,
    levels: Iterable[tuple[str, str, str | None]] = (),
) -> Variable:
    return Variable(
        name,
        label,
        Kind.LOAN,
        definition,
        unit,
        source,
        former,
        tuple(Level(*level) for level in levels),
    )


def _macro(
    name: str, label: str, definition: str, *, unit: str, source: str, former: str | None = None
) -> Variable:
    return Variable(name, label, Kind.MACRO, definition, unit, source, former)


def _series(
    name: str, label: str, definition: str, *, unit: str, source: str, former: str | None = None
) -> Variable:
    return Variable(name, label, Kind.SERIES, definition, unit, source, former)


def _structure(name: str, label: str, definition: str, *, former: str | None = None) -> Variable:
    return Variable(name, label, Kind.STRUCTURE, definition, former=former)


_VARIABLES: Final[tuple[Variable, ...]] = (
    # ----- the loan, at origination ------------------------------------------------------
    _loan(
        "credit_score",
        "Credit score",
        "The borrower's credit score at origination; in the cells, the midpoint of its band.",
        unit="points",
        source="classic_fico, origination file (9999 = not available)",
        former="fico_s",
    ),
    _loan(
        "original_ltv",
        "Loan-to-value at origination",
        "The loan amount over the property value when the loan was written.",
        unit="%",
        source="original_ltv, origination file (999 = not available)",
        former="orig_ltv",
    ),
    _loan(
        "original_cltv",
        "Combined loan-to-value at origination",
        "All liens on the property over its value when the loan was written.",
        unit="%",
        source="original_cltv, origination file (999 = not available)",
        former="orig_cltv",
    ),
    _loan(
        "debt_to_income",
        "Debt-to-income at origination",
        "The borrower's monthly debt payments over monthly income when the loan was written.",
        unit="%",
        source="original_dti, origination file (999 = not available)",
        former="dti",
    ),
    _loan(
        "term_years",
        "Original term",
        "The contractual term: 15 years up to 190 months, 30 years above.",
        unit="years",
        source="original_loan_term, origination file",
    ),
    _loan(
        "original_balance",
        "Original balance",
        "The unpaid principal balance when the loan was written.",
        unit="$",
        source="original_upb, origination file",
        former="orig_upb",
    ),
    _loan(
        "log_original_balance",
        "Original balance (log)",
        "The natural logarithm of the original balance.",
        unit="log $",
        source="original_upb, origination file",
        former="log_orig_upb",
    ),
    _loan(
        "note_rate",
        "Note rate",
        "The interest rate the borrower pays, fixed at origination.",
        unit="%",
        source="original_interest_rate, origination file",
    ),
    _loan(
        "origination_spread",
        "Note rate over the market rate at origination",
        "The note rate minus the 30-year market mortgage rate of the origination month.",
        unit="pp",
        source="original_interest_rate and MORTGAGE30US",
        former="orig_spread",
    ),
    _loan(
        "refinance_incentive",
        "Refinancing incentive",
        "The note rate minus the market mortgage rate now: how much a refinance would save.",
        unit="pp",
        source="original_interest_rate and MORTGAGE30US",
        former="refi_incentive",
    ),
    _loan(
        "insurance_coverage",
        "Mortgage insurance coverage",
        "The share of the loss the mortgage insurer covers.",
        unit="%",
        source="mortgage_insurance_percentage, origination file (999 = not available)",
        former="mi_percent",
    ),
    _loan(
        "estimated_ltv_change",
        "Estimated loan-to-value change",
        "Freddie Mac's own estimate of current loan-to-value, minus loan-to-value at "
        "origination. Not modelled: coverage runs from 0.8% of the 1999 vintage to 94% of 2021.",
        unit="pp",
        source="estimated_loan_to_value, performance file (999 = not available)",
        former="eltv_drift",
    ),
    _loan(
        "purpose",
        "Loan purpose",
        "Why the loan was taken out.",
        source="loan_purpose, origination file: P, C, N and R",
        levels=(
            ("purchase", "Purchase", None),
            ("cash_out_refinance", "Cash-out refinance", "refinance_cashout"),
            ("rate_term_refinance", "Rate-and-term refinance", "refinance_rate_term"),
        ),
    ),
    _loan(
        "occupancy",
        "Occupancy",
        "How the borrower uses the property.",
        source="occupancy_status, origination file: P, S and I",
        levels=(
            ("owner_occupied", "Owner-occupied", None),
            ("second_home", "Second home", None),
            ("investment_property", "Investment property", "investor"),
        ),
    ),
    _loan(
        "mortgage_insurance",
        "Mortgage insurance",
        "Whether the loan carries private mortgage insurance.",
        source="mortgage_insurance_percentage above zero, origination file",
        former="has_mi",
        levels=(
            ("insured", "Insured", "Y"),
            ("uninsured", "Not insured", "N"),
        ),
    ),
    _loan(
        "buyer_type",
        "Buyer",
        "Whether the borrower is buying a home for the first time.",
        source="first_time_homebuyer_indicator, origination file (9 = not available)",
        former="first_time_buyer",
        levels=(
            ("first_time", "First-time buyer", "Y"),
            ("repeat", "Repeat buyer", "N"),
        ),
    ),
    _loan(
        "channel",
        "Origination channel",
        "Who originated the loan. Broker and correspondent are one level: until 2008 half the "
        "book was coded third party, not otherwise specified.",
        source="channel, origination file: R, B, C and T",
        levels=(
            ("retail", "Retail", None),
            ("broker_or_correspondent", "Broker or correspondent", "third_party"),
        ),
    ),
    _loan(
        "region",
        "Census region",
        "The census region of the property's state.",
        source="property_state, origination file",
        levels=(
            ("Northeast", "Northeast", None),
            ("Midwest", "Midwest", None),
            ("South", "South", None),
            ("West", "West", None),
            ("Other", "Territories and unknown states", None),
        ),
    ),
    _loan(
        "property_type",
        "Property type",
        "The kind of property the loan is secured on.",
        source="property_type, origination file: SF, PU, CO, MH and CP",
        levels=(
            ("single_family", "Single-family home", None),
            ("planned_unit_development", "Planned unit development", "planned_unit"),
            ("condominium", "Condominium", "condo"),
            ("manufactured_or_coop", "Manufactured home or co-op", "other"),
        ),
    ),
    _loan(
        "units",
        "Units",
        "The number of dwelling units in the property.",
        source="number_of_units, origination file",
        levels=(
            ("one_unit", "One unit", "1"),
            ("two_to_four_units", "Two to four units", "2-4"),
        ),
    ),
    _loan(
        "loan_size",
        "Loan size",
        "Whether the loan exceeds the national conforming limit, as a high-cost area allows.",
        source="super_conforming_flag, origination file",
        former="super_conforming",
        levels=(
            ("conforming", "Conforming", "N"),
            ("super_conforming", "Super-conforming", "Y"),
        ),
    ),
    _loan(
        "borrower_count",
        "Borrowers",
        "The number of borrowers on the loan.",
        source="number_of_borrowers, origination file",
        former="n_borrowers",
        levels=(
            ("one", "One borrower", "1"),
            ("two_or_more", "Two or more borrowers", "2+"),
        ),
    ),
    # ----- macro covariates: each read three months back ------------------------------------
    _macro(
        "ltv_change",
        "Loan-to-value change since origination",
        "Loan-to-value at origination carried by the national house price index to the "
        "observation month, minus loan-to-value at origination. Zero at origination; positive "
        "when prices have fallen.",
        unit="pp",
        source="CSUSHPINSA",
        former="cltv_drift",
    ),
    _macro(
        "unemployment_change",
        "Unemployment change since origination",
        "The unemployment rate now minus the rate in the origination month.",
        unit="pp",
        source="UNRATE",
        former="unemp_gap",
    ),
    _macro(
        "financial_conditions",
        "Financial conditions",
        "The Chicago Fed National Financial Conditions Index: zero is average, positive is "
        "tighter than average.",
        unit="index",
        source="NFCI",
        former="nfci_lagged",
    ),
    _macro(
        "policy_rate_change",
        "Policy rate change since origination",
        "The federal funds rate now minus the rate in the origination month.",
        unit="pp",
        source="FEDFUNDS",
        former="policy_rate_gap",
    ),
    _macro(
        "mortgage_rate_decline",
        "Mortgage rate fall since origination",
        "The market mortgage rate in the origination month minus the rate now, on the 15-year "
        "rate for 15-year loans. Positive when refinancing has become cheaper.",
        unit="pp",
        source="MORTGAGE30US, MORTGAGE15US",
        former="rate_gap",
    ),
    _macro(
        "house_price_growth",
        "House price growth",
        "The year-on-year change in the national house price index.",
        unit="fraction",
        source="CSUSHPINSA",
        former="hpi_growth",
    ),
    _macro(
        "yield_curve_slope",
        "Yield curve slope",
        "The 10-year Treasury yield minus the 2-year.",
        unit="pp",
        source="T10Y2Y",
        former="term_spread",
    ),
    _macro(
        "corporate_bond_spread",
        "Corporate bond spread",
        "Moody's Baa corporate yield minus the 10-year Treasury yield.",
        unit="pp",
        source="BAA10Y",
        former="credit_spread",
    ),
    _macro(
        "inflation_rate",
        "Inflation",
        "The year-on-year change in consumer prices.",
        unit="fraction",
        source="CPIAUCSL",
        former="inflation",
    ),
    _macro(
        "inflation_change",
        "Inflation change since origination",
        "Year-on-year inflation now minus year-on-year inflation in the origination month.",
        unit="fraction",
        source="CPIAUCSL",
        former="inflation_gap",
    ),
    _macro(
        "equity_return",
        "Equity return",
        "The year-on-year change in the Nasdaq Composite index.",
        unit="fraction",
        source="NASDAQCOM",
    ),
    _macro(
        "equity_volatility",
        "Equity volatility",
        "The CBOE volatility index.",
        unit="index",
        source="VIXCLS",
        former="vix",
    ),
    _macro(
        "volatility_change",
        "Volatility change since origination",
        "The CBOE volatility index now minus its value in the origination month.",
        unit="index points",
        source="VIXCLS",
        former="vix_gap",
    ),
    _macro(
        "consumer_sentiment",
        "Consumer sentiment",
        "The University of Michigan consumer sentiment index.",
        unit="index",
        source="UMCSENT",
        former="sentiment",
    ),
    _macro(
        "housing_starts_growth",
        "Housing starts growth",
        "The year-on-year change in housing units started.",
        unit="fraction",
        source="HOUST",
        former="starts_growth",
    ),
    # ----- the monthly FRED series the macro covariates are built from ----------------------
    _series(
        "unemployment_rate",
        "Unemployment rate",
        "Civilian unemployment rate, seasonally adjusted.",
        unit="%",
        source="UNRATE",
    ),
    _series(
        "house_price_index",
        "House price index",
        "S&P CoreLogic Case-Shiller U.S. National Home Price Index.",
        unit="index, January 2000 = 100",
        source="CSUSHPINSA",
        former="hpi",
    ),
    _series(
        "mortgage_rate_30y",
        "30-year mortgage rate",
        "Average 30-year fixed mortgage rate.",
        unit="%",
        source="MORTGAGE30US",
    ),
    _series(
        "mortgage_rate_15y",
        "15-year mortgage rate",
        "Average 15-year fixed mortgage rate.",
        unit="%",
        source="MORTGAGE15US",
    ),
    _series(
        "financial_conditions_index",
        "Financial conditions index",
        "Chicago Fed National Financial Conditions Index.",
        unit="index, 0 = average",
        source="NFCI",
        former="nfci",
    ),
    _series(
        "fed_funds_rate",
        "Federal funds rate",
        "Effective federal funds rate.",
        unit="%",
        source="FEDFUNDS",
        former="policy_rate",
    ),
    _series(
        "treasury_10y",
        "10-year Treasury yield",
        "10-year Treasury constant maturity yield.",
        unit="%",
        source="DGS10",
    ),
    _series(
        "treasury_10y_2y_spread",
        "10-year minus 2-year Treasury spread",
        "10-year minus 2-year Treasury constant maturity yield.",
        unit="pp",
        source="T10Y2Y",
        former="term_spread",
    ),
    _series(
        "baa_treasury_spread",
        "Baa corporate spread",
        "Moody's Baa corporate bond yield relative to the 10-year Treasury yield.",
        unit="pp",
        source="BAA10Y",
        former="credit_spread",
    ),
    _series(
        "consumer_price_index",
        "Consumer price index",
        "Consumer price index for all urban consumers, seasonally adjusted.",
        unit="index",
        source="CPIAUCSL",
        former="cpi",
    ),
    _series(
        "nasdaq_composite",
        "Nasdaq Composite",
        "Nasdaq Composite index.",
        unit="index",
        source="NASDAQCOM",
        former="equity_index",
    ),
    _series(
        "vix_index",
        "VIX",
        "CBOE volatility index.",
        unit="index",
        source="VIXCLS",
        former="vix",
    ),
    _series(
        "consumer_sentiment_index",
        "Consumer sentiment index",
        "University of Michigan consumer sentiment index.",
        unit="index, 1966 Q1 = 100",
        source="UMCSENT",
        former="sentiment",
    ),
    _series(
        "housing_starts",
        "Housing starts",
        "Privately owned housing units started, seasonally adjusted annual rate.",
        unit="thousands",
        source="HOUST",
    ),
    _series(
        "observed_mortgage_delinquency_rate",
        "Mortgage delinquency rate, commercial banks",
        "Delinquency rate on single-family residential mortgages at all commercial banks. "
        "A reference, never a covariate.",
        unit="%",
        source="DRSFRMACBS",
    ),
    # ----- the structure of the panel -------------------------------------------------------
    _structure("loan_id", "Loan", "The loan identifier. Never published."),
    _structure("age", "Loan age", "Months since origination; the model's time scale."),
    _structure("period", "Observation month", "The calendar month a loan-month is observed in."),
    _structure(
        "origination_month",
        "Origination month",
        "The month the loan was written, as months since year zero.",
        former="orig_month",
    ),
    _structure(
        "origination_period",
        "Origination month",
        "The month the loan was written, as a calendar period.",
        former="orig_period",
    ),
    _structure("vintage", "Vintage", "The quarter the loan was written in."),
    _structure("event", "Default", "Whether the loan-month ended in default."),
    _structure("prepaid", "Prepayment", "Whether the loan-month ended in a voluntary payoff."),
    _structure(
        "loan_months",
        "Loan-months",
        "How many loan-months a cell stands for: the weight of every fit.",
        former="n",
    ),
)


def _index(variables: Iterable[Variable]) -> dict[str, Variable]:
    found: dict[str, Variable] = {}
    for variable in variables:
        if variable.name in found:
            message = f"Two variables are named {variable.name!r}."
            raise ValueError(message)
        found[variable.name] = variable
    return found


#: Every variable, by its current name.
VARIABLES: Final[dict[str, Variable]] = _index(_VARIABLES)

#: Labels of lifelines' parameter blocks, which keep lifelines' own names.
PARAMETERS: Final[dict[str, str]] = {
    "lambda_": "Scale",
    "rho_": "Shape",
    "alpha_": "Scale",
    "beta_": "Shape",
    "mu_": "Location",
    "sigma_": "Scale",
    "Intercept": "Intercept",
}

#: Labels of the distribution families.
DISTRIBUTIONS: Final[dict[str, str]] = {
    "weibull": "Weibull",
    "loglogistic": "Log-logistic",
    "lognormal": "Log-normal",
    "exponential": "Exponential",
}

#: The order in which a former name is resolved when two kinds shared it. ``vix``,
#: ``sentiment``, ``term_spread`` and ``credit_spread`` were each the name of a FRED series
#: and of the covariate built from it; a covariate is what a model table means.
_KIND_PRECEDENCE: Final[tuple[Kind, ...]] = (Kind.LOAN, Kind.MACRO, Kind.STRUCTURE, Kind.SERIES)


def variable(name: str, *, kind: Kind | None = None) -> Variable:
    """The variable with this name, or the one that used to have it.

    ``kind`` settles a former name two kinds shared; without it, a covariate wins over the
    series it was built from.
    """
    current = VARIABLES.get(name)
    if current is not None and (kind is None or current.kind is kind):
        return current
    kinds = _KIND_PRECEDENCE if kind is None else (kind,)
    for wanted in kinds:
        for candidate in _VARIABLES:
            if candidate.former == name and candidate.kind is wanted:
                return candidate
    message = f"No variable is or was named {name!r}."
    raise KeyError(message)


def label(name: str, *, kind: Kind | None = None) -> str:
    """What a reader is shown for a variable, current or former name.

    A name the registry does not hold is shown as it is, with underscores as spaces, so a
    new column never breaks a page; the registry tests are what keep that from happening
    to a modelled variable.
    """
    try:
        return variable(name, kind=kind).label
    except KeyError:
        return name.replace("_", " ")


def level_label(name: str, code: object) -> str:
    """What a reader is shown for a categorical level, current or former codes."""
    try:
        return variable(name, kind=Kind.LOAN).level(code).label
    except KeyError:
        return str(code)


_TERM: Final = re.compile(r"C\((\w+), Treatment\('([^']*)'\)\)\[T\.([^\]]*)\]")


def term_label(term: str) -> str:
    """A formula term as a reader would say it.

    ``C(purpose, Treatment('purchase'))[T.cash_out_refinance]`` is "Loan purpose: Cash-out
    refinance (against Purchase)"; a continuous term is its variable's label; an intercept
    stays an intercept.
    """
    match = _TERM.fullmatch(term)
    if match:
        name, reference, level = match.groups()
        return f"{label(name)}: {level_label(name, level)} (against {level_label(name, reference)})"
    if term in PARAMETERS:
        return PARAMETERS[term]
    return label(term)


def former_names(kind: Kind | None = None) -> dict[str, str]:
    """Former name to current name, for the variables of ``kind``."""
    return {
        entry.former: entry.name
        for entry in _VARIABLES
        if entry.former is not None and (kind is None or entry.kind is kind)
    }


def former_levels() -> dict[str, dict[str, str]]:
    """For each categorical, former code to current code."""
    return {
        entry.name: {level.former: level.code for level in entry.levels if level.former}
        for entry in _VARIABLES
        if entry.levels
    }


def glossary(kinds: Iterable[Kind] = tuple(Kind)) -> list[dict[str, str]]:
    """One row per variable, for the site's Variables page."""
    wanted = set(kinds)
    return [
        {
            "Label": entry.label,
            "Name": f"`{entry.name}`",
            "What it is": entry.definition,
            "Unit": entry.unit,
            "Values": "; ".join(f"{level.label} (`{level.code}`)" for level in entry.levels),
            "Source": entry.source,
            "Formerly": f"`{entry.former}`" if entry.former else "",
            "Kind": entry.kind.value,
        }
        for entry in _VARIABLES
        if entry.kind in wanted
    ]


def labelled(values: Mapping[str, object]) -> dict[str, object]:
    """A mapping keyed by variable names, keyed by labels instead."""
    return {label(str(name)): value for name, value in values.items()}
