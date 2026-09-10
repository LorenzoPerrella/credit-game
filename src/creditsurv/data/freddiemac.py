"""Loader for the Freddie Mac Single-Family Loan-Level Dataset.

The right data for this project, and the reason the loan book here is otherwise
simulated: it sits behind a free registration on Clarity, so it cannot be fetched
as part of a reproducible pipeline. This module reads files the user has already
downloaded and maps them onto the same canonical panel the synthetic generator
produces, so everything downstream is unchanged.

**Nothing here touches the network.** The dataset is behind an authenticated
download and scraping it would breach the terms it is offered under. The column
layout, by contrast, is published openly, and the field lists below were extracted
from ``file_layout_july_2026.xlsx`` rather than transcribed by hand -- 66 fields
across two files is too many to copy reliably, and a single misplaced name shifts
every column after it while still parsing cleanly.

Each vintage comes as two pipe-delimited files with no header row:

``sample_orig_YYYY.txt``
    One row per loan, as underwritten.
``sample_svcg_YYYY.txt``
    One row per loan-month.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

#: Download page for the dataset. Registration is free but manual.
CLARITY_URL: Final = "https://claritydownload.fmapps.freddiemac.com/CRT/"

#: Origination file fields, in positional order, from the published layout.
ORIGINATION_COLUMNS: Final[tuple[str, ...]] = (
    "classic_fico",
    "first_payment_date",
    "first_time_homebuyer_indicator",
    "maturity_date",
    "msa",
    "mortgage_insurance_percentage",
    "number_of_units",
    "occupancy_status",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "channel",
    "prepayment_penalty_indicator",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_identifier",
    "loan_purpose",
    "original_loan_term",
    "number_of_borrowers",
    "seller_name",
    "super_conforming_flag",
    "pre_harp_loan_sequence_number",
    "special_eligibility_program",
    "harp_indicator",
    "property_valuation_method",
    "interest_only_indicator",
    "vantagescore_4",
)

#: Monthly performance file fields, in positional order.
PERFORMANCE_COLUMNS: Final[tuple[str, ...]] = (
    "loan_identifier",
    "period",
    "current_actual_upb",
    "current_loan_delinquency_status",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "defect_settlement_date",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_non_interest_bearing_upb",
    "due_date_of_last_paid_installment",
    "mi_recoveries",
    "net_sales_proceeds",
    "non_mi_recoveries",
    "total_expenses",
    "legal_costs",
    "maintenance_and_preservation_costs",
    "taxes_and_insurance",
    "miscellaneous_expenses",
    "actual_loss",
    "cumulative_modification_costs",
    "interest_rate_step_indicator",
    "payment_deferral_flag",
    "estimated_loan_to_value",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_plan",
    "current_period_modification_costs",
    "current_interest_bearing_upb",
    "mortgage_insurance_cancellation_indicator",
    "servicer_name",
    "bankruptcy_cramdown_costs",
)

#: Sentinels the dataset uses for "not available", by field. They are ordinary
#: numbers, so leaving them in place silently produces a portfolio whose average
#: credit score is several thousand.
_MISSING_SENTINELS: Final[dict[str, float]] = {
    "classic_fico": 9999,
    "original_dti": 999,
    "original_ltv": 999,
    "original_cltv": 999,
}

#: Zero-balance codes that terminate a loan through credit loss rather than
#: repayment. 01 is a voluntary payoff and is censoring, not an event.
DEFAULT_ZERO_BALANCE_CODES: Final[frozenset[str]] = frozenset({"02", "03", "09", "15"})
PREPAYMENT_ZERO_BALANCE_CODE: Final = "01"

#: Delinquency at which a loan is treated as defaulted: three missed payments.
DEFAULT_DELINQUENCY_MONTHS: Final = 3

_CENSUS_REGIONS: Final[dict[str, tuple[str, ...]]] = {
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
}
_STATE_TO_REGION: Final[dict[str, str]] = {
    state: region for region, states in _CENSUS_REGIONS.items() for state in states
}

_PURPOSE = {
    "P": "purchase",
    "N": "refinance_rate_term",
    "C": "refinance_cashout",
    "R": "refinance_rate_term",
}
_OCCUPANCY = {"P": "owner_occupied", "S": "second_home", "I": "investor"}
_CHANNEL = {"R": "retail", "B": "broker", "C": "correspondent", "T": "correspondent"}


class FreddieMacDataMissingError(FileNotFoundError):
    """Raised when the downloaded files are not where they were expected."""


def _require(path: Path, kind: str) -> None:
    if not path.exists():
        message = (
            f"No Freddie Mac {kind} file at {path}.\n"
            "This dataset is not downloadable programmatically: it sits behind a free "
            f"registration at {CLARITY_URL}.\n"
            "Download a sample vintage, then point --orig and --svcg at "
            "sample_orig_YYYY.txt and sample_svcg_YYYY.txt.\n"
            "To run without it, use the synthetic source instead:\n"
            "  uv run creditsurv build-data --source synthetic"
        )
        raise FreddieMacDataMissingError(message)


def _blank_sentinels(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column, sentinel in _MISSING_SENTINELS.items():
        if column in cleaned.columns:
            cleaned.loc[cleaned[column] >= sentinel, column] = np.nan
    return cleaned


def read_origination(path: Path) -> pd.DataFrame:
    """Read one ``sample_orig_YYYY.txt``."""
    _require(path, "origination")
    frame = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=list(ORIGINATION_COLUMNS),
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )
    numeric = [
        "classic_fico",
        "original_dti",
        "original_ltv",
        "original_cltv",
        "original_upb",
        "original_interest_rate",
        "original_loan_term",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return _blank_sentinels(frame)


def read_performance(path: Path) -> pd.DataFrame:
    """Read one ``sample_svcg_YYYY.txt``."""
    _require(path, "performance")
    frame = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=list(PERFORMANCE_COLUMNS),
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )
    frame["loan_age"] = pd.to_numeric(frame["loan_age"], errors="coerce")
    frame["current_actual_upb"] = pd.to_numeric(frame["current_actual_upb"], errors="coerce")
    return frame


def _delinquency_months(status: pd.Series) -> pd.Series:
    """Numeric months delinquent, treating the non-numeric codes as not-in-default.

    The field is alphanumeric: ``RA`` marks an REO acquisition and ``XX`` an unknown
    status. Coercing blindly turns both into NaN, which compares false and so reads
    as performing -- correct for ``XX`` and wrong for ``RA``, which is why the
    zero-balance code is checked alongside it rather than instead of it.
    """
    return pd.to_numeric(status, errors="coerce")


def to_canonical_panel(
    origination: pd.DataFrame,
    performance: pd.DataFrame,
    *,
    default_delinquency: int = DEFAULT_DELINQUENCY_MONTHS,
) -> pd.DataFrame:
    """Join the two files into the canonical loan-month panel.

    Loan age comes from the performance file rather than being derived from dates,
    and the origination month is recovered as ``period - loan_age``. The dataset has
    no origination date field -- only a first payment date, which sits one or two
    months later depending on the servicer -- so deriving age from it would put a
    portfolio's seasoning out by a month in a way that varies loan by loan.
    """
    events = _default_flags(performance, default_delinquency)

    panel = performance.loc[:, ["loan_identifier", "period", "loan_age"]].copy()
    panel["period"] = pd.PeriodIndex(pd.to_datetime(performance["period"], format="%Y%m"), freq="M")
    panel = panel[panel["loan_age"] >= 0].reset_index(drop=True)
    events = events.loc[panel.index]

    panel = panel.rename(columns={"loan_identifier": "loan_id", "loan_age": "age"})
    panel["age"] = panel["age"].astype("int64")
    panel["orig_period"] = panel["period"] - panel["age"]
    panel["event"] = events["defaulted"].to_numpy()
    panel["prepaid"] = events["prepaid"].to_numpy()

    attributes = _loan_attributes(origination)
    panel = panel.merge(attributes, on="loan_id", how="inner")
    return _truncate_at_first_event(panel)


def _default_flags(performance: pd.DataFrame, default_delinquency: int) -> pd.DataFrame:
    delinquency = _delinquency_months(performance["current_loan_delinquency_status"])
    zero_balance = performance["zero_balance_code"].fillna("")
    return pd.DataFrame(
        {
            "defaulted": (delinquency >= default_delinquency).fillna(False)
            | zero_balance.isin(DEFAULT_ZERO_BALANCE_CODES),
            "prepaid": zero_balance == PREPAYMENT_ZERO_BALANCE_CODE,
        },
        index=performance.index,
    )


def _loan_attributes(origination: pd.DataFrame) -> pd.DataFrame:
    score = origination["classic_fico"]
    attributes = pd.DataFrame(
        {
            "loan_id": origination["loan_identifier"],
            "credit_score": score,
            "fico_s": (score - 700.0) / 50.0,
            "orig_ltv": origination["original_ltv"],
            "dti": origination["original_dti"],
            "orig_upb": origination["original_upb"],
            "log_orig_upb": np.log(origination["original_upb"]),
            "note_rate": origination["original_interest_rate"],
            "purpose": origination["loan_purpose"].map(_PURPOSE),
            "occupancy": origination["occupancy_status"].map(_OCCUPANCY),
            "channel": origination["channel"].map(_CHANNEL),
            "region": origination["property_state"].map(_STATE_TO_REGION),
            "first_time_buyer": origination["first_time_homebuyer_indicator"].where(
                origination["first_time_homebuyer_indicator"].isin(["Y", "N"]), "N"
            ),
        }
    )
    # A loan missing a covariate cannot be modelled, and imputing underwriting
    # characteristics would invent the very thing being measured.
    return attributes.dropna().reset_index(drop=True)


def _truncate_at_first_event(panel: pd.DataFrame) -> pd.DataFrame:
    """Cut each loan at its first terminating month.

    Servicing files continue reporting after a default -- through foreclosure,
    disposition and loss settlement -- so a loan can carry many rows flagged as
    defaulted. Left alone that breaks the panel invariant of at most one event per
    loan, and counts a single default many times over in the likelihood.
    """
    ordered = panel.sort_values(["loan_id", "age"], kind="stable").reset_index(drop=True)
    terminating = ordered["event"] | ordered["prepaid"]
    first_event = ordered.loc[terminating].groupby("loan_id", observed=True)["age"].min()

    terminal_age = ordered["loan_id"].map(first_event)
    keep = terminal_age.isna() | (ordered["age"] <= terminal_age)
    kept = ordered[keep].reset_index(drop=True)

    is_terminal = kept["age"] == kept["loan_id"].map(first_event)
    kept["event"] = kept["event"] & is_terminal
    kept["prepaid"] = kept["prepaid"] & is_terminal
    return kept


def load_sample(origination_path: Path, performance_path: Path) -> pd.DataFrame:
    """Read one vintage and return it as a canonical loan-month panel."""
    return to_canonical_panel(
        read_origination(origination_path), read_performance(performance_path)
    )
