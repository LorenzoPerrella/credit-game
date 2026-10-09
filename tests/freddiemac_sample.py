"""The pandas loader for a downloaded Freddie Mac sample, used to build the fixtures.

**It lives here because nothing in the package calls it.** `tests/fixtures.py` writes the
fixtures in Freddie Mac's own pipe-delimited format and reads them back through this loader,
so the parsing, the sentinels and the code maps are exercised by every test that needs data;
the production path reads the same files with DuckDB (`data.book`) from parquet written by
`data.ingest`.

And while it was under `src/`, it closed the one import cycle in the codebase: it needs the
event definition from `data.book`, which needs the record layout from `data.ingest`, which
needs `data.freddiemac`. Out here the cycle does not exist and `data.freddiemac` imports
nothing at all.

The canonical panel it produces is the **episode** representation -- one boolean per cause
per loan-month -- which is what `data.panel` reads. The cell table's three-state `outcome`
column is the production spelling of the same fact; see `data.book`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from creditsurv.data.book import _STATE_TO_REGION
from creditsurv.data.book import (
    DEFAULT_DELINQUENCY as _DEFAULT_DELINQUENCY,
)
from creditsurv.data.book import (
    DEFAULT_ZERO_BALANCE as _DEFAULT_ZERO_BALANCE,
)
from creditsurv.data.book import (
    PREPAYMENT_ZERO_BALANCE as _PREPAYMENT_ZERO_BALANCE,
)
from creditsurv.data.freddiemac import (
    CLARITY_URL,
    ORIGINATION_COLUMNS,
    PERFORMANCE_COLUMNS,
)

if TYPE_CHECKING:
    from pathlib import Path

#: Sentinels the dataset uses for "not available", by field. They are ordinary
#: numbers, so leaving them in place silently produces a portfolio whose average
#: credit score is several thousand.
_MISSING_SENTINELS: Final[dict[str, float]] = {
    "classic_fico": 9999,
    "original_dti": 999,
    "original_ltv": 999,
    "original_cltv": 999,
}

#: The event definition, from the one place it is argued: `creditsurv.data.book`.
#:
#: It was written out again here, and the two copies had already come apart in type -- a
#: `frozenset` against a tuple, a bare string against a one-element tuple -- while agreeing on
#: the codes. They cannot disagree now, which matters because they decide what a default is.
#:
#: This module's comment used to add that "01 is a voluntary payoff and is censoring, not an
#: event". That stopped being true when prepayment became a competing risk: 01 is an event of its
#: own cause, and 16 and 96 are the censoring.
DEFAULT_ZERO_BALANCE_CODES: Final[frozenset[str]] = frozenset(_DEFAULT_ZERO_BALANCE)
PREPAYMENT_ZERO_BALANCE_CODE: Final = _PREPAYMENT_ZERO_BALANCE[0]
DEFAULT_DELINQUENCY_MONTHS: Final = _DEFAULT_DELINQUENCY

_PURPOSE = {
    "P": "purchase",
    "N": "rate_term_refinance",
    "C": "cash_out_refinance",
    "R": "rate_term_refinance",
}
_OCCUPANCY = {"P": "owner_occupied", "S": "second_home", "I": "investment_property"}
#: As the aggregation maps it: broker, correspondent and third party are one level.
_CHANNEL = {
    "R": "retail",
    "B": "broker_or_correspondent",
    "C": "broker_or_correspondent",
    "T": "broker_or_correspondent",
}
_BUYER = {"Y": "first_time", "N": "repeat"}


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
            "The layout spec is public even though the data is not:\n"
            "  https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset"
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
    panel["origination_period"] = panel["period"] - panel["age"]
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
            "original_ltv": origination["original_ltv"],
            "debt_to_income": origination["original_dti"],
            "original_balance": origination["original_upb"],
            "log_original_balance": np.log(origination["original_upb"]),
            "note_rate": origination["original_interest_rate"],
            "purpose": origination["loan_purpose"].map(_PURPOSE),
            "occupancy": origination["occupancy_status"].map(_OCCUPANCY),
            "channel": origination["channel"].map(_CHANNEL),
            "region": origination["property_state"].map(_STATE_TO_REGION),
            # Mapped as the aggregation maps it: a 9, "not available", drops the loan rather than
            # reading as a repeat buyer.
            "buyer_type": origination["first_time_homebuyer_indicator"].map(_BUYER),
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
