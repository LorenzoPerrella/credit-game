"""Fixture data written in the Freddie Mac file format.

Tests read the same pipe-delimited files the real loader reads, rather than a
panel handed to them directly. That costs a little machinery and buys two things:
the loader is exercised on every test that needs data, and there is no second
schema to keep in step with the first.

:func:`simulate_book` writes a book drawn from a **known** data-generating process,
which is what makes ``tests/test_recovery.py`` decisive. The process is deliberately
the exact estimand of the interval-censored likelihood: for an episode covering loan
age ``(a, a+1]`` with covariates ``x(a)``::

    lambda(a) = exp(x(a)'beta),  rho = exp(log_rho)
    P(default in the month | alive at a) = 1 - exp(-[((a+1)/lambda)**rho - (a/lambda)**rho])

so a recovered coefficient that misses its confidence interval indicates a bug
rather than a misspecified model. Because the draw is now written out and read back
through the loader, the same test also covers the parsing, the missing-value
sentinels and the truncation at the first terminating month.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from creditsurv.data.freddiemac import (
    ORIGINATION_COLUMNS,
    PERFORMANCE_COLUMNS,
    load_sample,
)
from creditsurv.features import add_macro_covariates

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.random import Generator

MAX_AGE_MONTHS = 60

#: Our canonical levels mapped back to the codes the dataset actually stores.
_PURPOSE_CODE = {"purchase": "P", "refinance_rate_term": "N", "refinance_cashout": "C"}
_OCCUPANCY_CODE = {"owner_occupied": "P", "second_home": "S", "investor": "I"}
_CHANNEL_CODE = {"retail": "R", "broker": "B", "correspondent": "C"}
_REGION_STATE = {"Northeast": "NY", "Midwest": "IL", "South": "TX", "West": "CA"}

_CATEGORICAL_MIX: dict[str, dict[str, float]] = {
    "purpose": {"purchase": 0.52, "refinance_rate_term": 0.30, "refinance_cashout": 0.18},
    "occupancy": {"owner_occupied": 0.87, "second_home": 0.05, "investor": 0.08},
    "channel": {"retail": 0.55, "broker": 0.18, "correspondent": 0.27},
    "region": {"South": 0.38, "West": 0.24, "Midwest": 0.21, "Northeast": 0.17},
    "first_time_buyer": {"N": 0.76, "Y": 0.24},
}


@dataclass(frozen=True)
class TrueParams:
    """Coefficients of the data-generating process.

    They act on ``log(lambda)``, the accelerated-failure-time scale, so a *positive*
    value lengthens survival and therefore lowers risk. Categorical reference levels
    carry no entry and are implicitly zero, matching the treatment coding the model
    uses.
    """

    intercept: float
    log_rho: float
    continuous: dict[str, float]
    categorical: dict[str, dict[str, float]] = field(default_factory=dict)
    prepayment_intercept: float = 5.1
    prepayment_log_rho: float = 0.15


DEFAULT_PARAMS = TrueParams(
    intercept=5.4,
    log_rho=0.26,
    continuous={"fico_s": 0.34, "cltv_drift": -0.020, "unemp_gap": -0.105},
    categorical={},
)


# --------------------------------------------------------------------------------------
# Writing individual records
# --------------------------------------------------------------------------------------


def origination_row(
    loan_id: str,
    *,
    fico: str = "740",
    dti: str = "32",
    ltv: str = "78",
    upb: str = "210000",
    rate: str = "4.25",
    purpose: str = "P",
    occupancy: str = "P",
    channel: str = "R",
    state: str = "CA",
    first_time: str = "N",
    first_payment: str = "201503",
    term: str = "360",
) -> str:
    """One line of ``orig_YYYYQn.txt``. Unset fields stay empty, as they do upstream."""
    values = dict.fromkeys(ORIGINATION_COLUMNS, "")
    values.update(
        classic_fico=fico,
        first_payment_date=first_payment,
        first_time_homebuyer_indicator=first_time,
        occupancy_status=occupancy,
        original_dti=dti,
        original_upb=upb,
        original_ltv=ltv,
        original_cltv=ltv,
        original_interest_rate=rate,
        channel=channel,
        property_state=state,
        loan_identifier=loan_id,
        loan_purpose=purpose,
        original_loan_term=term,
    )
    return "|".join(values[name] for name in ORIGINATION_COLUMNS)


def performance_row(
    loan_id: str,
    period: str,
    age: str,
    *,
    delinquency: str = "0",
    zero_balance: str = "",
    upb: str = "200000",
) -> str:
    """One line of ``perf_YYYYQn.txt``.

    An unset field is written empty, which is what the real files contain and what
    pyarrow then reads as null. That distinction matters: SQL treating an absent
    zero-balance code as null rather than as "not a default" propagates a nullable
    event flag all the way to the fitter.
    """
    values = dict.fromkeys(PERFORMANCE_COLUMNS, "")
    values.update(
        loan_identifier=loan_id,
        period=period,
        current_loan_delinquency_status=delinquency,
        loan_age=age,
        zero_balance_code=zero_balance,
        current_actual_upb=upb,
    )
    return "|".join(values[name] for name in PERFORMANCE_COLUMNS)


def write_files(
    directory: Path, origination: list[str], performance: list[str], *, tag: str = "2015Q1"
) -> tuple[Path, Path]:
    """Write both files under the dataset's own naming convention."""
    directory.mkdir(parents=True, exist_ok=True)
    orig_path = directory / f"orig_{tag}.txt"
    perf_path = directory / f"perf_{tag}.txt"
    orig_path.write_text("\n".join(origination) + "\n")
    perf_path.write_text("\n".join(performance) + "\n")
    return orig_path, perf_path


# --------------------------------------------------------------------------------------
# Simulating a book from a known process
# --------------------------------------------------------------------------------------


def _draw_categorical(name: str, size: int, rng: Generator) -> np.ndarray:
    levels = list(_CATEGORICAL_MIX[name])
    weights = np.array(list(_CATEGORICAL_MIX[name].values()))
    return rng.choice(levels, size=size, p=weights / weights.sum())


def _originations(n_loans: int, macro: pd.DataFrame, rng: Generator) -> pd.DataFrame:
    """Draw loans with correlated credit quality.

    A Gaussian copula ties weak scores to high leverage and high debt burden, so the
    covariates are collinear enough for selection to be a real exercise.
    """
    usable = macro.index[6:]
    orig_period = pd.PeriodIndex(rng.choice(usable, size=n_loans), freq="M")

    correlation = np.array([[1.00, -0.45, -0.35], [-0.45, 1.00, 0.30], [-0.35, 0.30, 1.00]])
    factors = rng.multivariate_normal(np.zeros(3), correlation, size=n_loans)

    credit_score = np.clip(700 + 50 * factors[:, 0], 580, 820)
    market_rate = pd.Series(orig_period).map(macro["mortgage_rate_30y"]).to_numpy(dtype=float)

    loans = pd.DataFrame(
        {
            "loan_id": [f"F{index:011d}" for index in range(n_loans)],
            "orig_period": orig_period,
            "credit_score": credit_score,
            "fico_s": (credit_score - 700.0) / 50.0,
            "orig_ltv": np.clip(75 + 12 * factors[:, 1], 30, 100),
            "dti": np.clip(36 + 8 * factors[:, 2], 10, 55),
            "orig_upb": np.exp(12.2 + 0.42 * rng.normal(size=n_loans)),
            "note_rate": market_rate + 0.35 - 0.22 * factors[:, 0],
        }
    )
    for name in _CATEGORICAL_MIX:
        loans[name] = _draw_categorical(name, n_loans, rng)
    return loans


def _monthly_hazard(age: np.ndarray, scale: np.ndarray, log_rho: float) -> np.ndarray:
    """Weibull conditional failure probability over the episode ``(a, a+1]``."""
    rho = np.exp(log_rho)
    increment = ((age + 1.0) / scale) ** rho - (age / scale) ** rho
    hazard: np.ndarray = 1.0 - np.exp(-increment)
    return hazard


def _linear_predictor(panel: pd.DataFrame, params: TrueParams) -> np.ndarray:
    predictor = np.full(len(panel), params.intercept, dtype=float)
    for name, coefficient in params.continuous.items():
        predictor += coefficient * panel[name].to_numpy(dtype=float)
    for name, effects in params.categorical.items():
        predictor += panel[name].map(effects).fillna(0.0).to_numpy(dtype=float)
    return predictor


def simulate_book(
    directory: Path,
    macro: pd.DataFrame,
    *,
    n_loans: int = 800,
    seed: int = 17,
    params: TrueParams = DEFAULT_PARAMS,
    max_age_months: int = MAX_AGE_MONTHS,
    with_prepayment: bool = False,
) -> tuple[Path, Path, TrueParams]:
    """Simulate a book from ``params`` and write it in the dataset's own format.

    Expansion and hazard evaluation happen in one vectorised pass, then each loan is
    truncated at its first terminating month; stepping loan by loan gives the same
    distribution far more slowly.
    """
    rng = np.random.default_rng(seed)
    loans = _originations(n_loans, macro, rng)

    ages = np.arange(max_age_months, dtype=np.int64)
    panel = loans.loc[loans.index.repeat(max_age_months)].reset_index(drop=True)
    panel["age"] = np.tile(ages, len(loans))
    panel["period"] = panel["orig_period"] + panel["age"]
    panel = panel[panel["period"] <= macro.index.max()].reset_index(drop=True)
    panel = add_macro_covariates(panel, macro)

    age_values = panel["age"].to_numpy(dtype=float)
    scale = np.exp(_linear_predictor(panel, params))
    defaulted = rng.random(len(panel)) < _monthly_hazard(age_values, scale, params.log_rho)

    if with_prepayment:
        prepayment_scale = np.exp(np.full(len(panel), params.prepayment_intercept))
        prepaid = rng.random(len(panel)) < _monthly_hazard(
            age_values, prepayment_scale, params.prepayment_log_rho
        )
    else:
        prepaid = np.zeros(len(panel), dtype=bool)

    panel["defaulted"] = defaulted
    panel["prepaid"] = prepaid & ~defaulted
    panel = panel.sort_values(["loan_id", "age"], kind="stable").reset_index(drop=True)

    terminating = panel["defaulted"] | panel["prepaid"]
    terminal_age = panel.loc[terminating].groupby("loan_id", observed=True)["age"].min()
    mapped = panel["loan_id"].map(terminal_age)
    panel = panel[mapped.isna() | (panel["age"] <= mapped)].reset_index(drop=True)

    return _write_book(directory, loans, panel, params)


def _write_book(
    directory: Path, loans: pd.DataFrame, panel: pd.DataFrame, params: TrueParams
) -> tuple[Path, Path, TrueParams]:
    """Render a simulated book as the two pipe-delimited files."""
    kept = set(panel["loan_id"])
    origination = [
        origination_row(
            str(row.loan_id),
            fico=f"{row.credit_score:.0f}",
            dti=f"{row.dti:.0f}",
            ltv=f"{row.orig_ltv:.0f}",
            upb=f"{row.orig_upb:.0f}",
            rate=f"{row.note_rate:.3f}",
            purpose=_PURPOSE_CODE[str(row.purpose)],
            occupancy=_OCCUPANCY_CODE[str(row.occupancy)],
            channel=_CHANNEL_CODE[str(row.channel)],
            state=_REGION_STATE[str(row.region)],
            first_time=str(row.first_time_buyer),
            first_payment=str(row.orig_period).replace("-", ""),
        )
        for row in loans.itertuples(index=False)
        if row.loan_id in kept
    ]

    performance = [
        performance_row(
            str(row.loan_id),
            str(row.period).replace("-", ""),
            str(row.age),
            # Three missed payments is the dataset's own default marker; the
            # terminating month also carries an REO zero-balance code, because the
            # loader is required to honour either.
            delinquency="3" if row.defaulted else "0",
            zero_balance="09" if row.defaulted else ("01" if row.prepaid else ""),
        )
        for row in panel.itertuples(index=False)
    ]

    orig_path, perf_path = write_files(directory, origination, performance)
    return orig_path, perf_path, params


def build_panel(
    directory: Path,
    macro: pd.DataFrame,
    **kwargs: object,
) -> tuple[pd.DataFrame, TrueParams]:
    """Simulate a book, write it, and read it back through the real loader.

    The round trip is the point. A test that received a panel directly would never
    touch the parsing, the missing-value sentinels or the truncation at the first
    terminating month, and those are where a loader breaks.
    """
    orig_path, perf_path, params = simulate_book(directory, macro, **kwargs)  # type: ignore[arg-type]
    panel = add_macro_covariates(load_sample(orig_path, perf_path), macro)
    return panel, params


def write_archives(
    root: Path,
    year: int,
    quarters: dict[int, tuple[list[str], list[str]]],
) -> Path:
    """Build the nested archive layout the real download has.

    ``historical_data_YYYY.zip`` containing ``historical_data_YYYYQn.zip``, each
    containing ``orig_YYYYQn.txt`` and ``perf_YYYYQn.txt``. Tests build this rather
    than a flat directory because the nesting is exactly what the ingest has to
    handle, and a flat fixture would exercise none of it.
    """
    import zipfile

    root.mkdir(parents=True, exist_ok=True)
    outer_path = root / f"historical_data_{year}.zip"
    with zipfile.ZipFile(outer_path, "w") as outer:
        for quarter, (origination, performance) in sorted(quarters.items()):
            tag = f"{year}Q{quarter}"
            inner_bytes = root / f"_inner_{tag}.zip"
            with zipfile.ZipFile(inner_bytes, "w") as inner:
                inner.writestr(f"orig_{tag}.txt", "\n".join(origination) + "\n")
                inner.writestr(f"perf_{tag}.txt", "\n".join(performance) + "\n")
            outer.write(inner_bytes, f"historical_data_{tag}.zip")
            inner_bytes.unlink()
    return outer_path
