"""Synthetic loan panel driven by real macroeconomic history.

No loan-level survival panel is public without registration, so the loan book is
simulated. The macro paths it responds to are real, pulled from FRED, which is
what gives the simulation its shape: vintages originated into 2006 meet the
housing collapse, vintages originated into 2019 meet the pandemic.

The generating process is deliberately *the exact estimand of the model that
will be fitted to it*. For an episode covering loan age ``(a, a+1]`` with
covariates ``x(a)``::

    lambda(a) = exp(x(a)'beta)
    rho       = exp(log_rho)
    P(default in the month | alive at a) = 1 - exp(-[((a+1)/lambda)**rho - (a/lambda)**rho])

which is precisely the per-episode contribution the interval-censored likelihood
maximises. That equivalence is the point: it makes the recovery test in
``tests/test_recovery.py`` decisive. A recovered coefficient that misses its
confidence interval indicates a bug, not a misspecified model.

Prepayment is simulated as a second, independent risk and then treated as
censoring, because the project models default only. That is an assumption, not a
fact -- see ``docs/methodology.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from creditsurv.features import add_macro_covariates

if TYPE_CHECKING:
    from numpy.random import Generator

#: Longest loan age simulated, in months. Five years of seasoning is enough to
#: cover the peak of the mortgage default curve without inflating the panel.
MAX_AGE_MONTHS: int = 60


@dataclass(frozen=True)
class TrueParams:
    """Coefficients of the data-generating process.

    Stored alongside every generated panel so the estimator can be checked
    against the truth rather than against its own plausibility.

    Coefficients act on ``log(lambda)``, the accelerated-failure-time scale, so a
    *positive* value lengthens survival and therefore lowers risk. Categorical
    reference levels carry no entry and are implicitly zero, matching the
    treatment coding the model uses.
    """

    intercept: float
    log_rho: float
    continuous: dict[str, float]
    categorical: dict[str, dict[str, float]] = field(default_factory=dict)
    prepayment_intercept: float = 5.1
    prepayment_log_rho: float = 0.15
    prepayment_refi_sensitivity: float = 0.55


#: Signs follow mortgage credit intuition: higher credit score and larger
#: balances survive longer; higher leverage, higher debt burden, a wider
#: origination spread, falling house prices, rising unemployment and tighter
#: financial conditions all shorten survival. The intercept is set so lifetime
#: default lands in the low single digits, as it does for agency-quality loans.
DEFAULT_PARAMS = TrueParams(
    intercept=6.72,
    log_rho=0.26,
    continuous={
        "fico_s": 0.34,
        "orig_ltv": -0.011,
        "dti": -0.013,
        "log_orig_upb": 0.06,
        "orig_spread": -0.22,
        "cltv_drift": -0.020,
        "unemp_gap": -0.105,
        "refi_incentive": 0.045,
        "nfci_lagged": -0.185,
    },
    categorical={
        "purpose": {"refinance_rate_term": 0.06, "refinance_cashout": -0.28},
        "occupancy": {"second_home": -0.14, "investor": -0.31},
        "channel": {"broker": -0.19, "correspondent": -0.11},
        "region": {"Northeast": 0.05, "Midwest": 0.02, "West": -0.07},
        "first_time_buyer": {"Y": -0.09},
    },
)

#: Portfolio mix. Proportions are plausible for an agency book rather than
#: calibrated to any particular disclosure.
_CATEGORICAL_MIX: dict[str, dict[str, float]] = {
    "purpose": {"purchase": 0.52, "refinance_rate_term": 0.30, "refinance_cashout": 0.18},
    "occupancy": {"owner_occupied": 0.87, "second_home": 0.05, "investor": 0.08},
    "channel": {"retail": 0.55, "broker": 0.18, "correspondent": 0.27},
    "region": {"South": 0.38, "West": 0.24, "Midwest": 0.21, "Northeast": 0.17},
    "first_time_buyer": {"N": 0.76, "Y": 0.24},
}


def _draw_categorical(name: str, size: int, rng: Generator) -> pd.Categorical:
    levels = list(_CATEGORICAL_MIX[name])
    weights = np.array(list(_CATEGORICAL_MIX[name].values()))
    drawn = rng.choice(levels, size=size, p=weights / weights.sum())
    return pd.Categorical(drawn, categories=levels)


def generate_originations(
    n_loans: int,
    macro: pd.DataFrame,
    rng: Generator,
    *,
    max_age_months: int = MAX_AGE_MONTHS,
) -> pd.DataFrame:
    """Draw origination records spread across the available macro history.

    Credit quality is correlated rather than independent: a Gaussian copula ties
    low credit scores to high leverage and high debt burden, which is what makes
    the covariates collinear enough to be worth selecting between.
    """
    # Leave room for the macro lag at the start; vintages near the end are
    # retained and simply observed for fewer months.
    usable = macro.index[6:]
    orig_period = pd.PeriodIndex(rng.choice(usable, size=n_loans), freq="M")

    correlation = np.array(
        [
            [1.00, -0.45, -0.35],
            [-0.45, 1.00, 0.30],
            [-0.35, 0.30, 1.00],
        ]
    )
    factors = rng.multivariate_normal(np.zeros(3), correlation, size=n_loans)

    credit_score = np.clip(700 + 50 * factors[:, 0], 580, 820)
    orig_ltv = np.clip(75 + 12 * factors[:, 1], 30, 100)
    dti = np.clip(36 + 8 * factors[:, 2], 10, 55)
    orig_upb = np.exp(12.2 + 0.42 * rng.normal(size=n_loans))

    # Risk-based pricing: weaker credit pays a wider spread over the market rate.
    fico_s = (credit_score - 700) / 50
    orig_spread = 0.35 - 0.22 * fico_s + 0.010 * (orig_ltv - 75) + 0.18 * rng.normal(size=n_loans)
    market_rate = pd.Series(orig_period).map(macro["mortgage_rate_30y"]).to_numpy(dtype=float)

    loans = pd.DataFrame(
        {
            "loan_id": np.arange(n_loans, dtype=np.int64),
            "orig_period": orig_period,
            "credit_score": credit_score,
            "fico_s": fico_s,
            "orig_ltv": orig_ltv,
            "dti": dti,
            "orig_upb": orig_upb,
            "log_orig_upb": np.log(orig_upb),
            "orig_spread": orig_spread,
            "note_rate": market_rate + orig_spread,
        }
    )
    for name in _CATEGORICAL_MIX:
        loans[name] = _draw_categorical(name, n_loans, rng)

    loans["max_age"] = max_age_months
    return loans


def _linear_predictor(panel: pd.DataFrame, params: TrueParams) -> np.ndarray:
    predictor = np.full(len(panel), params.intercept, dtype=float)
    for name, coefficient in params.continuous.items():
        predictor += coefficient * panel[name].to_numpy(dtype=float)
    for name, effects in params.categorical.items():
        # Reference levels are absent from `effects` and contribute zero.
        predictor += panel[name].map(effects).fillna(0.0).to_numpy(dtype=float)
    return predictor


def _monthly_hazard(age: np.ndarray, scale: np.ndarray, log_rho: float) -> np.ndarray:
    """Weibull conditional failure probability over the episode ``(a, a+1]``."""
    rho = np.exp(log_rho)
    increment = ((age + 1.0) / scale) ** rho - (age / scale) ** rho
    hazard: np.ndarray = 1.0 - np.exp(-increment)
    return hazard


def simulate_panel(
    loans: pd.DataFrame,
    macro: pd.DataFrame,
    params: TrueParams,
    rng: Generator,
    *,
    max_age_months: int = MAX_AGE_MONTHS,
) -> pd.DataFrame:
    """Expand originations into a loan-month panel and draw outcomes.

    The expansion is built in full and the hazards evaluated in one vectorised
    pass, then each loan is truncated at its first terminating month. This is far
    faster than stepping loan by loan and produces the same distribution.
    """
    ages = np.arange(max_age_months, dtype=np.int64)
    panel = loans.loc[loans.index.repeat(max_age_months)].reset_index(drop=True)
    panel["age"] = np.tile(ages, len(loans))
    panel["period"] = panel["orig_period"] + panel["age"]

    # Administrative censoring: the macro panel ends, so observation ends.
    panel = panel[panel["period"] <= macro.index.max()].reset_index(drop=True)
    panel = add_macro_covariates(panel, macro)

    age_values = panel["age"].to_numpy(dtype=float)
    default_scale = np.exp(_linear_predictor(panel, params))
    default_hazard = _monthly_hazard(age_values, default_scale, params.log_rho)

    prepayment_scale = np.exp(
        params.prepayment_intercept
        - params.prepayment_refi_sensitivity * panel["refi_incentive"].to_numpy(dtype=float)
    )
    prepayment_hazard = _monthly_hazard(age_values, prepayment_scale, params.prepayment_log_rho)

    defaulted = rng.random(len(panel)) < default_hazard
    prepaid = rng.random(len(panel)) < prepayment_hazard

    panel["default_fires"] = defaulted
    # Default takes precedence when both fire in the same month. The choice only
    # affects the handful of loans where two independent draws coincide.
    panel["terminates"] = defaulted | prepaid

    panel = panel.sort_values(["loan_id", "age"], kind="stable").reset_index(drop=True)
    terminal_age = panel.loc[panel["terminates"]].groupby("loan_id", observed=True)["age"].min()
    panel["terminal_age"] = panel["loan_id"].map(terminal_age)

    still_open = panel["terminal_age"].isna()
    keep = still_open | (panel["age"] <= panel["terminal_age"])
    panel = panel[keep].reset_index(drop=True)

    is_terminal_row = (~still_open[keep.to_numpy()].to_numpy()) & (
        panel["age"].to_numpy() == panel["terminal_age"].to_numpy()
    )
    panel["event"] = is_terminal_row & panel["default_fires"].to_numpy()
    panel["prepaid"] = is_terminal_row & ~panel["default_fires"].to_numpy()

    return panel.drop(columns=["default_fires", "terminates", "terminal_age", "max_age"])


def build_synthetic_panel(
    macro: pd.DataFrame,
    *,
    n_loans: int = 5000,
    seed: int = 42,
    params: TrueParams = DEFAULT_PARAMS,
    max_age_months: int = MAX_AGE_MONTHS,
) -> tuple[pd.DataFrame, TrueParams]:
    """Generate a loan-month panel and return it with the parameters that made it."""
    rng = np.random.default_rng(seed)
    loans = generate_originations(n_loans, macro, rng, max_age_months=max_age_months)
    panel = simulate_panel(loans, macro, params, rng, max_age_months=max_age_months)
    return panel, params
