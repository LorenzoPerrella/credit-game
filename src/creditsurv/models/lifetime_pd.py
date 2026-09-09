"""Lifetime probability of default, its term structure, and macro scenarios.

Three quantities matter in practice, and they are not the same thing:

**12-month PD** -- default probability over the next year, conditional on the loan
performing today. The IFRS 9 stage 1 measure.

**Lifetime PD** -- default probability over the remaining life, on the same
condition. The stage 2 measure.

**The PD term structure** -- the marginal probability of defaulting in each future
month. Two portfolios can share a lifetime PD and differ completely here, and the
difference decides when losses arrive.

All three are conditional on surviving to the reporting date. lifelines exposes
``conditional_after`` for exactly that, and it is used where covariates are fixed.
It is *not* enough when they are not.

## Why time-varying covariates change the calculation

With time-varying covariates a lifetime PD cannot be read off a single predicted
curve, because the curve depends on covariate values that do not exist yet. Two of
them -- ``cltv_drift`` and ``unemp_gap`` -- are zero at origination *by
construction*, so holding covariates at their current values quietly assumes house
prices never move again and unemployment never changes.

That assumption is not conservative, it is simply wrong, and it flatters the
answer: measured against Kaplan-Meier it overstated five-year survival by eight
percentage points. So the covariates are projected forward and the monthly hazards
chained:

    S(t) = prod over a < t of (1 - h(a, x(a)))

which is what :func:`survival_along_path` does. Most implementations freeze the
covariates instead; this is the difference between a scenario-responsive model and
one that merely looks like one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from creditsurv.data.panel import AGE, LOAN_ID
from creditsurv.features import add_macro_covariates
from creditsurv.models.aft import episode_hazards

if TYPE_CHECKING:
    from collections.abc import Sequence

    from creditsurv.models.aft import FitResult


def hazard_paths(
    result: FitResult,
    panel: pd.DataFrame,
    covariates: Sequence[str],
) -> pd.DataFrame:
    """Monthly hazard for every loan-month, as a loan-by-age table."""
    frame = panel.loc[:, list(covariates)]
    ages = panel[AGE].to_numpy(dtype=int)
    hazard = episode_hazards(result, frame, ages)

    table = pd.DataFrame({LOAN_ID: panel[LOAN_ID].to_numpy(), AGE: ages, "hazard": hazard})
    return table.pivot(index=LOAN_ID, columns=AGE, values="hazard")


def survival_along_path(
    result: FitResult,
    panel: pd.DataFrame,
    covariates: Sequence[str],
) -> pd.DataFrame:
    """Survival for each loan, chained along its own covariate path.

    Returns loans as rows and loan ages as columns, where column ``a`` holds the
    probability of surviving *through* age ``a``.
    """
    hazards = hazard_paths(result, panel, covariates)
    survival = (1.0 - hazards).cumprod(axis=1)
    survival.columns = pd.Index([int(age) + 1 for age in hazards.columns], name=AGE)
    return survival


def conditional_pd(
    survival: pd.DataFrame,
    *,
    as_of_age: int = 0,
    horizon_months: int | None = None,
) -> pd.Series:
    """Default probability over ``horizon_months``, given performing at ``as_of_age``.

    ``1 - S(as_of + horizon) / S(as_of)``. With ``as_of_age = 0`` this is the PD
    quoted at origination; with a later age it is the reporting-date measure IFRS 9
    asks for, on a loan that has already survived that long.
    """
    columns = list(survival.columns)
    if as_of_age not in {0, *columns}:
        message = f"as_of_age {as_of_age} is outside the projected horizon."
        raise ValueError(message)

    end_age = columns[-1] if horizon_months is None else as_of_age + horizon_months
    if end_age not in columns:
        message = f"Horizon reaches age {end_age}, beyond the projected {columns[-1]}."
        raise ValueError(message)

    survival_at_end = survival[end_age]
    survival_at_start = 1.0 if as_of_age == 0 else survival[as_of_age]
    return (1.0 - survival_at_end / survival_at_start).rename("pd")


def pd_term_structure(survival: pd.DataFrame) -> pd.DataFrame:
    """Marginal and cumulative default probability by loan age.

    The marginal column answers "of the loans written today, what share default in
    month ``a``" -- unconditional on surviving to ``a``, which is what a loss
    forecast needs. The hazard column is the conditional rate instead.

    The survival table must be **balanced**: every loan present at every age. A
    ragged table produced from an observed panel silently averages different loans
    at different ages, and the composition change looks like a falling hazard near
    the horizon when it is only the longest-observed loans remaining. Use
    :func:`project_panel` to build a balanced path.
    """
    missing = int(survival.isna().to_numpy().sum())
    if missing:
        message = (
            f"Survival table has {missing} missing cells, so loans enter and leave "
            "between ages and the term structure would mix different populations. "
            "Build a balanced path with project_panel first."
        )
        raise ValueError(message)

    mean_survival = survival.mean(axis=0)
    cumulative = 1.0 - mean_survival
    marginal = cumulative.diff().fillna(cumulative.iloc[0])
    hazard = marginal / mean_survival.shift(1).fillna(1.0)

    return pd.DataFrame(
        {
            "survival": mean_survival,
            "cumulative_pd": cumulative,
            "marginal_pd": marginal,
            "hazard": hazard,
        }
    )


# --------------------------------------------------------------------------------------
# Macroeconomic scenarios
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """A forward path for the macro series, expressed as shocks to the last value.

    Each entry maps a series to a sequence of deviations applied from the first
    projected month onwards. Shorter sequences hold their final value, so a
    permanent shock needs only its ramp.

    ``hpi`` shocks are proportional -- a house price index is a level, and a
    fifteen percent fall means the same thing at any index value. Everything else
    is additive, since rates and indices are already in comparable units.
    """

    name: str
    shocks: dict[str, Sequence[float]] = field(default_factory=dict)
    proportional: frozenset[str] = frozenset({"hpi"})


#: Nothing happens: every series holds its last observed value. A random walk is
#: the honest default when no forecast is available, and it is what the
#: unconditional backtest uses.
BASELINE = Scenario(name="baseline")

#: A recession resembling 2008 in shape rather than magnitude: unemployment climbs
#: over a year and stays high, house prices fall for two years, credit tightens.
ADVERSE = Scenario(
    name="adverse",
    shocks={
        "unemployment_rate": [*np.linspace(0.0, 4.0, 12), *([4.0] * 24)],
        "hpi": [*np.linspace(0.0, -0.20, 24), *([-0.20] * 12)],
        "nfci": [*np.linspace(0.0, 1.5, 6), *([1.5] * 30)],
        "mortgage_rate_30y": [*np.linspace(0.0, 1.0, 6), *([1.0] * 30)],
    },
)


def extend_macro(macro: pd.DataFrame, months: int, scenario: Scenario = BASELINE) -> pd.DataFrame:
    """Append a projected path to an observed macro panel.

    The projection is a random walk from the last observation, plus the scenario's
    shocks. A random walk is not a forecast, and is not offered as one -- it is the
    assumption that makes the *relative* effect of a scenario interpretable without
    smuggling in a view on the economy.
    """
    if months <= 0:
        message = "months must be positive."
        raise ValueError(message)

    last = macro.iloc[-1]
    future_index = pd.period_range(macro.index[-1] + 1, periods=months, freq="M")
    projected = pd.DataFrame(
        np.tile(last.to_numpy(dtype=float), (months, 1)),
        index=future_index,
        columns=macro.columns,
    )

    for series, path in scenario.shocks.items():
        if series not in projected.columns:
            continue
        shock = np.asarray(path, dtype=float)
        if len(shock) < months:
            shock = np.concatenate([shock, np.full(months - len(shock), shock[-1])])
        shock = shock[:months]
        if series in scenario.proportional:
            projected[series] = float(last[series]) * (1.0 + shock)
        else:
            projected[series] = float(last[series]) + shock

    return pd.concat([macro, projected])


def project_panel(
    loans: pd.DataFrame,
    macro: pd.DataFrame,
    *,
    as_of: pd.Period,
    horizon_months: int,
) -> pd.DataFrame:
    """Build the forward loan-month panel a book would experience from ``as_of``.

    ``loans`` is one row per loan carrying its origination attributes and its
    current ``age`` at the reporting date. Calendar time advances from ``as_of``
    while loan age advances from wherever each loan already is, which is what makes
    the panel forward-looking.

    Getting this wrong is easy and quiet. Advancing calendar time from each loan's
    *origination* instead replays its actual history, so a projected macro path
    appended after the reporting date is never reached and every scenario returns
    the same answer -- a model that looks stable when it is simply not being asked
    the question.
    """
    if AGE not in loans.columns:
        message = f"loans must carry a {AGE!r} column giving each loan's age at as_of."
        raise ValueError(message)

    last_period = as_of + (horizon_months - 1)
    if last_period > macro.index.max():
        message = (
            f"Macro panel ends {macro.index.max()} but the horizon reaches {last_period}. "
            "Extend it with extend_macro first."
        )
        raise ValueError(message)

    steps = np.arange(horizon_months, dtype=np.int64)
    projected = loans.loc[loans.index.repeat(horizon_months)].reset_index(drop=True)
    current_age = projected[AGE].to_numpy(dtype=np.int64)
    projected[AGE] = current_age + np.tile(steps, len(loans))
    projected["period"] = pd.PeriodIndex(
        [as_of + int(step) for step in np.tile(steps, len(loans))], freq="M"
    )
    return add_macro_covariates(projected, macro)


def scenario_lifetime_pd(
    result: FitResult,
    loans: pd.DataFrame,
    macro: pd.DataFrame,
    covariates: Sequence[str],
    *,
    horizon_months: int = 60,
    scenarios: Sequence[Scenario] = (BASELINE, ADVERSE),
) -> pd.DataFrame:
    """Lifetime PD for each loan under each scenario.

    The covariates are rebuilt from the projected macro path before scoring, which
    is the whole point: a model whose covariates are frozen produces the same
    number under every scenario and looks stable when it is merely blind.
    """
    as_of = macro.index.max() + 1
    results = {}
    for scenario in scenarios:
        extended = extend_macro(macro, horizon_months + 1, scenario)
        panel = project_panel(loans, extended, as_of=as_of, horizon_months=horizon_months)
        survival = survival_along_path(result, panel, covariates)
        results[scenario.name] = conditional_pd(survival)

    return pd.DataFrame(results)
