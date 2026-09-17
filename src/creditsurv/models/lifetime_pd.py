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

from creditsurv.data.panel import AGE, LOAN_ID, WEIGHT
from creditsurv.features import add_macro_covariates
from creditsurv.models.aft import episode_hazards

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from creditsurv.models.aft import FitResult


def hazard_paths(
    result: FitResult,
    panel: pd.DataFrame,
    covariates: Sequence[str],
) -> pd.DataFrame:
    """Monthly hazard for every loan-month, as a loan-by-step table.

    Columns count **months since the start of each loan's own path**, not absolute
    loan age. For new business the two coincide, so the distinction is invisible
    there -- and it decides everything on a book of mixed seasoning. A portfolio
    holding one loan at age 3 and another at age 47 spans fifty age columns of
    which each loan fills twelve, so a table indexed by age is mostly empty and
    every horizon read off its final column returns nothing.
    """
    frame = panel.loc[:, list(covariates)]
    ages = panel[AGE].to_numpy(dtype=int)
    hazard = episode_hazards(result, frame, ages)

    table = pd.DataFrame(
        {LOAN_ID: panel[LOAN_ID].to_numpy(), AGE: ages, "hazard": hazard}
    ).sort_values([LOAN_ID, AGE], kind="stable")
    table["step"] = table.groupby(LOAN_ID, observed=True).cumcount()
    return table.pivot(index=LOAN_ID, columns="step", values="hazard")


def survival_along_path(
    result: FitResult,
    panel: pd.DataFrame,
    covariates: Sequence[str],
) -> pd.DataFrame:
    """Survival for each loan, chained along its own covariate path.

    Loans are rows; columns count months from the start of the projection, so
    column ``k`` is the probability of still performing ``k`` months on. For loans
    projected from origination that is also their age, which is why the difference
    only shows up on a seasoned book.
    """
    hazards = hazard_paths(result, panel, covariates)
    survival = (1.0 - hazards).cumprod(axis=1)
    survival.columns = pd.Index([int(step) + 1 for step in hazards.columns], name="month")
    return survival


def conditional_pd(
    survival: pd.DataFrame,
    *,
    as_of_month: int = 0,
    horizon_months: int | None = None,
) -> pd.Series:
    """Default probability over ``horizon_months``, given performing at ``as_of_month``.

    ``1 - S(as_of + horizon) / S(as_of)``, where both are counted in months from
    the start of the projection. With ``as_of_month = 0`` this is the PD quoted at
    the reporting date; a later value conditions on the loan having survived that
    much further, which is the measure IFRS 9 asks for on a seasoned loan.
    """
    columns = list(survival.columns)
    if as_of_month not in {0, *columns}:
        message = f"as_of_month {as_of_month} is outside the projected horizon."
        raise ValueError(message)

    end_month = columns[-1] if horizon_months is None else as_of_month + horizon_months
    if end_month not in columns:
        message = f"Horizon reaches month {end_month}, beyond the projected {columns[-1]}."
        raise ValueError(message)

    survival_at_end = survival[end_month]
    survival_at_start = 1.0 if as_of_month == 0 else survival[as_of_month]
    return (1.0 - survival_at_end / survival_at_start).rename("pd")


def origination_book(encoded: pd.DataFrame, macro: pd.DataFrame, size: int) -> pd.DataFrame:
    """The commonest origination profiles, as a book to be scored from today.

    Calibration asks what the regressors are worth on a book, so the book has to be
    one that exists. Cells at age zero are exactly the origination profiles the
    portfolio was written in, and their counts say how much of it each accounts for
    -- so the largest ``size`` of them, carried with their weights, describe the book
    far better than the same number of individual loans drawn arbitrarily.

    They are then dated to the present: age zero at the last macro period, which asks
    what these profiles would be worth if written today rather than replaying the
    history they were actually written in.
    """
    book = encoded.loc[encoded[AGE] == 0].nlargest(size, WEIGHT).reset_index(drop=True)
    book[AGE] = 0
    book["period"] = macro.index.max() + 1
    # Named here rather than left to the projection, because the weights have to be
    # indexed by the same label the scored results come back under.
    book[LOAN_ID] = [f"row_{index:09d}" for index in range(len(book))]
    return book


def pd_term_structure(survival: pd.DataFrame) -> pd.DataFrame:
    """Marginal and cumulative default probability by loan age.

    The marginal column answers "of the loans on the books today, what share default
    in month ``k``" -- unconditional on surviving to ``a``, which is what a loss
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

#: A recession resembling 2008 in shape rather than magnitude, shocked on the series
#: the fitted model actually reads.
#:
#: The first version shocked unemployment, house prices, financial conditions and the
#: thirty-year mortgage rate. By the time the specification settled on ``cltv_drift``,
#: ``unemp_gap``, ``vix`` and ``inflation``, two of those four legs fed no covariate at
#: all and two of the model's covariates had no path -- ``vix``, the largest
#: standardised effect, among them. The published "adverse lifetime PD 1.46x baseline"
#: therefore understated the model's own sensitivity, which moves 6.6x in-sample between
#: 2005 and 2009. It was drift rather than a decision: the scenario predated the
#: specification by three days. ``tests/test_scenarios.py`` now fails if the two part.
#:
#: The selection on the whole training half then replaced the specification, and the same
#: test caught the scenario out again: ``vix`` was gone from the model, and ``nfci_lagged``,
#: ``policy_rate_gap``, ``sentiment`` and ``starts_growth`` had arrived with no path. The
#: volatility leg is removed and four legs are added, each the move its series made from
#: July 2007 to its extreme.
#:
#: Unemployment climbs four points over a year and holds; house prices fall a fifth over
#: two years; consumer prices fall two percent over a year, the deflation of 2009, which
#: ``inflation_gap`` reads as stress. Financial conditions tighten by 3.4 over sixteen
#: months, as the NFCI did to November 2008, and ease back over the next year. The policy
#: rate is cut by 95% over thirty months, as 5.26% became 0.11% by 2010 -- proportional,
#: so it meets the zero bound instead of crossing it. Consumer sentiment falls 39% over
#: sixteen months, as 90.4 became 55.3, and housing starts 65% over twenty-one, as 1.35
#: million became 478 thousand; both are levels, and proportional for the same reason
#: ``hpi`` and ``cpi`` are.
ADVERSE = Scenario(
    name="adverse",
    shocks={
        "unemployment_rate": [*np.linspace(0.0, 4.0, 12), *([4.0] * 24)],
        "hpi": [*np.linspace(0.0, -0.20, 24), *([-0.20] * 12)],
        "cpi": [*np.linspace(0.0, -0.02, 12), *([-0.02] * 24)],
        "nfci": [*np.linspace(0.0, 3.4, 16), *np.linspace(3.4, 0.0, 12), *([0.0] * 8)],
        "policy_rate": [*np.linspace(0.0, -0.95, 30), *([-0.95] * 6)],
        "sentiment": [*np.linspace(0.0, -0.39, 16), *([-0.39] * 20)],
        "housing_starts": [*np.linspace(0.0, -0.65, 21), *([-0.65] * 15)],
    },
    proportional=frozenset({"hpi", "cpi", "policy_rate", "sentiment", "housing_starts"}),
)


def scenario_legs(scenario: Scenario, sources: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """One row per shocked series: how far it moves, how soon, and what reads it.

    ``sources`` maps each covariate the model reads to the series it is built from. The
    calibration report used to describe the adverse path in prose, and the prose outlived
    the path: after the reselection it still promised a volatility spike to a model that no
    longer read volatility. A table built from the scenario cannot describe another one.
    """
    rows = []
    for series, path in scenario.shocks.items():
        shock = np.asarray(path, dtype=float)
        peak = int(np.argmax(np.abs(shock)))
        proportional = series in scenario.proportional

        def shown(value: float, *, as_share: bool = proportional) -> str:
            return f"{value:+.0%}" if as_share else f"{value:+.1f}"

        readers = sorted(name for name, used in sources.items() if series in used)
        rows.append(
            {
                "series": series,
                "move": shown(float(shock[peak])),
                "month reached": peak + 1,
                "by month 36": shown(float(shock[min(len(shock), 36) - 1])),
                "read by": ", ".join(readers) if readers else "nothing",
            }
        )
    return pd.DataFrame(rows)


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
    horizon_months: int,
) -> pd.DataFrame:
    """Build the forward loan-month panel a book would experience.

    ``loans`` is one row per loan carrying its origination attributes, its current
    ``age``, and the ``period`` its projection starts from. Calendar time advances
    from that period while loan age advances from wherever the loan already is,
    which is what makes the panel forward-looking.

    The start period is per loan rather than shared, because the reporting date is
    not the same for every question. Scoring the book on the shelf projects every
    loan from one reporting date; scoring cohorts written later projects each from
    its own origination. A single shared date silently misplaces the second case in
    calendar time, and the covariates then describe an economy those loans never saw.

    Getting the direction wrong is quieter still. Advancing calendar time from each
    loan's origination replays its actual history, so a projected macro path
    appended after the reporting date is never reached and every scenario returns
    the same answer -- a model that looks stable when it is not being asked the
    question.
    """
    for column in (AGE, "period"):
        if column not in loans.columns:
            message = f"loans must carry a {column!r} column."
            raise ValueError(message)

    start_periods = pd.PeriodIndex(loans["period"])
    last_period = start_periods.max() + (horizon_months - 1)
    if last_period > macro.index.max():
        message = (
            f"Macro panel ends {macro.index.max()} but the horizon reaches {last_period}. "
            "Extend it with extend_macro first."
        )
        raise ValueError(message)

    if LOAN_ID not in loans.columns:
        # A book rebuilt from aggregated cells has no loan id: a row is a covariate
        # combination standing for many loans. The row position is the identity the
        # projection needs, zero-padded so the lexical order the pivot imposes is
        # also the book's order -- otherwise every statistic weighted by the book
        # would silently pair the wrong weight with the wrong row.
        loans = loans.assign(**{LOAN_ID: [f"row_{index:09d}" for index in range(len(loans))]})

    steps = np.arange(horizon_months, dtype=np.int64)
    tiled = np.tile(steps, len(loans))
    projected = loans.loc[loans.index.repeat(horizon_months)].reset_index(drop=True)
    projected[AGE] = projected[AGE].to_numpy(dtype=np.int64) + tiled
    projected["period"] = start_periods.repeat(horizon_months) + tiled
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
    starting = loans.copy()
    starting["period"] = macro.index.max() + 1
    results = {}
    for scenario in scenarios:
        extended = extend_macro(macro, horizon_months + 1, scenario)
        panel = project_panel(starting, extended, horizon_months=horizon_months)
        survival = survival_along_path(result, panel, covariates)
        results[scenario.name] = conditional_pd(survival)

    return pd.DataFrame(results)
