"""Running a backtest: fit on the past, predict forward, compare with what happened.

Two macro modes, and the distinction matters more than any metric here.

**Conditional** feeds the macro path that actually occurred over the test window.
It isolates the credit model: given the right economy, does it rank and price
correctly? This is the fair test of the model as a model.

**Unconditional** feeds only what was knowable at the reporting date -- a random
walk from the last observation. It tests the whole system as it would really have
been used, macro forecasting error included. It is much harsher, and much closer to
the truth about how such a model performs in the field.

Reporting only the conditional result would overstate what the model can do.
Reporting only the unconditional one would blame the credit model for the
impossibility of forecasting the economy. The gap between them is the interesting
number: it says how much of the error is credit and how much is macro.

We have no vintage-dated macro forecasts, so the unconditional mode uses a random
walk. That is an approximation, not a reconstruction of what a forecaster would
have said, and it is recorded as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import pandas as pd

from creditsurv.backtest.metrics import (
    brier_score,
    calibration_slope_intercept,
    calibration_table,
    discrimination,
    stability_report,
    weighted_calibration,
    weighted_gini,
)
from creditsurv.config import TIME_VARYING_CONTINUOUS
from creditsurv.data.panel import AGE, EVENT, LOAN_ID, WEIGHT, to_interval_censored
from creditsurv.models.aft import Likelihood, episode_hazards, fit_aft
from creditsurv.models.lifetime_pd import (
    BASELINE,
    conditional_pd,
    extend_macro,
    project_panel,
    survival_along_path,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from creditsurv.backtest.splits import Split

PERIOD = "period"


class MacroMode(StrEnum):
    """Which macro path the model is given over the test window."""

    CONDITIONAL = "conditional"
    UNCONDITIONAL = "unconditional"


@dataclass(frozen=True)
class BacktestResult:
    """Everything one split produced."""

    split: str
    as_of: pd.Period
    macro_mode: MacroMode
    horizon_months: int
    n_loans: int
    n_defaults: int
    metrics: dict[str, float] = field(default_factory=dict)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    stability: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Why this fold produced nothing, when it produced nothing.
    error: str = ""

    def summary(self) -> dict[str, object]:
        return {
            "split": self.split,
            "as_of": str(self.as_of),
            "macro_mode": self.macro_mode.value,
            "horizon_months": self.horizon_months,
            "loans": self.n_loans,
            "defaults": self.n_defaults,
            **({"error": self.error} if self.error else {}),
            **{key: round(value, 5) for key, value in self.metrics.items()},
        }


def _state_at_reporting_date(test: pd.DataFrame) -> pd.DataFrame:
    """One row per test loan, holding its attributes as the test window opens."""
    ordered = test.sort_values([LOAN_ID, AGE], kind="stable")
    return ordered.groupby(LOAN_ID, observed=True).first().reset_index()


def _observed_outcome(test: pd.DataFrame, horizon_months: int) -> pd.DataFrame:
    """Did each loan default within the horizon, and was it watched the whole time?

    The ``complete`` flag matters more than it looks. A loan that prepays or runs
    off the end of the panel after six months cannot default in months seven to
    twenty-four, but it still sits in the denominator as a non-default. Comparing a
    twenty-four month predicted PD against that understates the realised rate and
    makes the model look like it over-predicts. Here it dragged actual-over-expected
    down to 0.47 on folds where the model was not in fact badly calibrated.

    Discrimination is unaffected -- the concordance index takes the censoring
    indicator and handles it -- so only calibration is restricted to the closed
    cohort: loans watched for the full horizon, plus those that defaulted inside it.
    """
    ordered = test.sort_values([LOAN_ID, AGE], kind="stable")
    start_age = ordered.groupby(LOAN_ID, observed=True)[AGE].transform("min")
    within = ordered[ordered[AGE] < start_age + horizon_months]

    grouped = within.groupby(LOAN_ID, observed=True)
    defaulted = grouped[EVENT].max().astype(bool)
    observed_months = grouped[AGE].max() - grouped[AGE].min() + 1
    outcome = pd.DataFrame(
        {
            "defaulted": defaulted,
            "months_observed": observed_months,
            "complete": defaulted | (observed_months >= horizon_months),
        }
    )
    return outcome.reset_index()


def run_split(
    split: Split,
    macro: pd.DataFrame,
    covariates: Sequence[str],
    formula: str,
    *,
    horizon_months: int = 12,
    macro_mode: MacroMode = MacroMode.CONDITIONAL,
    distribution: str = "weibull",
    likelihood: Likelihood = Likelihood.INTERVAL_CENSORED,
) -> BacktestResult:
    """Fit on one split's training half and score its test half.

    Dispatches on what the panel is. A loan-level panel is scored loan by loan,
    which is the richer test; an aggregated one has no loans to follow across the
    boundary, and is scored on exposure instead. Which one arrives is decided by the
    split, not by an argument, because passing the wrong flag would score an
    aggregated panel as though its cells were subjects -- and that produces numbers
    rather than an error.
    """
    if LOAN_ID not in split.train.columns:
        return _run_cell_split(
            split,
            macro,
            covariates,
            formula,
            horizon_months=horizon_months,
            macro_mode=macro_mode,
            distribution=distribution,
            likelihood=likelihood,
        )

    fitted = fit_aft(
        to_interval_censored(split.train),
        covariates,
        formula,
        distribution=distribution,
        likelihood=likelihood,
    )

    if macro_mode is MacroMode.UNCONDITIONAL:
        # Only what was knowable at the reporting date, extended by a random walk.
        known = macro[macro.index <= split.as_of]
        scoring_macro = extend_macro(known, horizon_months + 2, BASELINE)
    else:
        scoring_macro = macro

    # Each loan is projected from the month its own test window opens, which is
    # the reporting date for an existing book and the origination month for a
    # cohort written later.
    state = _state_at_reporting_date(split.test)
    projected = project_panel(state, scoring_macro, horizon_months=horizon_months)
    survival = survival_along_path(fitted, projected, covariates)
    predicted = conditional_pd(survival).rename("predicted_pd")

    observed = _observed_outcome(split.test, horizon_months).set_index(LOAN_ID)
    aligned = observed.join(predicted, how="inner").dropna()

    if aligned.empty:
        message = f"Split {split.name!r} produced no scoreable loans."
        raise ValueError(message)

    # Discrimination uses every loan, since the concordance index accounts for
    # censoring. Calibration uses only the closed cohort, because a partially
    # observed loan cannot have realised a full-horizon outcome.
    closed = aligned[aligned["complete"]]
    if closed.empty:
        message = f"Split {split.name!r} has no loans observed for the full horizon."
        raise ValueError(message)

    metrics = {
        **discrimination(aligned["months_observed"], aligned["defaulted"], aligned["predicted_pd"]),
        "brier": brier_score(closed["defaulted"], closed["predicted_pd"]),
        **calibration_slope_intercept(closed["defaulted"], closed["predicted_pd"]),
        "closed_cohort": float(len(closed)),
    }

    return BacktestResult(
        split=split.name,
        as_of=split.as_of,
        macro_mode=macro_mode,
        horizon_months=horizon_months,
        n_loans=len(aligned),
        n_defaults=int(aligned["defaulted"].sum()),
        metrics=metrics,
        calibration=calibration_table(closed["defaulted"], closed["predicted_pd"]),
        stability=stability_report(
            split.train, split.test, covariates, time_varying=TIME_VARYING_CONTINUOUS
        ),
    )


def _run_cell_split(
    split: Split,
    macro: pd.DataFrame,
    covariates: Sequence[str],
    formula: str,
    *,
    horizon_months: int,
    macro_mode: MacroMode,
    distribution: str,
    likelihood: Likelihood,
) -> BacktestResult:
    """Score an aggregated panel: fit on the cells before the date, test on those after.

    What changes is the unit. There is no loan to project forward, because a cell is
    a count of loan-months sharing a covariate combination -- so instead of building
    a forward path per loan, the fitted monthly hazard is evaluated on the test cells
    at the covariates they actually carry, and compared with the defaults they
    actually recorded.

    That is a weaker test than the loan-level one in a specific way: it cannot ask
    whether the model ranks *loans*, only whether it ranks *exposure*, and it cannot
    chain a multi-month horizon per subject. It is a stronger test in another: it
    covers the whole population rather than a sample of it.

    Under the unconditional mode the macro path is cut at the reporting date. On
    cells that shows up as a covariate rebuild rather than a projection: the two
    macro-derived covariates are recomputed from what was knowable, and the
    difference between the modes is again the cost of not knowing the economy.
    """
    fitted = fit_aft(
        split.train,
        covariates,
        formula,
        distribution=distribution,
        likelihood=likelihood,
        weights_col=WEIGHT,
    )

    test = split.test
    if test.empty:
        message = f"Split {split.name!r} has no exposure after {split.as_of}."
        raise ValueError(message)

    # Only the first horizon of the test window, so the number means the same thing
    # as the loan-level one: how the model did over a stated period, not over
    # however much data happened to be left.
    within = test[test[PERIOD] <= split.as_of + horizon_months]
    if within.empty:
        message = f"Split {split.name!r} has no exposure within {horizon_months} months."
        raise ValueError(message)

    if macro_mode is MacroMode.UNCONDITIONAL:
        within = _rebuild_macro_covariates(within, macro, split.as_of, horizon_months)
        if within.empty:
            message = f"Split {split.name!r} has no exposure left after rebuilding covariates."
            raise ValueError(message)

    hazard = pd.Series(
        episode_hazards(fitted, within.loc[:, list(covariates)], within[AGE].to_numpy(dtype=int)),
        index=within.index,
    )
    exposure = within[WEIGHT].astype(float)
    events = exposure * within[EVENT].astype(bool)

    expected = float((hazard * exposure).sum())
    actual = float(events.sum())
    metrics = {
        "gini": weighted_gini(hazard, events, exposure),
        "expected_defaults": expected,
        "actual_defaults": actual,
        "actual_over_expected": actual / expected if expected > 0 else float("nan"),
        "loan_months": float(exposure.sum()),
    }

    return BacktestResult(
        split=split.name,
        as_of=split.as_of,
        macro_mode=macro_mode,
        horizon_months=horizon_months,
        n_loans=int(exposure.sum()),
        n_defaults=int(actual),
        metrics=metrics,
        calibration=weighted_calibration(hazard, events, exposure),
        stability=stability_report(
            split.train,
            within,
            covariates,
            time_varying=TIME_VARYING_CONTINUOUS,
            weights_col=WEIGHT,
        ),
    )


def _rebuild_macro_covariates(
    cells: pd.DataFrame, macro: pd.DataFrame, as_of: pd.Period, horizon_months: int
) -> pd.DataFrame:
    """Recompute the macro-derived covariates from what was knowable at ``as_of``.

    The unconditional mode's claim is that the model is scored without hindsight, so
    the covariates that read the macro path must be rebuilt rather than reused. That
    the rebuild is possible at all is a property of the cell key: it carries the
    origination quarter and the loan age, which is everything the derivation needs.

    ``cells_to_episodes`` is called rather than the arithmetic repeated, so the
    covariate is built by the same code in the backtest as in the fit. Two
    implementations of one definition is how a backtest ends up flattering a model.
    """
    from creditsurv.data.panel import cells_to_episodes

    known = extend_macro(macro[macro.index <= as_of], horizon_months + 2, BASELINE)
    return cells_to_episodes(cells, known)


def run_backtest(
    splits: Sequence[Split],
    macro: pd.DataFrame,
    covariates: Sequence[str],
    formula: str,
    *,
    horizon_months: int = 12,
    macro_modes: Sequence[MacroMode] = (MacroMode.CONDITIONAL, MacroMode.UNCONDITIONAL),
    distribution: str = "weibull",
) -> tuple[pd.DataFrame, list[BacktestResult]]:
    """Run every split under every macro mode and summarise.

    Returns the summary table alongside the full results, because the calibration
    and stability tables are what explain a summary row that looks wrong.
    """
    results: list[BacktestResult] = []
    for split in splits:
        for mode in macro_modes:
            try:
                results.append(
                    run_split(
                        split,
                        macro,
                        covariates,
                        formula,
                        horizon_months=horizon_months,
                        macro_mode=mode,
                        distribution=distribution,
                    )
                )
            except (ValueError, ZeroDivisionError) as error:
                # A thin fold -- too few loans, or none that defaulted -- is a fact
                # about the data rather than a reason to abandon the other folds.
                #
                # ZeroDivisionError belongs here explicitly: lifelines raises it
                # from concordance_index when a fold contains no admissible pairs,
                # and it descends from ArithmeticError rather than ValueError, so
                # catching ValueError alone lets it through. That is exactly how a
                # small nightly run died while every test stayed green.
                #
                # The reason is carried on the result. A silently empty row is a
                # much worse outcome than a loud one: it surfaces later as a
                # KeyError three functions away, with nothing pointing back here.
                results.append(
                    BacktestResult(
                        split=split.name,
                        as_of=split.as_of,
                        macro_mode=mode,
                        horizon_months=horizon_months,
                        n_loans=0,
                        n_defaults=0,
                        error=f"{type(error).__name__}: {error}",
                    )
                )

    summary = pd.DataFrame([result.summary() for result in results])
    return summary, results


def macro_mode_gap(summary: pd.DataFrame, metric: str = "actual_over_expected") -> pd.DataFrame:
    """How much the model changes once the macro path is no longer known.

    The difference between the two modes attributes error: what remains under
    ``conditional`` belongs to the credit model, what appears under
    ``unconditional`` is the cost of not knowing the economy in advance.

    The default metric is a calibration one on purpose. Measured on this data the
    gap in **discrimination** is essentially zero -- concordance moves by 0.0005
    between the two modes -- while calibration moves materially, actual over
    expected going from 0.74 to 0.84 at one reporting date. That is what should
    happen: a macro path shifts every loan's PD in the same direction, so the
    ranking survives and the level does not. Reading the gap on concordance would
    conclude that macro forecasting does not matter, when what it actually shows is
    that the wrong tool was used to look for it.
    """
    if metric not in summary.columns:
        message = (
            f"No {metric!r} column in the summary; every fold failed. "
            "Check the error column for the reason."
        )
        raise ValueError(message)

    wide = summary.pivot_table(
        index=["split", "as_of"], columns="macro_mode", values=metric, aggfunc="first"
    )
    if {"conditional", "unconditional"} <= set(wide.columns):
        wide["gap"] = wide["conditional"] - wide["unconditional"]
    return wide.reset_index()
