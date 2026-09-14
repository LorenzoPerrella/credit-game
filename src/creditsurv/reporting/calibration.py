"""Generates the calibration report: what the regressors are worth.

"Calibration" means two different things in credit risk and both belong here, so
each is labelled:

**Parameter estimation** -- what each regressor is worth, its uncertainty, and what
a unit of it does to the probability of default.

**Predictive calibration** -- whether predicted default rates match realised ones.

A model can be excellent at the second and meaningless at the first, and vice
versa. Reporting one as though it were the other is a common way to overstate a
model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from creditsurv.features import MACRO_DERIVED
from creditsurv.models.aft import coefficient_table
from creditsurv.models.lifetime_pd import (
    conditional_pd,
    extend_macro,
    pd_term_structure,
    project_panel,
    scenario_lifetime_pd,
    survival_along_path,
)
from creditsurv.reporting import charts
from creditsurv.reporting.builder import Report, provenance

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from creditsurv.models.aft import FitResult


def _average(values: pd.Series, weights: pd.Series | None) -> float:
    """Mean over the book, weighting rows that stand for more than one loan.

    A row of an aggregated book is a *number of loans* sharing a covariate
    combination, not a loan. Averaging such rows unweighted answers a question
    nobody asked -- the mean over distinct combinations -- and systematically
    overstates rare ones, which in a credit book are the risky ones.
    """
    if weights is None:
        return float(values.mean())
    # Aligned on the index, never on position: ``values`` comes back from a pivot and
    # is ordered by loan id, which is not the order the book was handed over in.
    aligned = weights.reindex(values.index)
    if aligned.isna().any():
        message = "Weights do not cover every scored row; they must be indexed by loan id."
        raise ValueError(message)
    return float(np.average(values.to_numpy(dtype=float), weights=aligned.to_numpy(dtype=float)))


def covariate_steps(
    frame: pd.DataFrame, names: Sequence[str], *, weights_col: str | None = None
) -> dict[str, float]:
    """One standard deviation of each covariate, in the data the model was fitted to.

    Exposure-weighted when a weight is given: a row of the fitting panel stands for a
    number of loan-months, so an unweighted deviation would describe the binning.
    """
    weights = None if weights_col is None else frame[weights_col].to_numpy(dtype=float)
    steps: dict[str, float] = {}
    for name in names:
        if name not in frame.columns:
            continue
        values = frame[name].to_numpy(dtype=float)
        mean = float(np.average(values, weights=weights))
        steps[name] = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
    return steps


def marginal_effects(
    fitted: FitResult,
    loans: pd.DataFrame,
    macro: pd.DataFrame,
    covariates: Sequence[str],
    continuous: Sequence[str],
    *,
    horizon_months: int = 12,
    weights: pd.Series | None = None,
    steps: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Change in PD from a one standard deviation move in each covariate.

    Coefficients are not comparable across covariates measured in different units,
    and a time ratio is not the currency a credit decision is denominated in. This
    re-expresses each effect as the change in probability of default it produces,
    holding everything else where it is.

    Where the shock is applied depends on how the covariate is built, and getting
    this wrong is silent. Static characteristics are shocked on the loan record
    before the path is projected. Macro-derived covariates cannot be: projecting
    the panel *recomputes* them from the macro series, so a shock applied to the
    loan record is overwritten and the covariate reports exactly zero effect --
    which is what the first version of this table showed for all four of them,
    flatly contradicting their own coefficients. Which covariates count as derived is
    read from ``MACRO_DERIVED``; the validation found an older, pre-selection list in
    its place, so ``inflation`` was shocked on the record and reported as exactly zero.

    **The size of the move comes from the fitting data, not from the projection.**
    Under the random-walk baseline a macro *level* such as ``vix`` is flat across the
    whole projection and identical for every loan, so its deviation there is exactly
    zero and the covariate was skipped -- the table omitted the macro covariate with the
    largest standardised effect for that reason alone. ``steps`` carries one standard
    deviation of each covariate in the data the model was fitted to (see
    :func:`covariate_steps`); the projected panel is used only when it is not supplied.
    """
    derived = set(MACRO_DERIVED)
    extended = extend_macro(macro, horizon_months + 2)
    baseline_panel = project_panel(loans, extended, horizon_months=horizon_months)
    baseline = _average(
        conditional_pd(survival_along_path(fitted, baseline_panel, covariates)), weights
    )

    rows = []
    for name in continuous:
        if name not in baseline_panel.columns:
            continue
        if steps is not None and name in steps:
            step = float(steps[name])
        else:
            step = float(baseline_panel[name].std())
        if not np.isfinite(step) or step == 0.0:
            continue

        if name in derived:
            # Recomputed during projection, so shock the projected panel.
            panel = baseline_panel.copy()
            panel[name] = panel[name] + step
        else:
            shifted = loans.copy()
            shifted[name] = shifted[name] + step
            panel = project_panel(shifted, extended, horizon_months=horizon_months)

        shocked = _average(conditional_pd(survival_along_path(fitted, panel, covariates)), weights)
        rows.append(
            {
                "covariate": name,
                "kind": "time-varying" if name in derived else "static",
                "one_sd": step,
                "baseline_pd": baseline,
                "shocked_pd": shocked,
                "change_pp": (shocked - baseline) * 100.0,
            }
        )

    table = pd.DataFrame(rows)
    return table.reindex(table["change_pp"].abs().sort_values(ascending=False).index)


def generate(
    fitted: FitResult,
    loans: pd.DataFrame,
    macro: pd.DataFrame,
    covariates: Sequence[str],
    continuous: Sequence[str],
    *,
    reports_dir: Path,
    horizon_months: int = 60,
    weights: pd.Series | None = None,
    steps: dict[str, float] | None = None,
) -> Path:
    """Write ``calibration.md`` and its figures.

    ``weights`` names how many loans each row of ``loans`` stands for, which is what
    an aggregated book carries instead of one row per loan.

    ``steps`` is one standard deviation of each covariate in the data the model was
    fitted to, from :func:`covariate_steps`. Without it the marginal-effects table takes
    the deviation from the projected panel, where a macro level is flat and drops out.
    """
    figures = reports_dir / "figures"
    report = Report(
        "Calibration: what the regressors are worth",
        subtitle="Generated by `creditsurv report`. Numbers come from the run, not the prose.",
    )

    report.heading("Reading an AFT coefficient").text(
        """
This is an **accelerated failure time** model, so a coefficient acts on the
logarithm of survival *time*. A positive coefficient lengthens expected survival
and therefore **lowers** risk.

`exp(coef)` is consequently a **time ratio**, not a hazard ratio. A time ratio of
1.4 means a loan is expected to last forty percent longer, not that it is forty
percent more likely to default. Reading the column as a hazard ratio inverts the
sign of every conclusion in this report, which is why the table below names it for
what it is.
"""
    )

    table = coefficient_table(fitted)
    figure = charts.coefficients(table, figures / "coefficients.png")
    report.heading("Coefficient estimates").table(
        table.reset_index(), caption="Coefficients on log survival time", decimals=4
    ).figure(figure, "Coefficient estimates with 95% confidence intervals")

    report.text(
        """
Interpretation of the two parameter blocks: `lambda_` carries the covariate effects
on the scale of survival time, while `rho_` is the shape of the baseline hazard. A
`rho_` intercept above zero means the hazard *rises* with loan age, which is the
seasoning pattern mortgages are expected to show.
"""
    )

    effects = marginal_effects(
        fitted, loans, macro, covariates, continuous, weights=weights, steps=steps
    )
    report.heading("Marginal effects on probability of default").text(
        """
Coefficients are not comparable across covariates measured in different units, and
a time ratio is not the currency a credit decision is made in. Each row below moves
one covariate by one standard deviation and reports the change in twelve-month PD
in percentage points, holding everything else where it is.
"""
    ).table(effects, decimals=4)

    extended = extend_macro(macro, horizon_months + 2)
    panel = project_panel(loans, extended, horizon_months=horizon_months)
    survival = survival_along_path(fitted, panel, covariates)
    structure = pd_term_structure(survival)

    structure_figure = charts.term_structure(structure, figures / "term_structure.png")
    report.heading("PD term structure").text(
        """
Two portfolios can share a lifetime PD and differ entirely here, and the difference
decides when losses arrive. The cumulative and marginal series are shown in
separate panels rather than on a shared frame with two y-axes: they differ by an
order of magnitude, and a second axis lets the choice of scales decide how related
the two appear.
"""
    ).figure(structure_figure, "Cumulative and marginal PD by month")
    report.table(
        structure.reset_index().iloc[:: max(len(structure) // 12, 1)],
        caption="Term structure, sampled across the horizon",
    )

    report.key_values(
        {
            "12-month PD": _average(conditional_pd(survival, horizon_months=12), weights),
            f"lifetime PD ({horizon_months}m)": _average(conditional_pd(survival), weights),
        }
    )

    scenarios = scenario_lifetime_pd(
        fitted, loans, macro, covariates, horizon_months=min(horizon_months, 36)
    )
    scenario_figure = charts.scenario_comparison(scenarios, figures / "scenarios.png")
    uplift = _average(scenarios["adverse"], weights) / _average(scenarios["baseline"], weights)

    report.heading("Macroeconomic scenarios").text(
        f"""
The adverse path is shaped like 2008 rather than scaled to it: unemployment climbs
four points over a year and stays there, house prices fall a fifth over two years,
volatility jumps thirty points within a quarter and settles ten above where it began,
and the price level ends two per cent below the baseline. The baseline is a random
walk from the last observation, which is **not a forecast** and is not offered as
one: it is what makes the relative effect of a scenario interpretable without
smuggling in a view on the economy.

Only series the fitted model reads are shocked. The first version of this path also
tightened credit conditions and raised mortgage rates, which no covariate in the
formula reads, and left volatility and inflation flat, although in the published fit
volatility carried the largest effect of any macro covariate. Two legs of the scenario
did nothing and two covariates never moved. A test now fails whenever the shocked
series and the formula part company, so a change to either has to change the other.

Because the covariates are time-varying, the scenario is applied by projecting the
covariate paths and chaining conditional survival, not by re-scoring frozen
covariates. That distinction is the whole exercise: an earlier version of this code
advanced calendar time from each loan's origination rather than from the reporting
date, so the projected macro path was never reached and both scenarios returned
almost the same answer. The model looked stable when it was simply not being asked
the question.

**Adverse lifetime PD is {uplift:.2f}x the baseline.**
"""
    ).figure(scenario_figure, "Distribution of lifetime PD under each scenario")
    report.table(
        scenarios.describe().reset_index(names="statistic"),
        caption="Lifetime PD by scenario",
    )

    report.text(
        provenance(
            [
                "`uv run creditsurv report`",
                f"Distribution: {fitted.distribution}; likelihood: {fitted.likelihood.value}",
                f"Book scored: {len(loans):,} rows"
                + ("" if weights is None else f" standing for {weights.sum():,.0f} loans")
                + f"; horizon: {horizon_months} months",
            ]
        )
    )
    return report.write(reports_dir / "calibration.md")
