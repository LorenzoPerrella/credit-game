"""Command line interface.

Each command does one thing and says what it found. ``report`` runs the whole
pipeline and writes the documents in ``docs/reports/``.

Fitted models are deliberately not persisted between commands. Pickling a lifelines
fitter couples the artefact to the installed version of several libraries, and a
stale model file that loads without complaint is a worse failure than refitting.
Commands that need a model fit one.
"""

from __future__ import annotations

# typer resolves annotations at runtime, so Path cannot move into the
# type-checking block the way ruff would prefer -- the CLI stops working.
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated

import typer

from creditsurv.config import (
    CATEGORICAL_REFERENCE,
    MACRO_SERIES,
    STATIC_CONTINUOUS,
    TIME_VARYING_CONTINUOUS,
    default_formula,
    reports_dir,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Lifetime PD with parametric survival models.",
)


def default_covariates() -> list[str]:
    """Every column the default formula is allowed to read."""
    return [*STATIC_CONTINUOUS, *TIME_VARYING_CONTINUOUS, *CATEGORICAL_REFERENCE]


def _echo_table(frame: pd.DataFrame, *, index: bool = False) -> None:
    typer.echo(frame.to_string(index=index))


@app.command("fetch-macro")
def fetch_macro(
    refresh: Annotated[bool, typer.Option(help="Ignore the cache and re-download.")] = False,
) -> None:
    """Download and cache the FRED macroeconomic series."""
    from creditsurv.data.fred import load_macro_panel

    panel = load_macro_panel(refresh=refresh)
    typer.echo(f"Macro panel: {len(panel)} months, {panel.index.min()} to {panel.index.max()}")
    typer.echo(f"Series: {', '.join(spec.column for spec in MACRO_SERIES)}")


@app.command("build-data")
def build_data(
    orig: Annotated[Path, typer.Option(help="orig_YYYYQn.txt from the dataset.")],
    svcg: Annotated[Path, typer.Option(help="perf_YYYYQn.txt from the dataset.")],
) -> None:
    """Build the loan-month panel from Freddie Mac files and save it.

    The dataset is not downloadable programmatically -- registration is free but
    manual -- so the two files are named explicitly rather than guessed at.
    """
    from creditsurv.data.fred import load_macro_panel
    from creditsurv.data.freddiemac import load_sample
    from creditsurv.data.store import save_panel
    from creditsurv.features import add_macro_covariates

    macro = load_macro_panel()
    # Macro covariates are derived by the same code for every source, so a
    # difference in results can never come from a difference in feature building.
    panel = add_macro_covariates(load_sample(orig, svcg), macro)
    path = save_panel(panel)

    loans = panel["loan_id"].nunique()
    defaults = int(panel["event"].sum())
    typer.echo(f"Panel: {len(panel):,} loan-months, {loans:,} loans -> {path}")
    typer.echo(f"Defaults: {defaults:,} ({defaults / loans:.2%} of loans)")


@app.command()
def fit(
    dist: Annotated[str, typer.Option(help="weibull or loglogistic.")] = "weibull",
    likelihood: Annotated[
        str, typer.Option(help="interval_censored or right_censored.")
    ] = "interval_censored",
) -> None:
    """Fit the model and print its coefficients."""
    from creditsurv.data.panel import to_interval_censored
    from creditsurv.data.store import load_panel
    from creditsurv.models.aft import Likelihood, coefficient_table, fit_aft

    encoded = to_interval_censored(load_panel())
    result = fit_aft(
        encoded,
        default_covariates(),
        default_formula(),
        distribution=dist,
        likelihood=Likelihood(likelihood),
    )
    typer.echo(
        f"{result.distribution} / {result.likelihood.value}: "
        f"{result.n_episodes:,} episodes, {result.n_events:,} defaults, "
        f"AIC {result.aic:,.1f}, {result.elapsed_seconds:.1f}s"
    )
    _echo_table(coefficient_table(result).round(4), index=True)


@app.command()
def compare() -> None:
    """Compare distributional forms and test the shape assumption."""
    from creditsurv.data.panel import to_interval_censored
    from creditsurv.data.store import load_panel
    from creditsurv.models.selection import (
        distribution_comparison,
        marginal_comparison,
        shape_depends_on_covariates,
    )

    panel = load_panel()
    encoded = to_interval_censored(panel)
    covariates = default_covariates()
    formula = default_formula()

    typer.echo("Marginal (univariate) fits:")
    _echo_table(marginal_comparison(panel).round(2))
    typer.echo("\nRegression fits on identical episodes:")
    _echo_table(distribution_comparison(encoded, covariates, formula).round(2))
    typer.echo("\nDoes the hazard's shape vary with covariates?")
    _echo_table(shape_depends_on_covariates(encoded, covariates, formula, "fico_s").round(4))


@app.command()
def backtest(
    horizon: Annotated[int, typer.Option(help="Months to predict forward.")] = 12,
    folds: Annotated[int, typer.Option(help="Number of walk-forward reporting dates.")] = 3,
) -> None:
    """Run the walk-forward backtest under both macro modes."""
    from creditsurv.backtest.runner import macro_mode_gap, run_backtest
    from creditsurv.backtest.splits import walk_forward
    from creditsurv.data.fred import load_macro_panel
    from creditsurv.data.store import load_panel

    panel = load_panel()
    macro = load_macro_panel()
    summary, _ = run_backtest(
        walk_forward(panel, _reporting_dates(panel, folds)),
        macro,
        default_covariates(),
        default_formula(),
        horizon_months=horizon,
    )
    _echo_table(summary.round(4))
    typer.echo("\nCalibration gap between macro modes:")
    _echo_table(macro_mode_gap(summary).round(4))


def _reporting_dates(panel: pd.DataFrame, folds: int) -> Sequence[pd.Period]:
    """Evenly spaced reporting dates inside the observed window.

    The window is trimmed at both ends: the earliest dates have too little history
    to fit on, and the latest leave no room to predict forward into.
    """
    periods = panel["period"].sort_values().unique()
    start, stop = int(len(periods) * 0.45), int(len(periods) * 0.9)
    step = max((stop - start) // max(folds, 1), 1)
    return [periods[index] for index in range(start, stop, step)][:folds]


@app.command()
def report(
    horizon: Annotated[int, typer.Option(help="Months for the PD term structure.")] = 60,
    backtest_horizon: Annotated[int, typer.Option(help="Months to predict forward.")] = 12,
    folds: Annotated[int, typer.Option(help="Walk-forward reporting dates.")] = 3,
    loans: Annotated[int, typer.Option(help="Loans to score for the PD report.")] = 500,
) -> None:
    """Run the full pipeline and write the reports."""
    from creditsurv.backtest.runner import run_backtest
    from creditsurv.backtest.splits import walk_forward
    from creditsurv.data.fred import load_macro_panel
    from creditsurv.data.panel import at_origination, to_interval_censored
    from creditsurv.data.store import load_panel
    from creditsurv.models.aft import fit_aft
    from creditsurv.reporting import backtesting, calibration, methodology

    panel = load_panel()
    macro = load_macro_panel()
    encoded = to_interval_censored(panel)
    covariates = default_covariates()
    formula = default_formula()
    destination = reports_dir()

    typer.echo("Fitting...")
    fitted = fit_aft(encoded, covariates, formula)

    typer.echo("Writing methodology report...")
    written = [
        methodology.generate(panel, encoded, fitted, covariates, formula, reports_dir=destination)
    ]

    typer.echo("Writing calibration report...")
    book = at_origination(panel).head(loans).copy()
    book["age"] = 0
    book["period"] = macro.index.max() + 1
    written.append(
        calibration.generate(
            fitted,
            book,
            macro,
            covariates,
            [*STATIC_CONTINUOUS, *TIME_VARYING_CONTINUOUS],
            reports_dir=destination,
            horizon_months=horizon,
        )
    )

    typer.echo("Running backtest...")
    summary, results = run_backtest(
        walk_forward(panel, _reporting_dates(panel, folds)),
        macro,
        covariates,
        formula,
        horizon_months=backtest_horizon,
    )
    written.append(backtesting.generate(summary, results, reports_dir=destination))

    typer.echo("\nWritten:")
    for path in written:
        typer.echo(f"  {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
