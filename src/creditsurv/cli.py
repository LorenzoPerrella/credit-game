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


@app.command()
def ingest(
    years: Annotated[
        str | None, typer.Option(help="Year or range, e.g. 2006 or 1999-2026.")
    ] = None,
    force: Annotated[bool, typer.Option(help="Re-convert quarters already done.")] = False,
) -> None:
    """Convert the downloaded archives to parquet.

    Idempotent: a quarter already converted is skipped, so an interrupted run costs
    only the quarter it was in the middle of.
    """
    import logging

    from creditsurv.data.ingest import discover_years
    from creditsurv.data.ingest import ingest as run_ingest

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    selected = _parse_years(years) if years else discover_years()
    typer.echo(f"Ingesting {len(selected)} vintage year(s): {selected[0]}-{selected[-1]}")

    manifest = run_ingest(selected, force=force)
    performance = sum(entry["perf"] for entry in manifest.values())
    origination = sum(entry["orig"] for entry in manifest.values())
    typer.echo(f"\n{len(manifest)} quarters: {origination:,} loans, {performance:,} loan-months")


def _parse_years(spec: str) -> list[int]:
    if "-" in spec:
        first, last = (int(part) for part in spec.split("-", 1))
        return list(range(first, last + 1))
    return [int(spec)]


@app.command()
def portfolio() -> None:
    """Describe the book: outstanding, new lending, mix, drift, and the macro path.

    Written before any model is fitted. It is the description that makes the
    modelling legible -- and it is where several problems were found that staring at
    coefficients would not have surfaced.
    """
    import logging

    from creditsurv.data.fred import load_macro_panel
    from creditsurv.portfolio import (
        covariate_evolution,
        default_rate_by_period,
        origination_mix,
        originations_by_period,
        outstanding_by_period,
    )
    from creditsurv.reporting import charts

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    figures = reports_dir() / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    macro = load_macro_panel()

    typer.echo("New lending...")
    lending = originations_by_period()
    charts.new_lending(lending, figures / "new_lending.png")

    typer.echo("Book outstanding...")
    outstanding = outstanding_by_period()
    charts.outstanding_book(outstanding, figures / "outstanding_book.png")

    typer.echo("Origination mix...")
    charts.origination_mix_over_time(
        origination_mix("purpose"),
        figures / "mix_purpose.png",
        title="New lending by purpose",
    )

    typer.echo("Underwriting drift...")
    charts.underwriting_over_time(covariate_evolution(), figures / "underwriting_over_time.png")

    typer.echo("Macro series...")
    charts.macro_panel(macro, figures / "macro_panel.png")

    typer.echo("Realised default rate...")
    charts.default_rate_and_unemployment(
        default_rate_by_period(), macro, figures / "default_vs_unemployment.png"
    )

    typer.echo(
        f"\n{int(lending['loans'].sum()):,} loans, "
        f"${lending['amount'].sum() / 1e12:.2f}tn originated; "
        f"peak {int(outstanding['contracts'].max()):,} contracts outstanding"
    )
    typer.echo(f"Figures written to {figures}")


@app.command()
def profile(
    covariate: Annotated[str | None, typer.Option(help="Profile one covariate in detail.")] = None,
) -> None:
    """Screen the covariates before aggregating.

    This runs first, and the order is the point. Screening before the group-by means
    the cut points, the merges and the exclusions are decided from what the data
    looks like; screening after it means they were assumed, and a mis-binned
    covariate can only be found later by noticing its coefficient came out backwards.
    """
    import logging

    from creditsurv.data.aggregate import _CATEGORICAL, _SOURCE
    from creditsurv.profiling import (
        is_monotonic,
        profile_categorical,
        profile_continuous,
        propose_cut_points,
        screen_categoricals,
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if covariate in _CATEGORICAL:
        _echo_table(profile_categorical(covariate).round(5))
        return
    if covariate in _SOURCE:
        edges = propose_cut_points(covariate)
        typer.echo(f"Quantile cut points: {edges}")
        table = profile_continuous(covariate, edges)
        _echo_table(table.round(5))
        typer.echo(f"Monotonic in default rate: {is_monotonic(table)}")
        return
    if covariate is not None:
        message = f"Unknown covariate {covariate!r}."
        raise typer.BadParameter(message)

    typer.echo("Categorical covariates, whole history:")
    _echo_table(screen_categoricals().round(5))


@app.command()
def aggregate(
    report_cardinality: Annotated[
        bool, typer.Option(help="Report the collapse without saving.")
    ] = False,
) -> None:
    """Collapse the ingested panel into weighted cells.

    Episodes agreeing on every covariate and on their position in time are
    exchangeable, so they become one row carrying a count. At this scale that is not
    an optimisation: a fit over billions of rows is out of reach, a fit over weighted
    cells is a minute.
    """
    import logging

    from creditsurv.data.aggregate import build_cells, cardinality_report
    from creditsurv.data.store import save_cells

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if report_cardinality:
        _echo_table(cardinality_report().round(2))
        return

    cells = build_cells()
    path = save_cells(cells)
    typer.echo(f"{len(cells):,} cells covering {int(cells['n'].sum()):,} loan-months")
    typer.echo(f"Saved to {path}")


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
    as_of: Annotated[str, typer.Option(help="Reporting date, e.g. 2024-12.")] = "2024-12",
    horizon: Annotated[int, typer.Option(help="Months to predict forward.")] = 24,
    walk_forward_folds: Annotated[
        int, typer.Option(help="Repeat at several dates instead of one.")
    ] = 0,
) -> None:
    """Backtest at a single reporting date, or walk forward across several.

    One date by default, and a late one. Everything up to it trains the model, which
    is the point: a credit model wants every loan-month it can get, and holding back
    a decade to see the same result at four dates is a poor trade. The walk-forward
    is still there for when the question is whether a result held across regimes
    rather than what the model can do.
    """
    import logging

    import pandas as pd

    from creditsurv.backtest.runner import macro_mode_gap, run_backtest
    from creditsurv.backtest.splits import as_of_split, walk_forward
    from creditsurv.data.fred import load_macro_panel
    from creditsurv.data.panel import cells_to_episodes
    from creditsurv.data.store import load_cells

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    macro = load_macro_panel()
    panel = cells_to_episodes(load_cells(), macro)
    reporting_date = pd.Period(as_of, freq="M")

    if walk_forward_folds:
        splits = walk_forward(panel, _reporting_dates(panel, walk_forward_folds))
    else:
        splits = [as_of_split(panel, reporting_date)]

    for split in splits:
        typer.echo(str(split.describe()))

    summary, _ = run_backtest(
        splits, macro, default_covariates(), default_formula(), horizon_months=horizon
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
