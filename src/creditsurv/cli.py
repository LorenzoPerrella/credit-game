"""Command line interface.

Each command does one thing and says what it found. ``report`` runs the whole
pipeline and writes the documents in ``docs/reports/``.

Fitted models are deliberately not persisted between commands. Pickling a lifelines
fitter couples the artefact to the installed version of several libraries, and a
stale model file that loads without complaint is a worse failure than refitting.
Commands that need a model fit one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

import typer

from creditsurv.config import (
    CATEGORICAL_REFERENCE,
    MACRO_SERIES,
    ORDINAL,
    STATIC_CONTINUOUS,
    TIME_VARYING_CONTINUOUS,
    default_formula,
    reports_dir,
)

if TYPE_CHECKING:
    import pandas as pd

    from creditsurv.models.aft import FitResult

#: The reporting date every command cuts at, unless one is given. Late on purpose: a
#: credit model wants every loan-month it can get in training, and the test window only
#: has to be long enough to judge it. Shared by `fit`, `backtest` and `report` so that
#: all three mean the same model by the same name.
DEFAULT_AS_OF = "2024-12"

#: Where `fit --save` and `report` leave the coefficient table, and where the notebooks
#: read it.
COEFFICIENTS_FILE = "coefficients.csv"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Lifetime PD with parametric survival models.",
)


def default_covariates() -> list[str]:
    """Every column the default formula is allowed to read."""
    return [*STATIC_CONTINUOUS, *TIME_VARYING_CONTINUOUS, *ORDINAL, *CATEGORICAL_REFERENCE]


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


@app.command("prune-archives")
def prune_archives(
    years: Annotated[
        str | None, typer.Option(help="Year or range, e.g. 2006 or 1999-2026.")
    ] = None,
    yes: Annotated[bool, typer.Option(help="Skip the confirmation prompt.")] = False,
) -> None:
    """Delete the downloaded archives whose parquet is verified complete.

    The one irreversible step in this pipeline, and re-downloading costs hours behind a
    manual registration -- so it is a separate command and never a tail appended to the
    ingest, where a parse gone wrong would take the only copy with it.

    An archive is deleted only when **every** quarter of its year passes three separate
    checks: the manifest records it finished, both parquet files exist, and their row
    counts still match what the manifest recorded. The third is the one that catches a
    file truncated or overwritten since, which the existence of a file does not.

    Nothing is deleted without showing what would go and asking. Run it after a
    complete fit rather than straight after the ingest: that the parquet parses is not
    the same as that it is usable.
    """
    import logging

    import pandas as pd

    from creditsurv.data.ingest import audit_archives

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    audits = audit_archives(_parse_years(years) if years else None)
    if not audits:
        typer.echo("No archives found.")
        return

    _echo_table(pd.DataFrame([audit.describe() for audit in audits]))
    safe = [audit for audit in audits if audit.safe_to_delete]
    blocked = [audit for audit in audits if not audit.safe_to_delete]

    for audit in blocked:
        typer.echo(
            f"{audit.year}: NOT deleting -- {len(audit.missing)} missing, "
            f"{len(audit.mismatched)} row counts disagree"
        )

    if not safe:
        typer.echo("\nNothing is verified complete. Nothing deleted.")
        return

    freed = sum(audit.bytes_on_disk for audit in safe) / 1024**3
    typer.echo(f"\n{len(safe)} archive(s) verified complete, {freed:.1f} GB.")
    if not yes and not typer.confirm("Delete them? This cannot be undone"):
        typer.echo("Nothing deleted.")
        return

    for audit in safe:
        audit.path.unlink()
        typer.echo(f"  deleted {audit.path.name}")
    typer.echo(f"Freed {freed:.1f} GB.")


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


def _episodes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The weighted episode panel every model command works from, and the macro path.

    Aggregated cells rather than a loan-month panel: the book is 2.9 billion
    loan-months and 15.8 million cells, and only one of those two is a table a
    fitter can be handed. Expansion is deterministic, so this is the same panel the
    loan-level path would produce, carrying counts instead of repeated rows.
    """
    from creditsurv.data.fred import load_macro_panel
    from creditsurv.data.panel import cells_to_episodes
    from creditsurv.data.store import load_cells

    macro = load_macro_panel()
    return cells_to_episodes(load_cells(), macro), macro


@app.command()
def fit(
    dist: Annotated[str, typer.Option(help="weibull or loglogistic.")] = "weibull",
    likelihood: Annotated[
        str, typer.Option(help="interval_censored or right_censored.")
    ] = "interval_censored",
    as_of: Annotated[
        str, typer.Option(help="Fit on everything up to this month. Empty for all of it.")
    ] = DEFAULT_AS_OF,
    save: Annotated[bool, typer.Option(help="Write the coefficients under docs/reports.")] = True,
) -> None:
    """Fit the model and print its coefficients.

    ``--as-of`` defaults to the same reporting date the backtest cuts at, so this
    command and ``report`` produce the **same model** -- which matters because both
    write the same coefficient file, and two commands quietly disagreeing about which
    model is "the" model is a good way to publish a table nobody can reproduce. Pass
    an empty string to fit the whole panel instead.

    The coefficient table is saved by default: a fit on this population is hours, and
    the notebooks and reports should not each pay for one. It is a generated artefact
    -- regenerate it, do not edit it.
    """
    import logging

    import pandas as pd

    from creditsurv.backtest.splits import cell_split
    from creditsurv.data.panel import WEIGHT
    from creditsurv.models.aft import Likelihood, coefficient_table, fit_aft

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Already encoded: cells_to_episodes writes the interval bounds as it expands,
    # because the bounds are a function of the cell's age band and its event flag.
    encoded, _ = _episodes()
    if as_of:
        encoded = cell_split(encoded, pd.Period(as_of, freq="M")).train
        typer.echo(f"Training on {int(encoded[WEIGHT].sum()):,} loan-months up to {as_of}.")

    result = fit_aft(
        encoded,
        default_covariates(),
        default_formula(),
        distribution=dist,
        likelihood=Likelihood(likelihood),
        weights_col=WEIGHT,
    )
    typer.echo(
        f"{result.distribution} / {result.likelihood.value}: "
        f"{result.n_episodes:,} cells, {result.n_events:,} defaults, "
        f"AIC {result.aic:,.1f}, {result.elapsed_seconds:.1f}s"
    )
    table = coefficient_table(result)
    _echo_table(table.round(4), index=True)

    if save:
        destination = reports_dir() / COEFFICIENTS_FILE
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination)
        typer.echo(f"\nCoefficients written to {destination}")


@app.command()
def compare() -> None:
    """Compare distributional forms and test the shape assumption."""
    import logging

    from creditsurv.data.panel import WEIGHT
    from creditsurv.models.selection import (
        distribution_comparison,
        marginal_comparison,
        shape_depends_on_covariates,
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    encoded, _ = _episodes()
    covariates = default_covariates()
    formula = default_formula()

    typer.echo("Marginal (univariate) fits:")
    _echo_table(marginal_comparison(encoded, weights_col=WEIGHT).round(2))
    typer.echo("\nRegression fits on identical episodes:")
    _echo_table(distribution_comparison(encoded, covariates, formula, weights_col=WEIGHT).round(2))
    typer.echo("\nDoes the hazard's shape vary with covariates?")
    _echo_table(
        shape_depends_on_covariates(
            encoded, covariates, formula, "fico_s", weights_col=WEIGHT
        ).round(4)
    )


@app.command()
def backtest(
    as_of: Annotated[str, typer.Option(help="Reporting date, e.g. 2024-12.")] = DEFAULT_AS_OF,
) -> None:
    """Fit once on everything up to the reporting date, then predict against realised.

    One cut and one fit. There is no second calibration anywhere in this command:
    what comes after the date is scored by the model that never saw it, and the
    comparison is expected defaults against the ones that happened.
    """
    import logging

    import pandas as pd

    from creditsurv.backtest.runner import run_backtest

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    episodes, _ = _episodes()

    split, fitted, result = run_backtest(
        episodes, pd.Period(as_of, freq="M"), default_covariates(), default_formula()
    )
    typer.echo(str(split.describe()))
    typer.echo(f"Fitted in {fitted.elapsed_seconds / 60:.1f} minutes on the training half.\n")

    _echo_table(pd.DataFrame([result.summary()]))
    typer.echo("\nBy decile of predicted risk:")
    _echo_table(result.calibration.round(6))


@app.command()
def report(
    horizon: Annotated[int, typer.Option(help="Months for the PD term structure.")] = 60,
    as_of: Annotated[str, typer.Option(help="Reporting date for the backtest.")] = DEFAULT_AS_OF,
    loans: Annotated[int, typer.Option(help="Origination profiles to score.")] = 500,
    extra_fits: Annotated[
        bool, typer.Option(help="Also compare distributions and test the shape.")
    ] = True,
    reuse: Annotated[
        bool, typer.Option(help="Reuse a cached fit matching this specification exactly.")
    ] = False,
) -> None:
    """Run the pipeline and write the reports, from a single fit.

    The model is fitted **once**, on everything up to the reporting date, and that one
    model is what all three reports describe: the methodology report characterises it,
    the calibration report prices with it, and the backtest scores it on the months it
    has never seen. Nothing here refits it.

    That the same model appears in all three is the point. A report describing a model
    fitted on everything, next to a backtest of a different model fitted on a subset,
    invites the reader to attribute one's performance to the other.

    ``--no-extra-fits`` drops the two model-selection sections that each cost a further
    fit, taking the whole run to a single one. The reports then say the sections were
    skipped rather than omitting them silently.
    """
    import logging

    import pandas as pd

    from creditsurv.backtest.runner import run_backtest
    from creditsurv.backtest.splits import cell_split
    from creditsurv.data.panel import WEIGHT
    from creditsurv.models.aft import coefficient_table
    from creditsurv.reporting import backtesting, calibration, methodology

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    episodes, macro = _episodes()
    covariates = default_covariates()
    formula = default_formula()
    destination = reports_dir()

    reporting_date = pd.Period(as_of, freq="M")
    split = cell_split(episodes, reporting_date)

    typer.echo(f"Fitting on {int(split.train[WEIGHT].sum()):,} loan-months up to {as_of}...")
    fitted = cast(
        "FitResult", _fit_once(split.train, covariates, formula, as_of=as_of, reuse=reuse)
    )

    coefficients = destination / COEFFICIENTS_FILE
    coefficients.parent.mkdir(parents=True, exist_ok=True)
    coefficient_table(fitted).to_csv(coefficients)

    typer.echo("Writing methodology report...")
    written = [
        methodology.generate(
            split.train,
            split.train,
            fitted,
            covariates,
            formula,
            reports_dir=destination,
            weights_col=WEIGHT,
            extra_fits=extra_fits,
        )
    ]

    typer.echo("Writing calibration report...")
    book = _origination_book(split.train, macro, loans)
    written.append(
        calibration.generate(
            fitted,
            book,
            macro,
            covariates,
            [*STATIC_CONTINUOUS, *TIME_VARYING_CONTINUOUS],
            reports_dir=destination,
            horizon_months=horizon,
            weights=book.set_index("loan_id")[WEIGHT],
        )
    )

    typer.echo("Backtesting against what happened...")
    _, _, result = run_backtest(episodes, reporting_date, covariates, formula, fitted=fitted)
    written.append(backtesting.generate(split, result, reports_dir=destination))

    typer.echo("\nWritten:")
    for path in [*written, coefficients]:
        typer.echo(f"  {path}")


def _fit_once(
    train: pd.DataFrame,
    covariates: list[str],
    formula: str,
    *,
    as_of: str,
    reuse: bool,
) -> object:
    """Fit the training half, reusing a cached model when one matches exactly.

    A fit on this population is two and a half hours, and the run that discovered that
    completed one and was then killed while writing its reports -- throwing away the
    expensive part and keeping nothing. The fit is therefore saved the moment it
    succeeds, before anything downstream can fail.

    Reuse is **opt-in**. The fingerprint covers the specification and the panel's size,
    which does not catch a re-aggregation that happens to leave the row count alone, so
    silently reusing would eventually mean reporting on a model built from data that no
    longer exists.
    """
    from creditsurv.data.panel import WEIGHT
    from creditsurv.data.store import fit_fingerprint, load_fit, save_fit
    from creditsurv.models.aft import fit_aft

    described = {
        "as_of": as_of,
        "formula": formula,
        "distribution": "weibull",
        "weights_col": WEIGHT,
        "rows": len(train),
        "loan_months": int(train[WEIGHT].sum()),
    }
    fingerprint = fit_fingerprint(**described)

    if reuse:
        cached = load_fit(fingerprint)
        if cached is not None:
            typer.echo(f"  reusing the cached fit {fingerprint}")
            return cached
        typer.echo(f"  no cached fit {fingerprint}; fitting")

    fitted = fit_aft(train, covariates, formula, weights_col=WEIGHT)
    typer.echo(f"  {fitted.elapsed_seconds / 60:.1f} minutes")
    path = save_fit(fitted, fingerprint, {**described, "minutes": fitted.elapsed_seconds / 60})
    typer.echo(f"  saved to {path}")
    return fitted


def _origination_book(encoded: pd.DataFrame, macro: pd.DataFrame, size: int) -> pd.DataFrame:
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
    from creditsurv.data.panel import AGE, LOAN_ID, WEIGHT

    book = encoded.loc[encoded[AGE] == 0].nlargest(size, WEIGHT).reset_index(drop=True)
    book[AGE] = 0
    book["period"] = macro.index.max() + 1
    # Named here rather than left to the projection, because the weights have to be
    # indexed by the same label the scored results come back under.
    book[LOAN_ID] = [f"row_{index:09d}" for index in range(len(book))]
    return book


if __name__ == "__main__":  # pragma: no cover
    app()
