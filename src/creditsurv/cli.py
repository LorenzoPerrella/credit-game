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
    from collections.abc import Callable, Sequence

    import pandas as pd

    from creditsurv.backtest.splits import Split
    from creditsurv.models.aft import FitResult, Likelihood

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
    from creditsurv.names import readable

    typer.echo(readable(frame.reset_index() if index else frame).to_string(index=False))


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
            f"{len(audit.mismatched)} row counts disagree, "
            f"{len(audit.unreadable)} unreadable"
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
    import json
    import logging

    from creditsurv.data.fred import load_macro_panel
    from creditsurv.data.ingest import load_manifest
    from creditsurv.data.panel import AGE, EVENT, WEIGHT, default_rate_by_observation_month
    from creditsurv.data.store import DEFAULT_POLICY, cells_path, load_cells
    from creditsurv.portfolio import (
        book_summary,
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
        variable="purpose",
    )

    typer.echo("Underwriting drift...")
    charts.underwriting_over_time(covariate_evolution(), figures / "underwriting_over_time.png")

    typer.echo("Macro series...")
    charts.macro_panel(macro, figures / "macro_panel.png")

    typer.echo("Realised default rate...")
    defaults = default_rate_by_period()
    charts.default_rate_and_unemployment(defaults, macro, figures / "default_vs_unemployment.png")

    # Every number docs/portfolio.md quotes (S7). The modelled figures are the cells' own,
    # once the book has been aggregated, so they are the ones every report works from.
    cells = (
        load_cells(DEFAULT_POLICY, columns=["origination_month", AGE, WEIGHT, EVENT])
        if cells_path(DEFAULT_POLICY).exists()
        else None
    )
    performance_rows = sum(int(entry["perf"]) for entry in load_manifest().values())
    summary = book_summary(lending, outstanding, performance_rows=performance_rows, cells=cells)
    if cells is not None:
        # D1's evidence, recomputed on the book the model is fitted to: the realised
        # default rate month by month, where forbearance once made May 2020 a factor of 29.
        rates = default_rate_by_observation_month(cells)
        rates.to_csv(reports_dir() / "monthly_default_rate.csv", index=False)
    destination = reports_dir() / "portfolio_summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    typer.echo("")
    for key, value in summary.items():
        typer.echo(f"{key}: {value:,}" if isinstance(value, int | float) else f"{key}: {value}")
    typer.echo(f"Figures written to {figures}; the summary to {destination}")


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
    report_incomplete: Annotated[
        bool, typer.Option(help="Report, by vintage, the loans the cells leave out.")
    ] = False,
    report_exits: Annotated[
        bool,
        typer.Option(help="Report how loans leaving by a reperforming sale or removal count."),
    ] = False,
    moratorium: Annotated[
        str, typer.Option(help="exclude, censor or ignore: what a moratorium delinquency is.")
    ] = "exclude",
) -> None:
    """Collapse the ingested panel into weighted cells.

    Episodes agreeing on every covariate and on their position in time are
    exchangeable, so they become one row carrying a count. At this scale that is not
    an optimisation: a fit over billions of rows is out of reach, a fit over weighted
    cells is a minute.

    ``--moratorium`` decides what a delinquency the borrower was not required to cure
    counts as. Each policy writes its own table, so the two treatments can be compared
    instead of the second silently replacing the first. See ``MoratoriumPolicy``.
    """
    import logging

    from creditsurv.data.aggregate import (
        MoratoriumPolicy,
        build_cells,
        cardinality_report,
        credit_adjacent_exits,
        incomplete_cases,
    )
    from creditsurv.data.store import save_cells

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if report_cardinality:
        _echo_table(cardinality_report().round(2))
        return
    if report_incomplete:
        # D4: what the complete-case rule drops, and whether it defaults like what it keeps.
        table = incomplete_cases(policy=MoratoriumPolicy(moratorium))
        destination = reports_dir() / "incomplete_cases.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
        _echo_table(table.round(4))
        typer.echo(f"Written: {destination}")
        return
    if report_exits:
        # D5: whether censoring a reperforming sale or a removal loses a default.
        table = credit_adjacent_exits(policy=MoratoriumPolicy(moratorium))
        destination = reports_dir() / "credit_adjacent_exits.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
        totals = table.groupby("code")[
            ["loans", "defaulted_first", "censored_earlier", "censored_at_exit"]
        ].sum()
        _echo_table(totals)
        typer.echo(f"Written: {destination}")
        return

    policy = MoratoriumPolicy(moratorium)
    cells = build_cells(policy=policy)
    path = save_cells(cells, policy.value)
    typer.echo(f"{len(cells):,} cells covering {int(cells['loan_months'].sum()):,} loan-months")
    typer.echo(f"Moratorium policy: {policy.value}. Saved to {path}")


#: The moratorium option, shared by every command that reads cells. One definition, so the
#: policy a model was fitted under is the policy its reports, coefficients and cache entry
#: are named for -- and a comparison of the two treatments cannot quietly read one table
#: twice.
MoratoriumOption = Annotated[
    str, typer.Option(help="exclude, censor or ignore: which cell table to read.")
]


def _episodes(moratorium: str = "exclude") -> tuple[pd.DataFrame, pd.DataFrame]:
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
    # Narrowed to the model's covariates: the whole macro family is nine columns nothing
    # reads, and a row is never dropped for a series nothing reads.
    episodes = cells_to_episodes(load_cells(moratorium), macro, covariates=default_covariates())
    return episodes, macro


def _split(moratorium: str, as_of: pd.Period) -> tuple[Split, pd.DataFrame]:
    """The training and test halves at ``as_of``, and the macro path.

    Built from the cells half by half, so the whole panel never exists beside its halves:
    on the exact calendar key that would be ~12 GB before the first fit, on a 16 GB
    machine. See :func:`creditsurv.backtest.splits.split_cells`.
    """
    from creditsurv.backtest.splits import split_cells
    from creditsurv.data.fred import load_macro_panel
    from creditsurv.data.store import load_cells

    macro = load_macro_panel()
    split = split_cells(load_cells(moratorium), macro, as_of, covariates=default_covariates())
    return split, macro


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
    moratorium: MoratoriumOption = "exclude",
    streamed: Annotated[
        bool, typer.Option(help="Read the rows from the cell file instead of holding them.")
    ] = False,
    workers: Annotated[
        int, typer.Option(help="Processes the likelihood is evaluated in, when streamed.")
    ] = 1,
    block_rows: Annotated[
        int, typer.Option(help="Cells read at a time, when streamed.")
    ] = 1_000_000,
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

    from creditsurv.data.panel import WEIGHT
    from creditsurv.models.aft import Likelihood, coefficient_table, fit_aft

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if streamed:
        result = _fit_streamed(
            as_of=as_of,
            moratorium=moratorium,
            distribution=dist,
            workers=workers,
            block_rows=block_rows,
        )
    else:
        # Already encoded: cells_to_episodes writes the interval bounds as it expands,
        # because the bounds are a function of the cell's age band and its event flag.
        if as_of:
            encoded = _split(moratorium, pd.Period(as_of, freq="M"))[0].train
            typer.echo(f"Training on {int(encoded[WEIGHT].sum()):,} loan-months up to {as_of}.")
        else:
            encoded, _ = _episodes(moratorium)

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
def compare(moratorium: MoratoriumOption = "exclude") -> None:
    """Compare distributional forms and test the shape assumption."""
    import logging

    from creditsurv.data.panel import WEIGHT
    from creditsurv.models.selection import (
        distribution_comparison,
        marginal_comparison,
        shape_depends_on_covariates,
        shape_formula,
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    encoded, _ = _episodes(moratorium)
    covariates = default_covariates()
    formula = default_formula()

    typer.echo("Marginal (univariate) fits:")
    _echo_table(marginal_comparison(encoded, weights_col=WEIGHT).round(2))
    typer.echo("\nRegression fits on identical episodes:")
    _echo_table(distribution_comparison(encoded, covariates, formula, weights_col=WEIGHT).round(2))
    typer.echo("\nDoes the hazard's shape vary with covariates?")
    _echo_table(
        shape_depends_on_covariates(
            encoded,
            covariates,
            formula,
            shape_formula(covariates, CATEGORICAL_REFERENCE)[1],
            weights_col=WEIGHT,
        ).round(4)
    )


@app.command()
def backtest(
    as_of: Annotated[str, typer.Option(help="Reporting date, e.g. 2024-12.")] = DEFAULT_AS_OF,
    moratorium: MoratoriumOption = "exclude",
) -> None:
    """Fit once on everything up to the reporting date, then predict against realised.

    One cut and one fit. There is no second calibration anywhere in this command:
    what comes after the date is scored by the model that never saw it, and the
    comparison is expected defaults against the ones that happened.
    """
    import logging

    import pandas as pd

    from creditsurv.backtest.runner import backtest_split

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    split, _ = _split(moratorium, pd.Period(as_of, freq="M"))

    fitted, result = backtest_split(split, default_covariates(), default_formula())
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
    moratorium: MoratoriumOption = "exclude",
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

    from creditsurv.backtest.runner import backtest_split
    from creditsurv.data.panel import WEIGHT
    from creditsurv.models.aft import coefficient_table
    from creditsurv.models.lifetime_pd import origination_book
    from creditsurv.reporting import backtesting, calibration, methodology
    from creditsurv.reporting.calibration import covariate_steps

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    covariates = default_covariates()
    formula = default_formula()
    destination = reports_dir()

    reporting_date = pd.Period(as_of, freq="M")
    split, macro = _split(moratorium, reporting_date)

    typer.echo(f"Fitting on {int(split.train[WEIGHT].sum()):,} loan-months up to {as_of}...")
    fitted = cast(
        "FitResult",
        _fit_once(
            split.train,
            covariates,
            formula,
            as_of=as_of,
            reuse=reuse,
            moratorium=moratorium,
            start=_selection_start(as_of, moratorium),
        ),
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
            fit=_cached_fit(as_of=as_of, moratorium=moratorium),
        )
    ]

    typer.echo("Writing calibration report...")
    book = origination_book(split.train, macro, loans)
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
            steps=covariate_steps(
                split.train, [*STATIC_CONTINUOUS, *TIME_VARYING_CONTINUOUS], weights_col=WEIGHT
            ),
        )
    )

    typer.echo("Backtesting against what happened...")
    _, result = backtest_split(split, covariates, formula, fitted=fitted)
    written.append(backtesting.generate(split, result, reports_dir=destination))

    typer.echo("\nWritten:")
    for path in [*written, coefficients]:
        typer.echo(f"  {path}")


@app.command()
def views(
    as_of: Annotated[
        str, typer.Option(help="The reporting date the fit was made at.")
    ] = DEFAULT_AS_OF,
    moratorium: MoratoriumOption = "exclude",
    loans: Annotated[int, typer.Option(help="Origination profiles for the projections.")] = 500,
    horizon: Annotated[int, typer.Option(help="Months for the term structure.")] = 60,
    model: Annotated[bool, typer.Option(help="The views that need the fitted model.")] = True,
    portfolio: Annotated[bool, typer.Option(help="The views of the book itself.")] = True,
) -> None:
    """Compute the tables behind the documentation site, and write them to ``docs/tables``.

    **Never fits.** The model views score the fit ``creditsurv report`` saved -- the Weibull of
    the selected specification, and every other family the report compared when its fit is in
    the cache -- once on the training half and once on the test window, and open every table
    by segment. The portfolio views read the ingested book and the cells. The selection views
    are the record ``creditsurv select`` wrote.

    The tables are aggregates, small enough to commit: the site is built from them in CI,
    where the data they come from cannot go.
    """
    import logging

    import pandas as pd

    from creditsurv.backtest.runner import predicted_hazard
    from creditsurv.config import tables_dir
    from creditsurv.data.aggregate import MoratoriumPolicy
    from creditsurv.data.fred import load_macro_panel
    from creditsurv.data.panel import AGE, EVENT, WEIGHT
    from creditsurv.data.store import fit_fingerprint, load_cells, load_fit
    from creditsurv.models.aft import CONVERGENT_DISTRIBUTIONS
    from creditsurv.models.aft import FitResult as Fitted
    from creditsurv.models.lifetime_pd import origination_book
    from creditsurv.views.model import (
        calibration_views,
        coefficient_view,
        covariates_over_time,
        projection_views,
    )
    from creditsurv.views.portfolio import portfolio_views
    from creditsurv.views.selection import selection_views
    from creditsurv.views.tables import write_views

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    destination = tables_dir()
    reporting_date = pd.Period(as_of, freq="M")

    if model:
        covariates, formula = default_covariates(), default_formula()
        typer.echo("Splitting the cells...")
        split, macro = _split(moratorium, reporting_date)
        fingerprint = fit_fingerprint(
            **_fit_description(split.train, formula, as_of=as_of, moratorium=moratorium)
        )
        fitted = load_fit(fingerprint)
        if not isinstance(fitted, Fitted):
            typer.echo(
                f"No cached fit {fingerprint} of this specification. Run `creditsurv report` "
                "first: views never fit."
            )
            raise typer.Exit(1)

        typer.echo("Scoring the training half and the test window...")
        train_hazard = predicted_hazard(fitted, split.train, covariates).to_numpy()
        test_hazard = predicted_hazard(fitted, split.test, covariates).to_numpy()
        families = {fitted.distribution: train_hazard}
        for distribution in CONVERGENT_DISTRIBUTIONS:
            if distribution == fitted.distribution:
                continue
            other = load_fit(
                fit_fingerprint(
                    **_fit_description(
                        split.train,
                        formula,
                        as_of=as_of,
                        moratorium=moratorium,
                        distribution=distribution,
                    )
                )
            )
            if isinstance(other, Fitted):
                typer.echo(f"  and the cached {distribution} fit")
                families[distribution] = predicted_hazard(other, split.train, covariates).to_numpy()

        typer.echo("Calibration, backtest, coefficients and projections by segment...")
        tables = [
            *calibration_views(
                split, train_hazard=train_hazard, test_hazard=test_hazard, families=families
            ),
            coefficient_view(
                fitted, split.train, [*STATIC_CONTINUOUS, *TIME_VARYING_CONTINUOUS, *ORDINAL]
            ),
            covariates_over_time(split, TIME_VARYING_CONTINUOUS),
            *projection_views(
                fitted,
                origination_book(split.train, macro, loans),
                macro,
                covariates,
                horizon_months=horizon,
            ),
        ]
        write_views(tables, destination, fit=fingerprint)
        typer.echo(f"  {len(tables)} views of fit {fingerprint}")
        del split, train_hazard, test_hazard, families, tables

    if portfolio:
        typer.echo("The book by segment, the lending, the vintage curves and the macro series...")
        cells = load_cells(moratorium, columns=["origination_month", AGE, WEIGHT, EVENT])
        write_views(
            portfolio_views(cells, load_macro_panel(), policy=MoratoriumPolicy(moratorium)),
            destination,
        )
        del cells

    write_views(selection_views(reports_dir()), destination)
    typer.echo(f"Written: {destination}")


@app.command()
def select(
    as_of: Annotated[
        str, typer.Option(help="Select on everything up to this month.")
    ] = DEFAULT_AS_OF,
    moratorium: MoratoriumOption = "exclude",
) -> None:
    """Run the variable selection on the training half, and write what it chose.

    Steps 5 to 9 of ``docs/variable_selection.md`` on the whole population up to
    ``--as-of``: correlation, variance inflation, univariate screening, backward
    elimination and stability. On the whole population that is days, and it resumes --
    every fit is saved as it lands -- so a run that stops picks up where it was.

    The report goes to ``docs/reports/selection.md`` with ``selection.json`` beside it, the
    record the configuration is tested against. See ``creditsurv.models.procedure``.
    """
    import logging

    import numpy as np
    import pandas as pd

    from creditsurv.config import MACRO_CANDIDATES
    from creditsurv.data.fred import load_macro_panel
    from creditsurv.data.panel import (
        AGE_START,
        EVENT,
        EXACT_OBSERVATION,
        LOWER_BOUND,
        UPPER_BOUND,
        WEIGHT,
        cells_to_episodes,
        episode_step,
        observation_months,
    )
    from creditsurv.data.store import cells_identity, load_cells
    from creditsurv.models.procedure import (
        BASE_CATEGORICAL,
        CANDIDATE_CATEGORICAL,
        LOAN_CONTINUOUS,
        LOAN_ORDINAL,
        Fits,
        run_selection,
    )
    from creditsurv.reporting import selection

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    reporting_date = pd.Period(as_of, freq="M")
    candidates = [
        *LOAN_CONTINUOUS,
        *LOAN_ORDINAL,
        *MACRO_CANDIDATES,
        *BASE_CATEGORICAL,
        *CANDIDATE_CATEGORICAL,
    ]

    # The training half only, and nothing of the test half is ever built: selection runs
    # on the months the model may see. The half is taken out and the whole table let go
    # before the expansion, which adds fifteen macro columns and peaks at several GB more.
    cells = load_cells(moratorium)
    cut = reporting_date.year * 12 + reporting_date.month - 1
    step = episode_step(cells)
    selected = cells.iloc[np.flatnonzero(observation_months(cells).to_numpy() <= cut)]
    del cells
    train = cells_to_episodes(selected, load_macro_panel(), covariates=candidates, step=step)
    del selected
    halves = (train["origination_month"].to_numpy() // 12) % 2 == 0

    # Only what a fit or a covariance reads. The calendar columns, the episode end and the
    # vintage label are a quarter of the frame on ~60 million rows, and every fit in the
    # selection holds its stored design beside it.
    keep = {*candidates, AGE_START, LOWER_BOUND, UPPER_BOUND, EXACT_OBSERVATION, WEIGHT, EVENT}
    for column in [name for name in train.columns if name not in keep]:
        del train[column]

    identity = cells_identity(moratorium)
    typer.echo(f"Selecting on {len(train):,} cells, {int(train[WEIGHT].sum()):,} loan-months.")

    fits = Fits(train, identity=identity, as_of=as_of, moratorium=moratorium)
    record = run_selection(train, fits, halves=halves)
    written = selection.generate(record, reports_dir=reports_dir())

    time_varying = tuple(name for name in record.selected.continuous if name in MACRO_CANDIDATES)
    typer.echo(f"\nFormula: {record.selected.formula}")
    typer.echo(f"TIME_VARYING_CONTINUOUS = {time_varying}")
    typer.echo(f"Written: {written}")


@app.command()
def moratorium(
    as_of: Annotated[str, typer.Option(help="Reporting date for both fits.")] = DEFAULT_AS_OF,
) -> None:
    """Fit and backtest the model with forbearance excluded, and again with it censored.

    D1 of the validation: 17% of default events were moratoria, not credit, and the two ways
    of removing them keep different things. The choice is made on what each does to the
    coefficients and the backtest, written to ``docs/reports/moratorium.md``.

    One policy at a time, releasing its panel before the next. Both fits are cached under
    their policy, so ``report --reuse`` picks up the chosen one instead of refitting it.
    """
    import logging

    import pandas as pd

    from creditsurv.backtest.runner import backtest_split
    from creditsurv.reporting.moratorium import generate, outcome

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    reporting_date = pd.Period(as_of, freq="M")
    covariates, formula = default_covariates(), default_formula()

    outcomes = []
    for policy in ("exclude", "censor"):
        typer.echo(f"\n{policy}:")
        split, _ = _split(policy, reporting_date)
        fitted = cast(
            "FitResult",
            _fit_once(split.train, covariates, formula, as_of=as_of, reuse=True, moratorium=policy),
        )
        _, result = backtest_split(split, covariates, formula, fitted=fitted)
        outcomes.append(outcome(policy, split, fitted, result))
        typer.echo(str(result.summary()))
        del split, fitted, result

    written = generate(outcomes[0], outcomes[1], reports_dir=reports_dir())
    typer.echo(f"\nWritten: {written}")


@app.command("check-calendar")
def check_calendar(moratorium: MoratoriumOption = "exclude") -> None:
    """The validation's M1 test: defaults by month from the cells, against the loan-months.

    With the origination quarter in the key the reconstructed series peaked two months out
    of step with the true one. With the month in the key the two must be the same series,
    and their correlation must peak at a lag of zero. Written to
    ``docs/reports/calendar_check.csv``. A pass over every performance file.
    """
    import logging

    from creditsurv.data.aggregate import MoratoriumPolicy, defaults_by_month
    from creditsurv.data.panel import defaults_by_observation_month
    from creditsurv.data.store import load_cells
    from creditsurv.explore import lagged_correlation

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    truth = defaults_by_month(policy=MoratoriumPolicy(moratorium))
    rebuilt = defaults_by_observation_month(load_cells(moratorium))

    correlation = lagged_correlation(truth, rebuilt)
    months = truth.index.union(rebuilt.index)
    moved = (truth.reindex(months, fill_value=0) - rebuilt.reindex(months, fill_value=0)).abs()
    destination = reports_dir() / "calendar_check.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    correlation.rename_axis("lag").to_frame().to_csv(destination)
    typer.echo(correlation.round(6).to_string())
    typer.echo(
        f"Peak at lag {correlation.idxmax()}; {int(moved.sum()):,} of {int(truth.sum()):,} "
        "defaults filed in a different month."
    )
    typer.echo(f"Written: {destination}")


def _fit_streamed(
    *,
    as_of: str,
    moratorium: str,
    distribution: str,
    workers: int,
    block_rows: int,
) -> FitResult:
    """Fit from the cell file, in ``workers`` processes, and save it under its fingerprint.

    The rows are never held: each process reads its share of the cells, expands it, and keeps
    it compactly. The fit is cached under the same description a fit made in memory is, so
    `report` and `views` find it by the specification rather than by how it was made.
    """
    import pandas as pd

    from creditsurv.data.fred import load_macro_panel
    from creditsurv.data.panel import WEIGHT, CellBlocks
    from creditsurv.data.store import cells_path, fit_fingerprint, save_fit
    from creditsurv.models.aft import fit_streamed

    formula = default_formula()
    cut = None
    if as_of:
        reporting_date = pd.Period(as_of, freq="M")
        cut = reporting_date.year * 12 + reporting_date.month - 1
    source = CellBlocks(
        str(cells_path(moratorium)),
        load_macro_panel(),
        tuple(default_covariates()),
        rows=block_rows,
        months=(None, cut),
    )
    typer.echo(f"Reading {source.source} in {workers} process(es)...")
    result = fit_streamed(
        source,
        default_covariates(),
        formula,
        distribution=distribution,
        weights_col=WEIGHT,
        workers=workers,
    )
    record = result.blocks
    assert record is not None
    described = _fit_description(
        (result.n_episodes, int(record.loan_months)),
        formula,
        as_of=as_of,
        moratorium=moratorium,
        distribution=distribution,
    )
    fingerprint = fit_fingerprint(**described)
    path = save_fit(result, fingerprint, {**described, "minutes": result.elapsed_seconds / 60})
    typer.echo(
        f"  {record.rows:,} cells in {record.blocks} blocks, {record.stored_bytes / 1e9:.2f} GB "
        f"stored, {record.evaluations} evaluations, {result.elapsed_seconds / 60:.1f} minutes"
    )
    typer.echo(f"  saved as {fingerprint} to {path}")
    return result


def _fit_description(
    counted: pd.DataFrame | tuple[int, int],
    formula: str,
    *,
    as_of: str,
    moratorium: str,
    distribution: str = "weibull",
    weights_col: str | None = "loan_months",
    ancillary: str | None = None,
    likelihood: Likelihood | None = None,
) -> dict[str, object]:
    """What a report's fit is cached under, and so how any command finds it again.

    ``counted`` is the panel, or its row and loan-month counts when the rows were never held
    as a frame -- a fit streamed from the cell file counts them as it reads.

    The moratorium policy is in it because the two treatments can produce panels of similar
    size, and a censor fit silently reused for exclude would compare a model with itself.
    The ancillary formula and the likelihood enter only when they are not the defaults, so
    the report's own Weibull keeps the name it has always had.
    """
    from creditsurv.models.aft import Likelihood as Likelihoods

    if isinstance(counted, tuple):
        rows, loan_months = counted
    else:
        rows = len(counted)
        loan_months = int(counted[weights_col].sum()) if weights_col else rows
    described: dict[str, object] = {
        "as_of": as_of,
        "moratorium": moratorium,
        "formula": formula,
        "distribution": distribution,
        "weights_col": weights_col,
        "rows": rows,
        "loan_months": loan_months,
    }
    if ancillary is not None:
        described["ancillary"] = ancillary
    if likelihood is not None and likelihood is not Likelihoods.INTERVAL_CENSORED:
        described["likelihood"] = likelihood.value
    return described


def _fit_once(
    train: pd.DataFrame,
    covariates: list[str],
    formula: str,
    *,
    as_of: str,
    reuse: bool,
    moratorium: str = "exclude",
    start: pd.Series | None = None,
) -> object:
    """Fit the training half, reusing a cached model when one matches exactly.

    ``start`` is where the optimiser begins -- the selection's own fit of the specification,
    say -- and changes how long the fit takes, not where it ends.

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

    described = _fit_description(train, formula, as_of=as_of, moratorium=moratorium)
    fingerprint = fit_fingerprint(**described)

    if reuse:
        cached = load_fit(fingerprint)
        if cached is not None:
            typer.echo(f"  reusing the cached fit {fingerprint}")
            return cached
        typer.echo(f"  no cached fit {fingerprint}; fitting")

    fitted = fit_aft(train, covariates, formula, weights_col=WEIGHT, initial_point=start)
    typer.echo(f"  {fitted.elapsed_seconds / 60:.1f} minutes")
    path = save_fit(fitted, fingerprint, {**described, "minutes": fitted.elapsed_seconds / 60})
    typer.echo(f"  saved to {path}")
    return fitted


def _cached_fit(*, as_of: str, moratorium: str) -> Callable[..., FitResult]:
    """``fit_aft``, saved the moment each fit lands and read back when asked for again.

    For the extra fits a report makes -- the other distribution families and the shape
    test -- which ran 78 and 25 minutes on the training half and used to be thrown away, so
    that regenerating a report's prose cost them again. The fingerprint has the fields
    ``_fit_once`` uses, plus the ancillary formula and the likelihood when they are not the
    defaults, so the Weibull the report already holds is found under its own name.
    """

    def fit(
        encoded: pd.DataFrame,
        covariates: Sequence[str],
        formula: str,
        *,
        distribution: str = "weibull",
        likelihood: Likelihood | None = None,
        weights_col: str | None = None,
        ancillary: str | None = None,
        initial_point: pd.Series | None = None,
    ) -> FitResult:
        from creditsurv.data.store import fit_fingerprint, load_fit, save_fit
        from creditsurv.models import aft

        likelihood = likelihood or aft.Likelihood.INTERVAL_CENSORED
        described = _fit_description(
            encoded,
            formula,
            as_of=as_of,
            moratorium=moratorium,
            distribution=distribution,
            weights_col=weights_col,
            ancillary=ancillary,
            likelihood=likelihood,
        )
        fingerprint = fit_fingerprint(**described)
        cached = load_fit(fingerprint)
        if isinstance(cached, aft.FitResult) and cached.log_likelihood < 0:
            typer.echo(f"  reusing the cached {distribution} fit {fingerprint}")
            return cached
        result = aft.fit_aft(
            encoded,
            covariates,
            formula,
            distribution=distribution,
            likelihood=likelihood,
            weights_col=weights_col,
            ancillary=ancillary,
            initial_point=initial_point,
        )
        save_fit(result, fingerprint, {**described, "minutes": result.elapsed_seconds / 60})
        return result

    return fit


def _selection_start(as_of: str, moratorium: str) -> pd.Series | None:
    """The coefficients the selection ended on, when it chose on the same half and table.

    ``None`` when no selection has been recorded, when it selected for another date or
    moratorium policy, or when its fit is no longer in the cache: the fit then starts from
    lifelines' own seed, as it always did. A record older than the cell table cannot match,
    because the fit is cached under the table's size and time of writing.
    """
    import json

    from creditsurv.data.store import cells_identity
    from creditsurv.models.procedure import selected_fit
    from creditsurv.reporting.selection import SUMMARY_FILE

    path = reports_dir() / SUMMARY_FILE
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    if summary.get("as_of") != as_of or summary.get("moratorium") != moratorium:
        return None
    fitted = selected_fit(
        identity=cells_identity(moratorium),
        as_of=as_of,
        moratorium=moratorium,
        formula=str(summary["formula"]),
    )
    if fitted is None:
        return None
    typer.echo("  starting from the selection's fit of the specification it chose")
    params: pd.Series = fitted.fitter.params_
    return params


if __name__ == "__main__":  # pragma: no cover
    app()
