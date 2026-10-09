"""Command line interface.

Each command does one thing and says what it found. ``report`` runs the whole
pipeline and writes the documents in ``docs/reports/``.

Fitted models are deliberately not persisted between commands. Pickling a lifelines
fitter couples the artefact to the installed version of several libraries, and a
stale model file that loads without complaint is a worse failure than refitting.
Commands that need a model fit one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Final, cast

import typer

from creditsurv.config import (
    CATEGORICAL_REFERENCE,
    DEFAULT_CAUSE,
    DISTRIBUTION,
    MACRO_SERIES,
    STATIC_CONTINUOUS,
    TIME_VARYING_CONTINUOUS,
    default_covariates,
    default_formula,
    reports_dir,
)

if TYPE_CHECKING:
    import pandas as pd

    from creditsurv.backtest.splits import Split
    from creditsurv.models.aft import FitResult

#: The reporting date every command cuts at, unless one is given: the end of the
#: **development window** of `docs/rules.md`. Shared by `fit`, `backtest` and `report` so
#: that all three mean the same model by the same name.
#:
#: It was 2024-12 until September 2026, chosen late because a credit model wants every
#: loan-month it can get. That left no room between estimation and the test window, so the
#: level could only be anchored on data the coefficients had already seen or on the window
#: being judged. Three years now sit between them: estimation to 2021-12, anchoring on
#: 2022-01 to 2024-12, and the test window from 2025-01, each seeing only what the ones
#: before it did.
DEFAULT_AS_OF: Final = "2021-12"

#: Where `fit --save` and `report` leave the coefficient table, and where the notebooks
#: read it.
COEFFICIENTS_FILE: Final = "coefficients.csv"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Lifetime PD with parametric survival models.",
)


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


@app.command(name="prune-encodings")
def prune_encodings(
    stale_only: Annotated[
        bool, typer.Option(help="Keep the readings of the cell table that is on disk.")
    ] = True,
    name: Annotated[
        list[str] | None, typer.Option(help="Delete these readings by name, whatever their table.")
    ] = None,
    yes: Annotated[bool, typer.Option(help="Skip the confirmation prompt.")] = False,
) -> None:
    """Delete cached readings of the cell file, by default only the ones nothing can hit.

    A reading is **1.0 GB** and is named by the cell table's identity -- its file, size and time
    of writing -- so rebuilding the table makes every reading of the old one dead weight that
    nothing will ever look for again. Rule 2 needs six readings a campaign, two causes by three
    samples, so a rebuild can leave six gigabytes behind with no way to notice.

    Deleting a **current** one is different: it costs 11.6 minutes to take again, and a run that
    stops picks up at the fit it was on rather than at the reading. So the sweep is the stale
    ones unless `--no-stale-only` says otherwise, and nothing goes without being shown first.

    `--name` takes one by name, which is the case the sweep cannot reason about: a reading of the
    table on disk at a reporting date nobody will ask about again is current and useless at the
    same time, and only the person who made it knows which. The table printed first carries the
    date for exactly that.

    A separate command, like `prune-archives`, and for the same reason: never a tail appended to
    something else, where one mistake takes the only copy with it.
    """
    import logging

    import pandas as pd

    from creditsurv.models.engine.cache import audit_readings, remove_reading

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    readings = audit_readings()
    if not readings:
        typer.echo("No cached readings.")
        return

    _echo_table(pd.DataFrame([reading.describe() for reading in readings]))
    wanted = set(name or ())
    unknown = wanted - {reading.name for reading in readings}
    if unknown:
        typer.echo(f"No such reading(s): {', '.join(sorted(unknown))}")
        raise typer.Exit(1)
    going = [
        reading
        for reading in readings
        if reading.name in wanted or not (wanted or (stale_only and reading.current))
    ]
    kept = [reading for reading in readings if reading not in going]
    for reading in kept:
        typer.echo(f"{reading.name}: keeping -- it is a reading of the table on disk")
    if not going:
        typer.echo("\nNothing is stale. Nothing deleted.")
        return

    freed = sum(reading.bytes_on_disk for reading in going) / 1024**3
    current = sum(1 for reading in going if reading.current)
    typer.echo(
        f"\n{len(going)} reading(s), {freed:.1f} GB"
        + (
            f", of which {current} of the table on disk -- 11.6 minutes each to take again."
            if current
            else ", none of the table on disk."
        )
    )
    if not yes and not typer.confirm("Delete them? This cannot be undone"):
        typer.echo("Nothing deleted.")
        return

    for reading in going:
        remove_reading(reading.name)
        typer.echo(f"  deleted {reading.name}")
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
    from creditsurv.data.panel import (
        AGE,
        OUTCOME,
        WEIGHT,
        default_rate_by_observation_month,
    )
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
        load_cells(DEFAULT_POLICY, columns=["origination_month", AGE, WEIGHT, OUTCOME])
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
    extensions: Annotated[
        bool, typer.Option(help="Price each extension of the cell key, on nine quarters.")
    ] = False,
) -> None:
    """Screen the covariates before aggregating.

    This runs first, and the order is the point. Screening before the group-by means
    the cut points, the merges and the exclusions are decided from what the data
    looks like; screening after it means they were assumed, and a mis-binned
    covariate can only be found later by noticing its coefficient came out backwards.
    """
    import logging

    from creditsurv.data.book import CATEGORICAL, SOURCE
    from creditsurv.profiling import (
        is_monotonic,
        profile_categorical,
        profile_continuous,
        propose_cut_points,
        screen_categoricals,
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if extensions:
        # What a covariate in the key actually costs, against the ceiling declared in
        # docs/rules.md. A ceiling from the product of the level counts is not a cost.
        from creditsurv.profiling import extension_cost

        table = extension_cost()
        destination = reports_dir() / "key_extensions.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
        _echo_table(table.round(3))
        typer.echo(f"Written to {destination}")
        return

    if covariate in CATEGORICAL:
        _echo_table(profile_categorical(covariate).round(5))
        return
    if covariate in SOURCE:
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
        cardinality_report,
        credit_adjacent_exits,
        incomplete_cases,
        write_cells,
    )
    from creditsurv.data.book import MoratoriumPolicy
    from creditsurv.data.store import cells_path

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
    # Written a quarter at a time rather than concatenated: the held table peaked at 11.3 GB on
    # 91.6 million cells and wants about 21 at the 200 million finer bands would need.
    written = write_cells(policy=policy)
    typer.echo(f"{written:,} cells. Moratorium policy: {policy.value}.")
    typer.echo(f"Saved to {cells_path(policy.value)}")


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
    dist: Annotated[str, typer.Option(help="weibull or loglogistic.")] = DISTRIBUTION,
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
    block_rows: Annotated[int, typer.Option(help="Cells read at a time, when streamed.")] = 250_000,
    cause: Annotated[
        str, typer.Option(help="default or prepayment: which exit the model is of.")
    ] = DEFAULT_CAUSE,
    traced: Annotated[
        bool,
        typer.Option(
            "--traced/--written-out",
            help="Trace lifelines' likelihood with autograd instead of writing it out.",
        ),
    ] = False,
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
    from creditsurv.models.fits import fit_from_cells

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if streamed:
        result = fit_from_cells(
            as_of=as_of,
            moratorium=moratorium,
            distribution=dist,
            workers=workers,
            block_rows=block_rows,
            cause=cause,
            traced=traced,
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
    as_of: Annotated[str, typer.Option(help="Reporting date, e.g. 2021-12.")] = DEFAULT_AS_OF,
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
def windows(
    moratorium: MoratoriumOption = "exclude",
    workers: Annotated[int, typer.Option(help="Processes each window's fit is evaluated in.")] = 1,
    block_rows: Annotated[int, typer.Option(help="Cells read at a time.")] = 250_000,
    cuts: Annotated[
        str,
        typer.Option(help="Reporting dates to cut at, comma-separated. The rule's, by default."),
    ] = "",
    as_of: Annotated[
        str, typer.Option(help="End of the development window the level is anchored for.")
    ] = DEFAULT_AS_OF,
    anchor_window: Annotated[
        str, typer.Option(help="First and last month the level is anchored on, comma-separated.")
    ] = "",
) -> None:
    """Backtest the model at three cuts, anchor its level on 2022-24, and grade it.

    Every threshold is declared in ``docs/rules.md`` before this runs, and those are the
    defaults of every option here: the cuts at the end of 2018, 2020 and 2022 with 24 months
    after each, the anchoring window of 2022-01 to 2024-12, the master scale of eight grades
    and the criteria. The options exist so the command can be exercised on a fixture book,
    and the report states the windows it actually used.

    One fit per cut, streamed from the cell file and warm-started from the cut before, each
    estimated on everything up to its own cut so that no window is scored by a model that saw
    it. Nothing large is ever held: a window is read out of the parquet by its observation
    months, which is about 3% of the table, and the in-sample cycle is taken a calendar year
    at a time for the same reason.
    """
    import logging

    import pandas as pd

    from creditsurv.backtest.campaign import NoExposure, run_campaign
    from creditsurv.backtest.runner import backtest_windows
    from creditsurv.data.fred import load_macro_panel
    from creditsurv.models.anchoring import ANCHOR_WINDOW
    from creditsurv.reporting import windows as windows_report

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    first, last = ANCHOR_WINDOW if not anchor_window else anchor_window.split(",", 1)
    try:
        campaign = run_campaign(
            moratorium=moratorium,
            covariates=default_covariates(),
            formula=default_formula(),
            macro=load_macro_panel(),
            declared=backtest_windows() if not cuts else backtest_windows(cuts.split(",")),
            development_cut=pd.Period(as_of, freq="M"),
            anchor_window=(first, last),
            block_rows=block_rows,
        )
    except NoExposure as empty:
        # A window with no rows is a fact about the book, and the campaign raises it as one.
        # Turning it into a bad option is this layer's job, because only this layer has one.
        raise typer.BadParameter(str(empty)) from empty

    written = windows_report.generate(
        campaign.windows,
        cycle=campaign.cycle,
        anchor=campaign.anchor,
        level=campaign.level,
        grades=campaign.grades,
        acceptance=campaign.acceptance,
        reports_dir=reports_dir(),
    )
    _echo_table(campaign.windows)
    _echo_table(campaign.cycle)
    typer.echo(f"Written: {written}")


@app.command()
def family(
    moratorium: MoratoriumOption = "exclude",
    as_of: Annotated[
        str, typer.Option(help="End of the development window both families were selected on.")
    ] = DEFAULT_AS_OF,
    block_rows: Annotated[int, typer.Option(help="Cells read at a time.")] = 250_000,
) -> None:
    """Apply rule 2 of ``docs/rules.md``: which distribution family the model keeps.

    Each family's **selected** model is compared with the Aalen-Johansen cumulative incidence
    of default, which accounts for prepayment as a competing risk, over the loan ages carrying
    at least the exposure floor. A family whose selected model turns a declared sign is
    excluded whatever its fit; inside a tenth of a percentage point the Weibull is kept.

    Nothing is fitted here. Each selection cached the fit of the model it chose and this reads
    them, so a family whose selection has not been run is named and skipped. The prepayment
    hazard is the one the Weibull selection chose, held fixed across the comparison: a
    cumulative incidence needs both hazards, and what the rule is about is the default model.
    """
    import logging

    from creditsurv.data.fred import load_macro_panel
    from creditsurv.models.families import NotSelected, compare_families
    from creditsurv.reporting import family as family_report

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        comparison = compare_families(
            as_of=as_of,
            moratorium=moratorium,
            macro=load_macro_panel(),
            block_rows=block_rows,
        )
    except NotSelected as missing:
        # Which command to run is something only this layer can say.
        raise typer.BadParameter(str(missing)) from missing

    chosen, written = family_report.generate(
        comparison.gaps,
        comparison.signs,
        formulas=comparison.formulas,
        reports_dir=reports_dir(),
    )
    typer.echo(f"\nThe rule chooses: {chosen}")
    if chosen != DISTRIBUTION:
        typer.echo(
            f"config.DISTRIBUTION is {DISTRIBUTION}. Set it to {chosen} and re-run "
            "`creditsurv select` -- every fit of it is cached, so it costs minutes -- so that "
            "selection.json is the record of the published model."
        )
    typer.echo(f"Written: {written}")


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
    from creditsurv.models.fits import cached_fit, fit_once, selection_start
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
        fit_once(
            split.train,
            covariates,
            formula,
            as_of=as_of,
            reuse=reuse,
            moratorium=moratorium,
            start=selection_start(as_of, moratorium),
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
            fit=cached_fit(as_of=as_of, moratorium=moratorium),
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
    block_rows: Annotated[int, typer.Option(help="Cells read at a time.")] = 250_000,
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

    from creditsurv.config import tables_dir
    from creditsurv.data.book import MoratoriumPolicy
    from creditsurv.data.fred import load_macro_panel
    from creditsurv.data.panel import AGE, OUTCOME, WEIGHT
    from creditsurv.data.store import load_cells
    from creditsurv.views.build import NoCachedFit, model_views
    from creditsurv.views.portfolio import portfolio_views
    from creditsurv.views.selection import selection_views
    from creditsurv.views.tables import write_views

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    destination = tables_dir()

    if model:
        try:
            built = model_views(
                as_of=as_of,
                moratorium=moratorium,
                covariates=default_covariates(),
                formula=default_formula(),
                macro=load_macro_panel(),
                block_rows=block_rows,
                loans=loans,
                horizon=horizon,
            )
        except NoCachedFit as missing:
            # Views never fit, and which command does is something only this layer can say.
            typer.echo(str(missing))
            raise typer.Exit(1) from missing
        write_views(built.tables, destination, fit=built.fingerprint)
        typer.echo(f"  {len(built.tables)} views of fit {built.fingerprint}")
        del built

    if portfolio:
        typer.echo("The book by segment, the lending, the vintage curves and the macro series...")
        cells = load_cells(moratorium, columns=["origination_month", AGE, WEIGHT, OUTCOME])
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
    dist: Annotated[
        str, typer.Option(help="weibull or loglogistic: the family the whole run uses.")
    ] = DISTRIBUTION,
    cause: Annotated[
        str, typer.Option(help="default or prepayment: which exit is being modelled.")
    ] = DEFAULT_CAUSE,
    workers: Annotated[
        int, typer.Option(help="Processes each fit's likelihood is evaluated in; traced only.")
    ] = 1,
    block_rows: Annotated[int, typer.Option(help="Cells read at a time.")] = 250_000,
    traced: Annotated[
        bool,
        typer.Option(
            "--traced/--written-out",
            help="Trace lifelines' likelihood with autograd, re-reading the rows for every fit.",
        ),
    ] = False,
) -> None:
    """Run the variable selection on the training half, and write what it chose.

    Steps 5 to 10 of ``docs/variable_selection.md`` on the whole population up to
    ``--as-of``: correlation, variance inflation, univariate screening, backward
    elimination, stability and materiality. On the whole population that is days, and it
    resumes -- every fit is saved as it lands -- so a run that stops picks up where it was.

    ``--dist`` takes the whole procedure through one family. Rule 2 of ``docs/rules.md``
    compares the two *selected* models rather than two fits of one specification, which is
    only possible because every rule of steps 8, 9 and 10 reads the family's own
    coefficients.

    ``--cause prepayment`` selects the competing model instead, on the same cells: a
    default becomes censoring, and the declared priors are rule 6's rather than the default
    model's. They are not the same priors and cannot be -- a credit score that lengthens
    survival shortens the time to repayment -- so a run with one map and the other cause
    would eliminate covariates for disagreeing with the wrong economics.

    The rows are read **once** and every fit of the run is made from that reading, through
    ``creditsurv.models.kernel``. On the production table a fit is 53 seconds of arithmetic
    behind 12.1 minutes of reading, and the selection used to pay the reading once per
    candidate -- about thirty times, with the fifteen step-7 fits each beginning by
    recomputing the identical base objective to twelve digits. ``--traced`` goes back to
    tracing lifelines' likelihood with autograd and re-reading for every fit: the same
    estimator, and what the equivalence tests hold the other to. ``--workers`` applies to it
    only; a reading is one process and needs no more, at fifteen bytes a row.

    The report goes to ``docs/reports/selection.md`` with ``selection.json`` beside it, the
    record the configuration is tested against. See ``creditsurv.models.procedure``.
    """
    import logging

    import pandas as pd

    from creditsurv.config import MACRO_CANDIDATES
    from creditsurv.data.fred import load_macro_panel
    from creditsurv.data.panel import month_ordinal
    from creditsurv.data.store import cells_identity
    from creditsurv.models.fits import Fits, cell_source
    from creditsurv.models.procedure import (
        BASE_CATEGORICAL,
        CANDIDATE_CATEGORICAL,
        LOAN_CONTINUOUS,
        LOAN_ORDINAL,
        run_selection,
    )
    from creditsurv.models.selection import EXPECTED_SIGNS, PREPAYMENT_SIGNS
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

    # **The training half is never an object.** It used to be expanded whole -- 91.6 million
    # cells, fifteen macro columns, a frame of many gigabytes -- and every one of the twenty-odd
    # fits then held its stored design beside it. Here the cell file is described instead, and
    # each fit reads its own share of the parquet in `workers` processes: the same
    # specification the two ways agrees to 9.4e-07 standard errors, at 4.7 GB against 15.
    #
    # Nothing of the test half is read either way: the window stops at the reporting date.
    cut = month_ordinal(reporting_date)
    source = cell_source(
        moratorium, load_macro_panel(), candidates, block_rows=block_rows, until=cut, cause=cause
    )

    # **Steps 5 and 6 no longer read anything of their own.** They need a weighted covariance
    # of the continuous candidates, and that pass was 19 columns over 72.7 million rows -- the
    # last one the encoding had not replaced: 4.6 minutes per selection against 19.7 seconds
    # off the keys, and cached by nothing. `run_selection` takes the sums off the reading it
    # makes for the fits anyway, so
    # the counts come with them and this command has nothing to count before it starts.
    identity = cells_identity(moratorium)
    typer.echo(
        f"Selecting {cause} from {source.source} with the {dist} family"
        + (f" in {workers} process(es), re-reading for every fit." if traced else ", read once.")
    )

    fits = Fits(
        None,
        identity=identity,
        as_of=as_of,
        moratorium=moratorium,
        distribution=dist,
        cause=cause,
        blocks=source,
        workers=workers,
        # Declared in `config` before any fit, with its own argument: "a macro covariate is a
        # function of the vintage quarter and the loan age, both already in the aggregation
        # key". That is exactly the split the kernel needs.
        calendar=None if traced else MACRO_CANDIDATES,
    )
    record = run_selection(
        None,
        fits,
        stability=True,
        signs=EXPECTED_SIGNS if cause == DEFAULT_CAUSE else PREPAYMENT_SIGNS,
    )
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
    from creditsurv.models.fits import fit_once
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
            fit_once(split.train, covariates, formula, as_of=as_of, reuse=True, moratorium=policy),
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

    from creditsurv.data.aggregate import defaults_by_month
    from creditsurv.data.book import MoratoriumPolicy
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


if __name__ == "__main__":  # pragma: no cover
    app()
