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

    from creditsurv.backtest.splits import Split
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

    import pandas as pd

    from creditsurv.data.fred import load_macro_panel
    from creditsurv.data.ingest import load_manifest
    from creditsurv.data.panel import AGE, EVENT, WEIGHT, default_rate_by_observation_month
    from creditsurv.data.store import DEFAULT_POLICY, cells_path
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
    cells_file = cells_path(DEFAULT_POLICY)
    cells = (
        pd.read_parquet(cells_file, columns=["orig_month", AGE, WEIGHT, EVENT])
        if cells_file.exists()
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
    typer.echo(f"{len(cells):,} cells covering {int(cells['n'].sum()):,} loan-months")
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
    halves = (train["orig_month"].to_numpy() // 12) % 2 == 0

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

    described = {
        "as_of": as_of,
        # In the fingerprint because the two treatments can produce panels of similar
        # size, and a censor fit silently reused for exclude would compare a model with
        # itself.
        "moratorium": moratorium,
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

    fitted = fit_aft(train, covariates, formula, weights_col=WEIGHT, initial_point=start)
    typer.echo(f"  {fitted.elapsed_seconds / 60:.1f} minutes")
    path = save_fit(fitted, fingerprint, {**described, "minutes": fitted.elapsed_seconds / 60})
    typer.echo(f"  saved to {path}")
    return fitted


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
