"""Collapsing 1.75 billion loan-months into a table a model can be fitted to.

Episodes that agree on every covariate and on their position in time are
exchangeable, so they can be replaced by one row carrying a count. That is the whole
trick, and at this scale it is not an optimisation but the only thing that makes the
problem tractable: a fit over two billion rows is out of reach, a fit over a million
weighted cells is a minute.

The work is done in DuckDB rather than pandas because the join, the truncation and
the group-by all have to happen out of core — the inputs are far larger than memory
and never need to be resident.

**The macro series are deliberately absent from the grouping key.** Since
``period = origination_period + age``, unemployment, house prices and financial conditions
are a deterministic function of two columns that are already in the key, so they can
be recomputed on the aggregate at no cost in cardinality. Putting them in the key
instead would multiply it by the number of distinct months and destroy the collapse.

An earlier measurement in this project found the same technique compressing
1.00x and concluded it was not worth having. That was true at 215,000 rows, where
the possible cells vastly outnumbered the rows. At 1.75 billion the ratio inverts.
The measurement was right for the regime it was taken in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from creditsurv.data.book import (
    CATEGORICAL,
    CATEGORICAL_LEVELS,
    EPISODE_MONTHS,
    SOURCE,
    MoratoriumPolicy,
    PathSpec,
    case_expression,
    connect,
    sources,
    state_of_the_book_sql,
)
from creditsurv.data.store import cells_writer

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import duckdb
from creditsurv.features import BIN_EDGES

_LOGGER: Final = logging.getLogger(__name__)


#: Fields taking exactly one value across the whole dataset. Recorded rather than
#: quietly omitted, so the next reader does not spend an afternoon adding them back.
DEGENERATE_FIELDS: Final[tuple[str, ...]] = (
    "amortization_type",  # FRM, 100%
    "interest_only_indicator",  # N, 100%
)


@dataclass(frozen=True)
class CellSpec:
    """Which covariates enter the aggregation, and how coarsely.

    Deliberately a parameter rather than a constant. The cell count is the product of
    every covariate's band count, so the specification *is* the cardinality, and it
    cannot be fixed before variable selection has said which covariates earn their
    place. Aggregating on everything available and selecting afterwards is the wrong
    order: it produces a table too large to fit, which is exactly what happened here
    on the first attempt.

    Measured on 1999Q1 (27.7 million loan-months):

    ==========================================  ==========  ============
    Specification                               Cells       Compression
    ==========================================  ==========  ============
    9 continuous + 9 categorical, monthly ages  12,752,331          2.2x
    same, quarterly ages                        12,752,331          2.2x
    same, banded ages                              919,634         30.1x
    4 coarse continuous + 3 categorical, banded     14,221      1,948.0x
    ==========================================  ==========  ============
    """

    continuous: dict[str, tuple[float, ...]]
    categorical: tuple[str, ...]
    episode_months: int = EPISODE_MONTHS

    def validate(self) -> None:
        unknown = set(self.continuous) - set(SOURCE)
        if unknown:
            message = f"Unknown continuous covariate(s): {sorted(unknown)}"
            raise ValueError(message)
        unknown = set(self.categorical) - set(CATEGORICAL)
        if unknown:
            message = f"Unknown categorical covariate(s): {sorted(unknown)}"
            raise ValueError(message)


#: Coarse by design: five bands a covariate, three categoricals, banded ages. Chosen
#: from the measurement above as the point where the whole population fits in roughly
#: a million cells. Variable selection may replace it.
#: The classing actually used to build cells, as a **subset of** ``BIN_EDGES``.
#:
#: Two schemes used to coexist and only one was in the production path. The
#: documentation justified a DTI break at 43, "a long-standing underwriting threshold",
#: while the model used 45; it named LTV breaks at 85 and 95 that the model did not
#: have. A reader checking the economic justification of the bands found a
#: justification that did not describe the model.
#:
#: They are reconciled by making this grid a strict subset of the documented one -- a
#: test enforces it -- so every boundary that exists is one the documentation argues
#: for, and the coarsening is the only thing left to explain.
#:
#: **Why coarser rather than unified on the full set.** The cell count is the product
#: of the band counts. BIN_EDGES gives 8 x 8 x 6 = 384 combinations against 80 here, a
#: 4.8x multiplier, and it would land on top of the 3.11x the exact calendar key costs
#: and the 1.23x of the two loan covariates -- seventeen times the table. The
#: thresholds dropped are the finer ones; the MI break at 80 and the underwriting
#: break at 43 survive, and those are the two the economics actually turns on.
PRODUCTION_EDGES: Final[dict[str, tuple[float, ...]]] = {
    "credit_score": (580.0, 660.0, 700.0, 740.0, 760.0, 820.0),
    "original_ltv": (30.0, 70.0, 80.0, 90.0, 100.0),
    "debt_to_income": (10.0, 28.0, 36.0, 43.0, 55.0),
}

#: The documented grid, whole: 8 bands of credit score, 8 of loan-to-value, 6 of
#: debt-to-income against 5 / 4 / 4 above.
#:
#: Read from BIN_EDGES rather than copied, so the coarse grid and the fine one cannot drift
#: apart and the fine one needs no separate justification: it *is* the documentation's grid.
#: What it costs is 384 combinations against 80, a 4.8x ceiling that the sparsity of the cell
#: space cuts to far less -- how much less is measured, not assumed, in `creditsurv profile`.
FINE_EDGES: Final[dict[str, tuple[float, ...]]] = {
    name: BIN_EDGES[name] for name in PRODUCTION_EDGES
}


class Extension(StrEnum):
    """An addition to the cell key, switchable one at a time.

    The cell count is the product of the band counts, so an extension cannot be adopted
    because it sounds right: it has to be measured against the ceiling in `docs/rules.md`,
    which is why each one is a flag rather than an edit to the specification. ``creditsurv
    profile`` prices them one by one and in combination, on nine quarters.
    """

    HARP = "harp"
    ORIGINATION_SPREAD = "origination_spread"
    DELINQUENCY_STATE = "delinquency_state"
    FINE_BANDS = "fine_bands"


#: The order the extensions are given up in if the measured table exceeds the 150 million
#: cell ceiling, fixed in `docs/rules.md` **before** the measurement, so that what survives is
#: not chosen by what came out large.
#:
#: HARP is not on the list. It is a correction of what the model covers -- 18% of a decade of
#: vintages, at three times the default rate of the loans kept -- not a refinement of it, and
#: the three-state outcome is the same kind of thing. The refinements go first: the finer
#: bands, then the origination spread, then the payment state.
GIVE_UP_ORDER: Final[tuple[Extension, ...]] = (
    Extension.FINE_BANDS,
    Extension.ORIGINATION_SPREAD,
    Extension.DELINQUENCY_STATE,
)


def extended(base: CellSpec, *extensions: Extension) -> CellSpec:
    """``base`` with each extension switched on.

    Order is irrelevant and repetition harmless, so a caller can build the whole lattice of
    specifications by feeding it subsets.
    """
    continuous = dict(base.continuous)
    categorical = list(base.categorical)
    # In declaration order, not the caller's, so the cell file's columns do not depend on
    # how the specification was written down.
    for extension in (member for member in Extension if member in set(extensions)):
        if extension is Extension.FINE_BANDS:
            continuous.update(FINE_EDGES)
        elif extension is Extension.ORIGINATION_SPREAD:
            # The note rate, not the spread: the spread is the rate against the market rate
            # of the origination month, and the month is already in the key, so the rate is
            # the only part of it a cell has to carry.
            continuous["note_rate"] = BIN_EDGES["note_rate"]
        elif extension.value not in categorical:
            categorical.append(extension.value)
    spec = CellSpec(
        continuous=continuous, categorical=tuple(categorical), episode_months=base.episode_months
    )
    spec.validate()
    return spec


#: The specification before the extensions of September 2026, kept so that each of them can
#: be priced against it rather than against a moving baseline.
BASE_SPEC: Final = CellSpec(
    continuous=PRODUCTION_EDGES,
    # ltv_change is absent on purpose: it is a function of original_ltv and the macro
    # path, both recoverable from the key, so carrying it would multiply the
    # cardinality for information already there.
    #
    # mortgage_insurance and buyer_type are present because the argument that excluded them
    # was wrong. It asserted a cost of "up to 16x the table" from the product of the
    # level counts; the cell space is sparse and the measured cost of the two together
    # is **1.23x**. Mortgage insurance is a classic credit predictor and already had an
    # expected sign in the code. The claim that the loan side could not afford more
    # covariates did not survive being measured.
    categorical=(
        "purpose",
        "occupancy",
        "term_years",
        "mortgage_insurance",
        "buyer_type",
    ),
)

#: What the cells are built with: the base key, the HARP level and the payment state.
#:
#: **Chosen by the give-up order, not by preference.** `creditsurv profile --extensions`
#: priced every extension on nine quarters against the 150 million cell ceiling
#: (`docs/reports/key_extensions.csv`), projecting each multiple onto the 63.6 million cells
#: the base key produced:
#:
#: ===========================================  ==========  ==================
#: Specification                                Multiple    Projected cells
#: ===========================================  ==========  ==================
#: harp                                             1.063          67,650,751
#: delinquency_state                                1.185          75,403,551
#: origination_spread                               2.128         135,439,974
#: fine_bands                                       2.276         144,854,977
#: all four                                         4.903         312,015,537
#: less the finer bands                             2.544         161,891,606
#: less the finer bands and the spread              1.264          80,416,459
#: ===========================================  ==========  ==================
#:
#: All four is twice the ceiling, and giving up the finer bands alone still leaves it over,
#: so the second rung goes too and the key stops here. The order was fixed in
#: `docs/rules.md` before any of this was measured, which is the only reason the answer is
#: not the one that happened to be convenient: the finer bands are individually affordable
#: at 144.9 million and go first anyway, because a band grid is a refinement and the HARP
#: level and the payment state are corrections of what the model covers.
#:
#: What it costs is the loan's own note rate, and with it the spread at origination and the
#: refinancing incentive. The market rate's fall since origination survives, free, and
#: carries the first-order part of both: within a month the note rates of this book span a
#: point, where the rate itself has moved several since 2021.
DEFAULT_SPEC: Final = extended(BASE_SPEC, Extension.HARP, Extension.DELINQUENCY_STATE)


def _age_expression(step: int) -> str:
    """Loan age collapsed to the start of its episode, in months.

    The lower edge rather than an index, so the value keeps the units of loan age and
    the episode bounds read straight off it.
    """
    return f"CAST(age / {step} AS INTEGER) * {step} AS age"


def _not_null_filter(spec: CellSpec) -> str:
    """WHERE clause dropping rows whose categorical mapping came back NULL."""
    if not spec.categorical:
        return ""
    conditions = " AND ".join(f"{name} IS NOT NULL" for name in spec.categorical)
    return f"WHERE {conditions}"


#: The loan's origination month, as an ordinal in months since year zero.
#:
#: This replaced the vintage *quarter*, and the difference is two months of calendar
#: on every macro covariate in the model. Loans in a quarter are not all written in its
#: first month -- the mean offset is **+2.15 months** -- so reconstructing the month
#: from the quarter reads every macro series that much late, and pushes the backtest
#: boundary two months inside the training half where ``assert_no_lookahead`` cannot
#: see it, because it checks the distorted quantity.
#:
#: Verified by cross-correlating the true monthly default series against the
#: reconstructed one: the maximum sat at a lag of **+2**, not 0.
#:
#: It costs 3.11x the cells, measured. The convention matches ``_months_to_periods``:
#: ``year * 12 + (month - 1)``.
_ORIGINATION_MONTH: Final = (
    "(period_key // 100) * 12 + (period_key % 100) - 1 - age AS origination_month"
)


def _select_columns(spec: CellSpec) -> str:
    """Every covariate column of the SELECT, as SQL.

    Assembled from a list rather than interpolated as separate blocks: an empty
    continuous or categorical set would otherwise leave a dangling comma and fail
    with a parser error that says nothing about the specification that caused it.
    """
    # A band of NULL is not a gap in the table: it is the loan not reporting the ratio,
    # which only a HARP refinance does, and the key carries `harp` to say so. What the
    # model does with it is decided on the model's side, in features.absorb_not_reported.
    columns = [
        case_expression(SOURCE[name], edges, name) for name, edges in spec.continuous.items()
    ]
    columns += [f"{CATEGORICAL[name]} AS {name}" for name in spec.categorical]
    columns.append(_age_expression(spec.episode_months))
    columns.append(_ORIGINATION_MONTH)
    # Three states, not a flag: a cell's loan-months ended in default, in a voluntary
    # repayment, or in neither. Prepayment is a competing risk, and a model of it needs to
    # tell the two exits apart.
    columns.append("outcome")
    return ",\n            ".join(columns)


def _cells_for_quarter(
    connection: duckdb.DuckDBPyConnection,
    perf_path: str,
    orig_path: str,
    vintage: str,
    spec: CellSpec,
    policy: MoratoriumPolicy,
) -> pd.DataFrame:
    """Aggregate one vintage quarter.

    Every loan appears in exactly one quarter's files -- verified, not assumed: the
    identifiers of 1999Q1 and 1999Q2 do not intersect at all. So a quarter can be
    collapsed on its own and the results concatenated, which keeps memory flat.

    The vintage is attached as a constant too, but only as a label: the key carries
    the **origination month**, derived from the data as ``period - age``. With the
    month and the age in the key the observation month follows exactly, which is what
    lets the macro series stay out of the key entirely and be read at the right date.
    """
    query = f"""
    WITH book AS (
        {state_of_the_book_sql(policy, harp_level=Extension.HARP in spec.categorical)}
    ), classed AS (
        SELECT
            {_select_columns(spec)}
        FROM book
    )
    SELECT '{vintage}' AS vintage, *, COUNT(*) AS loan_months
    FROM classed
    -- A categorical that mapped to NULL is a code nobody has looked at. The loan is
    -- dropped rather than aggregated into a NULL level, for the same reason a loan
    -- with no credit score is dropped: it cannot be modelled, and imputing the
    -- category would invent the thing being measured.
    {_not_null_filter(spec)}
    GROUP BY ALL
    """
    frame: pd.DataFrame = connection.execute(query, [perf_path, orig_path]).df()
    return frame


def _levels_for(spec: CellSpec, quarters: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Every text column's full level set for this build, known before a row is read.

    Three sources, and the third is the only one that is not a declaration: the mapped
    categoricals in the key come from :data:`CATEGORICAL_LEVELS`, the outcome from the three
    causes the aggregation writes, and ``vintage`` from the quarters being aggregated -- which
    are the files on disk, so they are known at the top of the run rather than discovered by it.
    """
    from creditsurv.config import CENSORED, DEFAULT_CAUSE, PREPAYMENT_CAUSE

    levels = {
        name: CATEGORICAL_LEVELS[name] for name in spec.categorical if name in CATEGORICAL_LEVELS
    }
    levels["outcome"] = tuple(sorted((DEFAULT_CAUSE, PREPAYMENT_CAUSE, CENSORED)))
    levels["vintage"] = tuple(sorted(quarters))
    return levels


def build_cells(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    spec: CellSpec = DEFAULT_SPEC,
    policy: MoratoriumPolicy = MoratoriumPolicy.EXCLUDE,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Aggregate the parquet panel into weighted cells, one quarter at a time.

    The key is the coarse-classed covariates together with the origination month, the
    loan age and the event flag. The weight is the loan-month count.

    ``policy`` decides what a delinquency the borrower was not required to cure counts
    as. It is a parameter rather than a constant because the two defensible treatments
    are not equivalent and the choice is settled by measuring the difference -- see
    :class:`MoratoriumPolicy`.
    """
    perf = sources(perf_source, "perf")
    orig = sources(orig_source, "orig")
    if not perf or not orig:
        message = "No ingested quarters found. Run `creditsurv ingest` first."
        raise FileNotFoundError(message)

    spec.validate()
    return _concatenate(list(_quarters_of_cells(perf, orig, spec, policy, connection)))


def _quarters_of_cells(
    perf: list[str],
    orig: list[str],
    spec: CellSpec,
    policy: MoratoriumPolicy,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> Iterator[pd.DataFrame]:
    """One quarter's cells at a time, each already in its final types and levels.

    Vintage is in the key and constant within a quarter, so the pieces are disjoint: nothing
    downstream needs a second group-by, and nothing needs to see two of them at once.
    """
    con = connection or connect()
    levels = _levels_for(spec, [Path(path).stem for path in perf])
    for perf_path, orig_path in zip(sorted(perf), sorted(orig), strict=True):
        vintage = Path(perf_path).stem
        cells = _compact(
            _cells_for_quarter(con, perf_path, orig_path, vintage, spec, policy), levels
        )
        _LOGGER.info(
            "%s: %d cells from %d loan-months", vintage, len(cells), int(cells["loan_months"].sum())
        )
        yield cells


def write_cells(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    spec: CellSpec = DEFAULT_SPEC,
    policy: MoratoriumPolicy = MoratoriumPolicy.EXCLUDE,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> int:
    """Aggregate the panel and write it, holding one quarter at a time. Returns the cells written.

    What :func:`build_cells` does, without ever holding the table. It used to keep every
    quarter's frame -- so the categorical levels could be unified across them -- then
    concatenate, which is two live copies of 4.76 GB at 91.6 million cells, and then let Arrow
    make a third while writing: the measured peak was **11.3 GB**. At 200 million cells the
    concatenation alone wants about 21 GB, on a 16 GB machine, which is what put the finer bands
    of `docs/rules.md` out of reach before any fit was attempted.

    The levels come from :data:`CATEGORICAL_LEVELS` instead, so a quarter arrives with the whole
    schema and parquet can append it.
    """
    perf = sources(perf_source, "perf")
    orig = sources(orig_source, "orig")
    if not perf or not orig:
        message = "No ingested quarters found. Run `creditsurv ingest` first."
        raise FileNotFoundError(message)

    spec.validate()
    written = 0
    with cells_writer(policy.value) as write:
        for cells in _quarters_of_cells(perf, orig, spec, policy, connection):
            write(cells)
            written += len(cells)
    _LOGGER.info("Collapsed to %d cells", written)
    return written


def _compact(cells: pd.DataFrame, levels: Mapping[str, tuple[str, ...]]) -> pd.DataFrame:
    """One quarter's cells, in the types they should have come back in.

    DuckDB returns text as Python strings, one object per value. On the quarter-keyed
    table three such columns were 44% of the episode frame; the exact key carries five
    over some 66 million cells. So they become categorical as each quarter arrives,
    before a table of strings can exist, and the origination month -- an ordinal near
    24,000 -- is kept as a 32-bit integer.

    ``levels`` gives each text column the **whole** set it can hold, not the part this quarter
    happened to see. That is what lets the quarters be written one at a time: they share a
    schema, so parquet can append them. Collecting the levels from the quarters instead and
    unifying them afterwards needs every quarter resident at once, which is where the 11.3 GB
    peak came from.
    """
    for column in cells.columns:
        if cells[column].dtype == object:
            declared = levels.get(column)
            cells[column] = pd.Categorical(
                cells[column], categories=None if declared is None else list(declared)
            )
            unmapped = cells[column].isna().sum() if declared is not None else 0
            if unmapped:
                message = (
                    f"{unmapped} cells of {column!r} hold a value outside its declared levels "
                    f"{tuple(declared or ())}. A level was added to the mapping and not to "
                    "CATEGORICAL_LEVELS."
                )
                raise ValueError(message)
    if "origination_month" in cells.columns:
        cells["origination_month"] = cells["origination_month"].astype("int32")
    return cells


def _concatenate(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack the quarters, keeping every categorical column categorical.

    pandas keeps a categorical through ``concat`` only when every piece declares the
    same levels, and otherwise turns the whole column back into strings: one quarter
    without an investor loan would undo :func:`_compact` for the entire table. The
    levels are unified first, and sorted, so they do not depend on reading order.
    """
    if not frames:
        message = "No cells to concatenate."
        raise ValueError(message)
    categorical = [
        column
        for column in frames[0].columns
        if isinstance(frames[0][column].dtype, pd.CategoricalDtype)
    ]
    for column in categorical:
        levels = sorted({level for frame in frames for level in frame[column].cat.categories})
        for frame in frames:
            frame[column] = frame[column].cat.set_categories(levels)
    return pd.concat(frames, ignore_index=True)


def cardinality_report(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    spec: CellSpec = DEFAULT_SPEC,
) -> pd.DataFrame:
    """How far the collapse actually gets, key by key.

    Run before fixing the grain, not after. The plan calls for reducing the vintage
    to a year, or the episode to a quarter, if the cells run past what a fit can
    carry — and that decision needs a number rather than an intuition.
    """
    perf = sources(perf_source, "perf")
    orig = sources(orig_source, "orig")
    con = connect()

    counted = con.execute(
        f"SELECT COUNT(*) FROM ({state_of_the_book_sql()})", [perf, orig]
    ).fetchone()
    rows = int(counted[0]) if counted else 0
    cells = build_cells(perf, orig, spec=spec, connection=con)
    return pd.DataFrame(
        [
            {
                "loan_months": rows,
                "cells": len(cells),
                "compression": rows / max(len(cells), 1),
                "weight_total": int(cells["loan_months"].sum()),
            }
        ]
    )


#: Origination fields whose absence drops a loan before the categorical keys are read.
_COMPLETE_CASE_FIELDS: Final[tuple[str, ...]] = ("credit_score", "original_ltv", "debt_to_income")


def incomplete_cases(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    spec: CellSpec = DEFAULT_SPEC,
    policy: MoratoriumPolicy = MoratoriumPolicy.EXCLUDE,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """The loans the cells leave out, vintage by vintage, and how they default.

    The validation's D4. A loan missing its credit score, loan-to-value or debt-to-income,
    or carrying a categorical code no mapping names, is dropped rather than imputed:
    imputing an underwriting characteristic invents the thing being measured. Dropping is
    harmless only if what goes is small or looks like what stays, and neither can be
    assumed -- the validation found the share varying by two orders of magnitude across
    vintages, almost all of it missing debt-to-income, and the dropped loans riskier.

    One row per vintage: loans, how many are dropped, how many lack each field (a loan can
    lack several), and the ever-default rate of the loans kept and of those dropped, under
    ``policy``'s event definition. A pass over every performance file, quarter by quarter.
    """
    perf = sources(perf_source, "perf")
    orig = sources(orig_source, "orig")
    if not perf or not orig:
        message = "No ingested quarters found. Run `creditsurv ingest` first."
        raise FileNotFoundError(message)

    missing = {field: f"{field} IS NULL" for field in _COMPLETE_CASE_FIELDS}
    missing.update({name: f"({CATEGORICAL[name]}) IS NULL" for name in spec.categorical})
    flags = ", ".join(f"BOOL_OR({condition}) AS no_{name}" for name, condition in missing.items())
    dropped = " OR ".join(f"no_{name}" for name in missing)
    counts = ", ".join(f"COUNT(*) FILTER (WHERE no_{name}) AS no_{name}" for name in missing)

    con = connection or connect()
    frames = []
    for perf_path, orig_path in zip(sorted(perf), sorted(orig), strict=True):
        vintage = Path(perf_path).stem
        query = f"""
        WITH book AS ({state_of_the_book_sql(policy, complete_only=False)}),
        loans AS (
            SELECT loan_identifier, BOOL_OR(event) AS defaulted, {flags}
            FROM book
            GROUP BY loan_identifier
        ),
        judged AS (SELECT *, ({dropped}) AS dropped FROM loans)
        SELECT
            '{vintage}' AS vintage,
            COUNT(*) AS loans,
            COUNT(*) FILTER (WHERE dropped) AS dropped,
            {counts},
            AVG(CASE WHEN NOT dropped THEN defaulted::INTEGER END) AS default_rate_kept,
            AVG(CASE WHEN dropped THEN defaulted::INTEGER END) AS default_rate_dropped
        FROM judged
        """
        frames.append(con.execute(query, [perf_path, orig_path]).df())
        _LOGGER.info("%s: incomplete cases counted", vintage)

    table = pd.concat(frames, ignore_index=True)
    table["dropped_share"] = table["dropped"] / table["loans"]
    table["relative_risk"] = table["default_rate_dropped"] / table["default_rate_kept"]
    return table


#: Zero-balance codes that end a loan without a loss and without the borrower choosing to
#: repay: a reperforming loan sale (16) and a removal (96). Both are censoring, and
#: :func:`credit_adjacent_exits` measures what that classification rests on.
CREDIT_ADJACENT_EXITS: Final = ("16", "96")


def credit_adjacent_exits(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    policy: MoratoriumPolicy = MoratoriumPolicy.EXCLUDE,
    codes: tuple[str, ...] = CREDIT_ADJACENT_EXITS,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """How the loans leaving by a reperforming sale or a removal were treated, by vintage.

    The validation's D5: code 16 is credit by definition, since a reperforming loan was
    delinquent once, and counting its sale as censoring could lose a default. Whether it
    does depends on what the book did first. A loan that reached 90 days has already
    defaulted and been cut there, and one that was modified was censored at the
    modification; only a loan still performing when it was sold is censored at the sale.

    One row per vintage and code, over every loan carrying the code, under ``policy``'s
    event definition: how many defaulted first, were censored earlier, or were censored at
    the exit itself. A pass over every performance file, quarter by quarter.
    """
    perf = sources(perf_source, "perf")
    orig = sources(orig_source, "orig")
    if not perf or not orig:
        message = "No ingested quarters found. Run `creditsurv ingest` first."
        raise FileNotFoundError(message)
    listed = ", ".join(f"'{code}'" for code in codes)

    con = connection or connect()
    frames = []
    for perf_path, orig_path in zip(sorted(perf), sorted(orig), strict=True):
        vintage = Path(perf_path).stem
        query = f"""
        WITH book AS ({state_of_the_book_sql(policy, complete_only=False)}),
        per_loan AS (
            SELECT
                loan_identifier,
                BOOL_OR(event) AS defaulted,
                BOOL_OR(left_the_book) AS exited
            FROM book
            GROUP BY loan_identifier
        ),
        coded AS (
            SELECT loan_identifier, MIN(zero_balance_code) AS code
            FROM read_parquet(?)
            WHERE zero_balance_code IN ({listed})
            GROUP BY loan_identifier
        ),
        judged AS (
            SELECT
                c.code,
                COALESCE(o.defaulted, FALSE) AS defaulted,
                COALESCE(o.exited, FALSE) AND NOT COALESCE(o.defaulted, FALSE) AS at_exit
            FROM coded c LEFT JOIN per_loan o USING (loan_identifier)
        )
        SELECT
            '{vintage}' AS vintage,
            code,
            COUNT(*) AS loans,
            COUNT(*) FILTER (WHERE defaulted) AS defaulted_first,
            COUNT(*) FILTER (WHERE NOT defaulted AND NOT at_exit) AS censored_earlier,
            COUNT(*) FILTER (WHERE at_exit) AS censored_at_exit
        FROM judged
        GROUP BY code
        ORDER BY code
        """
        frames.append(con.execute(query, [perf_path, orig_path, perf_path]).df())
        _LOGGER.info("%s: exits by code counted", vintage)
    return pd.concat(frames, ignore_index=True)


def defaults_by_month(
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    spec: CellSpec = DEFAULT_SPEC,
    policy: MoratoriumPolicy = MoratoriumPolicy.EXCLUDE,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.Series:
    """Defaults by calendar month, counted on the loan-months the cells are built from.

    The reference for the validation's M1 test. A cell carries its origination month and
    its age, and the month its defaults happened in is their sum; this counts the same
    defaults without the cells, under the same rules -- the complete-case filter, the
    categorical keys, the moratorium policy -- so the two series have to agree to the unit.
    """
    perf = sources(perf_source, "perf")
    orig = sources(orig_source, "orig")
    if not perf or not orig:
        message = "No ingested quarters found. Run `creditsurv ingest` first."
        raise FileNotFoundError(message)
    spec.validate()

    con = connection or connect()
    frames = []
    for perf_path, orig_path in zip(sorted(perf), sorted(orig), strict=True):
        query = f"""
        WITH book AS ({state_of_the_book_sql(policy)}), classed AS (
            SELECT {_select_columns(spec)}, period_key FROM book
        )
        SELECT
            (period_key // 100) * 12 + (period_key % 100) - 1 AS month,
            SUM(CASE WHEN outcome = 'default' THEN 1 ELSE 0 END) AS defaults
        FROM classed
        {_not_null_filter(spec)}
        GROUP BY 1
        """
        frames.append(con.execute(query, [perf_path, orig_path]).df())
    counts = pd.concat(frames, ignore_index=True).groupby("month")["defaults"].sum()
    return counts.astype("int64").rename("defaults")
