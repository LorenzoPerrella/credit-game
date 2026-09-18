"""Screening the covariates **before** anything is aggregated.

This step exists because the order matters, and the first version of this pipeline
had it backwards. Binning and grouping were done first, from a specification chosen
in advance, and the screening came afterwards — by which point the bands were baked
into two million cells and a mis-binned covariate could only be found by noticing its
coefficient had the wrong sign.

`nmds` runs the other way round, and is right to:

```
load → frequency screening → class merging → GROUP BY
```

Screening first means the cut points, the merges and the exclusions are decided from
what the data actually looks like, and the aggregation then encodes decisions already
made. It also means the decisions are recorded rather than assumed.

Everything here reads the ingested parquet directly, one quarter at a time, and works
on the whole history rather than a sample: 2.9 billion loan-months is too much to
hold but not too much to count.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

import pandas as pd

from creditsurv.data.aggregate import (
    _CATEGORICAL,
    _SOURCE,
    GIVE_UP_ORDER,
    _connect,
    _resolve,
    _state_of_the_book_sql,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import duckdb

    from creditsurv.data.aggregate import CellSpec, PathSpec

_LOGGER: Final = logging.getLogger(__name__)

#: A level holding less than this share of exposure is merged, following `nmds`.
MIN_CATEGORY_SHARE: Final = 0.05

#: A covariate whose largest level holds at least this much has nothing left to
#: estimate a contrast from.
DOMINANCE_THRESHOLD: Final = 0.99

#: The most cells the table may hold, fixed in `docs/rules.md` before any of this was
#: measured: about 2.4 times the 63.6 million the published table holds. What it protects is
#: the aggregation itself and the hours every later fit costs, not a fit's memory -- the
#: engine reads the file a batch at a time and does not care how long it is.
CELL_CEILING: Final = 150_000_000

#: Quantiles used to propose cut points for a continuous covariate.
_QUANTILES: Final = (0.05, 0.20, 0.40, 0.60, 0.80, 0.95)


def _accumulate(
    perf: Sequence[str],
    orig: Sequence[str],
    inner_select: str,
    group_by: str,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Run one screening query per quarter and add the pieces together.

    Quarter by quarter for the same reason the aggregation is: every loan lives in
    exactly one quarter's files, so the counts simply sum, and nothing has to be
    resident.
    """
    con = connection or _connect()
    frames = []
    for perf_path, orig_path in zip(sorted(perf), sorted(orig), strict=True):
        query = f"""
        WITH book AS ({_state_of_the_book_sql()})
        SELECT {inner_select},
               COUNT(*) AS loan_months,
               SUM(CASE WHEN event THEN 1 ELSE 0 END) AS events
        FROM book
        GROUP BY {group_by}
        """
        frames.append(con.execute(query, [perf_path, orig_path]).df())

    combined = pd.concat(frames, ignore_index=True)
    keys = [column for column in combined.columns if column not in ("loan_months", "events")]
    return (
        combined.groupby(keys, observed=True, dropna=False)[["loan_months", "events"]]
        .sum()
        .reset_index()
    )


def profile_categorical(
    column: str,
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Exposure share and default rate for each level of a categorical covariate.

    The two things a level is judged on. A level below the minimum share is merged
    rather than dropped; a covariate whose largest level dominates has nothing to
    contribute at all.
    """
    expression = _CATEGORICAL[column]
    table = _accumulate(
        _resolve(perf_source, "perf"),
        _resolve(orig_source, "orig"),
        f"{expression} AS level",
        "1",
        connection=connection,
    )
    table["share"] = table["loan_months"] / table["loan_months"].sum()
    table["default_rate"] = table["events"] / table["loan_months"]
    table["verdict"] = [
        "merge: below minimum share" if share < MIN_CATEGORY_SHARE else "keep"
        for share in table["share"]
    ]
    return table.sort_values("share", ascending=False).reset_index(drop=True)


def propose_cut_points(
    column: str,
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    quantiles: Sequence[float] = _QUANTILES,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> list[float]:
    """Quantiles of a continuous covariate, as candidate band edges.

    A starting point, not an answer. Data-driven cuts fit the sample they were taken
    from, so the ones actually used come from credit conventions — 80, 85, 90, 95 on
    loan-to-value because that is where mortgage insurance and pricing tiers change.
    These say where the mass is, so a convention that would leave a band nearly empty
    is visible before it is adopted.
    """
    con = connection or _connect()
    expression = _SOURCE[column]
    quantile_list = ", ".join(str(q) for q in quantiles)

    # One quarter is enough to place the quantiles; this is a starting point.
    perf = _resolve(perf_source, "perf")
    orig = _resolve(orig_source, "orig")
    query = f"""
    WITH book AS ({_state_of_the_book_sql()})
    SELECT QUANTILE_CONT({expression}, [{quantile_list}]) AS edges FROM book
    """
    result = con.execute(query, [sorted(perf)[0], sorted(orig)[0]]).fetchone()
    return [round(float(value), 3) for value in result[0]] if result else []


def profile_continuous(
    column: str,
    edges: Sequence[float],
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Exposure and default rate for a continuous covariate, band by band.

    Read for **monotonicity**. A covariate whose default rate rises and falls across
    its own bands is either mis-binned or is measuring something other than what its
    name says. That check is what caught the mark-to-market leverage carrying an
    untreated missing-value sentinel: credit score and loan-to-value ordered their
    own risk cleanly and this one did not.
    """
    expression = _SOURCE[column]
    clauses = " ".join(
        f"WHEN {expression} <= {edge} THEN {index}" for index, edge in enumerate(edges)
    )
    band = f"CASE WHEN {expression} IS NULL THEN -1 {clauses} ELSE {len(edges)} END"

    table = _accumulate(
        _resolve(perf_source, "perf"),
        _resolve(orig_source, "orig"),
        f"{band} AS band",
        "1",
        connection=connection,
    )
    labels = ["<missing>", *(f"<= {edge}" for edge in edges), f"> {edges[-1]}"]
    table["range"] = [labels[int(index) + 1] for index in table["band"]]
    table["share"] = table["loan_months"] / table["loan_months"].sum()
    table["default_rate"] = table["events"] / table["loan_months"]
    return table.sort_values("band").reset_index(drop=True)


def is_monotonic(profile: pd.DataFrame) -> bool:
    """Whether default rate moves in one direction across the populated bands.

    Missing is excluded: it is not a point on the covariate's scale, and including it
    would report a break that says nothing about the binning.
    """
    populated = profile[(profile["band"] >= 0) & (profile["loan_months"] > 0)]
    rates = populated.sort_values("band")["default_rate"]
    return bool(rates.is_monotonic_increasing or rates.is_monotonic_decreasing)


def screen_categoricals(
    columns: Sequence[str] = tuple(_CATEGORICAL),
    perf_source: PathSpec = None,
    orig_source: PathSpec = None,
) -> pd.DataFrame:
    """One row per categorical covariate: levels, dominance, and what to do.

    The whole-history version of the screening `nmds` runs before its group-by.
    """
    con = _connect()
    rows = []
    for column in columns:
        profile = profile_categorical(column, perf_source, orig_source, connection=con)
        largest = float(profile["share"].max())
        rare = profile[profile["share"] < MIN_CATEGORY_SHARE]
        rows.append(
            {
                "covariate": column,
                "levels": len(profile),
                "largest_share": largest,
                "rare_levels": len(rare),
                "verdict": (
                    "drop: degenerate"
                    if largest >= DOMINANCE_THRESHOLD
                    else ("merge rare levels" if len(rare) else "keep")
                ),
            }
        )
        _LOGGER.info("%s: %d levels, largest %.1f%%", column, len(profile), largest * 100)
    return pd.DataFrame(rows).sort_values("largest_share", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# What an extension of the cell key costs
# --------------------------------------------------------------------------------------

#: The quarters an extension is priced on, fixed before the measurement.
#:
#: Nine, three from each regime the book has lived through: the years before the crisis, the
#: crisis and the HARP window it opened, and the decade after. An extension's cost is a
#: *ratio* of cell counts, and a ratio measured only where the book is quiet would understate
#: what a key does in the years that fill it -- 2009 to 2015 is where HARP, the refinancing
#: waves and the widest spread of note rates all sit.
SAMPLE_QUARTERS: Final[tuple[str, ...]] = (
    "2001Q2",
    "2005Q3",
    "2007Q1",
    "2009Q3",
    "2012Q2",
    "2015Q1",
    "2018Q2",
    "2020Q4",
    "2022Q3",
)


def _quarter_sources(quarter: str) -> tuple[str, str]:
    from creditsurv.data.ingest import Quarter

    year, number = int(quarter[:4]), int(quarter[-1])
    return (
        str(Quarter(year, number).parquet_path("perf")),
        str(Quarter(year, number).parquet_path("orig")),
    )


def _priced_specifications() -> dict[str, CellSpec]:
    """Every specification the measurement prices, in the order it reports them.

    The base, each extension on its own -- which is the only way a cost can be attributed
    to one of them -- then all four together, and then the give-up ladder of
    ``docs/rules.md`` walked one rung at a time, so the first specification under the
    ceiling can be read straight off the table.
    """
    from creditsurv.data.aggregate import BASE_SPEC, Extension, extended

    specifications = {"base": BASE_SPEC}
    for extension in Extension:
        specifications[extension.value] = extended(BASE_SPEC, extension)
    kept = list(Extension)
    specifications["all"] = extended(BASE_SPEC, *kept)
    dropped: list[str] = []
    for given_up in GIVE_UP_ORDER:
        kept.remove(given_up)
        dropped.append(given_up.value)
        specifications["all less " + ", ".join(dropped)] = extended(BASE_SPEC, *kept)
    return specifications


def extension_cost(
    quarters: Sequence[str] = SAMPLE_QUARTERS,
    *,
    published_cells: int | None = None,
) -> pd.DataFrame:
    """What each extension of the cell key costs, in cells, measured rather than bounded.

    The number that matters is not a quarter's own cell count but the **ratio** to the same
    quarters aggregated on the key as it was, applied to the table that key produced. A
    ceiling from the product of the level counts is not a cost: most combinations never
    occur, which is how *mortgage insurance* and *buyer type* came out at 1.19x where an
    unmeasured sixteenfold had kept them out of every screen.

    ``published_cells`` is the row count of the cell table built on the base key, and the
    projection is that count times the ratio. Left out, it is read from the cell file on
    disk, and the projection is omitted if there is none.
    """
    from creditsurv.data.aggregate import build_cells

    counts: dict[str, dict[str, int]] = {}
    for quarter in quarters:
        perf, orig = _quarter_sources(quarter)
        for name, spec in _priced_specifications().items():
            cells = build_cells(perf, orig, spec=spec)
            counts.setdefault(name, {})[quarter] = len(cells)
            _LOGGER.info("%s, %s: %d cells", quarter, name, len(cells))

    base = sum(counts["base"].values())
    published = _published_cells() if published_cells is None else published_cells
    rows = []
    for name, per_quarter in counts.items():
        sampled = sum(per_quarter.values())
        ratio = sampled / base
        rows.append(
            {
                "specification": name,
                "cells_sampled": sampled,
                "multiple_of_base": ratio,
                "projected_cells": None if published is None else round(published * ratio),
                "within_ceiling": None if published is None else published * ratio <= CELL_CEILING,
            }
        )
    return pd.DataFrame(rows)


def _published_cells() -> int | None:
    """Rows in the cell table on disk, which the projection is a multiple of."""
    import pyarrow.parquet as pq

    from creditsurv.data.store import cells_path

    path = cells_path("exclude")
    if not path.exists():
        return None
    return int(pq.ParquetFile(path).metadata.num_rows)
