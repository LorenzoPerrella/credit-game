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
    _connect,
    _resolve,
    _state_of_the_book_sql,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import duckdb

    from creditsurv.data.aggregate import PathSpec

_LOGGER: Final = logging.getLogger(__name__)

#: A level holding less than this share of exposure is merged, following `nmds`.
MIN_CATEGORY_SHARE: Final = 0.05

#: A covariate whose largest level holds at least this much has nothing left to
#: estimate a contrast from.
DOMINANCE_THRESHOLD: Final = 0.99

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
