# Data preparation

How 40 GB of nested archives become a table a model can be fitted to. Read
[data_dictionary.md](data_dictionary.md) first for what the fields mean; this
document is about what happens to them.

## The problem

The Freddie Mac Single-Family Loan-Level Dataset ships as one zip per vintage year,
1999 to 2026. Each holds four quarterly zips; each of those holds two headerless
pipe-delimited files — `orig_YYYYQn.txt` with 31 fields, one row per loan, and
`perf_YYYYQn.txt` with 35 fields, one row per loan-month.

| | |
|---|---|
| Archives | 28 vintage years, 40 GB compressed |
| Extracted | roughly 245 GB |
| Performance rows | on the order of 1.75 billion |
| Machine | 8 cores, 16 GB RAM |

Nothing about that fits in memory, and re-parsing it on every run would be
intolerable. The pipeline is therefore three stages, and **the full panel is never
materialised at any of them**.

```
zip annidati  ──[1 ingest]──>  parquet  ──[2 aggregate]──>  celle pesate  ──>  fit
   245 GB                      ~10 GB                       ~10⁶ righe
```

## Stage 1 — Ingest (`creditsurv ingest`)

`src/creditsurv/data/ingest.py`. Per quarter:

1. `unzip -p` streams the inner archive to a temporary file. Python's `zipfile`
   would need the whole member resident before it could be opened as an archive in
   turn, and the largest are several gigabytes.
2. `pyarrow.csv.open_csv` reads that member **in batches**.
3. Each batch is written straight to parquet with `ParquetWriter`.

Peak memory is one batch, so a 4 GB quarter costs what a 40 MB one costs.

**Column selection happens during the parse**, through `include_columns`, so the
discarded fields are never materialised. Two groups are dropped deliberately:

| Dropped | Why |
|---|---|
| `vantagescore_4`, `pre_harp_loan_sequence_number` | Empty in every vintage checked |
| All loss columns (`actual_loss`, `net_sales_proceeds`, expenses, recoveries) | Populated only for defaulted loans, and only relevant to LGD — out of scope |
| `postal_code`, `seller_name`, `msa` | High cardinality, no signal for a default model at this granularity |

**Measured, not estimated:** 2024Q1 is 4.9 million rows in 4.8 seconds. 500 MB of
text becomes 28 MB of parquet — about eighteen-fold. Extrapolated over the dataset:
roughly half an hour, roughly 10 GB.

The stage is **idempotent**. A quarter whose parquet already exists is skipped, so an
interrupted run costs only the quarter it was in the middle of.

### The manifest

Row counts are recorded in `data/interim/manifest.json` as the data goes past, and a
quarter is written there only once it closes.

That matters more than it looks. Downstream stages read the files the manifest lists
rather than globbing the directory — because globbing picks up a parquet still being
written, and a half-written parquet fails with a message about magic bytes that says
nothing about the cause. This is not hypothetical: it happened here, with an ingest
running in one terminal and an aggregation in another.

The manifest is also what later lets the archives be deleted with something better
than optimism (see `creditsurv prune-archives`).

## Stage 2 — Cleaning and event definition

`src/creditsurv/data/aggregate.py`, in SQL, because the join and the truncation both
have to happen out of core.

### Missing-value sentinels

The dataset encodes "not available" as ordinary numbers. Left in place they parse
perfectly happily and produce a portfolio whose average credit score is several
thousand.

| Field | Sentinel |
|---|---|
| `classic_fico` | 9999 |
| `original_dti` | 999 |
| `original_ltv`, `original_cltv` | 999 |

A loan missing a covariate is **dropped, not imputed**: imputing an underwriting
characteristic invents the very thing being measured.

### Event definition

| Outcome | Condition |
|---|---|
| **Default** | `current_loan_delinquency_status` ≥ 3 (90+ days) **or** `zero_balance_code` in {02 third-party sale, 03 short sale, 09 REO, 15 note sale} |
| **Prepayment** | `zero_balance_code` = 01 — treated as censoring |

`current_loan_delinquency_status` is **alphanumeric**: `RA` marks an REO acquisition
and `XX` an unknown status. Casting to a number turns both into null, which compares
false and so reads as *performing* — correct for `XX`, wrong for `RA`. The
zero-balance code is therefore checked alongside it, not instead of it.

### Truncation at the first terminating month

Servicing files keep reporting after a default, through foreclosure, disposition and
loss settlement, so a defaulted loan carries several flagged rows. Each loan is cut
at its first terminating month.

Left alone this breaks the one-event-per-loan invariant and counts a single default
many times over in the likelihood — silently, since nothing raises.

### Loan age

Age comes from `loan_age` in the performance file, and the origination month is
recovered as `period − age`. The dataset has **no origination date** — only a first
payment date, which falls one or two months later depending on the servicer — so
deriving seasoning from it would put a portfolio out by a month in a way that varies
loan by loan. Negative ages, which the dataset does emit, are dropped.

## Stage 3 — Coarse classing and aggregation

Episodes agreeing on every covariate and on their position in time are exchangeable,
so they collapse to one row carrying a count, and the likelihood treats that count as
a frequency weight.

At this scale that is not an optimisation but the only thing that makes the problem
tractable: a fit over two billion rows is out of reach, a fit over a million weighted
cells is a minute.

### Cut points

Chosen from credit conventions rather than fitted to the sample. Loan-to-value breaks
sit at 80, 85, 90 and 95 because that is where mortgage insurance and pricing tiers
actually change; debt-to-income breaks at 36 and 43 because those are long-standing
underwriting thresholds. Data-driven cuts would fit this sample better and would have
to be refitted, and re-justified, on every new one.

Each band takes its **midpoint** as its value, so a binned covariate keeps the scale
of the one it replaces and its coefficient stays comparable with an unbinned fit.
Cut points live in `BIN_EDGES` in `src/creditsurv/features.py`.

### The grouping key

```
key    = coarse-classed continuous covariates
       × categorical covariates
       × vintage quarter
       × loan age
       × event
weight = COUNT(*) AS n
```

**The macro series are deliberately absent from the key.** Since
`period = orig_period + age`, unemployment, house prices and financial conditions are
a deterministic function of two columns already in the key, so they are recomputed on
the aggregate at no cost in cardinality. Putting them in the key would multiply it by
the number of distinct months and destroy the collapse entirely.

### The weight is a count, never an amount

Weighting by outstanding balance estimates a *value-weighted* default rate rather
than a borrower probability of default, and Basel and IFRS 9 both define PD per
obligor. It also breaks inference: lifelines derives standard errors by treating
weights as replication counts, and warns that non-integer weights bias them.

`fit_aft` rejects non-integer weights for this reason. If exposure should influence
the model, `log_orig_upb` is already a covariate.

### A note on when this technique pays

An earlier measurement in this project found the same aggregation compressing
**1.00×** and concluded it was not worth having. That was true at 215,000 rows, where
the possible cells vastly outnumbered the rows: the cell space grows multiplicatively,
and at nine continuous covariates it ran to 573 billion combinations.

At 1.75 billion rows the ratio inverts. The measurement was correct for the regime it
was taken in, and wrong as a generalisation — which is the usual failure mode of a
measurement taken once.

## Reproducing

```bash
uv run creditsurv fetch-macro
uv run creditsurv ingest                          # ~30 min, idempotent
uv run creditsurv aggregate --report-cardinality
```

`data/` is not committed. Everything under it is reproducible from the commands
above.
