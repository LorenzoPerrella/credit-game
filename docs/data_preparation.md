# Data preparation

How 40 GB of nested archives become a table a model can be fitted to. Read
[data_dictionary.md](data_dictionary.md) first for what the fields mean; this
document is about what happens to them.

## The problem

The Freddie Mac Single-Family Loan-Level Dataset ships as one zip per vintage year,
1999 to 2026. Each holds four quarterly zips; each of those holds two headerless
pipe-delimited files — `orig_YYYYQn.txt` with 31 fields, one row per loan, and
`perf_YYYYQn.txt` with 35 fields, one row per loan-month.

| | Measured |
|---|---|
| Archives | 28 vintage years (1999–2026), 40 GB compressed |
| Quarters | 107 |
| Loans | **48,827,197** |
| Loan-months | **2,876,284,955** |
| Parquet after ingest | 17 GB |
| Machine | 8 cores, 16 GB RAM |

Nothing about that fits in memory, and re-parsing it on every run would be
intolerable. The pipeline is therefore three stages, and **the full panel is never
materialised at any of them**.

```
zip  ──[1 ingest]──>  parquet  ──[2 profile]──>  decisions  ──[3 aggregate]──>  cells  ──>  fit
245 GB                 17 GB                        spec                      15.9·10⁶
```

**The screening comes before the group-by, and the order is the point.** This
pipeline had it backwards at first: binning and grouping were done together from a
specification chosen in advance, and the screening ran afterwards — by which time the
bands were baked into two million cells, and a mis-binned covariate could only be
found by noticing its coefficient had come out with the wrong sign. Which is exactly
how the ELTV sentinel was eventually found.

`nmds` runs the other way round:

```
load → frequency screening → class merging → GROUP BY
```

and its screening covers the whole history, filtered by population rather than by
date. This pipeline now does the same.

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
text becomes 28 MB of parquet — about eighteen-fold. Over the whole dataset: 2.88
billion rows, 17 GB of parquet.

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
| `estimated_loan_to_value` | 999 |

The last one was missed on the first pass, and the consequence is worth recording.
`estimated_loan_to_value` is in the *performance* file, and only the origination
file's sentinels had been handled. The median ELTV of the 2006 vintage is literally
999, so mark-to-market leverage came out as `999 − 75 = 924` for most of the panel;
sixty percent of exposure landed in one band and the coefficient came out with the
wrong sign. The exploratory default-rate-by-band table is what exposed it: `fico_s`
and `orig_ltv` were cleanly monotonic and this one was not.

A loan missing a covariate is **dropped, not imputed**: imputing an underwriting
characteristic invents the very thing being measured.

### Categorical codes are mapped from what is in the field, not from the layout

Every `CASE` in `_CATEGORICAL` lists its branches explicitly and has **no `ELSE`**, so
a code nobody has looked at becomes NULL and the loan is dropped rather than being
absorbed into whichever level happens to be last. The mappings were decided from a
distinct-and-count across seven vintages; see
[variable_selection.md](variable_selection.md) for the frequencies and the four
corrections that came out of it.

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

⚠️ **`loan_age` is not monotone within a loan.** It counts scheduled payments since
the loan was originated *or modified*, so a modification restarts it. Loan
`F06Q10092168` runs to age 192 at twenty months delinquent, is modified, and
reappears the next month at age 3 with a clean delinquency status.

Truncating each loan at the smallest terminating *age* — which is what this pipeline
did — therefore cuts at the wrong row as soon as a loan has been modified. Measured on
2006Q1:

| | before | after |
|---|---|---|
| Loan-months | 18,130,622 | 17,877,294 |
| Defaults | 40,512 | **41,153** |
| Duplicated `(loan, age)` | 473,697 | 415 |

So 2.6% of the panel was one loan counted twice at the same age, and 641 real
defaults — 1.6% of them — were being cut off. The 253,000 loan-months removed were
previously-distressed months re-filed at young ages, arriving *performing*, diluting
exactly the part of the hazard curve the model is most sensitive to. It affects 0.4%
of the 1999 vintage's loans, 5.2% of 2006's and 1.9% of 2021's — and not at random,
since a modified loan is by definition one that got into trouble.

Two changes: truncation orders by **calendar period**, which is monotone by
construction, and a **modification ends observation** the way a prepayment does, the
month before the flagged row rather than on it. 415 duplicated pairs survive in
2006Q1, from loans with a repeated age and no modification flag; deduplicating them
costs a sort over the whole panel, ten times the aggregation's runtime, and is not
worth it at 0.002%.

## Stage 2b — Screening (`creditsurv profile`)

`src/creditsurv/profiling.py`. Runs on the ingested parquet, quarter by quarter,
across the **whole history** — 2.9 billion loan-months is too much to hold but not
too much to count.

| What it reports | Rule |
|---|---|
| Exposure share per categorical level | below **5%** → merge, following `nmds` |
| Largest level's share | ≥ **99%** → degenerate, drop |
| Default rate per level and per band | read for **monotonicity** |
| Quantiles of a continuous covariate | candidate cut points |

**Monotonicity is the check that earns its place.** A covariate whose default rate
rises and falls across its own bands is either mis-binned or is measuring something
other than what its name says. On this data `fico_s` runs cleanly from 469 to 16
basis points across its bands, a factor of 29, and `orig_ltv` likewise — and
`cltv_drift` did not, which is what exposed the untreated sentinel.

**Quantiles are a starting point, not an answer.** Data-driven cuts fit the sample
they were taken from, so the ones actually used come from credit conventions. What
the quantiles are for is showing where the mass sits: on `orig_ltv` the 60th and 80th
percentiles both come back as **80**, because that is the threshold above which
mortgage insurance is required and originations pile up against it. A band boundary
placed there would split an enormous mass at exactly the wrong point, and it is worth
knowing that before choosing rather than after.

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
       × loan age band
       × event
weight = COUNT(*) AS n
```

Quarters are aggregated **one at a time**. Every loan appears in exactly one
quarter's files — verified rather than assumed: the identifiers of 1999Q1 and 1999Q2
do not intersect at all — so a quarter can be collapsed on its own and the results
concatenated, which keeps memory flat.

The alternative was tried first. Grouping all quarters at once builds one hash table
over hundreds of millions of loan identifiers; DuckDB spilled **20 GB into the
working tree** before it was stopped. The connection now also points its temporary
directory outside the repository.

### Episodes are monthly

`EPISODE_MONTHS = 1`, and the reason is not compression.

**The width is set by how often the covariates move.** The time-varying covariates
come from monthly series — unemployment, the house price index, financial conditions
— so an episode spanning a quarter asks the model to hold constant something the data
says changed three times. Monthly is what the data supports, so monthly is what this
is. The width is then read back off the spacing of the distinct ages rather than
passed in, so a cell table can never disagree with the width it was built with.

The final collapse, measured on the whole dataset:

| | |
|---|---|
| Loan-months in | **2,515,340,009** |
| Cells out | **15,858,492** |
| Compression | **159×** |

The one cost is fit time, and it is real: see [the methodology
report](reports/methodology.md) for what a fit on this table takes.

Wider bands were tried, and the measurement that ruled them out is worth recording
because it was nearly misread. On 1999Q1, monthly and quarterly episodes produced
**identical** cell counts — 12,752,331 each — which looks like proof that collapsing
on age buys nothing. It was an artefact: `cltv_drift` was still in the grouping key at
the time, and nothing can collapse on age while a covariate moves underneath it. With
the derived covariates taken out of the key the same quarter gives 226,000 cells
monthly against 80,000 quarterly — a real 2.8× — bought by holding a monthly covariate
constant for three months, which is not a trade worth making.

**The specification *is* the cardinality.** Cell count is the product of every
covariate's band count, so the choice of covariates decides whether the result fits at
all. Aggregating on everything available and selecting variables afterwards is the
wrong order, and produced a table of roughly a billion cells on the first attempt.
The aggregation is parameterised by `CellSpec` so that its output can follow the
selection rather than precede it.

### Why the macro side is free and the loan side is not

**No macro series is in the key, and none can be.** Since
`period = orig_period + age`, every macro-derived covariate is a deterministic
function of two columns the key already holds, so all thirteen of them are recomputed
on the aggregate **at no cost in cardinality whatsoever**. Putting one in the key
would multiply it by the number of distinct months and destroy the collapse.

That asymmetry decides the shape of the whole specification:

| Adding | Cost |
|---|---|
| One macro series | **zero cells** |
| `channel`, `region`, `first_time_buyer` (2 × 4 × 2 levels) | up to **16×** the table |
| One continuous covariate at 5 bands | up to **5×** the table |

Which is why the macro side carries fourteen series and thirteen candidate
covariates, and the loan side carries six. The loan characteristics left out —
`log_orig_upb`, `orig_spread`, `channel`, `region`, `first_time_buyer` — were not
dropped on their merits, and `docs/variable_selection.md` records what each would
cost against what it might be worth.

### The weight is a count, never an amount

Weighting by outstanding balance estimates a *value-weighted* default rate rather
than a borrower probability of default, and Basel and IFRS 9 both define PD per
obligor. It also breaks inference: lifelines derives standard errors by treating
weights as replication counts, and warns that non-integer weights bias them.

`fit_aft` rejects non-integer weights for this reason. If exposure should influence
the model, that is what a covariate is for — `log_orig_upb` is binned and ready in
`BIN_EDGES`, and waiting only on the cardinality budget to admit it to the key.

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
