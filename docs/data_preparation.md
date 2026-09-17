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
| Quarters | 109 |
| Loans | **49,186,171** |
| Loan-months reported | **2,881,397,251** |
| Parquet after ingest | 17 GB |
| Machine | 8 cores, 16 GB RAM |

Nothing about that fits in memory, and re-parsing it on every run would be
intolerable. The pipeline is therefore three stages, and **the full panel is never
materialised at any of them**.

```
zip  ──[1 ingest]──>  parquet  ──[2 profile]──>  decisions  ──[3 aggregate]──>  cells  ──>  fit
245 GB                 17 GB                        spec                      63.6·10⁶
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
discarded fields are never materialised. Every field dropped is dropped deliberately, with
its reason beside it in `ingest.py`:

| Dropped | Why |
|---|---|
| `vantagescore_4`, `pre_harp_loan_sequence_number` | Empty in every vintage checked |
| All loss columns (`actual_loss`, `net_sales_proceeds`, expenses, recoveries) | Populated only for defaulted loans, and only relevant to LGD — out of scope |
| `postal_code`, `seller_name`, `msa` | High cardinality, no signal for a default model at this granularity |
| `prepayment_penalty_indicator` | Near-constant `N` on conforming loans |
| `harp_indicator`, `special_eligibility_program` | Programme flags, not states of the loan. `harp_indicator` marks exactly the loans with no debt-to-income, which the complete-case rule drops: see [What dropping removes](#what-dropping-removes) |
| `property_valuation_method` | How a value was obtained, not what it is |

Until the validation (D2) the last four were missing from this table, and so were
`delinquency_due_to_disaster`, `borrower_assistance_plan` and `payment_deferral_flag`,
which were not being kept either. Those three are what tell a moratorium from a default,
and they are kept now: see [A moratorium is not a default](#a-moratorium-is-not-a-default).

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
| `mortgage_insurance_percentage` | 999 |
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

#### What dropping removes

Dropping is harmless only if what goes is small or looks like what stays. On the vintages it
read, the validation (D4) found the share varying by two orders of magnitude and the dropped
loans 8 to 16% riskier. `creditsurv aggregate --report-incomplete` now counts it over every
vintage into `docs/reports/incomplete_cases.csv`, with the ever-default rate of the loans kept
and of those dropped:

| Vintages | Loans | Dropped | Of which no DTI | Default rate, kept | Default rate, dropped | Relative risk |
|---|---|---|---|---|---|---|
| 1999Q1–2009Q1 | 18,933,200 | 2.6% | 85% | 6.20% | 6.48% | 1.05 |
| **2009Q2–2019Q1** | 15,949,082 | **18.0%** | **99.9%** | 1.60% | 4.71% | **2.94** |
| 2019Q2–2026Q1 | 14,303,645 | 0.03% | 22% | 1.29% | 1.44% | 1.11 |
| all | 49,185,927 | 6.9% | 97.6% | 3.35% | 4.97% | 1.48 |

Rates are pooled over the vintages of a row, so each loan counts once. Within the middle row
no vintage is below 2.3 or above 4.75, and the share dropped peaks at 39.2% in 2012Q2.

**The middle row is one programme.** It opens in the first quarter of HARP, the Home
Affordable Refinance Program, and closes after the programme expired at the end of 2018. In
the 2012Q2 origination file, 181,302 of the 181,356 loans without a debt-to-income carry
`harp_indicator = Y`, and not one HARP loan has one. Nearly all are no-cash-out refinances
(181,197), at a median LTV of 98 against 74 for the rest, a quarter of them above 124.

**The model therefore does not cover HARP refinances**: 2.87 million loans and 135,408
defaults, about three times as likely to default as the loans kept from the same vintages. An
imputed DTI would invent the one thing the programme waived. Covering them means a level of
their own in the key -- a HARP or missing-DTI indicator -- at the cost of a re-aggregation and
a new selection. Until that is decided the exclusion is stated rather than repaired, and it is
listed under Open in `CLAUDE.md`.

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
| **Default** | `current_loan_delinquency_status` ≥ 3 (90+ days) **or** `zero_balance_code` in {02 third-party sale, 03 short sale, 09 REO, 15 note sale} — unless the month is a moratorium |
| **Prepayment and other exits** | `zero_balance_code` in {01 prepaid, 16 reperforming loan sale, 96 removal} — censoring |
| **End of observation** | a modification, and under `censor` a moratorium — the month before the flagged row |

`current_loan_delinquency_status` is **alphanumeric**: `RA` marks an REO acquisition
and `XX` an unknown status. Casting to a number turns both into null, which compares
false and so reads as *performing* — correct for `XX`, wrong for `RA`. The
zero-balance code is therefore checked alongside it, not instead of it.

**Codes 16 and 96 are censoring, and what that rests on is measured.** The validation (D5)
found both unclassified, and code 16 credit by definition: a loan sold as reperforming was
delinquent once, so counting its sale as censoring could lose a default. Whether it does
depends on what the book did first -- a loan that reached 90 days has already defaulted and
been cut there, and one that was modified was censored at the modification. `creditsurv
aggregate --report-exits` counts the three outcomes over every vintage
(`docs/reports/credit_adjacent_exits.csv`):

| Code | Loans | Defaulted first | Censored earlier | Censored at the exit |
|---|---|---|---|---|
| 16, reperforming loan sale | 185,526 | **167,148** (90.1%) | 15,531 (8.4%) | 2,847 (1.5%) |
| 96, removal | 129,077 | 53,464 (41.4%) | 1,381 (1.1%) | 74,232 (57.5%) |

On 2006Q1 the counts are the validation's, 5,855 and 973. **Censoring the sale loses no
default**: nine reperforming loans in ten are counted as defaults at their first 90-day
month, and of the rest all but 2,847 had left observation at a modification. Those 2,847
were performing when sold and had never reached 90 days, which by this event definition is
not a default. A removal is different in kind -- most removed loans were performing -- and
the one thing censoring could hide there is a default about to happen when the loan was
taken out of the dataset: at most 74,232 loans, 0.15% of the book.

### A moratorium is not a default

The CARES Act required servicers to report loans in forbearance as delinquent, and
disaster relief works the same way. Such a loan passes the 90-day test without its
borrower having failed to pay anything, and the validation found moratoria behind **17%
of the default events**, 99% of which returned to performing. Three fields tell them
apart, and none of them was being kept at ingest:

| Field | Marker |
|---|---|
| `delinquency_due_to_disaster` | `Y` |
| `borrower_assistance_plan` | `F`, forbearance |
| `payment_deferral_flag` | `P`, `C` |

`T` and `R` in the assistance plan are loss mitigation, trial and repayment plans for a
borrower who did fall behind, and are not moratoria. On the 2019Q3 vintage **87%** of the
rows at 90+ days carry one of the three markers.

What a marker means is `MoratoriumPolicy`, and the two defensible answers keep different
things:

| Policy | The accommodated month | The months after it |
|---|---|---|
| `exclude` | not an event | the loan stays at risk, and a later genuine default counts |
| `censor` | ends observation | lost, with any default among them |
| `ignore` | an event, as before | kept for comparison only |

Each policy writes its own table, `cells_<policy>.parquet`. `creditsurv moratorium` fits
and backtests the same specification on both, so the choice rests on what it does to the
model; see [the moratorium report](reports/moratorium.md).

**The rule for choosing, written while both fits were still running.** `exclude` is the
prior. Forbearance went to the borrowers who asked for it, and those were on the whole the
borrowers under strain, so ending their observation at the first accommodated month removes
loans *because of* their risk: informative censoring, which biases the hazard down.
`exclude` keeps them at risk and counts the defaults that genuinely follow. The data can
overturn the prior only through the backtest -- if `exclude` fails an acceptance criterion
that `censor` passes, `censor` is chosen; otherwise `exclude` stands. Coefficient moves are
reported and not used to choose: the two fits estimate different dependent variables, and
a difference between them is not a defect of either.

**The result.** Both policies pass the overall actual over expected -- 1.061 excluded,
0.958 censored -- and the Gini, 0.558 and 0.551, and both fail the decile criterion, at
0.718 to 1.154 and 0.705 to 1.014. Neither fails a criterion the other passes, so by the
rule above `exclude` stands.

Those decile ranges took each decile's expected rate as a plain mean of its cells' hazards,
not weighted by loan-months, which was corrected later. On the selected model's test window
the correction moved the lowest ratio by 0.001 and the riskiest decile's by 0.13, so both
policies' ranges still start near 0.7, well below the 0.80 the criterion needs, and the
comparison the rule reads is unchanged. The ranges themselves have not been recomputed.

Two measurements say the prior deserved to be the prior. Censoring lost **26% of the test
window's defaults**, 56,413 against 76,380, for 3% less exposure: loans that took a
moratorium and genuinely defaulted afterwards, exactly the ones censoring was feared to
remove for their risk. And `inflation` changes sign between the two fits, from −0.42 to
+1.71. A covariate whose direction depends on how one year's forbearance is treated is
behaving as a calendar effect rather than an elasticity, which is what the validation said
of it (S5); the selection weighs it against its gap form.

**The spike is gone.** The validation's evidence was the monthly default rate: 90.4 bp in
May 2020 against 3.07 bp through 2019, a factor of 29 that no credit recession produces;
2008-09, a real one, reached 24.5 bp. Recomputed on the `exclude` cells
(`docs/reports/monthly_default_rate.csv`, written by `creditsurv portfolio`):

| | Monthly default rate |
|---|---|
| 2019, pooled | 2.94 bp |
| May 2020 | **2.00 bp** |
| 2020, pooled | 2.86 bp |
| Highest month of 2020-21 | 3.73 bp, November 2020 |
| Highest month of 2007-11 | **25.11 bp**, November 2009 |

The pandemic no longer looks like a crisis and 2009 still does. 2017, where the hurricanes
had added 16,706 events, now sits at 3.25 bp between 2016's 3.26 and 2018's 3.31. That
2020 comes out slightly *below* 2019 is itself a reading of the policy rather than of the
economy: forbearance postponed the defaults it did not prevent, and the refinancing wave
filled the book with new, low-hazard loans.

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

### Loans enter late, and the likelihood is told

A loan is not always first reported at age zero: up to **63% of the 1999 vintage** is
first observed above it. The episode likelihood conditions every loan-month on survival
to its own start, so a late entrant contributes only the months it was seen. That is left
truncation, handled rather than ignored.

It rests on an assumption worth stating, because nothing checks it: **the age at which a
loan enters is independent of when it defaults**, given its covariates. Entry selected on
risk -- loans reported only once they were in trouble, or only if they survived -- would
bias the hazard at young ages, and this data cannot say whether that happened.

The late entrants used to be absorbed silently in one place: the loan-level duration
distribution rebuilt from the cells for the Kaplan-Meier comparison, which cannot
represent a loan absent at the start and clipped the difference, on the belief that the
panel is reported contiguously from origination. The validation measured 4,605,963
loan-months absorbed that way. `net_entries` now measures them, and a warning names them
once they pass 1% of the panel.

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

Chosen from credit conventions rather than fitted to the sample. Data-driven cuts would
fit this sample better and would have to be refitted, and re-justified, on every new one.

Two sets exist, and they used to disagree. `BIN_EDGES` in `src/creditsurv/features.py` is
the fine classing the exploration reads; `PRODUCTION_EDGES` in
`src/creditsurv/data/aggregate.py` is what the cells are built from, and a test holds it
to a **subset** of the first -- coarser, because every band multiplies the table:

| Covariate | Production breaks |
|---|---|
| `fico_s`, (score − 700) / 50 | −2.4, −0.8, 0, 0.8, 1.2, 2.4 |
| `orig_ltv` | 30, 70, 80, 90, 100 |
| `dti` | 10, 28, 36, 43, 55 |

This document once justified a DTI break at 43 while the model cut at 45, and described
LTV breaks at 85 and 95 that no cell had. The 80 break is the one the economics turns on,
since mortgage insurance is required above it and originations pile up against it, and 43
is the qualified-mortgage limit.

Each band takes its **midpoint** as its value, so a binned covariate keeps the scale
of the one it replaces and its coefficient stays comparable with an unbinned fit.

### The grouping key

```
key    = coarse-classed continuous covariates
       × categorical covariates: purpose, occupancy, term, has_mi, first_time_buyer
       × origination month
       × loan age
       × event
weight = COUNT(*) AS n
```

Quarters are aggregated **one at a time**. Every loan appears in exactly one
quarter's files — verified rather than assumed: the identifiers of 1999Q1 and 1999Q2
do not intersect at all — so a quarter can be collapsed on its own and the results
concatenated, which keeps memory flat.

**The origination month, not the quarter.** The key used to carry the vintage quarter,
and every macro covariate was then read as if the loan had been written in the quarter's
first month. Loans are not all written in it -- the mean offset is +2.15 months -- so every
macro series was read two months late, and cells near the backtest date were filed on the
wrong side of it: some 21 million loan-months of look-ahead that `assert_no_lookahead`
could not see, because it checked the shifted month. The exact month costs 3.51× the
cells, measured on nine quarters.

`creditsurv check-calendar` checks that the move worked. It rebuilds the monthly default
series from the cells and sets it against the series counted straight from the performance
files. The validation's correlation had peaked two months out; it now peaks at a lag of
zero, and not one of the 1,536,686 defaults is filed in a different month
(`docs/reports/calendar_check.csv`).

Text keys come back from DuckDB as Python strings. Each quarter's are made categorical as
it arrives, and the levels are unified before the quarters are stacked, because pandas
turns a categorical column back into strings when two pieces disagree on its levels.

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

| | `exclude` | `censor` |
|---|---|---|
| Loan-months in | **2,535,194,125** | **2,507,132,461** |
| Cells out | **63,639,116** | **63,139,859** |
| Compression | **40×** | **40×** |

The quarter-keyed table was 15,858,492 cells, four times fewer, for the reasons given
under the grouping key.

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
| `has_mi` and `first_time_buyer` | **1.19×** the table, measured |
| The origination month in place of the quarter | **3.51×**, measured |
| One continuous covariate at 5 bands | up to **5×** the table |

The middle rows replace a figure nobody had measured. `channel`, `region` and
`first_time_buyer` were once kept out of the key as "up to sixteen times the table", the
product of their level counts, which is a ceiling and not a cost, since most combinations
of levels never occur together. Measured on nine quarters, `has_mi` and
`first_time_buyer` together cost 1.19×, and they are in the key.

That is still why the macro side carries fifteen candidate covariates and the loan side
eight. The loan characteristics left out -- `log_orig_upb`, `orig_spread`, `channel`,
`region` -- were not dropped on their merits, and `docs/variable_selection.md` records
what each would cost against what it might be worth.

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

### What the exact key costs downstream

Four times the cells is not four times the work; it is another machine's worth of memory.
A stock lifelines fit holds about 680 bytes a training row, 45-50 GB for this table, so
fits run through `creditsurv.models.blocks` a block at a time, and the panel is never
held beside its training and test halves. [CLAUDE.md](https://github.com/LorenzoPerrella/credit-game/blob/main/CLAUDE.md) keeps the rules that
follow from it.

## Reproducing

```bash
uv run creditsurv fetch-macro
uv run creditsurv ingest                            # ~30 min, idempotent
uv run creditsurv aggregate --report-cardinality
uv run creditsurv aggregate --moratorium exclude    # 11.3 GB at peak
uv run creditsurv aggregate --moratorium censor
uv run creditsurv moratorium                        # both treatments, fitted and backtested
uv run creditsurv select                            # the specification; days, resumable
uv run creditsurv report --extra-fits
```

`data/` is not committed. Everything under it is reproducible from the commands
above.
