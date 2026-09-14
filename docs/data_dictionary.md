# Data dictionary — the record layout

How the data is built, what one row means at each stage, and what every field
contains. Read this before the methodology: the modelling choices only make sense
once the grain of the data is clear.

## Provenance

| Layer | Source | Scale |
|---|---|---|
| Loan book | Freddie Mac Single-Family Loan-Level Dataset | 48,827,197 loans, 1999–2026 |
| Macroeconomic series | FRED (St. Louis Fed), public CSV endpoint, no API key | 14 series, 1997–2026 |

Both real, and the second is what gives the first its shape: vintages written into
2006 meet the housing collapse, those written into 2021 meet the rate rise.

The loan data is **not downloadable programmatically**. Registration is free but
manual, at <https://claritydownload.fmapps.freddiemac.com/CRT/>. Nothing in this
repository touches the network for it — scraping an authenticated download would
breach the terms it is offered under.

## Regenerating

```bash
uv run creditsurv fetch-macro    # populates data/raw/fred/
uv run creditsurv ingest         # 40 GB of archives to parquet, ~30 min, idempotent
uv run creditsurv aggregate      # collapse to weighted cells
```

`data/` is not committed and everything under it is reproducible from those three
commands. The FRED cache holds raw downloads only; all transformation is in code.

---

## The five layers

Each has a different **grain** — the thing one row represents — and confusing them
is the most common way these models go wrong.

| Layer | One row is | Measured size |
|---|---|---|
| 1. Raw macro | one month of one economic series | 354 months × 14 series |
| 2. Origination record | one loan, as underwritten | **48,827,197** |
| 3. Performance record | one loan in one calendar month | **2,876,284,955** |
| 4. Weighted cell | a covariate combination at one age, with a count | **15,858,492** |
| 5. Model matrix | one *episode*, carrying that count as a weight | same as layer 4 |

Layer 4 is where this project differs from a textbook treatment, and it is not an
optimisation: a fit over 2.9 billion rows is out of reach, and a fit over 15.9
million weighted cells takes minutes. See
[data_preparation.md](data_preparation.md) for why the collapse is exact.

---

## Layer 1 — Raw macroeconomic series

**Grain: one month of one series.** Monthly `PeriodIndex`, one column per series.

| Column | FRED id | Native | Unit | To monthly |
|---|---|---|---|---|
| `unemployment_rate` | `UNRATE` | Monthly | Percent | Last |
| `hpi` | `CSUSHPINSA` | Monthly | Index, Jan 2000 = 100 | Last |
| `mortgage_rate_30y` | `MORTGAGE30US` | Weekly | Percent | **Mean** |
| `mortgage_rate_15y` | `MORTGAGE15US` | Weekly | Percent | **Mean** |
| `nfci` | `NFCI` | Weekly | Index, 0 = average conditions | **Mean** |
| `policy_rate` | `FEDFUNDS` | Monthly | Percent | Last |
| `treasury_10y` | `DGS10` | Daily | Percent | **Mean** |
| `term_spread` | `T10Y2Y` | Daily | Percentage points | **Mean** |
| `credit_spread` | `BAA10Y` | Daily | Percentage points | **Mean** |
| `cpi` | `CPIAUCSL` | Monthly | Index | Last |
| `equity_index` | `NASDAQCOM` | Daily | Index | **Mean** |
| `vix` | `VIXCLS` | Daily | Index | **Mean** |
| `sentiment` | `UMCSENT` | Monthly | Index | Last |
| `housing_starts` | `HOUST` | Monthly | Thousands, annual rate | Last |

Daily and weekly series are **averaged**, not sampled. The rate a borrower lives
with over a month is the average of its days, not whichever day happened to fall
last — and averaging also absorbs the market holidays that leave 226 to 303 gaps in
each daily series since 1999.

**What `nfci` is.** The Chicago Fed National Financial Conditions Index, a weekly
summary of 105 measures of risk, liquidity and leverage across money, debt and
equity markets. Zero is average conditions over its own history; positive is
tighter than average. It is the single most useful macro covariate here because it
moves *before* unemployment does.

**`DRSFRMACBS`** (single-family mortgage delinquency rate, quarterly) is fetched as a
**reference series only** — used to sanity-check observed default rates against a
published aggregate, and never a covariate. It is an outcome, not a driver.

### Why the panel starts two years before the loans

`MACRO_START` is January 1997, and the loan data begins in 1999. A covariate lagged
three months and measured as a year-on-year change reaches fifteen months back, so
a macro panel starting alongside the loans would have no such covariate until April
2000 — and those rows are **dropped**, not imputed. That would silently remove the
opening months of every 1999-vintage loan while keeping the rest, which is left
truncation the likelihood is never told about. Reaching further back costs nothing.

### Two rules about missing data

**Interior gaps are forward-filled.** FRED has occasional holes — `UNRATE` has one
in this window — and monthly series published late leave one at the edge of a month.

**The trailing edge is truncated.** Series publish on different lags: `hpi` runs two
to three months behind `unemployment_rate`. Forward-filling that ragged edge would
invent macro observations that never existed, which then leak into every covariate
built on them. The panel ends at the last month for which *every* series has a real
observation, and logs what it discarded. On a current run it ends **2026-06**.

### Every series is lagged three months, for one of two reasons

| Lag | Series | Why |
|---|---|---|
| **Publication** | `unemployment_rate`, `hpi`, `nfci`, `cpi`, `sentiment`, `housing_starts` | Published in arrears and later revised: the value for month *t* is not known in *t* |
| **Transmission** | both mortgage rates, `treasury_10y`, `term_spread`, `credit_spread`, `equity_index`, `vix`, `policy_rate` | Quoted in real time and never revised, and still unable to cause a default in the month they are quoted |

Market quotes used to be read contemporaneously, on the argument that they are known in
real time. That is an argument about **availability**. The event is ninety days of missed
payments: a loan delinquent in month *t* missed its payments in *t−3*, *t−2* and *t−1*,
so nothing observed in *t* can be what caused it. The backtest showed the cost. Predicted
default spiked in April 2025 and March 2026, the two VIX peaks of the test window, with
actual over expected at 0.47 and 0.59, while realised default did not move.

The lag reaches every market series, not only the one that was noticed. `policy_rate` had
been in neither list, and so was never lagged at all.

---

## Layer 2 — Origination record

**Grain: one loan, as underwritten.** 31 pipe-delimited fields, **no header row** —
so a misplaced field name shifts every column after it while still parsing cleanly.
The field lists in `freddiemac.py` were extracted from the published layout
spreadsheet rather than transcribed: 66 fields across two files is too many to copy
reliably.

| Field | Source column | Unit / domain | Notes |
|---|---|---|---|
| `loan_id` | `loan_identifier` | — | Primary key |
| `credit_score` | `classic_fico` | 300–850 | **9999 means missing** |
| `fico_s` | derived | ≈ −2.4 to +3.0 | `(credit_score − 700) / 50`, the modelled form |
| `orig_ltv` | `original_ltv` | Percent | **999 means missing** |
| `orig_cltv` | `original_cltv` | Percent | Combined: catches second liens |
| `dti` | `original_dti` | Percent | **999 means missing** |
| `orig_upb` | `original_upb` | USD | |
| `note_rate` | `original_interest_rate` | Percent | |
| `orig_term` | `original_loan_term` | Months | 180 or 360 for almost all of the book |
| `mi_percent` | `mortgage_insurance_percentage` | Percent | 0 where uninsured |
| `purpose` | `loan_purpose` | P / C / N / R | purchase, cash-out, rate-term |
| `occupancy` | `occupancy_status` | P / S / I | owner, second home, investor |
| `channel` | `channel` | R / B / C / T | See the warning below |
| `region` | `property_state` | 4 census regions | Fifty dummies buy little |
| `first_time_buyer` | `first_time_homebuyer_indicator` | Y / N | 9 means missing |
| `property_type` | `property_type` | SF / PU / CO / MH / CP | |
| `units` | `number_of_units` | 1–4 | |
| `n_borrowers` | `number_of_borrowers` | 1–10 | |

**What `credit_score` is, and whether using it is circular.** It is a FICO score:
300–850, produced by a model calibrated on the probability of serious delinquency.
The worry is fair and the answer is that it is not circular, for three reasons that
all have to hold. It is measured **at inception** and never updated. It predicts a
**different outcome** — 90+ days late on *any* credit line, over roughly 24 months,
across the whole US population — from ours, which is default on *this mortgage* over
its lifetime. And it is an **input a lender actually has** at the decision point,
which is the test that matters for a model meant to be used.

What *would* be circular: an internal PD on the same event and horizon, the current
delinquency status, or `DRSFRMACBS`. All three are excluded.

**Missing-value sentinels are real numbers.** A credit score of 9999 and a DTI of
999 parse perfectly happily; left in place they produce a portfolio whose average
credit score is several thousand. Every one is blanked on read, and the list is in
`aggregate.py` next to the `NULLIF` that does it.

⚠️ **`channel` cannot be used at four levels.** Until 2008 about half of
originations are coded `T`, third-party not otherwise specified, and broker and
correspondent are near zero; from 2009 `T` vanishes and those two absorb it exactly.
That is a change in how Freddie Mac coded the field, not a change in how loans were
sold, and a model given four levels reads the coding change as a risk effect.
Retail's own share is stable throughout — 53.8% in 1999, 57.9% in 2021 — so the
binary split is the part that means the same thing in every vintage. It is collapsed
to **retail against third-party**.

---

## Layer 3 — Performance record

**Grain: one loan in one calendar month.** 35 fields, one row per loan per month of
servicing. 2.88 billion of them. Eleven columns are read:

| Field | Source | Notes |
|---|---|---|
| `loan_id` | `loan_identifier` | |
| `period` | `period` | YYYYMM |
| `age` | `loan_age` | **See the warning below** |
| `delinquency` | `current_loan_delinquency_status` | *Alphanumeric* |
| `zero_balance_code` | `zero_balance_code` | How the loan ended |
| `upb` | `current_actual_upb` | Outstanding balance |
| `eltv` | `estimated_loan_to_value` | **999 means missing** |
| `modification_flag` | `modification_flag` | Y this month, P thereafter |

⚠️ **`loan_age` restarts at a modification.** The field counts scheduled payments
since the loan was originated *or modified*. Loan `F06Q10092168` runs to age 192 at
twenty months delinquent, is modified in April 2022, and reappears the next month at
**age 3** with a clean delinquency status.

Believed, that gives one loan two episodes at the same age and re-files
previously-distressed months as performing ones at young ages. It affects 0.4% of
the 1999 vintage's loans, 5.2% of 2006's and 1.9% of 2021's — and not at random,
since a modified loan is by definition one that got into trouble. A **modification
therefore ends observation**, as a prepayment does: the modified contract is a
different loan. Truncation orders by calendar period, which is monotone by
construction, rather than by age, which is not.

⚠️ **`estimated_loan_to_value` is not used**, despite being the better measure of
mark-to-market leverage — it is Freddie's own per-loan valuation rather than a
national index. Coverage runs from 0.8% of the 1999 vintage to 94% of 2021, so a
model built on it would estimate a different quantity in every decade. The median
ELTV of the 2006 vintage is literally **999**.

⚠️ **`current_loan_delinquency_status` is alphanumeric.** `RA` marks an REO
acquisition and `XX` an unknown status. Coercing the column to a number turns both
into NaN, which compares false and so reads as *performing* — right for `XX`, wrong
for `RA`. The zero-balance code is checked alongside it, not instead of it.

### Event definition

| Outcome | Condition | Treatment |
|---|---|---|
| **Default** | `delinquency` ≥ 3 (90+ days) **or** `zero_balance_code` in {02, 03, 09, 15} | The event |
| **Prepayment** | `zero_balance_code` = 01 | Censoring |
| **Modification** | `modification_flag` in {Y, P} | Censoring, the month *before* |
| **Still performing at panel end** | — | Censoring |

The zero-balance codes are third-party sale, short sale, REO disposition and note
sale: four ways a loan ends through credit loss rather than repayment.

**Servicing files keep reporting after a default**, through foreclosure, disposition
and loss settlement, so a defaulted loan carries many flagged rows — and often a
zero-balance code at the very end that looks like a prepayment. Each loan is cut at
its first terminating month. Left alone this breaks one-event-per-loan and counts a
single default dozens of times in the likelihood.

---

## Layer 4 — Weighted cells

**Grain: a covariate combination, at one loan age, with a count.** This is what the
fitter is actually handed.

| Field | Type | Definition |
|---|---|---|
| `vintage` | str | Origination quarter, `YYYYQn`, read off the file name |
| `age` | int | Loan age in months, the start of the episode |
| `event` | bool | Whether this cell's loan-months ended in default |
| `n` | int | **How many loan-months the row stands for** |
| `fico_s`, `orig_ltv`, `dti` | float | Coarse-classed, carried at the band's midpoint |
| `purpose`, `occupancy`, `term_years` | category | Mapped levels |

Episodes agreeing on every covariate and on their position in time are
exchangeable, so they collapse into one row carrying a count, and the likelihood
treats that count as a frequency weight. 2.52 billion loan-months become 15.86
million cells — 159× — and the estimate is identical.

**`n` is a count of loan-months, never an amount.** Weighting by exposure would
answer a different question from the one Basel and IFRS 9 ask: a PD is defined per
obligor, so a $2m loan and a $200k loan each contribute one default. Weighting by
balance turns the estimate into a loss-weighted rate, which is a different quantity
with the same name. lifelines also warns that non-integer weights bias its variance
estimates, so the standard errors would be wrong as well.

**Band midpoints, not band indices.** A coarse-classed covariate keeps the units of
the one it replaces, so its coefficient stays comparable with an unbinned fit and
reads in the original scale.

### Covariates rebuilt after the collapse

These are **not** in the cell key, because each is a function of the vintage quarter
and the loan age — both of which the key already carries. So the macro side of the
specification is free: adding a series cannot change the size of the cell table by
one row, while adding a loan characteristic to the key can multiply it.

All thirteen are **built**; eight are **fitted**. The other five were given up by the
variable selection, for reasons set out with their measured evidence in
[variable_selection.md](variable_selection.md). They are still constructed, because
the selection has to be re-runnable and because a covariate that cannot be built
cannot be reconsidered.

| Covariate | Shape | Definition | In the model |
|---|---|---|---|
| `cltv_drift` | gap | `orig_ltv × hpi(orig)/hpi(now) − orig_ltv` — leverage gained or lost | ✅ |
| `unemp_gap` | gap | `unemployment(now) − unemployment(orig)` | ✅ |
| `policy_rate_gap` | gap | `policy_rate(now) − policy_rate(orig)` | ✅ |
| `rate_gap` | gap | `market_rate(orig) − market_rate(now)`, **switched by term** | ✅ |
| `nfci_lagged` | level | Financial conditions now | ✅ |
| `vix` | level | Implied volatility now | ✅ |
| `hpi_growth` | year-on-year | House prices | ✅ |
| `inflation` | year-on-year | CPI | ✅ |
| `credit_spread` | level | Baa − 10y now | ✗ collinear with `nfci_lagged`, which contains it |
| `term_spread` | level | 10y − 2y now | ✗ collinear with `policy_rate_gap` |
| `sentiment` | level | Consumer sentiment now | ✗ no marginal signal |
| `equity_return` | year-on-year | Nasdaq | ✗ no marginal signal |
| `starts_growth` | year-on-year | Housing starts | ✗ U-shaped; a linear term cannot carry it |

Three shapes, and the distinction is not cosmetic. A **gap** is zero at origination
by construction, so it carries the *movement* and leaves the level to the
origination covariates — which is what keeps the pair from being collinear. A
**level** is the state of the world the loan is living in, whatever it was written
into. A **year-on-year change** uses a twelve-month window because the monthly
change in these series is mostly noise.

**`rate_gap` is switched by term**: a fifteen-year loan is compared with the
fifteen-year rate. That is possible only because `term_years` is in the cell key.
It is the *market* component of the refinancing incentive, not the whole of it — the
full measure needs the loan's own note rate, which the key does not carry.

### Why loan-to-value is split in two

`orig_ltv` and mark-to-market LTV are *equal* at origination and stay strongly
correlated afterwards, so fitting both gives unstable coefficients. The pair is
decomposed into a level — `orig_ltv`, underwriting at origination — and a movement —
`cltv_drift`, how far house prices have carried the position since. The movement is
zero at origination by construction, so the two carry nearly independent
information.

### Fields deliberately excluded from the model

| Excluded | Reason |
|---|---|
| Origination vintage as a covariate | Reserved for the time split. As a covariate it absorbs the macro effects the model exists to estimate. |
| Current delinquency status | A mediator, not a predictor. Including it inflates every metric while destroying the model's use. |
| Contemporaneous macro | Look-ahead, and no mechanism: a default in *t* was caused before *t*. Every series is lagged three months. |
| `amortization_type`, `interest_only_indicator` | Exactly **one** value each across the whole dataset. |
| All loss and proceeds columns | Populated only for defaults, and they need LGD, which is out of scope. |

---

## Layer 5 — Model matrix

**Grain: one episode** — the same rows as layer 4, re-expressed as half-open
intervals `(age_start, age_stop]` for the fitter, still carrying `n`.

| Field | Type | Definition |
|---|---|---|
| `age_start` | float | Loan age at the start of the episode. Passed as `entry_col`. |
| `lower_bound` | float | Lower interval bound |
| `upper_bound` | float | Upper interval bound, `inf` when right-censored |
| `exact_observation` | bool | **Always False.** See below. |

### The encoding

| Case | `age_start` | `lower_bound` | `upper_bound` | Likelihood contribution |
|---|---|---|---|---|
| Survived the month | `a` | `a+1` | `inf` | `log S(a+1) + H(a) = log[S(a+1)/S(a)]` |
| Defaulted that month | `a` | `a` | `a+1` | `log[S(a) − S(a+1)] + H(a)` |

The `+H(a) = −log S(a)` term is the left-truncation contribution, and it is what
makes each episode **conditional on surviving to `a`**. The product over episodes is
exactly the discrete-time likelihood with time-varying covariates — in a fully
parametric model.

**`exact_observation` is always False, including on the default row.** This looks
wrong and is not. lifelines enforces `lower_bound == upper_bound` if and only if
that flag is true, and uses the flag to mean *the event time is known exactly*. Our
knowledge is never exact: monthly reporting gives the month, never the day. Genuine
interval knowledge is therefore a **censored row with finite bounds**, and right
censoring is the same construct with an infinite upper bound.

---

## Worked example

Loan **F06Q10000595**, a real record from the 2006Q1 archive. A cash-out refinance
in Kansas that defaults after nineteen months.

**Layer 2 — as underwritten**

```
loan_identifier  F06Q10000595   loan_purpose          C    original_loan_term     180
classic_fico              679   occupancy_status      P    first_payment_date  200603
original_ltv               90   channel               R    number_of_borrowers      2
original_cltv              90   property_state       KS    property_type           SF
original_dti               32   first_time_buyer      N    mi_percentage           12
original_upb          131,000   original_rate     6.375
```

A fifteen-year cash-out refinance at 90% leverage on a 679 score: adverse, but
insured and with two borrowers.

**Layer 3 — the servicing file, 32 rows**

```
period  age  dlq  zero_balance        upb
200602    0   00             -   131,000
...
200706   16   00             -   123,206
200707   17   01             -   123,206
200708   18   02             -   123,206
200709   19   03             -   123,206   <- default: 90+ days
200710   20   04             -   123,206
...
200808   30   14             -   123,206
200809   31   15            01         0   <- zero balance, twelve months later
```

Two things this shows. The file keeps reporting for **twelve months after the
default**, with the delinquency counter climbing to fifteen. And the final row
carries zero-balance code `01`, which on its own reads as a **prepayment** — a loan
that paid off cleanly. Without cutting at the first terminating month, this loan
would be counted as both a default and a prepayment, or as neither.

**Layer 4 — the cell it lands in**

```
vintage  fico_s  orig_ltv  dti  purpose            occupancy       term_years  age  event
2006Q1     -0.4      85.0   32  refinance_cashout  owner_occupied          15   19   True
```

Every continuous covariate is at its band's midpoint: a 679 score is `fico_s` −0.42,
which falls in the band (−0.8, 0.0] carried as −0.4; 90% LTV falls in (80, 90]
carried as 85. This cell is shared with every other loan of the 2006Q1 vintage that
matches on all seven fields and defaulted in its twentieth month — and `n` counts
them.

**Layer 5 — the episodes, with the covariates rebuilt**

```
age  start  stop  lower  upper  event  period   cltv_drift  unemp_gap  nfci   rate_gap  hpi_growth
  0    0.0   1.0    1.0    inf  False  2006-01       0.000        0.0  -0.54     -0.000       0.144
  1    1.0   2.0    2.0    inf  False  2006-02      -0.436        0.0  -0.55     -0.150       0.141
  2    2.0   3.0    3.0    inf  False  2006-03      -0.639       -0.1  -0.53     -0.258       0.135
 ...
 17   17.0  18.0   18.0    inf  False  2007-06      -1.603       -0.6  -0.59     -0.630      -0.003
 18   18.0  19.0   19.0    inf  False  2007-07      -1.575       -0.5  -0.56     -0.650      -0.008
 19   19.0  20.0   19.0   20.0   True  2007-08      -1.462       -0.6  -0.53     -0.524      -0.014
```

Nineteen surviving episodes each contributing `log[S(a+1)/S(a)]` with an unbounded
upper limit, then a terminal episode bracketing the default between ages 19 and 20.
Every row carries `age_start` as its truncation point and none is an exact
observation.

Read `hpi_growth` down the column: **+14.4% a year at origination, −1.4% by the
month of default.** The loan was written at the top of the market and defaulted as
it turned. No covariate fixed at origination can represent that, which is the whole
argument for the time-varying construction.

---

## Structural guarantees

`creditsurv.data.panel.validate_episodes` enforces these on a loan-level panel, and
they are tested in `tests/test_panel.py`:

1. No duplicated `(loan_id, age)` — a duplicate double-counts its likelihood contribution.
2. No gaps: ages advance in unit steps — a gap silently drops exposure.
3. At most one event per loan.
4. An event only on a loan's final month — a loan cannot keep paying after defaulting.
5. No negative ages.
6. Every loan starts at `age = 0` — a loan starting later is a selection effect the likelihood is never told about.

None of these raise on their own. All of them bias the fit. On the aggregated path
the loan id is gone, so rules 1 and 3 cannot be checked per loan — which is exactly
why the modification defect above had to be found in the source data instead, and
why there is now a test for it at the SQL level.

---

## Known limitations

**FRED serves the latest revision, not the vintage.** Values are as *currently*
restated, not as first published, so macro covariates carry mild look-ahead.
Point-in-time data would require ALFRED. The publication lags partially compensate;
they do not eliminate it.

**Prepayment is treated as independent censoring.** It is really a competing risk: a
loan that prepays can never default, and the two share drivers. Treating it as
censoring assumes that, conditional on covariates, prepayment carries no information
about default risk — which biases lifetime PD **upward**. A scope decision, recorded
rather than buried.

**Modification is treated the same way**, with the same caveat and a stronger reason
to accept it: the modified loan is contractually a different loan.

**No LGD or EAD**, so no expected loss. PD only.
