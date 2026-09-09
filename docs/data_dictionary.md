# Data dictionary — the record layout

How the input data is built, what one row means at each stage, and what every field
contains. Read this before the methodology: the modelling choices only make sense
once the grain of the data is clear.

## Provenance

| Layer | Source | Real or simulated |
|---|---|---|
| Macroeconomic series | FRED (St. Louis Fed), public CSV endpoint, no API key | **Real** |
| Loan book | `creditsurv.data.synthetic` | **Simulated** |
| Loan book (optional) | Freddie Mac Single-Family Loan-Level Dataset | Real, registration required |

The loan book is simulated because no loan-level survival panel is publicly
available without registration. The macro paths it responds to are real, which is
what gives it its shape — vintages originated into 2006 meet the housing
collapse. See [methodology.md](methodology.md) for why this is a deliberate
choice rather than a shortcut, and what it costs.

## Regenerating

```bash
uv run creditsurv fetch-macro                                  # populates data/raw/fred/
uv run creditsurv build-data --source synthetic --n-loans 5000 --seed 42
```

`data/` is not committed. Everything under it is reproducible from the commands
above; the FRED cache holds raw downloads only, with all transformation in code.

## The four layers

Data passes through four shapes. Each has a different **grain** — the thing one
row represents — and confusing them is the most common way these models go wrong.

| Layer | One row is | Typical size |
|---|---|---|
| 1. Raw macro | one month of one economic series | ~330 months |
| 2. Origination record | one loan, as underwritten | 5,000 loans |
| 3. Loan-month panel | one loan in one calendar month | ~200,000 rows |
| 4. Model matrix | one *episode* of one loan | same as layer 3 |

---

## Layer 1 — Raw macroeconomic series

**Grain: one month of one series.** Monthly `PeriodIndex`, one column per series.

| Column | Series | Native frequency | Unit | Aggregation to monthly |
|---|---|---|---|---|
| `unemployment_rate` | `UNRATE` | Monthly | Percent | Last observation |
| `hpi` | `CSUSHPINSA` | Monthly | Index (Jan 2000 = 100) | Last observation |
| `mortgage_rate_30y` | `MORTGAGE30US` | Weekly | Percent | **Mean** over the month |
| `nfci` | `NFCI` | Weekly | Index, 0 = average conditions | **Mean** over the month |

Weekly series are averaged rather than sampled: the rate a borrower experiences
over a month is the average of its weeks, not whichever week happened to fall
last.

`DRSFRMACBS` (mortgage delinquency rate, quarterly) is fetched separately as a
**reference series only**. It is used to sanity-check simulated default rates
against reality and is never a covariate — it is an outcome, not a driver.

### Two rules about missing data

These are different problems and are handled differently.

**Interior gaps are forward-filled.** FRED has occasional holes — `UNRATE` has
one in the 1999–2026 window — and quarterly series only report every third
month.

**The trailing edge is truncated.** Series publish on different lags: `hpi` runs
two to three months behind `unemployment_rate`. Forward-filling that ragged edge
would invent macro observations that never existed, which then leak into every
covariate built on them. The panel therefore ends at the last month for which
*every* series has a real observation, and logs what it discarded. On a current
run the panel ends **2026-06**, dropping later readings from three
faster-publishing series.

---

## Layer 2 — Origination record

**Grain: one loan, as underwritten.** Fixed at origination and never revised.

| Field | Type | Unit / domain | Notes |
|---|---|---|---|
| `loan_id` | int64 | — | Primary key |
| `orig_period` | Period[M] | — | Origination month, the vintage |
| `credit_score` | float | 580–820 | FICO-like score |
| `fico_s` | float | ≈ −2.4 to +2.4 | `(credit_score − 700) / 50`, the modelled form |
| `orig_ltv` | float | 30–100 | Loan-to-value at origination, percent |
| `dti` | float | 10–55 | Debt-to-income, percent |
| `orig_upb` | float | USD | Original unpaid principal balance |
| `log_orig_upb` | float | log USD | The modelled form |
| `orig_spread` | float | Percentage points | `note_rate − mortgage_rate_30y` at origination |
| `note_rate` | float | Percent | Contractual rate |
| `purpose` | category | purchase / refinance_rate_term / refinance_cashout | Reference: purchase |
| `occupancy` | category | owner_occupied / second_home / investor | Reference: owner_occupied |
| `channel` | category | retail / broker / correspondent | Reference: retail |
| `region` | category | Northeast / Midwest / South / West | Reference: South |
| `first_time_buyer` | category | Y / N | Reference: N |

`region` is four census regions rather than fifty states on purpose: the design
matrix multiplies against a panel of hundreds of thousands of rows, and fifty
dummies buy little.

Credit quality is **correlated, not independent**. A Gaussian copula ties weak
scores to high leverage and high debt burden (ρ = −0.45 between score and LTV,
−0.35 between score and DTI), and pricing is risk-based, so `orig_spread` widens
as credit weakens. This is what makes variable selection a real exercise rather
than a formality.

---

## Layer 3 — Canonical loan-month panel

**Grain: one loan in one calendar month.** The long format. Produced identically
by the synthetic generator and the Freddie Mac loader, so everything downstream
is written once.

Carries every layer-2 field, repeated per month, plus:

| Field | Type | Definition |
|---|---|---|
| `period` | Period[M] | Calendar month of the observation |
| `age` | int64 | Months since origination. **0 in the first month.** |
| `indexed_cltv` | float | `orig_ltv × hpi(orig_period) / hpi(period)` — mark-to-market LTV |
| `cltv_drift` | float | `indexed_cltv − orig_ltv`. **0 at origination by construction.** |
| `unemp_gap` | float | `unemployment_rate(period) − unemployment_rate(orig_period)` |
| `refi_incentive` | float | `note_rate − mortgage_rate_30y(period)` |
| `nfci_lagged` | float | Financial conditions index, lagged |
| `event` | bool | True only on the month of default |
| `prepaid` | bool | True only on the month of prepayment |

### Publication lags are applied per series, not uniformly

| Series | Lag | Why |
|---|---|---|
| `unemployment_rate` | 3 months | Published in arrears and revised |
| `hpi` | 3 months | Published ~2 months in arrears and revised |
| `nfci` | 3 months | Conservative; weekly in practice |
| `mortgage_rate_30y` | **None** | Weekly market quote, never revised, known in real time |

A blanket lag would be simpler and would misstate what was actually knowable. A
borrower comparing their note rate to today's market rate does not wait three
months to do it.

### Why loan-to-value is split in two

`orig_ltv` and `indexed_cltv` are *equal* at origination and stay strongly
correlated afterwards, so fitting both gives unstable coefficients. The pair is
decomposed into a level and a movement:

- **`orig_ltv`** — underwriting quality at origination
- **`cltv_drift`** — how far house prices have carried the position since

`cltv_drift` is zero at origination by construction, so the two carry close to
independent information. `indexed_cltv` is retained in the panel for inspection
but is **not** a model covariate.

### Fields deliberately excluded from the model

| Excluded | Reason |
|---|---|
| Origination vintage | Reserved for the out-of-time split. As a covariate it absorbs the macro effects the model exists to estimate. |
| Contemporaneous revised macro | Look-ahead. All revised series are lagged. |
| Current delinquency status | A mediator, not a predictor. Including it inflates every metric while destroying the model's actual use — predicting lifetime PD from origination. |

---

## Layer 4 — Model matrix

**Grain: one episode of one loan** — the same rows as layer 3, re-expressed as
half-open intervals `(age_start, age_stop]` for the fitter.

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
| Defaulted that month | `a` | `a` | `a+1` | `log[S(a) − S(a+1)] + H(a) = log[1 − S(a+1)/S(a)]` |

The product over episodes is exactly the discrete-time likelihood with
time-varying covariates — in a fully parametric model.

**`exact_observation` is always False, including on the default row.** This looks
wrong and is not. lifelines enforces `lower_bound == upper_bound` if and only if
that flag is true, and uses the flag to mean *the event time is known exactly*.
Our knowledge is never exact: monthly reporting tells us the month, never the
day. Genuine interval knowledge is therefore a **censored row with finite
bounds**, and right censoring is the same construct with an infinite upper bound.

---

## Event definition and censoring

| Outcome | `event` | Encoding | Meaning |
|---|---|---|---|
| Default | `True` | Finite `[a, a+1]` | Terminal. Interval-censored within its month. |
| Prepayment | `False` | `[a+1, inf)` | Terminal, treated as **censoring** |
| Still performing at panel end | `False` | `[a+1, inf)` | Administrative censoring |

**Prepayment is treated as independent censoring.** In reality it is a competing
risk: a loan that prepays can never default, and the two share drivers such as
`refi_incentive`. Treating it as censoring assumes that, conditional on
covariates, prepayment carries no information about default risk. This biases
lifetime PD **upward**. It is a scope decision — this project models a single
risk — and it is recorded in [methodology.md](methodology.md) rather than buried.

---

## Worked example

Loan 2994 from `--n-loans 3000 --seed 2026`: a weak-credit cash-out refinance
that defaults after six months.

**Layer 2 — as underwritten**

```
loan_id      2994          orig_period   2016-09      purpose     refinance_cashout
credit_score  597.8        fico_s        -2.045       occupancy   owner_occupied
orig_ltv       81.65       dti           40.47        channel     retail
orig_upb   179,705.67      log_orig_upb  12.099       region      West
note_rate       4.443      orig_spread    0.983       first_time_buyer  N
```

Everything about it is adverse: a score two standard deviations below average,
leverage above 80, debt-to-income above 40, and a 98bp spread over the market
rate — risk-based pricing telling the same story the score does.

**Layer 1 — macro at origination and at default**

```
period    unemployment_rate      hpi   mortgage_rate_30y    nfci
2016-09                 5.0  183.925               3.460  -0.346
2017-03                 4.4  186.504               4.196  -0.487
```

**Layer 3 — the loan-month panel**

```
 age  period   indexed_cltv  cltv_drift  unemp_gap  refi_incentive  nfci_lagged  event
   0  2016-09        81.652       0.000        0.0           0.983       -0.366  False
   1  2016-10        81.159      -0.493       -0.1           0.973       -0.360  False
   2  2016-11        80.877      -0.775        0.0           0.673       -0.366  False
   3  2016-12        80.751      -0.901        0.1           0.245       -0.346  False
   4  2017-01        80.721      -0.931        0.0           0.293       -0.370  False
   5  2017-02        80.630      -1.022       -0.2           0.276       -0.418  False
   6  2017-03        80.554      -1.097       -0.2           0.247       -0.460   True
```

Note `cltv_drift` starting at exactly 0 and turning negative as house prices
rise, and `refi_incentive` collapsing from 0.98 to 0.25 as market rates climb
towards the note rate. Both are genuinely time-varying and loan-specific — the
kind of covariate a static model cannot represent.

**Layer 4 — the model matrix**

```
 age_start  lower_bound  upper_bound  exact_observation
       0.0          1.0          inf              False
       1.0          2.0          inf              False
       2.0          3.0          inf              False
       3.0          4.0          inf              False
       4.0          5.0          inf              False
       5.0          6.0          inf              False
       6.0          6.0          7.0              False
```

Six surviving episodes, each contributing `log[S(a+1)/S(a)]` with an unbounded
upper limit, then the terminal episode bracketing the default between ages 6 and
7. Every row carries `age_start` as the truncation point, and no row is ever an
exact observation.

---

## Structural guarantees

`creditsurv.data.panel.validate_episodes` enforces these, and they are tested in
`tests/test_panel.py`:

1. No duplicated `(loan_id, age)` — a duplicate double-counts its likelihood contribution.
2. No gaps: ages advance in unit steps — a gap silently drops exposure.
3. At most one event per loan.
4. An event only ever on a loan's final month — a loan cannot keep paying after defaulting.
5. No negative ages.
6. Every loan starts at `age = 0` — a loan starting later is a selection effect the likelihood is never told about.

None of these raise on their own. All of them bias the fit.

---

## Known limitations

**FRED serves the latest revision, not the vintage.** Values are as *currently*
restated, not as first published, so macro covariates carry mild look-ahead.
Point-in-time data would require ALFRED. The publication lags above partially
compensate; they do not eliminate it.

**The loan book is simulated.** Coefficients recovered from it describe the
generating process, not the US mortgage market. The Freddie Mac connector exists
for anyone who wants the real thing.

**Prepayment is treated as censoring**, as described above.
