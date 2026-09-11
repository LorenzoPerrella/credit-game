# Variable selection

Which covariates enter the model, how they were chosen, and what was discarded at
each step. The procedure follows [`nmds`](../../nmds), including its thresholds; the
method is taken from it, not the code.

Read [data_preparation.md](data_preparation.md) first for how the data reaches this
point, and [data_dictionary.md](data_dictionary.md) for what the fields mean.

## Why a procedure at all

With 48.8 million loans and thirty-odd candidate covariates, almost anything will be
statistically significant. Significance is not the constraint — *stability* is. A
specification chosen by searching this sample will fit this sample and will not
survive the next vintage.

So the order is: economic reasoning first, statistics second, and an explicit rule at
every step, written down before the results are seen.

## Everything is exposure-weighted

The panel is aggregated into cells carrying a loan-month count. An unweighted
statistic over cells would weight a cell holding six loan-months the same as one
holding sixty thousand — it would describe the *binning*, not the data. Every
statistic in `creditsurv.explore` and `creditsurv.models.selection` takes a weight
column, and the tests are built so an unweighted implementation fails them.

## The steps

### 1. Fill rate — is the field there at all?

`explore.fill_rate`. **Threshold: missing > 99% → drop.**

Several fields in this dataset are empty in every vintage. `vantagescore_4` and
`pre_harp_loan_sequence_number` are entirely absent; the loss columns
(`actual_loss`, `net_sales_proceeds`, expenses, recoveries) are populated only for
defaulted loans and only matter for LGD, which is out of scope.

⚠️ Fill rates were verified on 2025Q4. `modification_flag` is empty there and will
**not** be empty in the 2008–2012 vintages, so it must be checked on a crisis vintage
before being trusted either way.

### 2. Concentration — is there anything to contrast?

`explore.concentration_report`. **Threshold: largest level ≥ 99% → degenerate.**

Deliberately 99%, not 90%. A covariate that is 87% owner-occupied still has 13% of a
very large book saying something, and with 48 million loans that minority is millions
of observations. Degenerate should mean *there is nothing left to estimate a contrast
from*, not merely *lopsided*.

Rare levels are merged rather than dropped, following `nmds`, which folds categories
below 5% into an "other" class.

### 2b. Distinct values — what is actually in the field?

`creditsurv profile`, and the step that has to come before any remapping.

Reading the codes from the file layout and writing a `CASE` is not the same as
counting them. Doing it the other way round — distinct and count first, across seven
vintages from 1999 to 2024 — produced four corrections to mappings that had already
been written:

| Field | Observed | What the assumption got wrong |
|---|---|---|
| `loan_purpose` | P 28.9%, N 44.8%, C 26.3%, **`9` 0.0%** | `9` is "not available". An `ELSE` branch folded it into `refinance_rate_term`. |
| `channel` | R 55.0%, **T 24.8%**, C 14.4%, B 5.8%, `9` 0.0% | `T` is a quarter of the book. It had been merged into `correspondent`, which is smaller. |
| `amortization_type` | **FRM 100%** | Modelled as a covariate. It has one value. |
| `interest_only_indicator` | **N 100%** | Same. |
| `property_type` | SF 75.7%, PU 17.2%, CO 6.4%, MH 0.5%, CP 0.2%, `99` | Tail below 5% left as its own levels. |
| `number_of_units` | 1 98.1%, 2 1.4%, 3 0.3%, 4 0.2%, `99` | Same. |
| `occupancy_status` | P 92.0%, I 4.8%, S 3.3% | Below the 5% rule, but kept — see below. |

Decisions taken, and why:

- **`9` and `99` become NULL, and the loan is dropped.** They are missing-value codes,
  not categories. Every `CASE` in `_CATEGORICAL` now lists its branches explicitly and
  has **no `ELSE`**, so an unmapped code becomes NULL rather than being absorbed into
  whichever level the author happened to put last.
- **`channel`: `T` kept separate.** Third-party origination, not otherwise specified,
  is 24.8% of exposure. Merging it into `correspondent` at 14.4% would have hidden a
  quarter of the portfolio inside a smaller category.
- **`amortization_type` and `interest_only_indicator` dropped.** One value each. They
  are listed in `DEGENERATE_FIELDS` rather than silently omitted, so the next reader
  does not spend an afternoon adding them back.
- **`property_type`: MH and CP merged into `other`**; `number_of_units`: 2, 3 and 4
  merged into `2-4`. Both tails are below 5%, which is the `nmds` rule.
- **`occupancy_status`: all three levels kept**, although investor (4.8%) and second
  home (3.3%) sit below the 5% rule. With 48.8 million loans that is 2.3 million and
  1.6 million loans respectively — the rule exists to stop a level having nothing to
  estimate from, and neither of these is anywhere near that. They are also
  economically distinct in a way that merging would destroy. **This is a departure
  from `nmds`, made deliberately and on the size of the book.**

### 3. Default rate by band — does the covariate order the risk?

`explore.default_rate_by_band`. No threshold; this one is read, not applied.

The rate is **events over exposure** — a monthly hazard — not defaults over loans.
Two bands can hold the same number of defaults and differ entirely in risk if one was
watched ten times as long, and across twenty-five vintages that is the rule rather
than the exception.

What to look for: monotonicity. A covariate whose default rate rises and falls across
its own bands is either mis-binned or is proxying something else.

### 4. Kaplan-Meier by stratum — can one curve serve?

`explore.survival_by_stratum` and `explore.curves_cross`.

This runs **before** any model is fitted, and answers a question the coefficients
cannot: do the strata's survival curves *cross*?

- Curves that separate and stay separated are exactly what a single model with
  covariates is for. The covariate shifts the scale; one functional form serves.
- Curves that **cross** cannot be reconciled by scaling one into the other. No
  covariate coefficient will fix that, and it needs investigating before proceeding.

The project's commitment is to **one survival function**. Where heterogeneity is not
absorbed by the scale, the `ancillary` formula lets the *shape* parameter depend on
covariates too — which is still a single parametric model, not a segmentation.

### 5. Weighted correlation — which pairs say the same thing?

`explore.weighted_correlation` and `explore.collinear_pairs`.
**Threshold: |ρ| > 0.8 → keep one.**

Pairs are **reported, not resolved**. Which of two collinear covariates to keep is a
judgement about what the model is *for*, and `nmds` makes it explicitly with a stated
priority ladder rather than letting a procedure pick.

This matters here because the macro set contains five interest rates
(`MORTGAGE30US`, `MORTGAGE15US`, `FEDFUNDS`, `DGS10`, `T10Y2Y`) that are collinear by
construction. **The priority order must be fixed before the results are seen**, or the
choice becomes a rationalisation.

### 6. Variance inflation — collinearity beyond pairs

`models.selection.stepwise_vif`. **Threshold: VIF > 10 → drop, one at a time.**

Correlation catches pairs; VIF catches a covariate reconstructible from several
others together. The elimination is stepwise because dropping one covariate changes
every remaining factor.

`priority` protects covariates in stated order, most protected last. Without it the
procedure drops whichever offender happens to have the larger factor — arbitrary, and
unstable across samples.

⚠️ `nmds`'s own implementation no longer runs: it uses
`LinearRegression(normalize=True)`, removed in scikit-learn 1.2. This one computes the
factor by weighted least squares directly.

### 7. Univariate screening — does it carry anything alone?

`models.selection.univariate_screening`. **Threshold: p > 0.05 → discard.**

One model per candidate, with a fixed set of covariates forced into every fit
(`always_include`), so each is judged on what it *adds* rather than on what it happens
to proxy.

A screen, not a decision. A covariate can be insignificant alone and matter in
combination, which is why the survivors still face backward elimination.

### 8. Backward elimination — two criteria, applied together

`models.selection.backward_elimination`. **Thresholds: p > 0.05, and the expected
sign.**

The sign constraint is the most useful thing in this procedure and the least common.
A covariate whose coefficient comes out economically backwards is eliminated **even
when significant**, because a wrong sign is not a weak result — it is a symptom,
usually of collinearity — and a model asserting that higher credit scores default
sooner will fit this sample and no other.

Expected signs are on the **accelerated failure time** scale, where a positive
coefficient lengthens survival and therefore *lowers* risk:

| Covariate | Expected | Reasoning |
|---|---|---|
| `fico_s` | **+** | Better credit survives longer |
| `orig_ltv`, `orig_cltv` | **−** | More leverage fails sooner |
| `dti` | **−** | Heavier debt burden fails sooner |
| `cltv_drift` | **−** | Leverage rising after origination fails sooner |
| `unemp_gap` | **−** | Unemployment above its origination level fails sooner |
| `nfci_lagged` | **−** | Tighter financial conditions fail sooner |
| `mi_percent` | **+** | Insured loans are underwritten to a stricter standard |

One covariate is removed per step, the model refitted, and the test repeated. A
backwards sign outranks any p-value: it says the specification is wrong, not that the
evidence is thin.

## Results of running it

On the whole population: **15,858,492 cells covering 2,515,340,009 loan-months, with
1,938,519 defaults.** No sampling at any step.

### Candidates

Seventeen, after the screening and remapping above:

| Group | Covariates |
|---|---|
| Origination | `fico_s`, `orig_ltv`, `dti`, `term_years` |
| Mark-to-market | `cltv_drift` |
| Macro, gap since origination | `unemp_gap`, `policy_rate_gap`, `rate_gap` |
| Macro, level now | `nfci_lagged`, `term_spread`, `credit_spread`, `vix`, `sentiment` |
| Macro, year-on-year | `hpi_growth`, `inflation`, `equity_return`, `starts_growth` |
| Categorical | `purpose`, `occupancy` |

### 5. Weighted correlation — nothing reaches 0.8

| Pair | ρ (exposure-weighted) |
|---|---|
| `rate_gap` ↔ `policy_rate_gap` | **−0.793** |
| `credit_spread` ↔ `vix` | +0.701 |
| `policy_rate_gap` ↔ `term_spread` | −0.686 |
| `nfci_lagged` ↔ `credit_spread` | +0.670 |
| `nfci_lagged` ↔ `hpi_growth` | −0.622 |
| `nfci_lagged` ↔ `starts_growth` | −0.616 |
| `rate_gap` ↔ `term_spread` | +0.617 |

**Not one pair crosses the threshold**, and the closest is a hair under it. The
priority ladder was fixed in advance precisely for this case and did not have to be
used — which is the right order of events, and worth recording because a ladder that
is never needed looks like wasted work until the one time it is.

### 6. Variance inflation — nothing reaches 10

| Covariate | VIF | | Covariate | VIF |
|---|---|---|---|---|
| `credit_spread` | **8.41** | | `starts_growth` | 2.36 |
| `policy_rate_gap` | 4.89 | | `sentiment` | 2.27 |
| `nfci_lagged` | 4.45 | | `equity_return` | 2.25 |
| `rate_gap` | 4.17 | | `unemp_gap` | 2.01 |
| `hpi_growth` | 3.37 | | `cltv_drift` | 1.82 |
| `term_spread` | 3.37 | | `orig_ltv` | 1.18 |
| `inflation` | 2.99 | | `term_years` | 1.14 |
| `vix` | 2.71 | | `fico_s` | 1.08 |
| | | | `dti` | 1.06 |

The stepwise pass eliminated nothing. **All seventeen candidates survive both
collinearity screens.**

### Why thirteen macro covariates are not collinear

This was the expected failure and it did not happen, so the reason matters.

As *time series* these are hopelessly collinear — five of them are interest rates, and
over 1999–2026 the Fed funds rate, the ten-year yield, the term spread and both
mortgage rates move as one thing. A pure time-series regression on them would be
unusable.

**But the covariates are not the series.** The design matrix is indexed by *vintage
and age*, not by calendar time, and three of the covariates are **gaps since
origination** rather than levels. Two loans observed in the same month — identical
`credit_spread`, identical `vix`, identical `hpi_growth` — have entirely different
`rate_gap` and `unemp_gap` if one was written in 2004 and the other in 2019.

So the panel spans two dimensions where a time series spans one, and the gap
construction is what projects the covariates onto the second. `rate_gap` has a VIF of
4.17 against its own underlying rate series being almost perfectly collinear with
three others. That is not a trick: it is the identifying variation a vintage panel
actually has, and it is the same reason the macro covariates can be estimated at all
rather than being absorbed by the baseline hazard.

The corollary is a warning. If the gaps were replaced by levels — `policy_rate` now
instead of its move since origination — the covariates would collapse onto calendar
time and the collinearity would return at full strength. The decomposition into level
and movement is doing load-bearing work in more than one place.

## What this procedure does not do

**No information value or weight of evidence.** `nmds` does not use them either.
Screening here is p-value based, plus VIF, plus correlation, plus frequency and
missing-rate rules.

**No stepwise forward search.** Adding covariates by significance is how a
specification gets fitted to a sample.

**No automatic resolution of collinearity.** The procedure reports; the priority order
is stated by hand.

## Where the thresholds come from

| Check | Threshold | Source |
|---|---|---|
| Missing rate | > 99% | `nmds` `q6_*` |
| Minimum category share | < 5% → merge | `nmds` `q6_*` |
| Dominance / degeneracy | ≥ 99% | this project (`nmds` has no explicit rule) |
| Correlation | > 0.8 | `nmds` `s1_*`, `## Nota correlazioni` |
| Variance inflation | > 10 | `nmds` `stepwise_vif` |
| Univariate significance | p > 0.05 | `nmds` `fit_single_aft` |
| Backward elimination | p > 0.05 + sign | `nmds` `survival_backward` |
