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
