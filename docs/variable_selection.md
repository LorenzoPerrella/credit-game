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

### 7-8. The fit, and what the signs said

One fit, 19 terms, **308.8 minutes** on 15,858,492 cells. It converged cleanly — 168
iterations, exit mode 0 — and produced a table in which **every covariate has
p = 0.0000**.

That is the single most important line in this document. At 2.5 billion loan-months a
p-value is not a filter: it separates nothing, because nothing is insignificant. The
univariate screen at p > 0.05 and the backward elimination's p-value arm are both
**inert on this data**, and every discriminating decision has to come from somewhere
else — the sign, the shape of the marginal relationship, or the economics.

So the diagnosis was made by comparing each covariate's **marginal** relationship with
default against its **conditional** coefficient. That comparison separates three things
a bare "wrong sign" cannot:

| | Marginal | Conditional | Diagnosis |
|---|---|---|---|
| Same direction, both strong | ✓ | ✓ | The covariate works; if it contradicts the prior, **the prior was wrong** |
| Right direction alone, flips with others | ✓ | ✗ | **Collinearity**, the classic case |
| No ordering at all | ✗ | (large, p=0) | **No content**; significance is an artefact of size |

### What survived, and what it is worth

The marginal default hazard across each covariate's own bands, events over exposure,
on the whole population. The ratio is first band to last:

| Covariate | First band | Last band | Ratio | Sign |
|---|---|---|---|---|
| `fico_s` | 28.11 bp | 3.23 bp | **8.7×** | + |
| `cltv_drift` | 6.12 bp | 49.65 bp | **8.1×** | − |
| `unemp_gap` | 3.89 bp | 28.52 bp | **7.3×** | − |
| `vix` | 4.40 bp | 32.23 bp | **7.3×** | − |
| `policy_rate_gap` | 26.99 bp | 4.13 bp | **6.5×** | + |
| `hpi_growth` | 20.61 bp | 3.86 bp | **5.3×** | + |
| `rate_gap` | 3.62 bp | 17.51 bp | **4.8×** | − |
| `inflation` | 17.56 bp | 4.83 bp | 3.6× | none |
| `dti` | 3.93 bp | 13.70 bp | 3.5× | − |
| `nfci_lagged` | 6.42 bp | 19.99 bp | 3.1× | − |
| `term_years` | 3.10 bp | 9.37 bp | 3.0× | none |
| `orig_ltv` | 4.75 bp | 14.35 bp | 3.0× | − |

Twelve covariates, plus `purpose` and `occupancy`.

### Two priors that were wrong, and the revision

`rate_gap` and `policy_rate_gap` came out against the signs written for them, and both
are kept with the prior **revised** rather than eliminated. This is the one place the
procedure was overruled, so it is set out in full.

Revising an expected sign after seeing the fit is precisely what the sign constraint
exists to prevent, and doing it on the strength of the fit would make the constraint
worthless. The justification here is not the fit. It is a mechanism that can be checked
without it — and the check is the marginal table above, which was computed
independently of the model and says the same thing, strongly and in order.

**What both priors assumed is a floating-rate transmission channel that a thirty-year
fixed-rate mortgage does not have.** The borrower's payment does not move when the
policy rate moves. Strip that channel out and the remaining sign is the opposite one in
each case.

`policy_rate_gap` — the Fed cuts in crises and tightens into strength, so a policy rate
far *below* the one the loan was written at means 2009 or 2020, not relief. The band
5.5pp below origination carries **26.99 bp** of monthly default, the highest of any
band of any covariate in this model; the band 5.5pp above carries 4.13 bp.

`rate_gap` — market rates below the note rate mean refinancing is open, and whoever can
refinance does, leaving the book as a prepayment, which this model treats as censoring.
Who stays is who *cannot*: impaired credit, no equity. The coefficient measures that
adverse selection, not the payment burden:

```
rate_gap   -4.00 → 3.62 bp    +0.25 →  6.20 bp
           -1.50 → 4.21 bp    +0.75 → 11.94 bp
           -0.75 → 3.48 bp    +1.50 → 16.64 bp
           -0.25 → 3.58 bp    +4.00 → 17.51 bp
```

Flat at 3.5 bp for as long as rates sit above the note rate; 4.8× higher across the
whole range where refinancing is attractive. **This is the competing-risk limitation
this project declares, appearing as a measurable and ordered effect rather than as a
caveat.** Treating prepayment as independent censoring is not innocuous here: it is
visibly wrong in the direction theory predicts, and the sign flip is how it announces
itself.

> **This argument was wrong, and is retracted below.** Both covariates are eliminated
> in step 9, and the marginal ordering that justified the revision turns out to be the
> macro cycle rather than the refinancing incentive. The section is kept as written
> because a procedure document that quietly deletes its own mistakes is not a record of
> anything.

## The five covariates given up

Recorded in `config.ELIMINATED`, with a test asserting none of them reappears in the
formula — a covariate that quietly comes back is a silent reversal of a documented
decision, and nothing else in the suite would notice.

### `credit_spread` and `term_spread` — collinearity, the textbook case

Both order default correctly **on their own** and flip once the others are present.

```
credit_spread  0.75 → 4.55 bp    2.75 → 10.73 bp       (4.1×, monotone)
               1.75 → 4.38 bp    3.50 → 15.13 bp
               2.25 → 4.54 bp    5.50 → 18.49 bp
```

Conditional coefficient **+0.123**: wider spreads, longer survival. The cause is not
mysterious. `nfci_lagged` is in the model, and the Chicago Fed's index is built from 105
indicators of risk, liquidity and leverage — **including Baa-Treasury spreads**. With
NFCI present, `credit_spread` is a residual, and ρ(`credit_spread`, `vix`) = +0.70
finishes the job. `term_spread` is the same story against `policy_rate_gap`
(ρ = −0.69), its other view of the same monetary cycle.

A wrong sign is not a weak result — it is a symptom, and this is what of.

**This rule is an addition to the `nmds` procedure, not a borrowing from it.** `nmds`
eliminates on a violated *prior*; `term_spread` never had one, so a prior-based rule
would have kept it. The rule applied here is a **marginal/conditional sign reversal**:
a covariate whose conditional coefficient contradicts its own unconditional
relationship with the outcome is carrying something other than what its name says. It
is stated so it can be applied consistently rather than invoked when convenient.

### `equity_return` and `sentiment` — no content

```
equity_return   -0.45 → 9.45 bp   +0.05 → 7.99 bp   +0.30 → 6.63 bp
                -0.10 → 7.05 bp   +0.15 → 8.34 bp   +0.95 → 8.70 bp

sentiment        57.5 → 7.76 bp    80.0 → 6.47 bp   105.0 → 8.74 bp
                 70.0 → 10.44 bp   90.0 → 5.10 bp
```

Flat and unordered across their entire range: 1.4× and 2.0×, against 8.7× for
`fico_s`. Every band holds between 8% and 32% of exposure, so this is not a small-band
artefact. **Neither covariate carries any univariate information about mortgage
default.**

And both are hugely significant. `equity_return` receives a coefficient of −0.44 at
p = 0.0000, on 2.5 billion loan-months, having no relationship with the outcome at all.

If this document makes one argument worth taking away, it is that one.

### `starts_growth` — real information, wrong functional form

```
starts_growth  -0.50 → 16.81 bp    +0.05 →  6.12 bp
               -0.20 →  6.78 bp    +0.20 →  7.57 bp
               -0.05 →  5.34 bp    +0.90 → 15.13 bp
```

**A U.** Both extremes carry three times the risk of the middle: the collapse of
construction (2008–2010) and the boom (2005–2006, 2021). A construction boom *is* the
top of the housing cycle, and loans written at the top are the worst in the dataset.

A linear term fits a straight line through a parabola and its slope means nothing —
which is why the conditional coefficient is large, significant and uninterpretable. The
information is real; the form cannot carry it.

It is eliminated **as a linear term**, and the way back in is stated rather than left
implicit: banded and entered as a categorical, the U is representable. That is what
`nmds` would have done, and it is the one place where its approach is straightforwardly
better than this one — see below.

### 9. Stability — does the coefficient survive a change of sample?

The twelve survivors were fitted, and three of them came out with the **wrong sign**:
`nfci_lagged`, `policy_rate_gap` and `hpi_growth`. The last is the alarming one, because
its marginal relationship is among the cleanest in the dataset — 20.61 bp to 3.86 bp,
monotone, 5.3× — and the fitted model now said rising house prices *shorten* survival.

The obvious suspect was the elimination itself: five covariates had been removed, and
perhaps the survivors were now carrying what those had absorbed. **That was tested and
it is not the cause.** On identical rows, the two specifications agree:

| | 17 covariates | 12 covariates |
|---|---|---|
| `nfci_lagged` | +0.0139 | +0.0839 |
| `policy_rate_gap` | +0.0796 | +0.0849 |
| `hpi_growth` | +3.3200 | +3.4021 |

The only other difference between the two runs was the **sample**: one was fitted on
everything, the other on everything up to 2024-12 — 6.3% less exposure. And when 6% of
a sample flips three signs, the problem is not the 6%.

#### What the effect sizes show

Raw coefficients hide this, because the covariates are on wildly different scales:
`inflation` has a standard deviation of 0.017 and `cltv_drift` one of 15.2. Ranked by
**effect of one standard deviation on log survival time**:

| Covariate | 1 sd effect | Under a change of sample |
|---|---|---|
| `vix` | **−0.3216** | stable |
| `cltv_drift` | **−0.2938** | stable |
| `inflation` | **+0.1305** | stable |
| `unemp_gap` | **−0.0943** | stable |
| `hpi_growth` | −0.0917 | **flipped** |
| `nfci_lagged` | +0.0526 | **flipped** |
| `policy_rate_gap` | −0.0406 | **flipped** |
| `rate_gap` | −0.0066 | effectively zero |

**The three that flip are three of the four smallest effects**, and each sits beside a
larger correlated covariate carrying the same economic information:

| Dimension | Kept | ρ | Flipped |
|---|---|---|---|
| Financial stress | `vix` (−0.32) | +0.55 | `nfci_lagged` (+0.05) |
| House prices | `cltv_drift` (−0.29) | −0.31 | `hpi_growth` (−0.09) |
| Interest rates | `rate_gap` (−0.007) | −0.79 | `policy_rate_gap` (−0.04) |

On the house-price pair this is literal rather than statistical: **`cltv_drift` is
constructed from the house price index.** The information enters the model twice, and
the second entry is a residual whose sign is noise.

#### The rule

> A covariate whose **standardised effect is small** and which sits beside a **larger
> correlated covariate carrying the same economic information** is not identified. Its
> sign is noise, and it will move when the sample does.

This is not the sign constraint, and it is worth being clear why not. A sign constraint
asks whether a coefficient points the right way; this asks whether it points anywhere at
all. Applied here it *explains* the earlier sign failures rather than chasing them: a
covariate-by-covariate elimination on a jointly identified block keeps rotating the
basis, and each refit produces a different set of wrong signs.

Neither correlation at 0.8 nor VIF at 10 catches it. `nfci_lagged` against `vix` is
ρ = 0.55, well inside both thresholds, and the covariate is still not identified —
because identification depends on the effect size relative to the shared variation, not
on the shared variation alone.

#### The specification this leaves

**One covariate per economic dimension**: housing (`cltv_drift`), labour (`unemp_gap`),
financial stress (`vix`), prices (`inflation`). Four, from the thirteen that were built.

Fitted on both samples, it holds:

| Covariate | Training (≤2024-12) | Whole population | Change |
|---|---|---|---|
| `fico_s` | +0.3838 | +0.3899 | 1.6% |
| `orig_ltv` | −0.0155 | −0.0157 | 1.3% |
| `dti` | −0.0186 | −0.0186 | **0%** |
| `term_years` | −0.0383 | −0.0378 | 1.3% |
| `cltv_drift` | −0.0154 | −0.0161 | 4.5% |
| `unemp_gap` | −0.0377 | −0.0386 | 2.4% |
| `vix` | −0.0288 | −0.0287 | **0.3%** |
| `inflation` | +5.0810 | +5.0072 | 1.5% |

No sign changes and nothing moving more than 4.5%, against `hpi_growth` going from
+0.75 to −1.31 on the same two samples one specification earlier. Every standardised
effect now sits between 0.09 and 0.47: there is no residual covariate left whose sign
could be noise.

#### The retraction

The argument made in *"Two priors that were wrong, and the revision"* above is
**withdrawn**. `rate_gap`'s conditional effect is −0.0066 per standard deviation, which
is zero; the 4.8× marginal ordering that was offered as independent evidence is the
macro cycle, the same thing that produced the spurious hump in the marginal hazard. The
mechanism — that a fixed-rate book has no payment channel — may well be true and is not
what the data was showing.

Recorded rather than edited away, and mirrored in the code beside where the revision
used to live. The lesson generalises: **a marginal relationship is evidence that a
covariate is correlated with the outcome, never that it is identified in a model.**

## Would `nmds` have done the same?

Four of the five, yes. The other decisions diverge, and it is worth being precise about
which.

| Decision | `nmds` | Here |
|---|---|---|
| `credit_spread` eliminated on a violated sign | **Yes** — `elimination_type="invalid_coefficient"` | Same |
| `hpi_growth`, `nfci_lagged`, `policy_rate_gap` eliminated on stability | **No** — it has no stability step; it would have eliminated them on sign and then refit into the next set of wrong signs | Eliminated on the standardised-effect rule |
| `term_spread` eliminated | **No** — it has no expected sign, so nothing catches it | Marginal/conditional reversal |
| `equity_return`, `sentiment` eliminated | **On sign**, if it had a prior for equity indices — its univariate screen is p > 0.05, which they pass at p = 0 | On absent marginal content |
| `starts_growth` as a linear term | **Would not arise** — `nmds` coarse-classes into bands, so the U is representable | Eliminated; banding is the way back |
| `rate_gap`, `policy_rate_gap` priors revised | **No** — it would have eliminated both | Revised, with the mechanism stated |

Two of these deserve more than a row.

**`nmds` bins everything, and this pipeline does not.** Its covariates enter as coarse
classes, which is why a U-shaped relationship is not a problem there and is one here:
a class-based specification represents any shape, at the cost of parameters and of a
binning decision per covariate. This pipeline bins the origination covariates and
enters the macro-derived ones as linear terms, which is cheaper and assumes monotonicity
— an assumption `starts_growth` violates and the others do not. **On this point `nmds`
is simply better**, and the fix is known rather than hypothetical.

**Revising the two priors is a departure, and the weaker of the two arguments here.**
`nmds` would have dropped `rate_gap` and `policy_rate_gap` and kept its discipline
intact, and a reader who thinks that is the right call would be applying the reference
procedure correctly. The case for keeping them rests on the marginal tables being
computed independently of the fit, on both mechanisms being checkable without it, and on
what would be lost: `rate_gap` is the only covariate in the model that exposes the
competing-risk limitation, and eliminating it would remove the evidence of a known
weakness rather than the weakness.

Both are recorded as revisions. Neither is presented as a prior.

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
