# Decision log

The choices that shaped the model, the alternatives set aside, and the reasoning that was
later withdrawn. Each links to where the evidence lives. A repository that shows only what
worked is not much use to the next person who has to decide the same things.

## Rules set before the results

Fourteen rules are written down before the fits they govern, so that no result can pick them --
`docs/rules.md` holds them all, with the measurement behind each. These four are the oldest:

| Decided in advance | Why it had to be in advance |
|---|---|
| The rule choosing the moratorium policy: `exclude` stands unless it fails a backtest criterion that `censor` passes | Two event definitions estimate different dependent variables; chosen afterwards, the choice would follow the prettier backtest |
| The acceptance criteria: actual over expected from 0.80 to 1.25, overall and in every decile, and a Gini above 0.45 | A backtest with no criterion can be read but not passed or failed |
| The order in which variance inflation removes covariates | Otherwise the procedure drops whichever offender has the larger factor, which is arbitrary and unstable across samples |
| The economic dimension of every candidate | The stability rule removes a covariate only beside a larger one of the same dimension; decided afterwards it would drop whatever came out inconvenient |

## Choices, and what they were chosen over

### The model

| Chosen | Over | Because | Evidence |
|---|---|---|---|
| A parametric survival model | a classifier; a Cox model | lifetime PD needs the timing, the censored loans, extrapolation past the data, a response to scenarios and a smooth term structure | [Methodology](methodology.md) |
| Interval-censored monthly episodes with left truncation | `CoxTimeVaryingFitter`; loan-level durations | it is exactly the discrete-time likelihood with time-varying covariates, in a fully parametric model | [Home](index.md#the-idea-in-one-table) |
| Nested tests and comparisons against Kaplan-Meier | the generalised gamma, which nests five families | it does not converge on the episode panel under any remedy tried, and on data generated from a Weibull it estimated the shape at 4.04 where the truth is 1 | [Methodology report](reports/methodology.md) |
| The Weibull | the log-logistic, which has the better likelihood | rule 2 on the Aalen-Johansen incidence, 0.1027 pp against 0.1498. Reopened in October 2026 against the decile criterion rule 2 never considered, and the log-logistic taken through its own selection under the bands: it reads **6 of 10** deciles in band against the Weibull's 5, so neither family reads it alone and the family is not the answer | [Rules](rules.md), [Variable selection](variable_selection.md#the-distribution-family-and-why-the-weibull-was-kept-against-a-better-likelihood) |
| One fit, scored out of time, never refitted | refitting at four dates under two macro assumptions | eight fits for one report, and a report on one model beside a backtest of another invites the reader to credit one with the other's performance | [Backtest report](reports/backtesting.md) |
| A calendar cut | holding out random loans | random loans still train on the months they are judged on | [Calibration & backtest](calibration.md) |
| Gini from the exposure-weighted Lorenz curve | a concordance index | the index needs pairs of subjects, and a cell is not a subject | [Backtest report](reports/backtesting.md#discrimination) |

### The data

| Chosen | Over | Because | Evidence |
|---|---|---|---|
| The whole population, collapsed into weighted cells | a sample | episodes identical on every covariate and position in time are exchangeable, so the collapse is exact | [Data preparation](data_preparation.md#stage-3-coarse-classing-and-aggregation) |
| Monthly episodes | quarterly ones | the covariates move monthly; quarterly episodes only looked as compact because a monthly covariate was still in the key | [Data preparation](data_preparation.md#episodes-are-monthly) |
| Screening before aggregation | aggregating from a specification chosen in advance | by the time screening ran, mis-binned covariates were baked into millions of cells | [Data preparation](data_preparation.md#the-problem) |
| Dropping a loan missing a covariate | imputing it | imputing an underwriting characteristic invents the thing being measured. The price was HARP, and it is **no longer paid**: rule 8 keeps the ratio missing and carries a `harp` level that absorbs the constant filling it, which is the dummy-variable adjustment rather than an imputation | [Data](data.md#what-is-out-of-scope), [rule 8](rules.md) |
| *loan-to-value at origination* (`original_ltv`, formerly `orig_ltv`) and *loan-to-value change since origination* (`ltv_change`, formerly `cltv_drift`) | the indexed loan-to-value, or the estimated one | the level and the movement are separable; the estimated loan-to-value covers 0.8% of the 1999 vintage and 94% of 2021 | [Data dictionary](data_dictionary.md#why-loan-to-value-is-split-in-two) |
| *origination channel* (`channel`) as retail against broker or correspondent | four levels | a coding change in 2009, not a market one | [Portfolio](portfolio.md#what-was-written) |
| Every macro series lagged three months, market quotes included | contemporaneous readings "known in real time" | a loan 90 days delinquent in a month missed its payments in the three before it | [Data dictionary](data_dictionary.md#every-series-is-lagged-three-months-for-one-of-two-reasons) |
| A count of loan-months as the weight | the balance | PD is per obligor | [Data preparation](data_preparation.md#the-weight-is-a-count-never-an-amount) |
| `exclude` for moratoria | `censor`; counting them as defaults | forbearance went to borrowers under strain, so censoring removes loans because of their risk | [Moratorium report](reports/moratorium.md) |
| The exact origination month in the key | the quarter | the quarter read every macro series 2.15 months late on average, 3.51 times the cells for the fix | [Data preparation](data_preparation.md#the-grouping-key) |

## Withdrawn

!!! failure "Two priors revised after the fit, then retracted"
    *mortgage rate fall since origination* (`mortgage_rate_decline`, formerly `rate_gap`) and *policy rate change since origination* (`policy_rate_change`, formerly `policy_rate_gap`) came out against their declared signs, and the first run
    kept both with the priors revised, on a mechanism -- a fixed-rate mortgage has no floating
    payment channel -- and on a marginal ordering of default rates by band offered as
    independent evidence. **Withdrawn**: the conditional effect of *mortgage rate fall since origination* was zero, and the
    marginal ordering was the macro cycle. The lesson is now a rule of the project: a marginal
    relationship is evidence that a covariate is correlated with the outcome, never that it
    is identified in a model.
    [The retraction](variable_selection.md#the-retraction) is kept beside the argument it
    withdraws.

!!! failure "The selection command did not run the rule its documentation described"
    Its first complete run checked declared signs and p-values and nothing else, so it kept
    *mortgage rate fall since origination*, *inflation* (`inflation_rate`, formerly `inflation`) and *equity return* (`equity_return`), whose signs in the full model contradicted
    their own. The rule was stated before that run; the omission was found by reading the run,
    and the run was committed as it came out so the order of events can be weighed.

!!! failure "A survival curve from frozen covariates"
    An earlier comparison with Kaplan-Meier predicted each loan's curve from its origination
    covariates, which freezes house prices and unemployment where they were at origination.
    It overstated five-year survival by eight percentage points. The curve now chains the
    hazard along each loan's realised covariate path.

!!! failure "A scenario the model never reached"
    An earlier projection advanced calendar time from each loan's origination rather than from
    the reporting date, so the projected macro path was never reached and both scenarios gave
    almost the same answer. Later the scenario moved two series no covariate read, and then
    kept describing volatility after the reselection removed it. The legs are now built from
    the scenario itself, and a test fails when the shocked series and the formula part.

!!! failure "Sentinels and truncation"
    The estimated loan-to-value's 999 sentinel was missed, so mark-to-market leverage came
    out as 924 for most of the panel and its coefficient took the wrong sign. Truncating
    loans at their smallest terminating age cut the wrong row after a modification and lost
    641 real defaults on 2006Q1. Both were found by tables that did not order risk the way
    the other covariates did.

!!! note "Point-in-time macro series were looked for, and are not available"
    The validation's observation that every macro series is the *revised* one, and the
    attempt to answer it. FRED's public graph endpoint accepts a vintage date and **ignores
    it**: `?id=UNRATE&vintage_date=2019-06-01` and `?id=UNRATE_20190601` both return 200 and
    both return the current series, running to the latest observation, with April 2019
    unemployment at today's 3.7. ALFRED's own endpoints answer 404 without a key and the
    API answers 400. So every backtest here reads revised data.

    What that costs is a backtest fair about the *model* and optimistic about the *data*:
    the unemployment rate a 2018 model would have been handed differs from the one it is
    scored with by a revision nobody could have known at the time. The finding is held to
    the network by a test, so the day the endpoint honours the parameter the suite says so
    rather than the limitation quietly outliving its reason.

## What the last stage decided, and what it could not

Prepayment, the family, HARP and payment history are closed, each by the thing named above as
what would close it -- the detail and the measured numbers are on the
[validation response](validation.md#what-the-next-stage-closed). Three decisions inside that
stage are worth recording for the same reason this log exists.

**The event definition is three-state now, and `panel.CENSORED` names the third.** Default,
prepayment, neither. The third state had no name for a while, which is how it stayed invisible
while two commands went on asking the cell table for a boolean `event` column that no longer
existed -- a fault that could not reach the tests, because both commands read the production
table rather than a fixture.

**A covariate may be in the model because removing it left something unfittable.** Rule 11, and
the prepayment model has one: without *unemployment change since origination* the polish stops
912 standard errors out with every step it wants refused below the parent's optimum, and the same
happens from SLSQP, from L-BFGS-B and from a cold start. That is a statement about lifelines'
likelihood, not about the covariate, and the elimination table says so rather than reporting a
clean removal.

**A banded covariate is read as bands.** Rule 12, written after the first backtest of the stage
and before the fit that answered it, which is the only order that makes it a rule rather than a
preference: the in-sample deciles said the risk spread was compressed by 40%, three covariates
that are already bands in the key were entering as a straight line through their midpoints, and
reading them as bands recovered 10.2% of the gap. What it did **not** do is close it. The reason
offered then was resolution inside the bands, and the section below records that this was measured
and is wrong: the finer grid bought 0.015 on the worst decile, and the cause was the hazard's
shape.

## What the shape stage found, and where this goes next

**Resolution was not the constraint, and the measurement that says so cost a table.** Rule 7's
ceiling was re-declared at 250 million cells from the engine's measured capacity *before* the
rebuild, the finer 8 / 8 / 6 grid was built at 194,340,956 cells, and a selection under it chose
the **same ten covariates with the same nine eliminations**. Actual over expected in the worst
in-sample decile moved **0.603 to 0.618** against a band of 0.80 to 1.25. The key went back to
5 / 4 / 4; the fine table is kept so the measurement is not paid twice.

**The compression was the hazard's shape.** Rule 14, written before the fit: the shape reads the
nine covariates **fixed at origination** and none of the six recomputed every loan-month, because
`period = cohort + age` holds identically and a covariate that moves over a loan's life moves with
the age -- which rule 14 already forbade the shape to read. All nine, because a subset would be a
hand-pick, and the first attempt was one: it named `credit_score` on an unmeasured belief, and the
declared procedure reached the same covariate **with the measurement that it is the right one**
(84% of the shape gain on 22% of the parameters) plus two the pick had not considered.

The shape takes the in-sample deciles from **5 of 10 in band to 9 of 10** and flips rule 5's cycle
criterion from fail to pass. It also overshoots: the error changes sign, from a predicted spread
of 79.6x against 125x realised to 214x against 139x.

**Four cells, and none reads the band.** {Weibull, log-logistic} x {scalar shape, the origination
side} was declared exhaustive before the last was fitted, so completing it is not a search, and
all four are published. The best misses the 1.25 ceiling by 0.165.

**And the residual has a cause.** Decile 1 is 89.2% the top credit-score band, and **83% of its
miss is the 2019, 2020 and 2021 originations** at an actual over expected of 3.413, where every
vintage to 2018 reads 1.082. Those loans stack every favourable reading at once -- top band,
policy rate at zero, house prices rising, housing starts up -- and a **multiplicative hazard has
no floor**: the prediction reaches 1.5 defaults per 100,000 loan-months a month where the book
does not go below about 5.

!!! failure "A correct inference retracted on a table that could not refute it"
    The missing floor was inferred from a fitted shape of 2 and `h(0) = 0`, correctly labelled an
    inference rather than a measurement. It was then **retracted** because the aggregate 0-to-11
    age band over-predicts at 0.755 -- and that reasoning does not hold, because the aggregate
    mixes all ten deciles and the young-age over-prediction belongs to the *risky* ones, whose
    fitted shape is below one. Inside decile 1 the first year under-predicts at 1.698. **A sum
    over one dimension cannot refute a claim about an interaction**, which is the same lesson this
    project already carries about diagnostics that go mute at scale, met from the other side.

### Where this goes next

The residual is a **documentable exception** rather than an open finding: a cause, a scope (three
vintages; without them 1.082), a magnitude (0.11% of the in-sample defaults) and a remediation
named rather than searched for. Whether that exception is accepted is a decision for the
validation, not for the model, and the test window stays unread until it is made -- rule 14 says
so, and the diagnosis makes a prediction about that window before anyone looks: the pattern is
young loans in the safest band under benign macro readings, and the window's young loans are the
2022 to 2024 originations.

A hazard with a floor is an additive baseline, which is neither of the two families rule 2
declared. It would need its own rule and its own motive, written before the fit rather than
after this one, and the grid deliberately stopped at four rather than reaching for a fifth
specification chosen by which came closest on the criterion it is judged by.
