# Decision log

The choices that shaped the model, the alternatives set aside, and the reasoning that was
later withdrawn. Each links to where the evidence lives. A repository that shows only what
worked is not much use to the next person who has to decide the same things.

## Rules set before the results

Four decisions were written down before the fits they govern, so that no result could pick
them:

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
| The Weibull, for now | the log-logistic, which has the better likelihood | changing family is a new selection; the gain against Kaplan-Meier is small; its falling long-age hazard is an extrapolation choice to be made on purpose | [Variable selection](variable_selection.md#the-distribution-family-and-why-the-weibull-was-kept-against-a-better-likelihood) |
| One fit, scored out of time, never refitted | refitting at four dates under two macro assumptions | eight fits for one report, and a report on one model beside a backtest of another invites the reader to credit one with the other's performance | [Backtest report](reports/backtesting.md) |
| A calendar cut | holding out random loans | random loans still train on the months they are judged on | [Calibration & backtest](calibration.md) |
| Gini from the exposure-weighted Lorenz curve | a concordance index | the index needs pairs of subjects, and a cell is not a subject | [Backtest report](reports/backtesting.md#discrimination) |

### The data

| Chosen | Over | Because | Evidence |
|---|---|---|---|
| The whole population, collapsed into weighted cells | a sample | episodes identical on every covariate and position in time are exchangeable, so the collapse is exact | [Data preparation](data_preparation.md#stage-3-coarse-classing-and-aggregation) |
| Monthly episodes | quarterly ones | the covariates move monthly; quarterly episodes only looked as compact because a monthly covariate was still in the key | [Data preparation](data_preparation.md#episodes-are-monthly) |
| Screening before aggregation | aggregating from a specification chosen in advance | by the time screening ran, mis-binned covariates were baked into millions of cells | [Data preparation](data_preparation.md#the-problem) |
| Dropping a loan missing a covariate | imputing it | imputing an underwriting characteristic invents the thing being measured; the price is HARP, now declared out of scope | [Data](data.md#what-is-out-of-scope) |
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

## Where this goes next

The findings still open -- decile calibration, the family, prepayment, HARP, point-in-time
macro -- are listed on the [validation response](validation.md#still-open). The next stage
keeps to survival models: prepayment as a competing risk with cumulative incidence, payment
history, HARP as a level of the key, a family rule and a materiality threshold for macro
covariates written before the fits, and a calibration anchored on a window of its own and
tested on several.
