# Model

## In short

| | |
|---|---|
| Family | Weibull accelerated failure time, shape <!-- value: model.rho --> -- chosen by [rule 2](rules.md), not by likelihood |
| Competing risk | Prepayment, modelled; lifetime PD is a cumulative incidence |
| Twelve-month PD, origination book today | **<!-- value: scenario.pd_12m -->** |
| Lifetime PD over 36 months, baseline | <!-- value: scenario.baseline --> |
| Lifetime PD over 36 months, adverse | <!-- value: scenario.adverse --> |
| Adverse against baseline | **<!-- value: scenario.multiple -->** |

```mermaid
flowchart TD
    A["At origination<br/>credit score, loan-to-value, debt-to-income, term<br/>purpose, occupancy, insurance, buyer, HARP"] --> M
    B["Along the loan's life, lagged three months<br/>house prices, unemployment, policy rate,<br/>yield curve, sentiment, housing starts"] --> M
    A --> P
    B2["Along the loan's life<br/>the fall in the market rate, house prices,<br/>volatility, equity return, financial conditions"] --> P
    M["default: log survival time"] --> Q["the monthly default hazard<br/>at each age"]
    P["prepayment: log survival time"] --> S["the monthly prepayment hazard<br/>at each age"]
    Q --> R["cumulative incidence of default:<br/>term structure and lifetime PD"]
    S --> R
```

**Two hazards, not one, and the lifetime PD is a cumulative incidence.** A loan that repays
cannot later default, so treating repayment as censoring assumes it is uninformative about
default -- and on this book it is the opposite of uninformative: there are 33,797,300
prepayments against 1,671,207 defaults, twenty times as many exits through the door the model
used to ignore. The lifetime PD is therefore the Aalen-Johansen quantity, the share that has
left through default by a given age with the competition accounted for, rather than `1 - S` from
a single hazard, which overstates it at long horizons.

The prepayment model is selected by the same procedure under its own declared priors -- a credit
score that lengthens survival shortens the time to repayment, so the two sets of priors cannot be
one -- and what survives says what one would hope: the fall in the market rate since origination
at -0.198 per standard deviation and equity volatility at -0.215 are its largest effects, while
inflation, its change, volatility change and sentiment were all eliminated for turning their
signs. A refinancing is decided by the rate on offer, not by the business cycle at large. The
record is in the [prepayment selection](reports/selection_weibull_prepayment.md).

## Reading a coefficient

This is an **accelerated failure time** model: a coefficient acts on the logarithm of
survival *time*. **Positive lengthens survival and lowers risk.** `exp(coef)` is a time ratio,
not a hazard ratio; reading it as one inverts every conclusion.

Coefficients are not comparable across covariates measured in different units, so the
continuous ones are shown as the effect of one standard deviation, exposure-weighted on the
training half. The categorical ones are each level against its reference.

<!-- figure: coefficients -->

??? abstract "Every coefficient, with its standard error and interval"
    <!-- table: coefficients -->

??? warning "Three readings to hold loosely"
    - *Financial conditions* (`financial_conditions`) is right-signed and stable across samples, and nearly nothing: in a
      stress scenario it contributes its sign and little else.
    - *Policy rate change since origination* (`policy_rate_change`) has no declared prior. A policy rate below where the loan was written
      shortens survival, which reads as the central bank cutting into recessions rather than
      as a payment channel a fixed-rate mortgage does not have.
    - Housing carries two covariates, the position, *loan-to-value change since origination*
      (`ltv_change`), and the construction cycle, *housing starts growth* (`housing_starts_growth`). The stability step separates a pair only when the smaller one
      changes sign, and neither did.

## The term structure

Cumulative PD month by month for the origination book dated to today -- the commonest
origination profiles, each weighted by the lending it stands for -- on the baseline path, by
segment.

<!-- figure: term_structure -->

Two portfolios can share a lifetime PD and differ entirely here, and the difference decides
when losses arrive.

## Scenarios

The adverse path is shaped like 2008 rather than scaled to it, and it moves every series a
model reads and no series one does not -- across **both** hazards, which is eleven of them. That
is not a tidiness point: when the default model lost inflation to a sign reversal, the scenario
went on shocking the consumer price index for a model that could not feel it, while four of the
eight covariates of the hazard competing with default stood still under stress. The baseline is a random walk from the last observation: **not a forecast**,
but what makes the relative effect of a scenario readable without a view on the economy.

<!-- table: adverse_legs -->

Because the covariates are time-varying, a scenario is applied by projecting the covariate
paths from today and chaining the conditional survival, not by rescoring frozen covariates.

<!-- figure: scenarios -->

??? abstract "Scenarios by segment, as a table"
    <!-- table: scenario_summary -->

!!! warning "What these PDs are not"
    **Prepayment is treated as independent censoring**, so a loan that prepays is assumed to
    have gone on defaulting at the model's hazard. Prepayment is a competing risk, and
    ignoring it overstates lifetime PD, most at long horizons. There is no LGD or EAD, so no
    expected loss.

## Where to read more

- [Calibration report](reports/calibration.md): generated by `creditsurv report`; marginal
  effects on twelve-month PD, and the distribution of lifetime PD under each scenario.
- [Calibration & backtest](calibration.md): whether the hazard behind these numbers matches
  what happened.
