# Rules written before the fits

Every threshold this model is judged by, fixed **before** the runs that produce the numbers
it is judged on. A rule chosen after the results is not a rule: it is a way of keeping
whatever came out. The validation's first finding against the previous model was exactly
that -- a backtest with no criterion can be read but not passed or failed -- and the answer
then was to adopt the criteria it proposed as they stood. The same discipline is extended
here to the things the next stage of the model decides: the windows, the family, what counts
as a material covariate, how the level is anchored, and how far the key may grow.

Written on 18 September 2026, against the model as PR #17 left it and before the cells were
rebuilt. Where a rule is a judgement rather than a measurement, the reasoning is given so
that disagreeing with it is possible.

**This file holds the rules as they now stand. [The register](rules_register.md) holds how they
got there** -- every amendment with its date and motive, every measurement a rule fired on, and
every retraction, chronologically. The two were one document until 17 October 2026, when the
second independent validation observed that it had become a lab notebook. Nothing was deleted;
a rule that fires still has to say what it did. What changed is that a reader can now find the
rule without reading its history.

**And an amendment that touches an acceptance criterion needs a second signatory.** The
amendments to date are dated and motivated, and they are also self-approved, which is the one
piece of this discipline that rests on nobody but the author.

## 1. The windows

| Window | Months | What it is for |
|---|---|---|
| Development | up to **2021-12** | Estimation: the selection and every coefficient |
| Anchoring | **2022-01 to 2024-12** | The level adjustment, and nothing else |
| Test | **2025-01** onwards | The out-of-time backtest, never seen before it is scored |
| Backtest cuts | end of **2018**, **2020**, **2022** | Each judged on the 24 months after it |

The previous model cut once, at 2024-12, and was judged on fifteen quiet months. Three cuts
put the same model in front of a tightening cycle (2018), a pandemic and a moratorium regime
(2020), and a rate shock (2022). Each window's model is estimated on everything up to its own
cut, so a window is never scored by a model that saw it.

**2020 and 2021 stay in estimation.** Dropping them would remove the one episode in the data
where a policy intervention broke the relationship between unemployment and default, and a
model that has never seen such a period cannot be said to handle one. They are reported
separately, so their effect on the fit is visible rather than hidden.

**And they carry a regime indicator, because keeping them unmarked attenuates the macro block.**
Amended 17 October 2026, before the selection it governs. Under `exclude` a moratorium month is
not an event and the loan stays at risk, which is right for the dependent variable -- but the
largest unemployment shock in the series, over eleven points in two months, arrives with defaults
suspended by law, and the coefficient records it. Measured across the declared cuts, the
coefficient on *unemployment change since origination* reads **−0.0920 at 2018-12** and **−0.0305
at 2021-12**: four points of unemployment multiply the hazard by **1.67** on the first and **1.18**
on the second, a **threefold** attenuation of the one sensitivity a mortgage stress test turns on.

So an indicator on **2020-03 to 2021-09** enters the model, and it enters the **protected block**
rather than the candidates. Its purpose is to *absorb* the regime so the macro coefficients are
read on the rest of the book -- not to test whether it belongs -- which is the same argument rule 8
makes for the HARP level. It is a function of the observation month alone, so it costs **zero
cells**, and the selection may not remove it.

## 2. The distribution family

Weibull and log-logistic are each taken through the whole selection, and the winner is the
one whose **selected model** sits closest to the non-parametric cumulative incidence of
default (Aalen-Johansen, which accounts for prepayment as a competing risk):

- the measure is the **mean absolute gap in percentage points**, over loan ages carrying at
  least 100,000 loan-months;
- if the two are within **0.1 percentage points** of each other, the **Weibull** is kept: its
  hazard does not fall at long ages, and the difference between the families is largest
  exactly where the data ends and the extrapolation begins;
- a family whose selected model turns a **declared sign** is excluded whatever its fit: a
  model that says tighter financial conditions lengthen survival is not a better model, it
  is a broken one.

**Applied, and reopened once, and the answer held.** The Weibull sits 0.1027 percentage points
from the observed incidence against the log-logistic's 0.1498, over 222 ages, inside the declared
tie and also the closer of the two. In October 2026 the rule was reopened against section 5's
decile criterion, which it had never considered and which the Weibull fails: the log-logistic went
through its own selection under rule 12's bands and read **6 of 10** deciles in band against the
Weibull's 5, so neither family reads that band alone and the family is not what it is failing on.
[The register](rules_register.md) carries the reopening and the grid of family by shape that
followed it.

**Two things are owed and declared here.** The 0.1027 against 0.1498 was measured on the
**linear** form, so the comparison is re-run under rule 12's band factors before any family claim
is republished. And the note rate's cost is re-priced on nine quarters against the *current*
published key before rule 9's own identity can be applied as written.

## 3. Materiality of a macro covariate

A macro covariate whose effect of one standard deviation on log survival time is **below 0.02
in absolute value** is removed, after the selection's own steps and before the model is
published. On the current model that would remove financial conditions, which is kept at
−0.004 and contributes its sign and little else.

The threshold is a judgement: 0.02 of log survival time is about a 2% change in expected time
to default per standard deviation of the covariate, which is the smallest effect this data
can distinguish from a difference in specification. Identification, not significance, is the
reason: at 60 million episodes every p-value is zero, and the validation's objection was that
covariates this small change sign when the sample does.

## 4. Anchoring the level

The level is anchored on the **anchoring window alone**, as a single multiplier on the
default hazard:

    k = actual defaults / expected defaults, over 2022-01 to 2024-12

applied to every loan-month, and nothing else changes: not the coefficients, not the shape of
the hazard, not the ranking. A multiplier is the weakest adjustment that can fix a level, and
keeping it to one number is what stops it from absorbing the model's errors by segment --
which would make the segment views meaningless.

The anchored model is then scored on the test window, which the anchoring never saw.

**A single multiplier cannot hold an error that trends, so the level is monitored.** Amended 17
October 2026. One number corrects a level; it cannot correct a slope. Measured on the declared
test window, the **anchored monthly** actual over expected falls from **1.386 in January 2025 to
0.771 in March 2026**, and the last month is already outside the band the overall figure passes
comfortably at 0.959. The overall figure passes partly because errors of opposite sign cancel:
by vintage, anchored, 2022 reads 1.49 and 2023 reads 1.31 while 2005-2007 read 0.46 to 0.54.

So, declared before the runs that follow:

- the **monthly** actual over expected is published beside the overall one, never instead of it;
- the model is **re-anchored** when a six-month rolling actual over expected sits outside **0.90
  to 1.10 for three consecutive months**;
- a re-anchoring is a new multiplier on a new window, recorded with its date, and never a
  coefficient change. A level that needs the coefficients moved is a finding, not a drift.

## 5. The criteria

Unchanged from the previous model, where they came from the independent validation:

| Criterion | Threshold |
|---|---|
| Actual over expected, overall | 0.80 to 1.25 |
| Actual over expected, every decile of predicted risk | 0.80 to 1.25 |
| Gini, exposure-weighted | above 0.45 |

Two are added, for the things the previous model was found not to test:

- **The cycle, in sample.** At least **70% of calendar years** with actual over expected
  between 0.80 and 1.25. The current model manages 0.47 to 1.60 across years, and a single
  out-of-time ratio near one says nothing beside that spread.
- **Twelve-month PD by grade.** A master scale of **eight grades**, on geometric thresholds
  of predicted twelve-month PD (each grade twice the previous one's floor); the scale passes
  when **seven of the eight** do.

**The Jeffreys interval is replaced by a materiality tolerance, and four criteria are added.**
Amended 17 October 2026, before the runs that follow, because the second validation found section
5 sound but incomplete.

- **A grade passes on a relative tolerance, not on Jeffreys.** With five million obligor-years in
  a grade the 95% interval is a few parts in a thousand wide, so it tests the arithmetic of the
  average and not the model -- which is what this project already says about the Kaplan-Meier band
  and the crossing test. A grade passes when its predicted twelve-month PD is within **25%**
  relative of what its cohort realised, the same width the actual-over-expected band already uses,
  and the magnitude of every miss is published whether it passes or not. **Ratified 19 October
  2026**, as an amendment to an acceptance criterion requires: what settles it is that the scale
  fails either way, so the tolerance cannot turn a failure into a pass. On the one grade table
  published at that date -- `docs/reports/windows.md`, which measures the *unshaped* fit -- the
  misses read +5.3%, +15.0%, +36.7%, +26.3%, +9.3%, −4.4%, −5.5%, +21.0%: **six of eight** inside
  the tolerance against **one of eight** inside Jeffreys, with grades 3 and 4 out on both. Six
  still fails the seven of eight this rule asks for. The Jeffreys bounds stay as published
  columns, because withdrawing a criterion is not a reason to stop reporting it, and the published
  model's own scale does not exist yet: it is measured when the shape reaches the campaign.

  **A grade with no realised defaults is judged on the interval instead**, declared 21 October
  2026 before the run that would have met it. A relative tolerance needs a realised rate to be
  relative to, and at zero defaults there is none: the ratio is infinite and the grade fails
  whatever it predicted. That is exactly the case the interval was chosen for in the first place
  -- "the top grades hold few defaults and sometimes none" -- and at zero defaults the interval
  has width where a ratio has none. Demonstrated: 0 defaults in 20,000 obligor-years against a
  predicted 4.8e-05 sits **inside** the interval, 2.5e-08 to 1.26e-04, and reads a relative miss
  of infinity. Each grade publishes which of the two judged it.

  An earlier draft of this bullet quoted +89%, +31%, +43%, +23%, +6%, −6%, −7%, +3% as "the
  published model". That profile matches no run on the record -- it predates 19 October and was
  never of the shaped fit. A criterion carrying an unsourced number is one nobody can check.
- **Calibration by age band, out of sample.** At least **five of seven** bands inside 0.80 to
  1.25 on the test window. For a lifetime PD the shape along age is the product itself, and a
  criterion the previous section did not contain was visible only to a reader who went looking.

  **The two figures this bullet carried are of two different models, and neither is the published
  one out of sample.** Corrected 19 October 2026. In sample the published fit `bc016657b2b40bfc`
  holds **6 of 7** bands, not 5: `docs/tables/ae_by_age_band.parquet` reads 0.727, 1.184, 1.248,
  1.074, 0.862, 0.937, 1.145, and only the first year misses. The **2 of 7** is
  `docs/validation.md:124`, the second validation's reading of the *scalar-shape* model, and no
  age-band table of the published model on the test window exists -- `backtest_views` emits months,
  deciles and the acceptance table and nothing by age. So this criterion was declared and could not
  be evaluated. It is measured by one further view, built from `age_bands` exactly as the in-sample
  one is, and the first reading of it is published whatever it says.
- **Cumulative PD by vintage.** The dispersion across origination years is published, because an
  aggregate that passes can hide vintages pulling in opposite directions.
- **A backtest on forecast macro, not realised.** The published backtest hands the model the macro
  path that actually happened. A user does not have it. One window is therefore scored on a macro
  path forecast from the information available at the cut, and the difference between the two is
  the part of the error that belongs to the covariates rather than to the model.

## 6. Prepayment, and its expected signs

Prepayment becomes a competing risk with its own cause-specific model, on the accelerated
failure time scale, where a **positive** coefficient lengthens the time to prepayment.

| Covariate | Expected | Reasoning |
|---|---|---|
| Refinancing incentive (note rate over the market rate) | **−** | A borrower who can refinance cheaper does so sooner |
| Credit score | **−** | Better credit can refinance, and refinances sooner |
| Loan-to-value change since origination | **+** | Leverage that has risen blocks a refinance |
| House price growth | **−** | Rising prices free equity and enable cash-out |
| Unemployment change since origination | **+** | A weaker labour market prepays less |

No prior is declared for the others; the reversal rule of the selection's step 8 applies to
them, as it does for default.

## 7. How far the key may grow

The cell table may reach **250 million cells**, raised from 150 on 10 October 2026 -- the memory
that bounded it is gone and the time has moved by an order of magnitude. [The
register](rules_register.md) carries the measurement and the motive, both written before the
rebuild. The 150 was about 2.4 times the 63.6 million the key produced when it was declared;
the extensions this rule admitted took the table to **91,575,827**. The engine reads it a batch at a time, so what the ceiling
protects is the aggregation itself and the time every later fit costs, not a fit's memory.

If the measured cost of the extensions exceeds it, they are given up in this order:

1. the finer bands (8 / 8 / 6 instead of 5 / 4 / 4);
2. the origination spread;
3. the lagged delinquency state.

HARP and the three-state outcome are not on the list: they are corrections of what the model
covers and of what it calls an event, not refinements of it.

**The order was broken once, and saying so is the correction.** Amended 17 October 2026. Raising
the ceiling admitted the **finer bands**, which are *first* on the list above -- that is, the
least priority -- before the origination spread, which is second. The amendment's text claimed
that admitting them first "reorders nothing"; that is the inverse of what an order of give-up
means, and the claim is withdrawn. What makes it recoverable rather than merely admitted is that
the finer bands were then **measured and given back**: they bought 0.015 on the worst in-sample
decile for 2.12x the table. So the order now stands unbroken in effect, and the next extension
priced against this ceiling is the **origination spread**, which at 184.4 million projected and
210.2 with the declared 1.14 margin fits under it and has never been tried -- while rules 6 and 9
both depend on the note rate it would restore.

## 8. What the HARP level obliges

HARP refinances enter the model with the key of September 2026, and Freddie Mac reports no
debt-to-income for them. The rule is fixed here, before the selection that would otherwise
decide it:

- the ratio stays **missing in the cell table**. Nothing is imputed there, and a reader
  counting HARP loans sees the gap the source has;
- the model fills it with a **constant** and carries the **HARP level**, which absorbs the
  constant whole. That is the dummy-variable adjustment, and it is exactly a "not reported"
  band written on the scale the covariate already uses: the slope is estimated on the loans
  that report the ratio, and the fill's value is arbitrary;
- therefore **no specification may contain the debt-to-income without the HARP level**. The
  selection cannot drop it: it is in the protected block, not among the candidates, and the
  code refuses the fit rather than trusting the rule to be remembered.

A HARP loan is about three times as likely to default as the loans kept, so the level is
also the one covariate here whose coefficient is predicted before it is estimated: a
**negative** coefficient on log survival time, and a positive one would say the program's
underwater borrowers were the safer ones.

## 9. Two of the three rate covariates, never all three

The note rate enters the key, and with it three ways of comparing it to the market:

    refinance incentive = origination spread + fall in the market rate since origination

an identity, not a correlation. A design holding all three is singular by construction, and
the correlation and variance-inflation passes would not catch it, since two of the three are
already in the model before the third arrives. Declared now:

- the **default** model carries the **origination spread** and the **fall in the market
  rate**: how the loan was priced at underwriting, and what the market has done since;
- the **prepayment** model carries the **refinancing incentive**, which is the quantity the
  decision to refinance actually turns on, and not the other two.

**This rule is currently moot, and that is a cost rather than a resolution.** The note rate is
not in the cell key -- rule 7's give-up order reached it -- so none of the three is available, and
both models carry the fall in the market rate instead. The prepayment model therefore has no
measure of the quantity a refinancing decision actually turns on, which is one reason it does not
hold out of sample. Rule 2's amendment calls for the spread to be re-priced against the current
key before this rule can be applied as written.

## 10. What the payment state may be used for

Added on 25 September 2026, after the first selection tried to fit it and before the runs
that follow. The state a month ago went into the cell key as a candidate covariate; it is
**not** one, and the reason is arithmetic.

Default is three missed payments, so a loan that opens the month two payments behind is one
month from the definition:

| State a month ago | Loan-months | Defaults | Monthly rate |
|---|---|---|---|
| Current | 2,740,161,293 | 22,276 | **0.0008%** |
| One month | 23,871,219 | 20,206 | 0.085% |
| Three or more | 4,269,824 | 109,964 | 2.58% |
| Two months | 5,337,846 | 1,518,761 | **28.45%** |

Two months behind is 0.2% of the exposure and **91% of every default in the book**, at a rate
35,000 times the current state's. The likelihood pushes its coefficient as far as the clipping
allows -- the first attempt reached 1e+80 and an objective of −4.7e275 -- and the model that
came out would answer *will this loan default next month*, which is a behavioural score.

There is a second objection, and it does not depend on the first. A lifetime PD has to
**project its covariates** over the remaining life. A macro series can be projected under a
scenario and an origination characteristic does not move; a payment state can be projected
only by the model that is trying to predict it. It is the outcome, one month early.

So: the state stays in the key, where it costs 1.19× and earns its place in the **views** --
it says where the defaults are, which is worth publishing -- and the **origination** model does
not read it.

**The rule was too broad, and it is split in two.** Amended 17 October 2026. Both objections
above hold against an origination model and the first is measured, so nothing there is withdrawn.
The second objection -- that the state cannot be projected -- **does not carry** against a
different use: conditioning on the state **at the valuation date** requires no projection of it,
and that is the current practice for the lifetime PD of stage-2 credit. A loan in order and a loan
two payments behind receive the same PD from this model, against observed monthly rates of 0.0008%
and 28.45%.

Therefore, declared:

- no model of the remaining life may read the payment state **as a covariate of the origination
  model**, which is what this rule has always meant and what its arithmetic supports;
- a **module conditioned on the state at the valuation date** -- a transition matrix, or a hazard
  by initial state with an effect that decays -- is permitted, is a different model, and is **out
  of current scope**;
- and the scope is therefore declared rather than implied: **this model covers performing pools.
  It is not an expected-loss model for past-due credit**, and no report may present it as one.

## 11. A removal that leaves a model nobody can fit

Added on 27 September 2026, after the prepayment selection failed seven times on the same
removal and before the run that follows. Backward elimination removes the worst covariate by the
rules of step 8; nothing said what to do when the model *after* the removal cannot be fitted.

It happens, and for a reason worth stating. lifelines' interval-censored likelihood clips the
survival difference at 1e-25 but adds the left-truncation term **unclipped**, so the conditional
probability it computes can exceed one and the objective can fall below anything a likelihood can
take. The trade-off needs no extreme parameter: on the prepayment model the truncation term has
only to reach **0.0176 on the mean**, the same order as the hazard itself, to make a spurious
minimum seventy times below the real one. Every optimiser found it, from every start, with the
coefficients bounded and the shape bounded.

So: **the candidate model is fitted before it is adopted.** If it cannot be fitted, the removal
is refused, the covariate stays, the reason is recorded in the elimination table, and the
next-worst covariate is offered instead. The run continues and the record says what happened.

This is a rule about the *tool*, not about the model, and it is stated here because it changes
what a published specification means: a covariate may be in the model because removing it left
something unfittable rather than because it earned its place. Any such covariate is named in
`selection_*.md`, and one that stays for this reason is a candidate for a family whose
likelihood does not have the flaw.

## 12. A banded covariate is read as bands, not as a line through them

**Written on 2026-09-30, after the first backtest of this branch and before the fit that answers
it.** The backtest is a finding; what follows is a change to an input, declared here first, and
judged by the criteria section 5 already fixed rather than by any new one.

**What the backtest found.** In sample -- on the very rows the model was fitted to -- the actual
over expected rises monotonically across deciles of predicted risk: 0.594, 0.577, 0.604, 0.670,
0.761, 0.859, 0.977, 1.098, 1.204, 1.000. The model predicts a ratio of **74.4x** between the
riskiest tenth and the safest; the book realises **125.3x**. The ranking is right -- the Gini is
0.54 out of time -- and the **spacing is compressed by about 40%**. One multiplier cannot mend
that: it shifts every decile by the same factor and the error is a slope.

**Why the form is the cause.** `credit_score`, `original_ltv` and `debt_to_income` are already
bands in the cell key -- five, four and four of them -- and they entered the formula as **linear
terms on the band's midpoint**. The model therefore asserts that log survival time is a straight
line from the 620 band to the 790 band. It is not: forty points of score lost at 620 are worth
more than forty lost at 760, and with 45.7% of the exposure in the top band the line follows the
middle and flattens both tails.

**The rule.** Each banded loan covariate enters as a **factor, one coefficient a band**, with the
band carrying the most loan-months as the reference. Measured on the cell table before this rule
was written, and therefore not chosen to suit any result:

| Covariate | Bands | Reference band | Its share of exposure |
|---|---|---|---|
| `credit_score` | 5 | **790** | 45.67% |
| `original_ltv` | 4 | **50** | 43.00% |
| `debt_to_income` | 4 | **19** | 35.70% |

`term_years` is left alone: it has two levels, where a factor and a line are the same model.

Seven parameters, no new covariate, and **no cells** -- the bands are the key already, which is
what makes this cheap enough to do at all. No monotonicity is imposed, because imposing the shape
one expects is how a form comes to flatter a model rather than describe it.

**What this rule does not claim.** The covariate *set* was chosen by steps 5 to 10 under the
linear form, so a set re-decided under bands could differ -- a covariate that survived on a
straight line might not on a bend, or the reverse. This rule changes how three covariates are
read, not who is in the model, and the re-selection under the new form is owed. Until it is run,
the published specification is *the set chosen under the linear form, read as bands*, and the
report says so.

**How it is judged.** By section 5, unchanged: actual over expected between 0.80 and 1.25 overall
and in every decile, Gini above 0.45, and the share of calendar years in band. The test window is
not read until the windows report is regenerated, and the comparison published is against the
numbers above, which are already recorded.

## 13. What a compiled kernel may be, and what it may not

Declared before any of it is written, because a second language in a credit model is a thing to
be bounded in advance rather than contained afterwards.

**The gate, first.** A compiled implementation of the likelihood is kept only if it computes a
value, a gradient and a Hessian over the whole training half -- 72,671,500 rows at 26
parameters -- in at most **9.90 s** against the **29.69** the NumPy kernel is measured at, with
resident memory no higher than **1.31 GB**. Three times, on the same rows and this machine. If
it does not pass, it is abandoned and the number is published in `docs/reports/engine.md`.
Measured against the NumPy, never against autograd: measuring it against autograd would credit
a compiled language with removing a tape that NumPy already removed.

**And the segregation, which is not negotiable with the gate.**

- **One** crate, `crates/creditsurv-kernel/`, built as a separate workspace member and installed
  through an optional extra. Nobody needs a Rust toolchain to run this project.
- At most **three** `#[pyfunction]`. The boundary is a function call, not an object graph.
- Across it pass **only numpy arrays of fixed dtype** -- `f64` for parameters and tables, `u32`
  for the codes, `u16` for the age, `u8` for the flags -- and back a scalar, an `f64[p]` and an
  `f64[p, p]`. No Python objects, no pandas, no lifelines.
- **Inside** it: no I/O, no logging, no configuration and **no rule of this document**. The loop
  over rows and nothing else.
- `models/kernel/terms.py` stays **normative**. It is what the equivalence tests compare
  against, and the extension is optional at import: a missing one is the NumPy path, not an
  error.
- The summation is **deterministic**: a fixed chunk order, no unordered reduction, no FMA
  reassociation. Two runs agree bit for bit, as the Python path already does.
- CI runs the suite **twice on both Python legs**, with and without the extension, so neither
  path can rot.

What the gate buys, if it passes: a warm candidate's 5.67 minutes become about two and a
selection's 3.5 hours about 1.6. What it cannot buy is a different answer -- the estimator is
lifelines' own likelihood on the same rows, and the damped Newton polish certifies every fit to
under a thousandth of a standard error whichever backend evaluated it.

**The gate was not passed** -- 2.80x on a Hessian against a declared 3x -- **and the kernel is
kept under an exception declared at the time**, on the condition that the logic be airtight rather
than the number be three. [The register](rules_register.md) carries the exception, what was done
to meet its condition, and the threads gate that followed it.

## 14. The shape reads the risk, and which covariates it may read

Declared 10 October 2026 before the fits it governs, amended 17 October after the grid it called
for was complete. [The register](rules_register.md) carries the four cells, the measurements and
the two retractions. This is what stands.

**Why the shape and not the bands.** The backtest's decile criterion failed, and in sample too:
0.63, 0.60, 0.63, 0.70, 0.79, 0.88, 0.98, 1.09, 1.19, 1.00 -- five deciles of ten outside 0.80 to
1.25, the predicted spread 80x between the extremes where the book realises 130x, the ranking
sound at a Gini of 0.561. In a Weibull AFT the hazard ratio between two loans is
`exp(rho * delta)`, **constant in age**, because the Weibull is the only family that is both
accelerated-failure-time and proportional-hazards -- so one shape parameter sets the entire risk
spread, and `rho` is not free to serve it because it also fits the age profile. The finer bands
were built to test resolution instead and bought **0.015** on the worst decile.

**The line, which is an identity and not a judgement.** The shape may not depend on the loan
**age**: the hazard would no longer integrate to a coherent survival function, and a lifetime PD
is that integral. And `period = cohort + age` holds identically, so **a covariate whose value
moves over a loan's life moves with the age** -- admitting one to the shape is admitting age
dependence under another name. The engine's own partition is the same partition.

| | Covariates | In the shape |
|---|---|---|
| Fixed at origination | `credit_score`, `original_ltv`, `debt_to_income`, `term_years`, `purpose`, `occupancy`, `harp`, `mortgage_insurance`, `buyer_type` | **admissible** |
| Recomputed every loan-month | `ltv_change`, `unemployment_change`, `policy_rate_change`, `yield_curve_slope`, `consumer_sentiment`, `housing_starts_growth` | **refused** |

The six are refused on the identity, not on materiality: `ltv_change` is the strongest covariate
in the model, which is why the reason has to be structural rather than a preference.

**The set is the whole of the admissible side, because any subset is a pick.** All nine, in the
forms rule 12 declares. The first attempt was a subset -- it named `credit_score` on an unmeasured
belief that it is the primary risk dimension, where `credit_score` sits in the protected block and
has never passed the univariate screen -- and the declared procedure then reached the same
covariate **with the measurement that it is the right one** (84% of the shape gain on 22% of the
parameters), plus two the pick had not considered.

**And it is part of the specification, not a candidate.** Amended 17 October 2026, before the
selection that follows. The grid of family by shape was run to completion and published: the shape
takes the in-sample deciles from **5 of 10 in band to 9 of 10**, flips section 5's cycle criterion
from 69.6% of calendar years in band to **73.9%**, leaves the age profile unchanged at 5 of 7 --
and on the declared **test window** the log-logistic with it reads anchored A/E **0.927**, Gini
**0.548** and **10 of 10 deciles in band**, the first model in this project to pass all three of
section 5's criteria out of time.

So the shape-on-risk stops being a candidate model and becomes a property of the specification,
exactly as rule 12 made bands a property rather than a model: **the selection runs with the shape
reading the origination side**, and what it then chooses is the specification. One selection, one
adoption. The alternative -- publishing the candidate now and re-deriving it after the macro block
is corrected -- pays the adoption twice for a model that would change.

**What is still owed before anything is called selected**: rule 2's Aalen-Johansen comparison under
the bands, the three regime cuts, and the scenario.

**The criterion, unchanged.** Every in-sample decile inside 0.80 to 1.25; the likelihood ratio
reported and never decisive, for the reason rule 2 gives; the age calibration published beside the
deciles, now against section 5's own out-of-sample criterion; and the test window read once, at the
end, against criteria already recorded.

**One exception stands, on the record and not by waiver.** In sample the shape leaves decile 1 at
**1.676** against a ceiling of 1.25, and **83% of that miss is the 2019 to 2021 originations** at an
actual over expected of 3.413, where every vintage to 2018 reads 1.082. The cause is structural: a
**multiplicative hazard has no floor**, and those loans stack every favourable reading the model
has at once, so the prediction reaches 1.5 defaults per 100,000 loan-months a month where the book
does not go below about 5. The exception is bounded at **0.11%** of the in-sample defaults, and its
remedy is named rather than searched for -- an additive baseline, which is another family.

**And the decile reads three different numbers on three different windows, which is why it has been
reported wrongly twice.** For the published fit `bc016657b2b40bfc`:

| window | anchored | decile 1 |
|---|---|---|
| development, in sample | n/a | **1.676** |
| 2022-01 onwards (`backtest_by_decile`, the committed view) | no | **1.456** |
| **the declared test window, 2025-01 onwards** | **yes** | **1.150** |

Only the third row is this rule's criterion, and on it the exception **does not recur**: 10 of 10
deciles inside 0.80 to 1.25 at 0.803 to 1.150, with 8 of 8 grades inside section 5's tolerance. The
second row is not a criterion of anything -- it spans the anchoring window and the test window
together, on an unanchored hazard -- and it is what the site publishes under the name "acceptance
criteria by segment". That is a separate defect and it is recorded in the register.

The pair this paragraph carried until 19 October -- 1.415 in sample and 1.056 out -- is of the
**log-logistic candidate**, the cell of the grid described above and *not* the model rule 2 keeps.
The exception was accepted on a bound of 1.415 that the configured model does not honour in sample.

**Where it does recur is the 2020 cut, eightfold, and that is the finding this correction uncovers.**
Re-estimated with the shape, over 2021-22 the safest decile records **1,335 defaults against 164
expected -- A/E 8.12**, where without the shape it was 2.88. It is 93% vintages 2019-21, 76% in the
first year of life and 90% in the top score band; there the model predicts **0.49 defaults per
100,000 loan-months a month and 5.10 are realised**, and inside the moratorium months the regime
indicator lowers the prediction further, to eleven times under. In aggregate 1.4% of that window's
defaults -- but it is the PD of the best new production, at the moment most of it is written, and it
sits below any regulatory floor. **A shape near 2 gives the best loans a hazard starting at zero**,
which is the mechanism, so the remedy is a floor and not a coefficient: declared as an overlay,
calibrated on the realised rate of the safest segment (about 3 defaults per 100,000 loan-months a
month), with an additive baseline component or section 16's age baseline as the standing fix, tested
on the 2020 cut.

**And the model is a candidate until rule 2 is applied.** This rule says nothing is "called
selected" before rule 2's comparison under the bands, which is blocked on the prepayment model --
yet the specification is in the configuration and in every view on the site with no label saying so.
The label is owed.

## 15. An expected sign for every candidate, or it is excluded

Declared 17 October 2026, before the selection it governs.

**What went wrong without it.** Four of the six macro covariates in the published model carry no
declared sign: `policy_rate_change`, `yield_curve_slope`, `consumer_sentiment` and
`housing_starts_growth`. The selection's strongest filter is the sign against the prior, so for
those four it never fired. `policy_rate_change` came out at **+0.0708** on log survival time -- a
policy rate that has risen since origination lengthens survival -- read as the central bank cutting
into recessions, which is the sign of an era when rates rose only in expansions. Between 2022 and
2024 the rate rose more than five points for inflation, and the term lowered the predicted hazard
by **33% in 2023 and 36% in 2024**. Removing its contribution alone takes 2023 from 1.184 to 0.807,
2024 from 1.547 to 1.017, and the 2022 cut from **1.420 to 0.959** -- from failing to passing. It
is about **80%** of the level drift, and it carries 19% of the adverse scenario's stress, where in
a stagflation it would *reduce* the PD.

**The rule.** Every continuous candidate carries a declared expected sign, or it is not a
candidate. The existing policy -- that a sign should be declared "only where theory actually
commits to a direction", because a covariate listed with no clear prior would be eliminated for
disagreeing with a guess -- is **replaced** rather than argued with: a covariate for which theory
commits to no direction is one whose coefficient cannot be read, and a model of the economy made
of such terms responds to the economy by accident. The honest form of "no prior" is **not a
candidate**.

**The fifteen candidates, decided here and not during the selection.** Six already carried a
sign. Four gain one, because theory does commit:

| Candidate | Sign | Why theory commits |
|---|---|---|
| `mortgage_rate_decline` | **+** | a market rate below the note rate means refinancing into a lower payment is available, which is relief. Rule 6 gives the opposite prior for *prepayment*, where the same availability shortens the time to that exit |
| `corporate_bond_spread` | **−** | a wider spread is the price of credit risk rising: tighter credit, weaker economy |
| `consumer_sentiment` | **+** | households' own assessment of their capacity and willingness to pay |
| `housing_starts_growth` | **+** | housebuilding is housing demand and the liquidity of the collateral |

And five are **withdrawn as candidates**, because theory commits to no direction for them:

| Withdrawn | Why no direction | Status |
|---|---|---|
| `policy_rate_change` | on a fixed-rate book the borrower's payment does not move with the policy rate, so the channels are the market's -- lock-in and real-debt relief against affordability and cost of living -- and which dominates depends on whether rates rose with growth or with inflation, which this model does not observe | **in the published model** |
| `yield_curve_slope` | a slope is a forward-looking cycle proxy whose whole relation to default runs through a cycle this model already carries in unemployment and house prices | **in the published model**, and its sign flips across the cuts: +0.083 at 2018, then −0.032, −0.039, −0.048 |
| `inflation_rate` | inflation erodes the real value of a fixed nominal debt, which is relief, and erodes real income, which is stress | already eliminated for turning its sign |
| `inflation_change` | as the level | already eliminated |
| `equity_return` | a wealth channel for a minority of these borrowers, a cycle proxy otherwise | already eliminated for turning its sign |

**Two of the five are in the published model, and withdrawing the one that caused the failure is
the move this rule has to defend.** The defence is that the rule is general and was applied to all
fifteen uniformly: four gained a sign, five lost candidacy, and three of those five the selection
had already eliminated on its own. It is not that `policy_rate_change` performed badly --
performing badly is what the sign rule is *for*. It is that its coefficient cannot be read, because
the direction it should take depends on something the model does not see.

**When it takes effect.** At the next selection, not before. `TIME_VARYING_CONTINUOUS` is the
selection's output and `MACRO_CANDIDATES` its input, and a test holds the first inside the second;
withdrawing a candidate that the published specification contains would leave the configuration
inconsistent. So the withdrawal and the selection that re-derives the specification land together,
and until then the published model carries two covariates this rule says are not candidates --
which is the true state and is recorded rather than hidden.

Where a rate channel is wanted, it is a measure whose sign theory can give -- affordability, a
payment-to-income ratio, a payment shock -- and not a policy rate whose relation to default
reverses with the monetary regime.

## 16. The baseline along age may be flexible, and what makes it admissible

Declared 17 October 2026, before the fit.

**Why.** For a lifetime PD the shape along age *is* the product. The **scalar-shape** model held
**2 of 7** age bands inside 0.80 to 1.25 out of sample, and this rule was declared on that figure
-- a defect measured on a predecessor -- with the table for the published model named as the first
thing the implementing branch owed.

**That table exists as of 21 October 2026, and it holds 2 of 7 as well.** Measured on the declared
test window, anchored, by the view section 5 had declared and nothing could compute:

| band | A/E | | band | A/E |
|---|---|---|---|---|
| 0 to 11 | 0.814 | | 36 to 59 | 0.911 |
| 12 to 23 | **1.480** | | 60 to 119 | **0.621** |
| 24 to 35 | **1.347** | | 120 to 239 | **0.647** |
| | | | 240 to 399 | **0.725** |

**And it answers the question this rule asked before the fit, in the direction that justifies the
remedy rather than narrowing it.** The first year is *not* the only band out, and the misses do not
share a sign: the model **under-predicts the 12-to-35-month hump** by 48% and 35% and
**over-predicts the seasoned book** by a third to a half. A defect in one direction could be met by
something narrower than a free baseline -- a floor, a single factor. One that reverses cannot: no
monotone reparameterisation of a single shape fixes a curve that is too low at two years and too
high at eight. So the flexible baseline in log-time is admitted on its measured premise and not on
its predecessor's.

Two things this does not license. It is still **one shape per loan**, not a hazard that bends with
calendar time, because `period = cohort + age` holds identically. And the in-sample profile holds
**6 of 7** with only the first year out at 0.727, so the gain is out of sample and any in-sample
improvement is not the evidence. One shape is a compromise across vintages
-- 1999-2008 read 1.33 to 1.34 at 12 to 35 months where 2009-2018 read 0.90 to 0.95, and the 2022
vintage out of sample reads 1.75 -- and in the aggregate those cancel, while in the cumulative PD
of a new loan they move the losses through time. For an expected loss, *when* matters as much as
*how much*.

**Why it is admissible, which rule 14 might appear to forbid.** Rule 14 refuses to let the
**shape** depend on the age, because `exp(rho(t) * (log t - log lambda))` is not the integral of
the model's own hazard. A basis in log(age) in the **scale** block is a different object: for the
Weibull it gives `H = exp(g(log t) - rho * (A + B_macro))`, which is a single monotone-if-checked
function of time per loan, so the survival function stays coherent and the lifetime quantity stays
defined. And it costs **zero cells**, because `period = cohort + age` already puts age in the
calendar key -- the same identity rule 14 invokes to keep time-varying covariates out of the shape
puts a function of age alone safely into the scale.

**Two conditions, declared before the fit rather than discovered after it.**

- **One basis column is dropped**, because the basis is collinear with `rho * log t` in the
  log-time direction. Which one is a specification choice, recorded with the fit.
- **`g(log t)` is checked for monotonicity before publication.** Nothing imposes it today, and a
  cumulative hazard that falls is not one. A fit whose baseline is not monotone over the ages
  carrying exposure is refused, not published -- and the ages above the exposure floor are where
  the check applies, because beyond 180 months there are 489,000 loan-months at risk and then
  23,000. **This condition is withdrawn**: the amendment of 30 October 2026 below shows that the
  model's monthly hazard is non-negative for every `eta` profile, because the increment is taken at
  fixed covariates, so there is nothing for it to refuse.

**The form, fixed here.** **Piecewise first**, by rule 12's own mechanism: age bands as factors
with the band carrying the most loan-months as the reference. It reuses a mechanism this project
already has and whose reference discipline is held by a test. A restricted cubic basis is the
fallback, and only if bands do not reach section 5's out-of-sample age criterion -- not because it
fits better, since that is the argument rule 2 exists to refuse.

**Still in force on 30 October 2026**, and the amendment below is why it had to be restated: the
piecewise form was fitted, read as refused on this rule's monotonicity condition, and that reading
was withdrawn. The ordering this paragraph declared therefore stands unchanged -- bands first, the
restricted basis only if bands do not reach section 5's out-of-sample age criterion -- and that
criterion is what the band fit now owes.

### Four things settled before the fit, 24 October 2026

**The band column has to be built, because the one that exists is a reporting grouper.**
`views.segments.AGE_BANDS` cuts the age at 0, 12, 24, 36, 60, 120, 240, 400 and labels the result
with **strings** -- "0 to 11" -- for the views. Rule 12's own text says a banded level is a number
and that `Treatment('790.0')` is not a level of a float column, so the reporting grouper cannot be
the model's column. The model reads a **numeric** column carrying each band's lower edge, built
beside `moratorium_regime` in `features.add_macro_family`: that covariate is the precedent in every
respect, because it reads no macro series at all and is a function of the observation month alone,
where this one is a function of the age alone.

**The reference band is `0 to 11`, and the margin is thin enough to say so now.** On the estimation
table it carries **473,904,483** loan-months against `60 to 119`'s **469,648,978** -- a 0.9%
margin, effectively a tie, and the two would swap on a re-measurement. Rule 12's precedent is to
measure on the table the model is estimated on, which settles it; declaring it before the fit is
what stops a tie from being resolved by whichever answer turns out convenient. Measured on the
*test* window the answer would be `36 to 59`, which is exactly why the rule has to name the table.

**The in-sample age profile stops being evidence, and that is a consequence rather than a
decision.** A free coefficient for each band of the grouping section 5's criterion is scored on
makes the in-sample profile approximately 1.000 in every band by construction -- a saturated factor
reproduces the means of its own groups. So the published in-sample 6-of-7 will become 7-of-7 and
will say nothing. **Only the out-of-sample profile counts**, and the criterion was already written
that way. Anyone reading an in-sample improvement here as evidence is reading the arithmetic of a
saturated factor.

**The monotonicity condition is on the cumulative hazard, and it is not a claim about the hazard's
shape.** `H` must be non-decreasing because it is the integral of a non-negative hazard, which is
the whole of why this rule imposes it; `h = dH/dt` may rise, plateau and fall, and on this book it
does -- the observed hazard peaks near age 48 and reads 0.00093, 0.00090, 0.00088, 0.00085 at 48,
60, 84 and 120. A rule read as "the hazard must be monotone" would refuse the shape the data shows.
For the banded form the condition bites only at the band boundaries, where `g` steps: the step must
not take `H` down. It is checked numerically on the fitted model over the ages above the exposure
floor, using the same `hazard_by_age` that already publishes the at-risk counts the floor is
defined by.

### The band form was fitted, the monotonicity condition was read as violated, and that reading was wrong -- 30 October 2026

**What happened.** The selection under the banded baseline ran 14,007 s and chose fifteen terms
including `C(age_band, Treatment(0.0))`. Its band coefficients on the training half (fit
`a011cd438634371d`, log-likelihood -10,627,259.1489) are `0, -0.128311, -0.086231, +0.063848,
+0.278733, +0.318061, +0.288168`, so `eta` falls to twelve months and then rises at every boundary
through a hundred and twenty. That was read as a violation of this rule's monotonicity condition
and the fit was refused. **The refusal is withdrawn**, and the reason it was wrong is the more
useful half of this entry.

**The model's hazard is non-negative for every `eta` profile, by construction.** `episode_hazards`
(`models/aft.py:492-493`) evaluates the cumulative hazard at the times `a` and `a + 1` **with the
covariates the row itself carries**, so the increment is

    H(a+1 | x_a) - H(a | x_a) = exp(-rho * eta_a) * ((a+1)**rho - a**rho)

and both factors are positive: `exp` of a real number, and `(a+1)^rho - a^rho` for any `rho > 0`.
There is no `eta(a)` -- banded, spline or arbitrary -- that makes it negative. Verified through the
refused model itself over ages 0 to 359: the monthly hazard runs 7.3e-07 to 2.1e-03, nothing at or
below zero, and the implied cumulative hazard is **monotone increasing at every age**.

**What was computed instead** was `H(a | eta_b)` against `H(a | eta_{b+1})` -- the cumulative
hazards of two *different covariate vectors* at the same age, which is to say of two different
loans -- and the ratio between them was read as a jump in one loan's own curve. It is not one. The
model never places those two numbers side by side.

**So the condition as this rule wrote it is withdrawn.** It was declared on the object
`H = exp(g(log t) - rho * (A + B_macro))` read as "a single monotone-if-checked function of time
per loan", and that is not the object the implementation builds. This model is an AFT with
**time-varying covariates**, integrated month by month at fixed covariates within each month --
and `moratorium_regime`, `ltv_change` and `unemployment_change` have all made `eta` step between
one month and the next since the day they entered it. The age is no different from them, and a
condition that refuses a baseline along age would have to refuse those three as well.

**What is true, and what it is an argument for.** The band form makes the **hazard** discontinuous
in age: through the fitted model it reads 23.8 per 100,000 loan-months at age 35 and 17.4 at 36,
33.3 at 59 and 20.6 at 60. The hazard is positive and the cumulative hazard rises; what it is not
is continuous, so a term structure built from it has kinks at six ages chosen by a reporting
grouper. That is a reason to prefer a smooth basis, and it is a reason about **smoothness**, not
about coherence -- so it decides a tie rather than refusing a form.

**The form is settled by section 5's out-of-sample age criterion, exactly as this rule said
before the amendment, and that criterion has not been measured on the band fit.** Until it is,
publishing the band form and refusing it are both unjustified, so the configuration stays at the
published specification -- not because the band fit is refused, but because nothing has yet scored
it against the thing that admits it.

**What the data asks for, which is unaffected and was measurable all along.** `d log H / d log t`,
the quantity a Weibull holds *constant* at `rho`, off the Nelson-Aalen curve of the estimation
table: **1.67** at six months, **1.81** at twelve, 1.76 at twenty-four, 1.52 at thirty-six, 1.25
at sixty, 1.15 at eighty-four, **1.10** at a hundred and twenty. It halves. Fitting the straight
line costs **2.6x** the exposure-weighted mean square error of a single bend, 0.283 against 0.108.
One parameter is being asked to be two numbers, and that is why this rule exists.

**A form is probed before it is fitted, from 30 October 2026.** Any declared baseline form is
first fitted by exposure-weighted least squares to the book's own Nelson-Aalen `log H(log age)`
over the ages above the exposure floor, and the age profile it implies is registered **before**
the heavy fit. It costs milliseconds and reads only a committed table of observed counts. Declared
not to be evidence *for* a form -- it is in sample and marginal, where section 5's criterion is
out of sample -- only against one. **This is the instrument that would have cost least here**: not
because it would have caught a coherence failure, there being none, but because the implied
profile of a declared form is the cheapest thing in this project to look at and it was never
looked at.

## What these rules forbid

- Tuning any threshold on a test window, or choosing a cut after seeing a result.
- Anchoring on anything but the anchoring window.
- Choosing the family on the likelihood alone, or keeping one that turns a declared sign.
- Adding a covariate to the key without measuring what it costs first.
- Fitting the debt-to-income without the HARP level, or imputing the ratio it stands for.
- Putting the refinancing incentive, the origination spread and the fall in the market rate
  in one model.
- Letting the payment state into the **origination** model, or presenting this model as an
  expected-loss model for past-due credit.
- Removing a covariate without fitting the model that remains.
- Reading a banded covariate as a line through its midpoints.
- Letting a compiled kernel read a declared constant, touch a file, or be the only
  implementation of the likelihood.
- Letting the hazard's shape depend on the loan **age**, which would leave a hazard that does not
  integrate to a survival function, when the quantity being estimated is that integral.
- Letting a covariate that is **recomputed every loan-month** into the shape, which is that
  same age dependence under another name, because `period = cohort + age` holds identically.
- Searching for what to put in the shape by what improves the decile the model is judged on,
  or taking a subset of the admissible side when the whole of it is what can be declared.
- Carrying a continuous candidate with no declared expected sign.
- Fitting a flexible age baseline without registering the profile the probe implies for it first.
- Reading the cumulative hazard of **two covariate vectors at one age** as one loan's curve over
  time. The model takes its increment at fixed covariates, so the two numbers never meet; reading
  them as a jump is what refused a coherent fit on 29 October 2026.
- Correcting a level that trends with a single multiplier, and reporting the overall ratio
  without the monthly one beside it.
- Reading a grade's twelve-month PD against a Jeffreys interval on millions of obligor-years,
  which measures the arithmetic of the average rather than the model.
- Amending a rule that touches an acceptance criterion without a second signatory, or amending
  this file without the register recording what changed and when.
