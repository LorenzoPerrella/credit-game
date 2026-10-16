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

### The family is reopened, against a criterion this rule never considered

Amended 10 October 2026, before the log-logistic selection it calls for. **The motive first,
because stating it is what makes this legitimate rather than convenient.**

**The motive.** This rule chose the Weibull on the distance from the Aalen-Johansen incidence --
0.1027 percentage points against 0.1498, inside the 0.1 pp tie it had declared -- and kept the
Weibull for not falling at long ages. None of that measurement is withdrawn. What it never asked
is whether the selected model **calibrates by decile**, which is section 5's criterion, declared
before any of this, and which the Weibull fails in sample in five deciles of ten: 0.63, 0.60, 0.63,
0.70, 0.79, 0.88, 0.98, 1.09, 1.19, 1.00.

**And the argument that made the family look irrelevant to that failure was wrong.** It was put
as: the hazard ratio between two loans in an AFT is `exp(rho * delta)`, constant in age, so no
change of family can widen a compressed spread. **That holds only for the Weibull**, which is the
only family that is both accelerated-failure-time and proportional-hazards. For the log-logistic,
`h = (rho / t) * u / (1 + u)` with `u = (t / lambda)^rho`: the ratio is `exp(rho * delta)` while
`u` is small and tends to **1** as `u` grows. So the family does bear on the spread, and in two
directions at once -- its own curve bends, which may leave a larger `rho` than the Weibull's
compromise between a profile wanting 1.47 up to age 48 and 0.90 after, while its ratios compress
at long ages. Which dominates at the ages this book is actually at is empirical, and the argument
for not asking was a mistake.

**What makes asking it affordable, and therefore owed.** A selection run is **20.7 minutes** where
the four recorded runs averaged 10.5 hours. The whole question costs about an hour. A rule that
rests on a tie-break because the measurement was expensive is a rule to re-measure when it stops
being.

**What is fixed before the fits.**

- The log-logistic goes through **its own selection** under rule 12's band factors, as this rule
  has always required -- not the Weibull's specification refitted.
- Both families are judged by **section 5's decile band**, in sample: 0.80 to 1.25 in every
  decile.
- If both enter the band, the tie is broken by **this rule exactly as written above**: the mean
  absolute gap from the Aalen-Johansen incidence over the ages above the exposure floor, the 0.1
  pp tie, and the long-age preference for the Weibull. The rule is not replaced; it is handed the
  pair it was written to compare.
- If one enters, that one. If neither, the family is not the answer and the finding is published.
- The family and the shape are answered **separately**, because a model that changed both at
  once would leave neither measured -- and the **shape goes first**, because this reopening is
  *conditional* on the shape not answering the question. What motivates it is the Weibull failing
  section 5's decile band; if the Weibull reads that band with the shape rule 14 declares, the
  motive is gone and there is nothing here to reopen. Written before the log-logistic selection
  and after the first shape fit, which moved the deciles in band from 5 of 10 to **9 of 10**: that
  number says nothing about which family wins, so the order it settles is not a result being
  chosen.

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
  of predicted twelve-month PD (each grade twice the previous one's floor). A grade passes
  when the realised default rate of its cohort falls inside the **95% Jeffreys interval**
  around its predicted PD; the scale passes when **seven of the eight** do. Jeffreys because
  the top grades hold few defaults and a normal interval is wrong there.

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

The cell table may reach **250 million cells**, raised from 150 on 10 October 2026 for the
reasons set out below -- the memory that bounded it is gone and the time has moved by an order of
magnitude. The 150 was about 2.4 times the 63.6 million the key produced when it was declared;
the extensions this rule admitted took the table to **91,575,827**. The engine reads it a batch at a time, so what the ceiling
protects is the aggregation itself and the time every later fit costs, not a fit's memory.

If the measured cost of the extensions exceeds it, they are given up in this order:

1. the finer bands (8 / 8 / 6 instead of 5 / 4 / 4);
2. the origination spread;
3. the lagged delinquency state.

HARP and the three-state outcome are not on the list: they are corrections of what the model
covers and of what it calls an event, not refinements of it.

### What the measurement did to this rule

Added on 18 September 2026, after `creditsurv profile --extensions` and before any fit of
the new key, because a rule that fires has to say what it did.

All four extensions cost **4.90×** the base key on the nine sampled quarters, a projected
312 million cells. Giving up the finer bands leaves 161.9 million, still over, so the
second rung goes too and the key keeps **HARP and the lagged delinquency state**: 1.26×, a
projected 80.4 million cells (`docs/reports/key_extensions.csv`).

The finer bands are individually affordable, at 144.9 million, and go first regardless,
because the order was fixed by what each extension is *for* rather than by what it turned
out to cost.

### The ceiling is raised to 250 million cells, and why, declared before the rebuild

Amended 10 October 2026, before any finer-banded fit exists. **The motive is stated first,
because stating it is what makes this legitimate rather than convenient.**

**The motive.** The backtest fails its decile criterion, which was declared before the run and
is not to be tuned on the test window. The in-sample evidence says what kind of failure it is:
by decile of the log hazard, actual over expected runs **0.63, 0.60, 0.63, 0.70, 0.79, 0.88,
0.98, 1.09, 1.19, 1.00** -- the model over-predicts the safe deciles and under-predicts the
ninth, so its predicted risk is **compressed** across the spectrum while its ranking is sound
(Gini 0.561). That is in sample, so it is a statement about the specification and not about the
test window. And rule 12 is why the band grid is implicated: a banded covariate is read as
bands, so **five credit-score bands are the resolution of the model's risk ordering**, and the
safest band pools what the safe deciles are made of. The finer bands are 8 / 8 / 6 where the key
has 5 / 4 / 4.

**What the ceiling was for, and what has changed.** Rule 7 caps the table because of what a
table costs: "the aggregation itself and the time every later fit costs". Both have moved, and
one of them has dissolved.

* The aggregation held every quarter's frame and concatenated them -- 11.3 GB at 91.6 million
  cells and **about 21 GB at 200 million, on a 16 GB machine, which is what put the finer bands
  out of reach before any fit was attempted**. It writes a quarter at a time now, so its peak
  is one quarter: the widest is 1.4 million cells on the published key and **3.2 million with
  the finer bands, about 170 MB**. The memory reason is gone, not reduced.
* The time has moved by more than an order of magnitude. Measured on six quarters across the
  book: the published key aggregates in **33 minutes** and the finer-banded one in **39** --
  18% more time for 2.16x the cells, because the work is the SQL scan and not the output. A
  reading of the training half is **3.58 minutes** where it was 11.6, and a whole selection run
  is **20.7 minutes** where the four recorded runs averaged 10.5 hours.

**The new ceiling: 250 million cells.** Derived from a budget in time, declared here and
measured on this machine: an aggregation under **45 minutes**, a reading of the training half
under **10**, and a selection run under **70**. At 250 million cells those come to about 43
minutes, 9 and 60 -- scaled from the measurements above, since a reading and a fit are linear in
rows. The old 150 million was 2.4 times the 63.6 million the key then produced; this is 2.7
times the 91.6 million it produces now.

**What does not move.** The give-up order is unchanged -- the finer bands are first on it, so
admitting them first reorders nothing. Extensions are still priced **one at a time** against the
ceiling, so this admits the finer bands and says nothing about the origination spread, which was
184.4 million on its own and has never been priced beside them. And nothing is tuned on the test
window: the finer bands are judged on the **in-sample** decile calibration first, and the test
window is read once, at the end, against the criteria already recorded.

### How the two given-up extensions are re-priced, declared before the measurement

Added 9 October 2026, before running it, because the first pricing's projections are no longer
arithmetic that can be read.

**The anchoring moves, and it has to.** A projection is `published_cells x ratio`, where the
ratio is the sampled count of a specification over the sampled count of the key the published
table was built with. In September that key was the base and the table on disk was its 63.6
million cells, so the two agreed. The rebuild made the table **base + HARP + the payment
state**, 91,575,827 cells, while the ratios stayed measured against the base -- so a
base-anchored ratio applied to a table that is 1.264x the base over-counts by that factor.
The ratio is therefore measured against **the key the published table was built with**, and
applied to that table's own count.

**And the projection is read with a safety factor of 1.14**, because that is what the one
rebuild this project has done measured: a ratio estimated on nine quarters projected 80.4
million and produced 91.6, running **14% light**. An extension is affordable only if its
projection **times 1.14** is under the ceiling. The factor is on the projection and not on the
ceiling: **the ceiling stays 150 million**, as declared, and nothing here moves it.

**The give-up order does not move either.** If both extensions fit, the finer bands are still
taken first, for the reason the order was fixed on: a band grid is a refinement where the HARP
level and the payment state were corrections. If neither fits, rule 7 is confirmed on better
arithmetic and the key stops where it is.

### What the re-pricing found: neither fits, and the key stops where it is

Measured 9 October 2026, nine quarters, 31.4 minutes (`docs/reports/key_extensions.csv`):

| specification | x the published key | projected | times 1.14 | under 150 M |
|---|---|---|---|---|
| the published key | 1.000 | 91,575,827 | 104,396,443 | **yes** |
| **+ the finer bands** | 2.137 | 195,701,015 | 223,099,158 | **no** |
| **+ the origination spread** | 2.013 | 184,357,255 | 210,167,271 | **no** |

Neither is close. Both are over the ceiling **before** the margin is applied -- by 30% and 23%
-- so the decision does not rest on the 1.14 at all, and would not change if the margin were
dropped. **Rule 7 is confirmed and the key stops where it is**, with no rebuild of the cell
table warranted by this rule.

Two checks worth recording beside it. The September pricing **reproduces exactly** -- base
6,303,637 cells, HARP 6,701,001, the spread 13,415,718, the payment state 7,468,938, the finer
bands 14,348,301, all four 30,906,034 -- so the sampling is deterministic and nothing in the
aggregation has moved under it. And the published key prices at exactly 1.000 of itself,
projecting 91,575,827, which is the self-consistency the old anchoring had lost.

One thing the finer bands cost **less** than the first pricing suggested: 2.137x the published
key where they were 2.276x the base, because the published key is already finer and the two
overlap. Not nearly enough, and the direction is worth knowing for any later re-pricing.

This takes the loan's own note rate out of the key, and with it the origination spread and
the refinancing incentive. Rule 9 therefore has nothing left to divide: the default model
carries the fall in the market rate since origination, and so does the prepayment model, in
place of the incentive rule 6 names. It is the same comparison missing its constant term --
within one origination month this book's note rates span about a point, while the market
rate has moved several points since 2021 -- so the sign expected of the incentive is
expected of the fall, and the covariate that would have distinguished two loans written in
the same month at different prices is not available at this cell count.

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
it says where the defaults are, which is worth publishing -- and no model reads it. A 12-month
behavioural model is a different model with a different purpose, and this repository does not
hold one.

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

### The gate was not passed, and the exception is declared here

**Measured, 9 October 2026: 2.80x on a Hessian, against the 3x this rule asked for.** Three
readings on the whole training half, stable to 0.15%: a value, a gradient and a Hessian in
**9.73, 9.76, 9.74 s** against a NumPy baseline of 26.86 to 27.55, in 0.47 GB against the 1.31
ceiling. `docs/reports/engine.md` carries the table and why the ratio decides rather than the
absolute 9.90 s this rule also named -- the same NumPy code measures 26.86 to 29.69 s across a
session, so the absolute number is not reproducible and reading 9.74 as a pass would be choosing
the thermometer that suits.

**It is kept anyway, by the project owner's decision, and the condition attached to that decision
was that the logic be airtight rather than that the number be three.** What was done to meet it,
all measured:

* the two backends are compared **element by element** at four points, including two outside the
  data where every clip binds, for both families, at 1e-12 relative -- and on the production
  table, 72.7 million rows at the published specification, they agree to **1.2e-15** on the
  objective, 1.1e-13 on the gradient and 1.7e-13 of the Hessian's largest entry;
* the three sums that run over every row carry **Neumaier compensation**, because 72.7 million
  sequential additions into one `f64` had put the backends 4e-11 apart where this project's
  standard for two orderings of the same sum is 1.97e-16. It costs nothing measurable and the
  objective now reproduces to 1.2e-15;
* panics **unwind** rather than abort, so a bug raises a Python exception instead of killing an
  hour-old fit. Measured both ways: 10.19 s against 10.27, inside the noise;
* the bounds checks are **on**. Removing them bought 2% and segfaulted inside an ordinary fit;
* the arithmetic has unit tests in the crate -- the compensated sum, the capped exponential,
  the chain's shortcut on a zero factor -- and `cargo test` runs in CI beside `clippy -D
  warnings`;
* and **one difference is declared rather than fixed**: outside the data the log-logistic's
  Hessian overflows and the two backends reach a different flavour of non-finite in the same
  entries, `-inf` against `nan`. The objective agrees there to the last bit and a step to a
  non-finite curvature is refused by the polish either way, so the test holds the claim that
  decides a fit -- identical wherever the curvature is a number, and not a number wherever the
  other is not.

### Threads, declared before they are written

The plan made releasing the GIL conditional on the gate passing. It did not pass, and the kernel
was kept under the exception above, so the condition is restated here on its own terms rather
than inherited.

**The gate for threads, declared 9 October 2026 before any of it is written:** a value, a
gradient and a Hessian over the whole training half at least **twice** as fast as the
single-threaded **9.74 s**, so 4.87 s or better, on this machine's four physical cores. Below
that it is abandoned and the number is published, because a parallel sum is a reproducibility
liability and half a core's worth of speed does not pay for one.

**And the summation stays deterministic, which decides the design.** The rows are split into a
**fixed** number of contiguous parts, each part is summed in its own order, and the partial sums
are added **in the parts' own order** -- not as they finish. That is the lesson a pooled fit
already taught this project: one queue served every worker, `get` returned whichever finished
first, and two runs of the identical prepayment fit split at the twelfth digit and were five
significant figures apart forty evaluations later.

So the **thread count is part of what determines the answer**, exactly as the batch size is for a
reading: two runs at the same count agree **bit for bit**, a run at a different count agrees to
the last digits of a sum over 72.7 million terms, and the count is recorded with the fit rather
than taken from whatever machine happened to run it. No work-stealing, no unordered reduction, no
atomic accumulation.

## 14. The shape may read the risk, and what would make that worth keeping

Declared 10 October 2026, before the fit, because a model whose shape takes covariates is a
different model and not a tuning of this one.

**The failure.** The backtest's decile criterion -- actual over expected inside 0.80 to 1.25 in
every decile, fixed before the run -- fails, and it fails in sample too: 0.63, 0.60, 0.63, 0.70,
0.79, 0.88, 0.98, 1.09, 1.19, 1.00. The model's predicted risk is compressed, 80x between the
extreme deciles where the book realises 130x, while its ranking is sound at a Gini of 0.561.

**Why the shape is implicated, and the bands were not.** In a Weibull AFT the hazard ratio
between two loans is `exp(rho * delta)`, **constant in age**: the Weibull is the only family that
is both accelerated-failure-time and proportional-hazards, so a single shape parameter sets the
entire risk spread. And `rho` is not free to serve the spread, because it is also what fits the
age profile -- and the age profile asks for two different numbers. From the observed hazard's own
slopes, it wants about **1.47 up to age 48 and 0.90 after**: 0.00025 at age 3 to 0.00093 at 48 is
a factor 3.7 over 16 of log-time, and 0.00093 to 0.00085 at 120 is 0.91 over 2.5. One shape
compromises, the compromise pulls it down, and a smaller shape compresses every hazard ratio. The
finer bands were built to test resolution instead and bought 0.015 on the worst decile.

**What is not done, and why.** Letting the shape depend on the **age** would bend the baseline
where the data asks for it, and it is refused: the hazard would no longer integrate to a coherent
survival function, and a lifetime PD is that integral. Letting it depend on a **loan's** risk
keeps every loan's own (`lambda`, `rho`) and therefore its own survival curve, so the lifetime
quantity stays defined. That is the whole of the distinction.

**The specification, fixed here.** The shape reads `credit_score` as the bands rule 12 declares,
and nothing else. It is the primary risk dimension, it is the one whose grid rule 12 already
singled out, and one covariate is the smallest change that can answer the question. Not a search
over what else might enter: a model chosen by what improved the decile it was being judged on
would be the thing this document exists to prevent.

**The criterion, fixed here.**

* It is kept only if **every in-sample decile enters 0.80 to 1.25**, which is the band the
  backtest declares and the band the current model misses in five deciles of ten. Narrowing the
  spread of misses without entering the band is not a pass.
* The likelihood ratio against the scale-only model is reported and is **not** the criterion, for
  the reason rule 2 gives: on 72 million episodes a likelihood measures fit where the data is
  dense rather than where a lifetime PD spends its time.
* The age calibration is reported beside it, because a specification that fixed the deciles by
  making the age profile worse would have moved the problem rather than solved it.
* And it is judged **in sample**. The test window is read once, at the end, against the criteria
  already recorded -- and if the in-sample deciles do not enter the band, the test window is not
  read at all.

**What it costs, so that the cost is not an argument later.** The written-out kernel fits a shape
with one constant column and refuses more, by design: a row's likelihood depends on two scalars
and that is where its economy comes from. So this fit runs on the autograd path, which is the
slow one -- an hour or two against four minutes. That is a reason to make one declared attempt,
not a reason to make none.

### Which covariates the shape may read, and why that choice was not mine to make

Amended 10 October 2026, after the first shape fit converged and **before** any fit of the set
this amendment names. What it replaces is a hand-pick; the hand-pick stays on the record above,
because the reason it was wrong is the useful part.

**What the rule did, and what was wrong with it.** It named `credit_score` and nothing else, on
the ground that it is "the primary risk dimension". That was never measured. `credit_score` sits
in the selection's protected loan block and has therefore **never passed the univariate screen**;
among the covariates that did, `ltv_change` earns a likelihood ratio of **491,210** and
`unemployment_change` **361,544**, an order of magnitude above anything else in the table. The
claim that justified the pick is an assumption, and a specification resting on an unmeasured
belief about which covariate matters was chosen by hand.

**The line that is not a judgement.** This rule already forbids the shape to depend on the loan
**age**. `period = cohort + age` holds identically, so **a covariate whose value moves over a
loan's life moves with the age**, and admitting one to the shape is admitting age dependence
under another name. The engine's own partition is the same partition: every column of the design
is a function either of the loan combination, fixed at origination, or of the calendar key --
which *is* the tuple of the time-varying values together with the age.

| | Covariates | In the shape |
|---|---|---|
| Fixed at origination | `credit_score`, `original_ltv`, `debt_to_income`, `term_years`, `purpose`, `occupancy`, `harp`, `mortgage_insurance`, `buyer_type` | **admissible** |
| Recomputed every loan-month | `ltv_change`, `unemployment_change`, `policy_rate_change`, `yield_curve_slope`, `consumer_sentiment`, `housing_starts_growth` | **refused** |

The six are refused on the identity and not on materiality -- `ltv_change` is the strongest
covariate in the model, and that is exactly why the reason has to be structural rather than a
preference.

**And the set is the whole of the admissible side, because any subset is a pick.** The shape reads
all nine, in the forms rule 12 declares: the three banded covariates as factors against their own
reference bands, the categoricals as levels, `term_years` as the one term its two values allow.
About twenty parameters where there is one today. No screen, no per-covariate threshold, no order
of entry, nothing to stop early -- there is nothing here to tune, which is the property that makes
this a declaration instead of the search the list at the end of this document forbids.

**The fallback, declared here so that it is not chosen later.** The shape sits in an exponent and
needs only reach exp(5) to overflow the hazard, so twenty parameters in it are twenty ways to take
the objective with them. If the fit cannot be made to converge under the bound of 3 on the log
scale, that is published and the shape drops to **the underwriting block** -- `credit_score`,
`original_ltv`, `debt_to_income`, `term_years`, the four the cell key bands -- which is a nameable
set and not a choice among candidates. If that will not converge either, the shape stays scalar
and this rule is answered in the negative.

**The criterion does not move**: every in-sample decile inside 0.80 to 1.25, the likelihood ratio
reported and not decisive, the age calibration published beside it, and the test window read once
at the end or not at all.

### What the reopening found, and the one cell of the grid still unmeasured

Measured 10 October 2026. The log-logistic went through its own selection -- 35.2 minutes, and it
chose the Weibull's set **less `yield_curve_slope`**, five macro covariates against six -- and its
selected set was then fitted under rule 12's band factors exactly as `default_formula` applies
them to the Weibull's, with a **scalar** shape, because this rule compares selected models.

| specification | deciles in 0.80 to 1.25 | predicted spread | realised | direction |
|---|---|---|---|---|
| Weibull, scalar shape | 5 of 10 | 79.6x | 125.4x | compresses |
| **log-logistic, scalar shape** | **6 of 10** | **105.3x** | 134.3x | compresses |
| Weibull + shape on `credit_score` | 9 of 10 | 181.1x | 138.8x | over-disperses |
| Weibull + shape on all nine | 9 of 10 | 214.0x | 139.0x | over-disperses |

**Neither family enters the band on its own, so by this rule as written the family is not the
answer and the finding is published.** The log-logistic's misses are deciles 1 to 4 against the
Weibull's 1 to 5, and both miss low.

**What it did do is worth more than the verdict.** The log-logistic gains **36,374 log-likelihood
units** over the Weibull with **no extra parameters** -- one covariate fewer, in fact -- where
putting `credit_score` in the Weibull's shape gained 38,804 and cost four. Its own curve does
nearly all of what the Weibull needed a shape-on-risk for, free. Its shape comes out **1.5143**,
and the algebra written into this rule before any of it predicted that a log-logistic would need
about 1.55 at age 60 to match the Weibull's spread, because its ratios fall with age -- 3.63 at
age 6 to 1.37 at 600 at a gap of one in log scale. It fitted in **2.04 minutes** through the
compiled kernel, against 44.4 and 72.9 for the shape fits, which run on autograd.

**The two levers have opposite signs on the same error, and that is measured rather than argued.**
The family takes the predicted spread from 79.6x to 105.3x, toward a realised 134x and short of
it. The shape takes it from 79.6x to 181.1x and then 214.0x, past it. One under-corrects, the
other over-corrects.

**So the last cell of the grid is fitted, and that is a completion rather than a search.** The
family set is {Weibull, log-logistic} and the shape set is {scalar, the whole origination side
rule 14 declares}: **four cells, three measured, and the fourth is the only one left.** An
exhaustive two-by-two has nothing to stop early and nothing to choose -- which is the property
that separates it from the search this document's last section forbids. All four are published
whatever the fourth says.

**Declared before that fit:**

- The criterion is unchanged: every in-sample decile inside 0.80 to 1.25, the age calibration
  published beside it, the likelihood reported and not decisive.
- **If the fourth cell enters the band**, the family changes to the log-logistic -- and rule 2's
  *own* criterion is then owed under the band form, because the 0.1027 against 0.1498 that chose
  the Weibull was measured on the **linear** form. A family may not be switched on a criterion
  this rule does not use. The Aalen-Johansen comparison is re-run and published before anything
  is called selected.
- **If it does not**, no combination of these two levers meets a criterion fixed before any of
  them, that is what the model's record says, and the next question -- a hazard that does not
  leave the origin at zero, which is neither of these two families -- needs its own rule and its
  own motive, written then and not now.
- The grid stops at four. A fifth variation fitted because the fourth missed would be the search,
  and the honest report of a declared criterion unmet is the alternative to it.

### What the failing decile is made of, and the retraction that was itself wrong

Measured 17 October 2026, on the best cell of the grid, in the one pass the record above said was
owed. `exposure_totals` already takes a mapping of dimensions, so the cross is a Recipe and not new
code in the package.

**83% of the miss is three vintages.** Decile 1 reads 5,382 defaults against 3,804 expected, an
excess of 1,578. The **2019, 2020 and 2021** originations are 17% of its exposure and **34% of its
defaults**, at an actual over expected of **3.413** -- 1,311 of the 1,578. Everything else in the
decile, every vintage to 2018, reads **1.082**: inside the band, with room.

| decile 1, by vintage | A/E |
|---|---|
| 1999 to 2018 | **1.082** |
| 2019 | 3.330 |
| 2020 | 3.457 |
| 2021 | 3.377 |

**The same fact by age, because those vintages are the young loans.** Within decile 1 the first
year is 64% of the exposure and the only band that misses: **1.698**, where the book realises 2.6
defaults per 100,000 loan-months a month and the model predicts 1.5. From age 24 the model is
right -- 1.351, 0.995, 1.114 -- and the realised hazard of these loans is **flat at about 2.5 per
100,000 across their whole life**, where a shape above one forces the model's to start near zero
and climb.

**And the decile is not a mixture.** It is **89.2%** the top credit-score band, which reads 1.453,
with 8.0% in the 750 band at 1.327.

**So the cause is structural and it is nameable.** In a multiplicative hazard a favourable
covariate multiplies the hazard *down without limit*. The 2019 to 2021 originations stack every
favourable reading the model has at once -- the top credit band, the policy rate at zero, house
prices rising sharply so `ltv_change` is strongly negative, housing starts up -- and the predicted
hazard goes to the lowest value the model produces anywhere, about 1.5 per 100,000 a month. The
book does not go below about 5. **No covariate and no shape can put a floor under a multiplicative
hazard**; that takes an additive baseline, which is a different family and the grid stopped at
four.

**The retraction recorded above was wrong, and the inference it retracted was right.** The missing
floor was inferred from `rho = 2` and `h(0) = 0`, correctly labelled an inference; it was then
retracted because the **aggregate** 0-to-11 age band over-predicts at 0.755. That reasoning does
not hold: the aggregate mixes all ten deciles, and the young-age over-prediction belongs to the
*risky* ones, whose fitted shape is below one and whose hazard starts high and falls. Inside decile
1 the first year under-predicts at 1.698, which is the signature the original inference named. The
lesson is the one this document already carries about crossed dimensions -- a sum over one
dimension cannot refute a claim about an interaction -- and it cost a correct finding for a day.

**What this does and does not license.** It makes the residual a **documentable exception**: a
cause, a scope (three vintages; without them 1.082), a magnitude (1,578 defaults, **0.11%** of the
1,440,771 in sample), and a remediation that is named rather than searched for. It does **not**
license a fifth cell of the grid, and it does not license dropping those vintages from estimation
-- rule 1 keeps 2020 and 2021 deliberately, because they are the one episode where a policy
intervention broke the link between unemployment and default.

**And it carries a warning about the test window that is not good news.** The pattern is *young
loans in the safest band, under the most benign macro readings in the record*. The test window
scores 2025 onwards, where the young loans are the 2022 to 2024 originations. This diagnosis
therefore **predicts** that an out-of-time decile 1 will under-predict too, and says so before the
window is read rather than after.

### The grid is complete, and no cell of it enters the band

Measured 16 October 2026, the fourth cell fitted in 87.0 minutes and 8 damped Newton steps from
707 standard errors to **1.56e-08**. All four published, as declared.

| cell | deciles | the miss | age bands | years | predicted spread | realised |
|---|---|---|---|---|---|---|
| Weibull, scalar shape | 5 of 10 | 0.603 | 5 of 7 | 16 of 23 | 79.6x | 125.4x |
| log-logistic, scalar shape | 6 of 10 | 0.677 | 5 of 7 | - | 105.3x | 134.3x |
| Weibull + shape on nine | **9 of 10** | 1.546 | 5 of 7 | 17 of 23 | 214.0x | 139.0x |
| **log-logistic + shape on nine** | **9 of 10** | **1.415** | 5 of 7 | 17 of 23 | 206.3x | 145.5x |

**Neither lever, nor both together, reads the band.** The two scalar cells fail low -- deciles 1
to 5 and 1 to 4, the compression -- and the two shape cells fail in decile 1 alone and high. The
best of the four misses a ceiling of 1.25 by **0.165**.

**The levers do not cannibalise, and the log-logistic needs less shape.** The fourth cell is the
best likelihood of the four by 16,879 units over the Weibull with the same shape, on one parameter
fewer: -10,624,865.38 against -10,641,744.53, where the two scalar cells sit at -10,651,714.59 and
-10,688,088.43. And the shape it asks for is **22% smaller** -- `credit_score`'s 620 band reads
-0.324 against the Weibull's -0.413, and `rho` spans 0.920 to 2.038 against 0.840 to 2.017, the
low tail lifting. That is the algebra written into this rule before any of it was fitted: a family
whose own curve bends leaves the shape less to do.

**The age profile is 5 of 7 in all four cells, to the band.** Nothing either lever did touched it.
The two that fail are the same two every time -- **0 to 11 months** at 0.717 to 0.769 and **240 to
399** at 0.622 to 0.702 -- and both **over**-predict.

**That last number retracts the hazard-floor reading recorded above.** A model with no floor would
predict too *little* at young ages, and this one predicts too much: 0.000285 expected against
0.000215 realised over the first year, in every cell of the grid. So the safest decile
under-predicting is not the young-age behaviour showing through, and the inference drawn from
`rho = 2` and `h(0) = 0` does not survive its first check against another view. What is owed is
the same thing as before and now more clearly: **the decile-by-age cross**, one pass, which is the
only table that can say what the failing decile is made of. Until it is run the cause of that one
decile is unknown, and no further fit is warranted by a guess about it.

**Two things the grid did settle, which are worth keeping.** The compression the backtest found is
the shape, not the resolution and not the family: 5 of 10 deciles to 9 of 10, and rule 5's cycle
criterion from 69.6% of calendar years in band to **73.9%** against a 70% threshold, in both shape
cells. And the family is not a tie any more on the likelihood -- 36,374 units for one covariate
fewer -- though this rule says the likelihood is not the criterion, and the decile band it is
judged on separates the two families by one decile out of ten in the scalar cells and by 0.131 on
the miss in the shape cells.

**The grid stops at four, as declared.** The smallest miss of anything measured in this series --
**1.318** -- belongs to the `credit_score`-only shape that rule 14's amendment superseded as a
hand-pick. That is not a reason to adopt it. Choosing a specification by which one came closest on
the decile it is judged on is the thing the last section of this document forbids, and that
specification fails the criterion too. **A declared criterion is unmet by every cell of a declared
exhaustive grid, and that is what the model's record says.**

### What the criterion answered: the shape fixes the compression and then overshoots it

Measured 10 October 2026. Both fits on the whole training half, 72,671,500 cells, and both
judged by the band this rule fixed before either existed.

| decile | scale-only | shape on `credit_score` | shape on all nine |
|---|---|---|---|
| 1 | 0.632 | **1.318** | **1.546** |
| 2 | 0.603 | 0.938 | 1.000 |
| 3 | 0.629 | 0.863 | 0.920 |
| 4 | 0.699 | 0.860 | 0.879 |
| 5 | 0.789 | 0.913 | 0.915 |
| 6 | 0.885 | 0.956 | 0.960 |
| 7 | 0.976 | 0.993 | 0.993 |
| 8 | 1.086 | 1.030 | 1.025 |
| 9 | 1.188 | 1.042 | 1.038 |
| 10 | 0.995 | 0.997 | 0.994 |
| **in 0.80 to 1.25** | **5 of 10** | **9 of 10** | **9 of 10** |

**The compression was the shape.** The scale-only model predicts **79.6x** between the extreme
deciles where the book realises 125.4x, and the shape removes that under-dispersion entirely. The
age profile is **unchanged** at 5 of 7 bands in band, so the deciles were not bought by spending
the ages -- which is what this rule asked to be published beside. On the way, section 5's cycle
criterion flips from fail to pass: calendar years inside the band go from **69.6% to 73.9%**
against a 70% threshold, and that was not the target.

**And it overshoots.** The failure moves from the whole safe half to the safest tenth alone, and
it changes sign -- the model now predicts **less** risk there than the book realises. Predicted
spread 181.1x against 138.8x realised with `credit_score` in the shape, and **214.0x against
139.0x** with all nine.

**More shape is a better likelihood and a worse calibration**, which is what rule 2 says about the
likelihood and what this rule repeated. The nine covariates gain **7,540** log-likelihood units
over `credit_score` alone, on 14 further parameters, and take the failing decile from 1.318 to
1.546. `credit_score` carries **84% of the whole shape gain** -- 38,804 units of 46,344 -- on 22%
of the parameters; five of the other eight move `rho` by under 4% and are inert. The procedure
declared above reached the same covariate the hand-pick had named, together with the measurement
that it is the right one, and with two the hand-pick had not considered -- `original_ltv` at
-0.179 and `occupancy` at -0.108 on the log shape.

**So the criterion is not met and neither specification is published.** Both converged properly --
the nine took seven damped Newton steps from 621 standard errors to **0.000834**, at 19 shape
parameters with neither bound binding -- so the fallback declared above for a fit that cannot
converge never fired. What failed is the criterion, in the one decile where it had always failed.

**What the shape does say, and where the next question comes from.** `rho` runs from **0.840 to
2.017** across loan types, where 125 whole-model refits of this book had put a single number
between 1.07 and 1.62. The worst combinations get a shape **below one** -- a hazard that falls
with age, which is adverse selection burning off -- and the best get 2.02, a hazard that leaves
the origin from nothing, since a Weibull with `rho = 2` has `h(0) = 0`. The safest tenth of this
book defaults at **2.6 per 100,000 loan-months** and the model predicts 1.7, so **a missing hazard
floor is the reading these two numbers suggest** -- a shape given its freedom drives the floor
down, and the decile that fails is the one a floor would lift.

**That reading is an inference and not a measurement, and the difference is the next thing to
settle.** Every view here is a sum over one dimension at a time -- the deciles and the ages are
each accumulated separately and **never crossed** -- so nothing measured says the failing decile
is young loans rather than old ones in benign conditions. A decile-by-age table would say, and it
is one pass. Until it is run, the floor is a hypothesis with two consistent numbers behind it.

The magnitude, as this document requires beside every in-or-out verdict: **1,938 defaults against
3,552 expected, 0.135%** of the 1,440,771 in sample.

## What these rules forbid

- Tuning any threshold on a test window, or choosing a cut after seeing a result.
- Anchoring on anything but the anchoring window.
- Choosing the family on the likelihood alone, or keeping one that turns a declared sign.
- Adding a covariate to the key without measuring what it costs first.
- Fitting the debt-to-income without the HARP level, or imputing the ratio it stands for.
- Putting the refinancing incentive, the origination spread and the fall in the market rate
  in one model.
- Letting the payment state into any model of the remaining life.
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
