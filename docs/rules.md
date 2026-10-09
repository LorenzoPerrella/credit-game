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

The cell table may reach **150 million cells**. That was about 2.4 times the 63.6 million the
key produced when the ceiling was declared; the extensions this rule admitted took it to
**91,575,827**, so 1.6 times remains. The engine reads it a batch at a time, so what the ceiling
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
