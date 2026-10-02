# Working on credit-game

Lifetime PD with parametric survival models, fitted on the whole Freddie Mac book.
Read [README.md](README.md) for what it does. This file is the things that are not
obvious from the code and that a mistake in would cost hours or go unnoticed.

## The constraints that shape everything

**Memory decides what can be fitted; time decides what is worth fitting.** The cell key
carries the exact origination month, *mortgage insurance* (`mortgage_insurance`), *buyer type*
(`buyer_type`), the *HARP level* (`harp`) and the *payment state of the month before*
(`delinquency_state`): **91.6 million cells over 2.774 billion loan-months** under the
`exclude` policy, with 1,671,207 defaults and 33,797,300 prepayments.

- **What the key holds is decided by a measured cost against a declared ceiling.**
  `docs/rules.md` caps the table at 150 million cells and fixes the order the extensions are
  given up in; `creditsurv profile --extensions` prices them on nine quarters
  (`docs/reports/key_extensions.csv`). All four wanted extensions came to 4.90x the old key,
  a projected 312 million, so the finer bands and the note rate went and the HARP level and
  the payment state stayed. The projection said 80.4 million and the rebuild produced 91.6:
  **a ratio estimated on nine quarters ran 14% light**, which is the accuracy to expect of it.

- **Every interval-censored fit goes through `creditsurv.models.blocks`.** A stock
  lifelines fit holds ~680 bytes a row of autograd tape and design copies, which would be
  45-50 GB here. The engine evaluates lifelines' own likelihood a block at a time: on 9.5
  million rows, 2.55 GB and 14.9 minutes where lifelines took 13.45 GB and 24.7. It mirrors
  private lifelines code, so `tests/test_blocks.py` holds it to lifelines' coefficients,
  errors and log-likelihood. Re-run it before trusting a new lifelines.
- **Measure memory as phys_footprint** ("peak memory footprint" in `/usr/bin/time -l`),
  never `maxrss`: it misses compressed pages and understated the fit about 2.5x.
- **The training half is 59.7 million rows.** Held in memory it fitted in 91 minutes under
  `exclude` and 79 under `censor`, at a 15 GB footprint. **Streamed from the cell file in
  worker processes it fitted in 68.1 minutes at 4.7 GB** -- four processes, 250,000 cells a
  batch, 315 blocks, 1.79 GB of them stored -- and reproduced the in-memory fit to **9.4e-07
  standard errors**, the same log-likelihood, the same 59,663,961 cells and 1,460,306
  defaults. A warm start from it converges in one Newton step, 4.7 minutes.
- **Memory is the batch, not the blocks.** A reader peaks at what one batch costs to expand:
  1.53 GB at a million cells, 0.97 GB at 250,000. Six processes at a million reached 11.8 GB;
  four at 250,000 reach 4.7. The stored blocks are 30 bytes a row wherever they are.
- `creditsurv moratorium`, two fits and two backtests, took 4.8 hours.
- **`report` starts its fit where the selection ended.** Same specification, same rows, so
  the selection's cached fit is already the optimum, and Newton goes from there instead of
  SLSQP from lifelines' seed.
- **A rebuilt table invalidates every cached fit, and the old optima are still worth having.**
  `cells_identity` is the file's name, size and time of writing, so 36 of the 175 fits on disk
  are a selection run on a table since replaced -- 17.5 hours of them. `Fits._elsewhere` finds
  the same specification fitted on *another* table and hands it to the optimiser as a
  **starting point, never as a result**: where a fit ends is settled by the polish, on the
  gradient and curvature of these rows, under a thousandth of a standard error. Because such a
  start is a guess about a different table, a `Pinned` from it is retried cold, which is the
  one place that exception is not taken at its word.
- **The rows are read once a selection, not once a candidate.** A fit through the written-out
  kernel on the production table is **53 seconds of arithmetic behind 12.1 minutes of reading**,
  and every one of a selection's thirty fits used to pay that reading again -- the fifteen
  step-7 fits of one logged run each began by recomputing the identical base objective to
  twelve digits. `blocks.encode_blocks` reads with **no formula involved** and keeps fifteen
  bytes a row plus the key of every combination (1.09 GB for the training half); each
  candidate's design is then two tables built by putting its formula through **3,001** loan
  combinations and **153,309** calendar keys. `creditsurv select --traced` goes back to tracing
  autograd and re-reading, which is what the equivalence tests hold the other to; `--workers`
  applies to it only, because a reading is one process.
- **Nothing holds the panel any more, and nothing should start again.** Every command that
  used to expand the training half now reads the cell file a batch at a time: the fits
  (`models.blocks`), the selection (`Fits(blocks=...)`), the views (`views.streamed`) and the
  backtest windows. A whole selection was measured at **2.75 GB** across its parent and three
  workers, where one fit holding the half had been 15 GB. `split_cells` remains for a caller
  that genuinely needs both halves as frames; on this table that is nobody.
- **The views are sums.** Every calibration table is loan-months at risk, the defaults among
  them and the defaults expected, grouped by an age, a year, a segment or a decile -- and sums
  add over batches, so one pass fills every table for every segment and every family.
  `survival_by_age` and `actual_expected` are each split into the additive part and the
  derivation for that reason. A **decile** is the exception: it needs the whole distribution,
  so its boundaries come off a weighted histogram of the log hazard in a pass of its own, land
  within a bin of the true quantile, and cannot split a tie.
- **A cached fit is searchable.** The fingerprint hashes the row counts, so naming a fit meant
  expanding the rows to count them -- to find the model that would have scored them.
  `find_fits(**criteria)` reads the descriptions written beside the pickles instead, and tells
  a report's fit from a selection's by whether it carries `purpose`.
- **A window is read, not filtered.** `load_cells_window` selects on the observation month,
  which is `origination_month + age` and therefore not a column parquet can be asked about by
  name; DuckDB evaluates it while reading. Two years of observation are about 3% of the table,
  and `outcomes_by_age` and `load_largest_cells` do the same for a non-parametric curve and
  for the origination profiles.
- Aggregation peaks at ~11.3 GB, concatenating and writing the table. One heavy job at a
  time.
- A successful fit is saved to `data/processed/fits/<hash>.pickle` the moment it
  succeeds, with its description beside it as JSON. In `report` reuse is **opt-in**: that
  fingerprint covers the panel's row count, which does not catch a re-aggregation that
  leaves the count alone. `creditsurv select` fingerprints the cell file itself and
  resumes on its own.
- `report --no-extra-fits` drops the two model-selection sections that each cost a
  further fit. The reports then say the section was skipped.
- **Never leave a long run unsaved.** One run completed a 154-minute fit and was then
  killed writing its reports, keeping nothing. That is why the cache exists.

**Where a fit's time actually goes, measured -- and the first two figures of this were wrong
for a year.** On a 250,000-cell batch of the production table at the **26** parameters rule 12
produced: expanding the design 39.6 ms, a value with its gradient **234.9 ms**, a Hessian
**1,373 ms**. The earlier reading of 66 ms was taken at **19** parameters, before the band
factors, and was never retaken: it is 3.6x light, and it is where the apparent mystery of an
evaluation reading ~180 s on the training half came from. The Hessian is **5.8 times a
value-and-gradient**, not the 49 times a value that used to be quoted -- `minimize` is always
called with `jac=True`, so nothing in the optimiser's path ever buys a value alone. And the cost
per row **rises with the block size**, 0.83 to 1.33 µs a row from 49,000 to 968,000, because at
26 columns a million-row block allocates 208 MB of design and ~700 MB of tape per evaluation.
`docs/reports/engine.md` carries the table and the scaling to the whole training half.

**And the sentence this paragraph used to end with -- "speed lives in the evaluation, not in
Newton" -- was true only while a Hessian was unaffordable.** It is now about twice a
value-and-gradient, so **damped Newton goes first from wherever a fit starts**, cold or warm,
with the method chain behind it as the fallback. On the production table, the same
specification from lifelines' own seed: **10.18 minutes and 16 evaluations** against **43.79
and 142**, to the same log-likelihood of -10,691,177.6879. It begins 1.89e+03 standard errors
out, the first six evaluations are refused as not a likelihood -- the damped step probing the
clipped region -- and the damping ladder walks it in: 1.26e3, 706, 423, 207, 58.4, 6.61, 0.107,
3.31e-05. `polish=False` keeps the optimiser, because that mode exists to reproduce lifelines
exactly.

And the evaluation is not arithmetic-bound. Profiled, a value-and-gradient spent **42% in
autograd's tape, 32% in `pandas.take` and 20% copying**, with the likelihood's own exp and log a
minority. The pandas half was waste -- the design and the masks the likelihood filters by do not
change between evaluations -- and `_Slicer` removed it: **1.9x on every evaluation**. What is
left is autograd, and reducing that means owning an analytic gradient, which is the second
implementation of lifelines' likelihood this engine exists to avoid. **That line was crossed
deliberately on `perf/fit-engine`**, with the equivalence tests as the contract: every column of
the design is a function of the loan combination (3,001 of them) or of the calendar key (153,309)
and never of both, so `eta = A[i] + B[j]`, the interval is always one month and exact
observations never occur. Measured on the real cardinalities, one thread: a value-and-gradient
over 53.3 million rows in **10.2 s** and one with the Hessian in **29.9**, against 57.8 and 396
-- **5.1x and 11.8x** -- with the whole training half resident in 0.81 GB of fifteen-byte rows.
An earlier draft projected 18x and 41x from a prototype that computed the interval probability
with one exponential, which matches lifelines only where no clip binds; see
`docs/reports/engine.md`.

## lifelines' optimiser stops short of the optimum

SLSQP stops on a change of 1e-10 in the *mean* log-likelihood, a tolerance that takes no
account of how precisely the data pin a coefficient down. On four quarters of the book it
stopped up to **5.9 standard errors** from the maximum, 2.0 on sixteen, and finishing the
job gained 20 log-likelihood units. Every fit is therefore polished with Newton steps on
lifelines' own gradient and Hessian until less than 1e-3 standard errors remain.
`polish=False` reproduces lifelines exactly and exists for the equivalence tests.

**Warm starts need Newton.** From a nested model's coefficients SLSQP takes as many
evaluations as from a cold start, 27 against 27. The engine runs Newton directly instead:
0.7 minutes against 4.2 on four quarters, to the same optimum.

**And Newton needs damping.** lifelines clips the interval probability at 1e-25 and adds the
truncation term unclipped, so far from the data the objective -- a mean negative
log-likelihood, which cannot be negative -- goes negative and flat. Adding *loan-to-value
change since origination* (`ltv_change`, formerly `cltv_drift`) to the loan block, a warm
start's full Newton step went 8.31e5 standard errors, to -4604, and was taken because it was
lower; the fit fell back on SLSQP for 76 minutes, and a flatter cliff would have been
reported as the optimum. The polish now takes a step only to a value a likelihood can have,
damped (Levenberg-Marquardt) until it lowers the objective, and a fit that ends anywhere
else raises. The next warm start, adding *unemployment change since origination*
(`unemployment_change`, formerly `unemp_gap`), took six damped steps from 735 standard
errors out to 4e-6: **30 minutes against 76**. On three million rows the same start took 14
evaluations and 7 Hessians where SLSQP needed 91 evaluations, and ended 3e-5 standard errors
from the cold optimum.

**The region that is not a likelihood is reported as infinite, not at face value.** The
objective is a mean *negative* log-likelihood and cannot be negative; beyond a ridge lifelines'
surface falls away into a region that is not a likelihood at all, because the interval
probability is clipped and the left-truncation term is not. At face value that region is the
most attractive place on the surface, and on the prepayment model **six attempts in a row**
ended there -- warm and cold, SLSQP, L-BFGS-B and trust-constr alike, with the coefficients
bounded throughout, because it is `rho_` and not the coefficients that makes the cumulative
hazard explode: exp(5) is enough. Returned as `inf` with a zero gradient it is a wall, and every
method backtracks from it.

**The parameters are bounded, because lifelines leaves them unbounded, and the shape is the one
that matters.** Its `_bounds` are for the univariate fitters; an AFT model's parameters are free.
The cumulative hazard is `exp(rho * (log t - log lambda))`, so the **shape sits in an exponent**:
it needs only reach exp(5) to overflow the hazard and take the objective with it, while a scale
coefficient of the same size does nothing of the kind. Bounding the coefficients at 100 was
therefore aimed at the wrong parameter and changed nothing; the shape is bounded at **3 on the
log scale**, a shape of 0.05 to 20, where **125 converged fits on this book** put it between
1.07 and 1.62. Neither bound can bind here, and a fit that ends on one is refused as not
identified rather than published.

**A pooled fit adds the shares in the parts' own order, because it used to add them in the
order they arrived.** One queue serves every worker, so `get` returns whichever finished first.
Floating-point addition is not associative, so the same point summed in two arrival orders
differs in its last digit -- and an optimiser turns that into a different search: two runs of the
identical prepayment fit agreed to every printed digit for eighty evaluations, split at
0.065288491918 against 0.065288491919, and were five significant figures apart forty evaluations
later, each walking its own path over the same surface. Every multi-process fit made before this
was reproducible only to a tolerance, this one included, so a re-run of a cached fit may differ in
its last digits. `tests/test_streaming.py` now fits the same fixture three times in three
processes and compares **bit for bit**; it fails on arrival order.

**In a pool, the local objective sees a share of the rows -- so its value is a share of the
objective.** With four processes it is a quarter. Anything compared against the whole objective
therefore belongs to the pooled evaluator, and two things had been left in the local one: the
floor, which then refused every nested fit that was good (a candidate at 0.076 turned down
against a parent's 0.071), and the progress line, which reported a quarter of the objective in
every run made with workers and misread four separate diagnoses before it was noticed.

**A nested model cannot fit better than its parent, and that bound belongs inside the fit.**
Dropping a covariate cannot raise the maximised log-likelihood -- the parent could have set that
coefficient to zero, so every point of the child is a point of the parent. A backward-elimination
fit reporting an improvement has therefore not found a maximum but the region where lifelines'
clipped likelihood is unbounded below. The parent's optimum is handed to the objective as a
**floor**, which makes that region unreachable while the fit runs; refusing it afterwards is the
belt, and the difference matters -- the prepayment model's step 8 spent two hours and forty
minutes reaching 0.0122 against a parent's 0.0179 and was still going.

**lifelines' clipped region begins directly below the maximum, so the nested bound has no slack
to give.** The allowance was one log-likelihood unit "for the last digits of a sum over 72 million
terms", and that reason is measurably wrong: the same specification evaluated at the same
coefficients over 800 blocks instead of 443 reproduces the log-likelihood to **1.97e-16** relative,
three hundredths of a millionth of a unit. What settles the size is the other end. On the
prepayment model's step 8 the optimiser probed points reading 11.8, then 158, 183 and 213 units
better than its parent's optimum -- and the parent is not the one in the wrong: re-polished from
its own coefficients with the tolerance driven from 1e-03 to **1e-09** standard errors it takes two
more Newton steps and gains **-0.000 units**. Those probes are the shallow edge of the clipped
region, in the same line searches that further along read 10,699 units and 4.6 million. An
allowance wide enough to admit a probe 213 units better is one that lets a fit converge onto
clipped ground and be cached as an optimum, so the bound stays at a unit. Raising it to 1e-06 of
the log-likelihood on a first reading of the same run, which took the 11.8 for the scale of the
thing, was wrong and is reverted.

**An optimiser can circle the wall instead of stopping at it.** `_PINNED_REFUSALS` counts refusals
in a row, and step 8 produced a cycle of six or seven refusals with one accepted point among them:
the run never passed one, and the fit went to 211 evaluations and 4.3 hours to be refused as it
would have been at the start. Over a window the states separate -- 73% refused across the whole of
SLSQP, 88% once the cycle set in, 0% in the productive phase before it and never under 35%
after -- so forty evaluations three-quarters refused, **with nothing inside them improving on the
best point already found**, ends the fit. The second condition is what makes it safe: a search is
refused where it probes, not where it stands.

**And a fit refused by the floor is refused by every optimiser and every starting point.** When
the polish gives up, what turned its steps back decides what the failure means. Against
lifelines' clipped region it has failed where it stood, and another method from another start
may stand somewhere better -- that is what `_FALLBACK_METHODS` and the cold retry are for.
Against the **parent's optimum** it has found a boundary that sits in the same place for
everybody: on the prepayment model's step 8, SLSQP gave up 557 standard errors out, L-BFGS-B
from its own path gave up at 762, trust-constr was started next, and a cold attempt would have
repeated all three -- over twelve hours for one candidate, for an answer rule 11 gives either
way. `_Pinned` therefore records which kind of refusal it saw, and a polish stalled by the floor
raises `Pinned`, which both the method fallbacks and the cold retry let through.

A diagnosis that did not survive its test, kept because it is easy to reach again: that such a
stall means a **flat direction**, since the polish reported 557 standard errors while a refused
probe changed the objective by 0.2 units. It does not. Those 0.2 units were the gain of a step
damped by 1e+08, not of the Newton step, and the two quantities the comparison needed --
`0.5 * remaining**2` and the quadratic model's own prediction carried to log-likelihood units --
are **identically equal**, as a one-parameter test shows in three lines. The ratio is 1 whatever
the curvature, so it measures nothing.

**The polish's verdict is binding.** It measures the distance to the optimum and the engine
promises under a thousandth of a standard error; a fit the polish could not move is refused, not
published. Without that, one was **cached** at 6,850 standard errors out while the log said the
polish had finished it.

**The polish decides whether a fit worked, not the optimiser's flag.** lifelines caps SLSQP at
`ftol=1e-10, maxiter=200`, and on the prepayment model a fit reached the cap while stable to
**nine significant figures for twenty evaluations** -- at the answer, reported `success=False`.
Discarding it would have thrown away two and a half hours and started another method from
scratch. The other direction happened too: trust-constr reported `success=True` at a point
whose next evaluation read -8.97e+69. So a method is accepted when the polish can finish from
its point and certify the optimum, and rejected when it cannot -- which is the only test that
matches what the engine promises.

**SLSQP gives up on an ill-conditioned design; another optimiser does not.** It solves a
quadratic subproblem at each step, and where the curvature is bad it reports *"Rank-deficient
equality constraint subproblem"* and stops -- the prepayment model did that at step 8 of its
selection, from a **cold** start, at a finite objective of 56.58 with 23 iterations behind it,
on the same design the default model fits happily. What differs is the curvature of a
likelihood whose event rate is twenty times higher. The engine now tries **L-BFGS-B** and then
**trust-constr** when lifelines' own method stops, and records which one found the answer. The
estimator is unchanged -- the same likelihood on the same rows has the same optimum -- and the
damped Newton polish certifies the result is at it to under a thousandth of a standard error,
which is what makes trying another path safe rather than a different model.

## Rules that are silent when broken

**A loan without a debt-to-income is a HARP refinance, and nothing else.** Freddie Mac
waived the ratio for the programme, and in the kept book the gap and the programme are the
same set: on 2012Q2, 181,302 of the 181,356 loans without it carry the flag, and every other
loan without it is dropped by the complete-case rule. The cells keep the ratio **missing**;
the model fills it with a constant that the `harp` level absorbs whole, which is the
dummy-variable adjustment and not an imputation. `features.absorb_not_reported` raises rather
than fit the filled ratio without the level, and rule 8 of `docs/rules.md` says why. HARP is
238.8 million loan-months, 8.6% of the book, and 135,070 defaults.

**The payment state is the month before, never the month itself.** At three missed payments
the loan has defaulted by definition, so the state during the month *is* the event. The month
before is what a servicer knows in time to act, and it carries most of what there is to know:
of 1,671,207 defaults, 1,518,761 are loans that opened the month two payments behind. The
lag is taken on the raw code, not on a number, because `RA` -- an REO acquisition -- would
otherwise cast to the same NULL as "this is the loan's first month" and read as up to date.

**A moratorium is not a default.** CARES Act and disaster forbearance had to be reported
as delinquency, and made up 17% of the default events. The event definition is a
`MoratoriumPolicy`, each policy writes its own `cells_<policy>.parquet`, and a fit's
fingerprint names its policy. Two runs under different policies have different
dependent variables. `exclude` was chosen by a rule written before either fit
(`docs/reports/moratorium.md`); censoring threw away a quarter of the test window's
defaults.

**Every macro series is lagged three months, market quotes included.** A loan 90 days
delinquent in month t missed its payments in t-3 to t-1, so no reading from month t can
be what caused it. "Known in real time" is an argument about publication, not about
transmission.

**The weight is a count of loan-months, never an amount.** Basel and IFRS 9 define PD
per obligor, so a $2m loan and a $200k one each contribute one default. Weighting by
balance silently estimates a different quantity with the same name, and lifelines'
variance estimates assume integer replication counts.

**Categorical mappings have no `ELSE` branch.** An unmapped code becomes NULL and the
loan is dropped. Four mistakes came from doing it the other way: `9` and `99` are
"not available" codes, not categories, and an `ELSE` folds them into whichever level
was written last. Decide mappings from a distinct-and-count across vintages, never
from the layout spreadsheet.

**Truncation orders by calendar period, not by loan age.** `loan_age` restarts at a
modification, so it is not monotone within a loan. Ordering by age cuts the wrong row
and, on 2006Q1, lost 641 real defaults while double-counting 2.6% of the panel.

**Missing-value sentinels are real numbers.** 9999 for credit score, 999 for LTV, DTI,
estimated LTV and the mortgage insurance percentage. Left in place they produce a
portfolio whose average credit score is several thousand. The median ELTV of the 2006
vintage is literally 999.

**Macro covariates are free; loan covariates cost cells -- so measure the cost.** A macro
series is a function of the origination month and the loan age, both in the key, so it
costs **zero cells**. A loan covariate multiplies the table by what it actually costs:
*Mortgage insurance* and *buyer type* together cost 1.19x, measured on nine quarters, where an
unmeasured sixteenfold figure had kept them out of every screen.

## Checks that go mute at this scale

The recurring failure in this project is a diagnostic that was correct on thousands of
loans and says nothing on 48 million. Expect more of these.

- **The Kaplan-Meier band** collapses to a hundredth of a percentage point, so every
  smooth curve is outside it. Report the *magnitude* of the deviation, not in-or-out.
- **The crossing test** fired on tails of 2 loan-months and on gaps of 0.00002 at ages
  where no loan can default yet. It needs an exposure floor and a materiality floor.
- **Every p-value is 0.0000.** The univariate screen and the p-value arm of backward
  elimination are **inert**. Discrimination has to come from the sign, the shape of the
  marginal relationship, or the economics.
- **An out-of-time actual-over-expected cannot be read alone.** In sample it ran from
  0.27 to 2.78 by year, so the backtest publishes that dispersion beside it, and the
  acceptance criteria in `backtest/runner.py` are declared before any run.
- **The Jeffreys interval on a grade goes the same way.** The master scale asks that a grade's
  predicted twelve-month PD fall inside the 95% interval around what its loans did, and **0 of 8
  pass** -- on relative errors of 5% in grade 5 (0.005657 predicted against 0.005381), 8% in
  grade 7 and 30% in grade 4. With 5 million obligor-years in a grade the interval is a few parts
  in a thousand wide, so it is testing the arithmetic of the average, not the model. The rule was
  declared before the run and stands; what has to be published beside it is the **magnitude** of
  each miss, exactly as for the Kaplan-Meier band.

**A marginal relationship is evidence that a covariate is correlated with the outcome,
never that it is identified in a model.** This was got wrong twice — see
`docs/variable_selection.md`, where the retracted argument is kept under a notice
rather than deleted.

## The specification is an output

`creditsurv select` runs steps 5 to 9 of `docs/variable_selection.md` on the training
half and writes `docs/reports/selection.json`, and `tests/test_procedure.py` fails when
`config` and that record part. Two inputs to it are fixed before any fit and must stay
so: `MACRO_ELIMINATION_PRIORITY` and `ECONOMIC_DIMENSION`. Changed after the results,
either becomes a way of dropping whatever came out inconvenient. The selection never sees
the test window; stability compares loans originated in even and in odd years.

## The family was chosen, and the likelihood would have chosen differently

Rule 2 of `docs/rules.md` is applied by `creditsurv family` and it picked the **Weibull**: both
families went through their own selection, and the selected models sit 0.1027 and 0.1498
percentage points from the Aalen-Johansen cumulative incidence of default on average over the
222 loan ages above the exposure floor. Neither turns a declared sign, so neither is excluded,
and the 0.047 pp between them is inside the 0.1 pp tie the rule declared -- where the Weibull is
kept for not falling at long ages. It is also the closer of the two on the criterion itself.

The likelihood is the reason the rule is not the likelihood. On the specification before the
validation the Weibull led by 623,126 AIC points; on the one after it the log-logistic led by
83,961. A criterion that changes its mind with the specification is not a criterion, and AIC on
72 million episodes measures fit where the data is dense rather than where a lifetime PD spends
its time. `docs/reports/family.md` carries the age-by-age gap.

## Identification

`period = cohort + age` holds **identically**, so no two of calendar time, origination
vintage and loan age can be held fixed while the third moves. The baseline hazard's
shape is therefore not identified non-parametrically; it becomes identified only under
the restriction that calendar time enters through a few macro covariates rather than as
a free period effect. Any covariate-free diagnostic of the hazard shape is answering a
different question.

Where families nest, **test rather than rank**: the exponential is a Weibull with shape
one, so it costs a Wald test on a parameter already estimated, not a fit.

## Irreversible

`creditsurv prune-archives` deletes the 40 GB of downloads. Separate command, never
appended to the ingest, never run without the user asking. It verifies three conditions
per quarter; the decisive one is that parquet row counts still match the manifest.

## Conventions

- `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -m "not network"`
- **And `mypy` on the other leg of the matrix before pushing.** `tool.mypy` sets
  `python_version = "3.12"`, so a local run checks one of the two the CI runs, and numpy's
  stubs infer differently under 3.11: **twice now** a branch has gone green locally and failed
  CI on `no-any-return` alone, in a file whose logic was fine. `UV_PYTHON=3.11 uv sync -q &&
  UV_PYTHON=3.11 uv run mypy`, then sync back.
- **Notebooks carry evidence, not logic.** Every statistic is a tested function in the
  package; a notebook calls it and shows the result. If a cell contains an algorithm,
  the algorithm is in the wrong place.
- Tests use fixtures written in **Freddie Mac's own pipe-delimited format**, so the
  loader is exercised on every test that needs data. There is no synthetic panel.
- Comments say **why**, with the measurement that settled it. Commit messages carry the
  number that justified the change.
- `nmds` is the methodological reference. Where this project departs from it, the
  departure is stated and argued rather than made silently.

## Documentation

The site (`mkdocs.yml`, published to GitHub Pages from `main`) opens with short section pages
-- `docs/index.md`, `data.md`, `portfolio.md`, `methodology.md`, `model.md`,
`calibration.md`, `validation.md`, `decisions.md`, `reproduce.md` -- over the long documents
they summarise: `docs/data_dictionary.md` (record layout, a real loan traced through five
layers), `docs/data_preparation.md` (40 GB to a fittable table), `docs/variable_selection.md`
(what survived, what was given up, and what `nmds` would have decided), and the generated
reports under `docs/reports/` -- among them `moratorium.md`, where the event definition was
chosen, and `selection.md`, where the specification was.

**Figures, tables and numbers on the site are placed, never typed.** A page writes
`<!-- figure: name -->`, `<!-- table: name -->` or `<!-- value: name -->`; the hook in
`creditsurv.site.hooks` fills them at build time from `docs/tables`, the aggregates
`creditsurv views` computes locally from the cached fit and the parquet. CI cannot read the
data, so those tables are committed, and the build fails on a placeholder naming nothing, a
view the manifest lacks, or model views from more than one fit. `creditsurv views` never
fits: it stops when the report's fit is not in the cache. Scoring the training half for it
takes the footprint near 15 GB, so nothing else heavy runs beside it.

## Open

**The backtest fails its decile criterion.** Overall actual over expected 0.920 and Gini
0.561 pass; the deciles run 0.598 to 1.064: over-prediction in the safer deciles, down to
0.598 in the second, and a riskiest decile slightly under-predicted. The report first read
0.597 to 0.995, every decile below one, because each decile's expected rate was a plain mean
of cell hazards rather than weighted by loan-months. The criteria were fixed before the run,
so the model is not to be tuned to them on the test window.

***Occupancy* (`occupancy`) changes the hazard's shape, a little.** Its curves cross at 90
months, which no scale factor reconciles, and letting occupancy into the shape parameter is
significant -- likelihood ratio 220.6 on two parameters, p = 1e-48 -- and small: a third of
what *buyer type* earns at the screen (696) and under a thousandth of *loan-to-value change
since origination* (542,917). The model stays scale-only in occupancy.
