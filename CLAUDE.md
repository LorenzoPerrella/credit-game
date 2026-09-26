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
  SLSQP from lifelines' seed. A cell table rebuilt since has another identity, and the fit
  starts cold.
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

**Where a fit's time actually goes, measured.** On a 250,000-row block of the production table
with 19 parameters: expanding the design 41 ms, a value 28 ms, a value with its gradient 66 ms,
**a Hessian 1,381 ms -- 49 times the value**, because autograd takes it forward-over-reverse.
A cold fit is 50 to 100 SLSQP evaluations and two to six Newton steps, so **the optimiser's
path is ~85% of a fit and the Hessian ~15%**: speed lives in the evaluation, not in Newton.

And the evaluation is not arithmetic-bound. Profiled, a value-and-gradient spent **42% in
autograd's tape, 32% in `pandas.take` and 20% copying**, with the likelihood's own exp and log a
minority. The pandas half was waste -- the design and the masks the likelihood filters by do not
change between evaluations -- and `_Slicer` removed it: **1.9x on every evaluation**. What is
left is autograd, and reducing that means owning an analytic gradient, which is the second
implementation of lifelines' likelihood this engine exists to avoid.

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

**The coefficients are bounded, because lifelines leaves them unbounded.** Its `_bounds` are
for the univariate fitters; an AFT model's coefficients are free, and the objective is not a
likelihood everywhere. Every failed run of the prepayment model ended in that region --
coefficients of 1e+80 under SLSQP, `ABNORMAL` under L-BFGS-B, and a trust-constr point whose
next evaluation read **-8.97e+69**. The bound is 100 on standardised covariates: three orders
of magnitude outside anything a credit model can mean, ten inside where the objective breaks,
so it cannot bind at an optimum -- and a fit that ends on it is refused as not identified
rather than published.

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

**The Weibull against the log-logistic.** On the selected specification the log-logistic has
the better likelihood by 83,961 AIC points and sits slightly closer to Kaplan-Meier (1.21
points of survival on average against 1.26, -2.74 at 312 months against -3.20), but turns
*financial conditions* (`financial_conditions`, formerly `nfci_lagged`) against its prior;
on the specification before the validation the Weibull led by 623,126. Switching family
means `creditsurv select` with log-logistic fits, ~15 hours, since every rule of steps 8 and
9 reads the family's coefficients. See `docs/variable_selection.md`.

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
