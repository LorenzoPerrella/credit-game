# Working on credit-game

Lifetime PD with parametric survival models, fitted on the whole Freddie Mac book.
Read [README.md](README.md) for what it does. This file is the things that are not
obvious from the code and that a mistake in would cost hours or go unnoticed.

## The constraints that shape everything

**Memory decides what can be fitted; time decides what is worth fitting.** The cell key
carries the exact origination month, `has_mi` and `first_time_buyer`: 63.6 million cells
over 2.54 billion loan-months under the `exclude` policy.

- **Every interval-censored fit goes through `creditsurv.models.blocks`.** A stock
  lifelines fit holds ~680 bytes a row of autograd tape and design copies, which would be
  45-50 GB here. The engine evaluates lifelines' own likelihood a block at a time: on 9.5
  million rows, 2.55 GB and 14.9 minutes where lifelines took 13.45 GB and 24.7. It mirrors
  private lifelines code, so `tests/test_blocks.py` holds it to lifelines' coefficients,
  errors and log-likelihood. Re-run it before trusting a new lifelines.
- **Measure memory as phys_footprint** ("peak memory footprint" in `/usr/bin/time -l`),
  never `maxrss`: it misses compressed pages and understated the fit about 2.5x.
- **The training half is 59.7 million rows**, and a cold fit on it took 91 minutes under
  `exclude` and 79 under `censor`. `creditsurv moratorium`, two fits and two backtests,
  took 4.8 hours.
- **Never hold the panel beside its halves.** Build them from the cells with
  `split_cells`, which lets the table go before expanding either half, keep text keys
  categorical, and let `episode_hazards` narrow each block instead of copying the
  covariates out. Each of those was gigabytes: holding the table took the comparison of
  moratorium policies to a 17.3 GB footprint on this 16 GB machine.
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

## Rules that are silent when broken

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
`has_mi` and `first_time_buyer` together cost 1.19x, measured on nine quarters, where an
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

`docs/data_dictionary.md` (record layout, a real loan traced through five layers),
`docs/data_preparation.md` (40 GB to a fittable table), `docs/variable_selection.md`
(what survived, what was given up, and what `nmds` would have decided),
`docs/portfolio.md`, and the generated reports under `docs/reports/` -- among them
`moratorium.md`, where the event definition was chosen, and `selection.md`, where the
specification was.

## Open

**HARP refinances are outside the model.** Freddie Mac reports no debt-to-income for them,
so the complete-case rule drops them: 18% of the 2009Q2-2019Q1 vintages, about three times as
likely to default as the loans kept (`docs/data_preparation.md`). Covering them takes a level
of their own in the key -- a re-aggregation and a new selection -- never an imputed DTI.

**`occupancy` survival curves cross** at 90 months: investor loans die faster early and
slower late. No scale factor reconciles that, so the AFT assumption is violated for
that covariate. The remedy is `ancillary` on the shape parameter — still one parametric
model, not a segmentation — and it is untested because it costs a fit.
