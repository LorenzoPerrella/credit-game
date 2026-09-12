# Working on credit-game

Lifetime PD with parametric survival models, fitted on the whole Freddie Mac book.
Read [README.md](README.md) for what it does. This file is the things that are not
obvious from the code and that a mistake in would cost hours or go unnoticed.

## The one constraint that shapes everything

**A fit is 37 minutes** on 2.36 billion loan-months, and was 154 before the
specification was reduced. Anything that costs a fit has to earn it.

- `creditsurv report --reuse` reads a cached fit and regenerates everything in
  **10 minutes** instead of 47. Use it for any change that does not touch the
  specification or the panel.
- A successful fit is saved to `data/processed/fits/<hash>.pickle` the moment it
  succeeds, with its description beside it as JSON. Reuse is **opt-in**: the
  fingerprint covers the panel's row count, which does not catch a re-aggregation that
  leaves the count alone.
- `report --no-extra-fits` drops the two model-selection sections that each cost a
  further fit. The reports then say the section was skipped.
- **Never leave a long run unsaved.** One run completed a 154-minute fit and was then
  killed writing its reports, keeping nothing. That is why the cache exists.

## Rules that are silent when broken

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

**Missing-value sentinels are real numbers.** 9999 for credit score, 999 for LTV, DTI
and estimated LTV. Left in place they produce a portfolio whose average credit score
is several thousand. The median ELTV of the 2006 vintage is literally 999.

**Macro covariates are free; loan covariates are not.** A macro series is a function of
the vintage quarter and the loan age, both already in the aggregation key, so it costs
**zero cells**. Adding `channel`, `region` and `first_time_buyer` would multiply the
table sixteenfold. This asymmetry is why the specification looks so lopsided.

## Checks that go mute at this scale

The recurring failure in this project is a diagnostic that was correct on thousands of
loans and says nothing on 48 million. Two are fixed; expect more.

- **The Kaplan-Meier band** collapses to a hundredth of a percentage point, so every
  smooth curve is outside it. Report the *magnitude* of the deviation, not in-or-out.
- **The crossing test** fired on tails of 2 loan-months and on gaps of 0.00002 at ages
  where no loan can default yet. It needs an exposure floor and a materiality floor.
- **Every p-value is 0.0000.** The univariate screen and the p-value arm of backward
  elimination are **inert**. Discrimination has to come from the sign, the shape of the
  marginal relationship, or the economics.

**A marginal relationship is evidence that a covariate is correlated with the outcome,
never that it is identified in a model.** This was got wrong twice — see
`docs/variable_selection.md`, where the retracted argument is kept under a notice
rather than deleted.

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
`docs/portfolio.md`, and the generated reports under `docs/reports/`.

## Open

**`occupancy` survival curves cross** at 90 months: investor loans die faster early and
slower late. No scale factor reconciles that, so the AFT assumption is violated for
that covariate. The remedy is `ancillary` on the shape parameter — still one parametric
model, not a segmentation — and it is untested because it costs a fit.
