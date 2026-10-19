# Third validation round: the three corrections, then everything else

## Context

The third independent validation (19 October 2026, on `f77f440`) found that **the model is better
than its published numbers say, and some of its published numbers do not describe it at all**. It
closed the cross-sectional calibration finding -- on the declared test window the published model
holds **10 of 10 deciles between 0.803 and 1.150** and **8 of 8 grades** inside rule 5's tolerance,
for the first time in this project -- and found that none of that appears in any report, because the
reports measure a fit without the shape.

It ordered the work itself, and that order is this plan's order: correct the calculation before
measuring anything else -- **N1, N2, N3** -- then one round of `report`, `windows` and `views`. Until
those are done, no published level, scenario or PD figure describes the configured model.

Each of the three is confirmed at the artefact level, not inferred.

### N1 — `windows` fits a model nobody configured

Two cached fit descriptions differ in exactly one key, on identical rows:

| fingerprint | as_of | rows | ancillary | minutes |
|---|---|---|---|---|
| `bc016657b2b40bfc` (`report`) | 2021-12 | 72,671,500 | the configured shape | 27.1 |
| `b64681b34fa5d88b` (the campaign) | 2021-12 | 72,671,500 | **absent** | 5.2 |

`backtest/campaign.py` never passes the ancillary formula to `fit_streamed` or to
`fit_description`, so all **four** campaign fits -- three regime cuts and the development model --
estimate a scalar shape. Every number in `docs/reports/windows.md`, the anchoring multiplier and the
master scale among them, is of a specification the configuration does not declare.

The second half of the defect is that nothing could have caught it. `FitResult` carries no ancillary
field, so no report states its shape and no test can compare one; the freshness test added after the
second round covers three reports of eleven and compares the scale formula alone.

### N2 — the projection does not re-date the origination

`origination_book` dates its profiles to today by setting the period and the age, and leaves the
month they were written in. Every *change since origination* covariate is then measured over twenty
years in a book supposed to have been written this month. The proof is in the published table: a
book written in one future month carries **five vintage eras**, weighted 868,565 / 49,918 /
1,113,551 / 7,892 / 1,524,681.

### N3 — the origination book is not a portfolio

It is the 500 heaviest age-zero cells ever written, which is a mode rather than a book:

| | the projection book | Freddie Mac 2025-26 |
|---|---|---|
| purchase | 0.59% | **71.1%** |
| rate-term refinance | **87.8%** | 15.6% |
| insured | 0.47% | 34.1% |
| first-time buyer | none | 33.7% |
| investor or second home | none | 7.7% |
| credit score 760-820 | 95.9% | — |

The site's headline figures -- a 12-month PD of 0.01% and an adverse multiple of 3.28x -- and every
row of the calibration report's term structure are computed on that book.

### Two defects found while checking the round's findings

**A published table does not evaluate the criteria it is named after.** `views.build` scores its
backtest tables from `as_of + 1`, which is 2022-01 -- the anchoring window **and** the test window
together -- on an **unanchored** hazard. Rule 1 declares the anchoring window to be for the level
adjustment and nothing else, and the test window to begin 2025-01. So `acceptance_by_segment`
carries a `deciles_passed` column that reads False where the declared criterion reads True, and the
same decile has three readings: **1.676** in sample, **1.456** on that mixed unanchored window, and
**1.150** on the declared test window, which is the only one that is a criterion.

**A declared criterion that nothing computes.** Rule 5's out-of-sample age-band criterion was added
on 17 October; `views.backtest_views` emits months, deciles and the acceptance table and nothing by
age. It is one further view, built from `age_bands` exactly as the in-sample one is.

---

## Branch 0 — the rules and the register · zero compute · first, and blocking

No fit runs until the criteria it is judged by are on the record. N7's remedy, and three corrections
of numbers that were of other models: rule 14's citation of a candidate's deciles, rule 5's
per-grade profile and its age-band premise, and the 17 October amendments to rules 1, 4, 5, 7, 10,
15 and 16, none of which had been entered in the register. The 25% grade tolerance is ratified, on
the arithmetic that settles it: the scale fails under both criteria, so the tolerance cannot turn a
failure into a pass.

## Branch 1 — the shape in the campaign, and the tests that would have caught it

Thread the ancillary formula through `run_campaign` into `_fitted_to`, and from there into **both**
`fit_streamed` and `fit_description` -- passing it to only the fit would save a shaped model under an
unshaped fingerprint, which is worse than today. With it threaded, the development fit's fingerprint
becomes the one `report` has already cached, so a lookup makes that fit free and guarantees the two
commands describe the same model rather than two fits that agree.

Two smaller divergences go in the same change: the campaign declares its calendar covariates without
the protected block, so the moratorium indicator lands on the loan side where the selection puts it
on the calendar side; and `FitResult` gains an ancillary field, without which the shape is
unrecoverable from a cached fit.

Then every report states what it is -- formula, shape and generation date, which no report carries
today -- and the freshness tests cover every report a reporting module writes, comparing both
formulas, with a completeness assertion so the next report cannot escape the check the way
`windows.md` and `family.md` did. The out-of-sample age-band view lands here too.

## Branch 2 — the projection book

Re-date the origination month beside the period, with a test that fails on the defect: the book's
first projected month has its gap covariates identically zero and one vintage era. Then declare the
book. The replacement is a **declared origination window** -- the real originations of the last
twelve months, every profile, true weights -- read with the window reader the backtest already uses,
and its composition is published against the lending table, which is the check that would have
caught N3.

The **stock** book -- what is on the shelf at the reporting date, each loan at its own age -- is the
book an expected credit loss is computed on, and it is a new product rather than a correction. It
follows, with the two decisions it needs declared: whether the month's own defaults and prepayments
are dropped, and that only performing loans are scored, which is the scope rule 10 already declares.

## Branch 3 — one measurement round, then the pages that quote it

`windows`, then `report`, then `views`, one heavy job at a time. Then the documents each number
reaches: the navigation still files three current reports as superseded, three pages and `CLAUDE.md`
quote an anchoring multiplier two generations old, the family report predates the bands and the
shape with no notice, and the site's model carries no label saying it is a candidate until rule 2's
comparison is applied.

## → Fourth validation round

Three corrections and one measurement round later, every published level, scenario and PD describes
the configured model for the first time, which is the condition this round set. The branches that
follow change the model, and validating the corrections first is what stops them being designed
against numbers that turn out to be artefacts.

## The rest, in the round's order

| | What |
|---|---|
| **Rule 16, the age baseline** | P5. A basis in log(age) in the scale block, which costs no cells because age is already in the calendar key. The branch owes the published model's own out-of-sample age profile first: if the first year is the only band out, the remedy is narrower than a free baseline |
| **The declared PD floor** | N6. An overlay calibrated on the realised rate of the safest segment, about 3 defaults per 100,000 loan-months a month, tested on the 2020 cut where the safest decile reads an A/E of 8.12 |
| **The stock book** | The book an ECL is computed on, as scoped in branch 2 |
| **Sentiment, and the regime split** | N4 and N5: a gap form or a withdrawal, and a regime indicator with more than one level across nineteen months |
| **Prepayment, then lifetime incidence** | L2 then L1. Rule 12's bands were never applied to the prepayment model, and the lifetime quantity that needs it is already written and unused |
| **The level, governed** | Rule 4's monitoring, declared on 17 October and not implemented |
| **A backtest on forecast macro** | Rule 5's fourth added criterion |
