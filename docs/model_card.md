# Model card

What the published model may be used for, what it may not, and how to call it. Every number on
this page is placed from the committed views at build time, or dated.

## The model

A **Weibull accelerated failure time** model of the monthly default hazard, with its shape reading
the covariates fixed at origination (rule 14), fitted on every loan-month the Freddie Mac book
observed up to 2021-12 under the `exclude` moratorium policy. The fit is
<!-- value: views.fit -->. Its level is anchored by one multiplier, estimated on 2022-01 to
2024-12 and on nothing else (rule 4), and it is judged on the months after that, which neither
the fit nor the anchoring saw.

## What it is fit for

| Use | Verdict | On the declared test window |
|---|---|---|
| Ranking a performing pool | **fit** | Gini <!-- value: backtest.gini -->, against a threshold of <!-- value: acceptance.gini --> |
| Twelve-month PD of a performing pool | **fit** | actual over expected <!-- value: backtest.ae -->, every decile from <!-- value: backtest.deciles -->, against a band of <!-- value: acceptance.band --> |
| Lifetime PD and its term structure | **not fit** | <!-- value: backtest.age_bands --> age bands inside the band, where rule 5 asks for five of seven |
| Stress testing and scenarios | **not fit** | no backtest on forecast macro yet; the backtest grants the model the economy that happened |
| The PD of one loan in arrears | **not fit** | rule 10: the model does not read the payment state, so a loan two payments behind gets the PD of one in order |
| Regulatory PD (IRB) | out of scope | a different default definition, horizon and data requirements |

<!-- recorded: 2026-10-23 -->
The master scale passes on the same window: 8 of 8 populated grades inside rule 5's 25% relative
tolerance, from the [windows report](reports/windows.md), which is the shaped model's own
campaign.
<!-- /recorded -->

A **performing pool** is every loan not yet in default, arrears included: that is the population
the criteria were measured on, and the anchored level carries the defaults that pass through
arrears. The pool's PD is valid; a single loan's in arrears is not.

## How to call it

```bash
uv run creditsurv score                       # the book at risk in the table's last month
uv run creditsurv score --month 2025-06       # any month the table observes
uv run creditsurv score --publish             # also write the book by grade to docs/tables
```

Each cell of the book gets:

- `hazard`, the anchored monthly default probability at its own covariates;
- `pd_12m`, the twelve-month PD, `1 - (1 - hazard)^12`. It is the quantity the master scale was
  measured on, and the one this page calls fit;
- `pd_12m_floored`, the same with rule 17's regulatory floor of <!-- value: floor.regulatory -->.
  It is a second object, and no criterion reads it;
- `grade`, its place on the master scale.

The scored cells are profiles of loans and stay under `data/`. Only the book by grade, aggregated,
is ever written to `docs/tables`. The command never fits: it scores the fit `creditsurv report`
saved, and stops if that fit is not in the cache.

## The book on the shelf

The loans at risk in <!-- value: shelf.month -->: <!-- value: shelf.loans --> of them, at an
anchoring multiplier of <!-- value: anchor.multiplier -->.

| | |
|---|---|
| Twelve-month PD, the model's | <!-- value: shelf.pd_12m --> |
| Twelve-month PD, floored | <!-- value: shelf.pd_12m_floored --> |
| Loans the floor binds on | <!-- value: shelf.below_floor --> |
| Loans in arrears, valid only as part of the pool | <!-- value: shelf.in_arrears --> |

<!-- table: scored_book -->

Grade 1 is exactly the set of loans the floor binds on, because the master scale's first edge is
the same <!-- value: floor.regulatory --> as the floor.

## What it gets wrong, and how large

- **The shape along age.** It under-predicts the second and third years and over-predicts the
  seasoned book by about a third, which is why no lifetime PD is offered. Rule 16 declared two
  flexible forms along age, a band factor and a restricted cubic basis. Each was predicted,
  fitted and measured on this window, and both read 3 of 7 bands; the basis also took the
  safest decile out of the band. Neither was adopted, and the record is in
  `docs/reports/selection_age_basis.json` and the register of 5 November 2026.
- **The level across regimes.** Each cut is judged on the 24 months after it:
  <!-- recorded: 2026-10-23 -->
  0.4920 on 2019-20, where the CARES Act emptied the window of defaults; 1.3211 on 2021-22; and
  0.9849 on 2023-24, inside the band.
  <!-- /recorded -->
  The anchoring fixes the level on the window it is estimated on, and by construction cannot fix
  a regime it has not seen. `period = cohort + age` holds identically, so calendar time enters
  only through a few macro covariates, and a regime driven by another channel moves the level.
- **No hazard floor.** A multiplicative hazard can predict the safest new production below what
  the book has ever realised (rule 14). Rule 17's floor hides this in the regulatory number and
  leaves it in the model. The share it binds on is published above for that reason.
- **Prepayment is not netted out.** `pd_12m` annualises a cause-specific hazard, as the master
  scale's realised rate per obligor-year does. Over a year in which part of the book repays, the
  probability that a loan on the shelf today defaults is slightly lower than that.
