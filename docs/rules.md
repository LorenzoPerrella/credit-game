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

The cell table may reach **150 million cells**, about 2.4 times the current 63.6 million. The
engine reads it a batch at a time, so what the ceiling protects is the aggregation itself and
the time every later fit costs, not a fit's memory.

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

## What these rules forbid

- Tuning any threshold on a test window, or choosing a cut after seeing a result.
- Anchoring on anything but the anchoring window.
- Choosing the family on the likelihood alone, or keeping one that turns a declared sign.
- Adding a covariate to the key without measuring what it costs first.
- Fitting the debt-to-income without the HARP level, or imputing the ratio it stands for.
- Putting the refinancing incentive, the origination spread and the fall in the market rate
  in one model.
