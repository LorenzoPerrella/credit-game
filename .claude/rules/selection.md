---
paths:
  - "src/creditsurv/config.py"
  - "src/creditsurv/models/procedure.py"
  - "src/creditsurv/models/selection.py"
  - "src/creditsurv/models/rules.py"
  - "src/creditsurv/models/specification.py"
  - "src/creditsurv/models/families.py"
  - "docs/reports/selection*"
  - "tests/test_procedure.py"
  - "tests/test_selection.py"
---

# Selection and the specification

- **The specification is an output.** `creditsurv select` runs steps 5-10 of
  `docs/variable_selection.md` on the training half and writes `docs/reports/selection.json`.
  `tests/test_procedure.py` fails when `config` and that record part. Config follows the record:
  a config naming a term the selection dropped makes `report` start cold, because the selection's
  optimum is no longer a warm start.
- **Fixed before any fit and never after:** `MACRO_ELIMINATION_PRIORITY` and `ECONOMIC_DIMENSION`.
  Changed after the results, either is a way of dropping whatever came out inconvenient.
- **`PROTECTED_MACRO` is not a candidate.** It is never screened at step 7 and carries no prior.
  At step 6, however, `stepwise_vif` only **ranks** it last through `priority`, so a protected
  name can still be removed if every offender is protected. Make protection a floor before any
  protected term shares a dimension with another, because step 9's `not_identified` removes the
  smaller unstable covariate beside a larger one of the same dimension.
- **Rule 12's bands are factors**, read band by band, never as a line through midpoints.
  `worst` and `stability_table` skip them, since they have no single sign.
- **Rule 14's shape reads only covariates fixed at origination.** `default_shape()` filters both
  lists, and reads `ORIGINATION_CATEGORICAL`, the set offered, not `CATEGORICAL_REFERENCE`, the
  scale's output.
- **Rule 16, the baseline along age.** The model's monthly hazard is positive for **every** `eta`
  profile, because `episode_hazards` takes its increment `H(a+1 | x_a) - H(a | x_a)` at fixed
  covariates within the month. A monotonicity condition on `H(t | fixed eta)` was declared,
  applied, used to refuse a coherent fit, and withdrawn (see rule 16 and the register). A form
  along age is judged by section 5's out-of-sample age-band criterion and nothing else.
- **The family was chosen by rule 2, not the likelihood**: Weibull, inside a declared 0.1 pp tie.
  AIC switched its preference with the specification.
- **A marginal relationship shows correlation with the outcome, never identification in a
  model.** This was got wrong twice (see `docs/variable_selection.md`).
- **Where families nest, test rather than rank.** The exponential is a Weibull with shape one: a
  Wald test on a parameter already estimated, not a fit. A term that bends the hazard along age
  changes what "shape one" means.
- **`period = cohort + age` holds identically**, so the baseline's shape is identified only
  because calendar time enters through a few macro covariates. A covariate recomputed every
  loan-month moves with the age and may not enter the shape.
- **One shape sets the whole risk spread** in a Weibull AFT, which is also PH. The finer bands
  bought 0.015 on the worst decile and the key went back to 5/4/4.
