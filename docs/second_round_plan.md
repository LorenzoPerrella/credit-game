# Second validation round: the branches that close it

## Context

The second independent validation (17 October 2026, on `56ffa88`) reproduced **every published
number to the fourth digit with a scorer that imports nothing from the package**, closed 14 of
the 17 first-round findings including all five high-severity ones, and found **zero discrepancies
between published and recomputed**. The engine, the data and the procedure are not in question.

What it also found is that **the model is not yet fit for professional use**, and said precisely
where:

| Use | Verdict | Blocked by |
|---|---|---|
| Research, benchmark, challenger | **fit** | — |
| Portfolio risk ranking | with reservations | P6 (within-year Gini falling ten years) |
| 12-month PD, performing pools (IFRS 9 stage 1) | **not yet** | P1 level drift, caused by P2 + P3 |
| Lifetime PD (IFRS 9 stage 2) | **not fit** | L1 `1−S` at 36m, L2 prepayment, P5 age, L3 state |
| Stress testing and scenarios | **not fit** | P3, P2, P4 |

The stated purpose is **prediction of industrial, professional quality; every other assumption is
subordinate to it.** So this plan is ordered by what moves a fitness verdict, not by what is
cheapest — with the exception of two items that are both cheap *and* on the critical path.

Three things reframe the work against the validator's own priority order:

1. **The candidate passes.** The log-logistic with the shape on the nine origination covariates
   reads anchored A/E **0.927**, Gini **0.548**, **10 of 10 deciles in band** on the declared test
   window — the first model in the project to pass all three rule-5 criteria. The prediction this
   repository recorded in `rules.md`, that an out-of-sample decile 1 would under-predict too, **did
   not come true**: it reads 1.056, and the young vintages inside it 1.043 and 0.992. That
   retraction is owed.
2. **L1 is an arc, not a rebuild.** `models/lifetime_pd.py:128-251` already holds
   `competing_paths`, `lifetime_incidence` and `incidence_term_structure`, all tested;
   `models/families.py:89-131` already combines both causes' hazards in production. Nothing hands a
   second `FitResult` to `projection_views`, and `views/build.py` has no `cause` vocabulary at all.
3. **P5 needs no engine change.** Age is already in the calendar key
   (`factorisation.py:233`), so a basis in log(age) in the **scale** block is a function of `j` and
   the factorisation classifies it automatically. `H = exp(g(log t) − rho·(A[i] + B_macro[j]))` is
   a flexible baseline in log-time with zero cells and no change to `terms.py`, `likelihood.py`,
   `factorisation.py` or the crate. The most expensive-looking finding is among the cheapest.

**Decisions taken before planning:** the shape becomes a declared rule rather than a published
candidate, so one selection and one adoption; the note rate is **measured** on nine quarters
before any rebuild is considered; the payment-state module is **deferred** with the scope declared.

---

## Branch 1 — `rules/second-round` · zero compute · **first, and blocking**

Every criterion the later branches are judged by, written before the fits. This is the project's
own discipline and the validator's C5 remedy. No fit runs until this lands.

**Split the document.** `docs/rules.md` has become a lab notebook: rule 14 carries sections dated
10, 17, 16 and 10 October in that order, with a retraction *and* the retraction of the retraction
both present, and the 16 October section still saying the cause of decile 1 is unknown. Split into
`docs/rules.md` (current rules only) and `docs/rules_register.md` (chronological: amendments,
measurements, retractions). Require a second signatory for any amendment touching an acceptance
criterion.

**New rule — the shape reads the origination side.** Promote what rule 14 measured from a
candidate to the specification, exactly as rule 12 did for bands: the shape reads the nine
covariates fixed at origination and none of the six recomputed every loan-month. Record the
retracted prediction about decile 1 beside it.

**New rule — an expected sign for every candidate, or it is excluded.** Declare the four missing
ones in `EXPECTED_SIGNS` (`src/creditsurv/models/selection.py:442-464`): `policy_rate_change`,
`yield_curve_slope`, `consumer_sentiment`, `housing_starts_growth`. Each with its justifying
comment. This is P2's remedy and it is consequential: it moves them from the marginal/conditional
reversal rule to the **top-precedence** backwards-sign rule (`models/rules.py:170-178`), so the
specification will change. Also fix the comment at `selection.py:459-463`, which says
`policy_rate_change` was eliminated as inside the noise — only `mortgage_rate_decline` is in
`config.ELIMINATED`; `policy_rate_change` survived and is the model's second macro effect.

**Rule 1 amended — a moratorium regime indicator**, 2020-03 to 2021-09, in the **protected**
block (`models/procedure.py:106-110` `BASE_CATEGORICAL`), not among the candidates. The purpose is
to *absorb* the moratorium so the macro coefficients read cleanly, not to test whether it belongs —
the same argument `config.py:101-105` makes for `harp`. P3's remedy.

**Rule 9 amended — sign stability across time cuts**, beside the parity split. State the
identification caveat in the rule: `period = cohort + age` holds identically, so a time cut removes
the very calendar variation the macro block is identified from, and the two splits are not
interchangeable diagnostics.

**Rule 5 extended** — the validator judges it "good, incomplete". Add: out-of-sample age-band
calibration; cumulative PD by vintage; a backtest on **forecast** macro rather than realised; and
replace the Jeffreys interval on grades with a declared **per-grade materiality tolerance**, since
at five million obligor-years per grade Jeffreys tests the arithmetic of the average — which this
repository already says about Kaplan-Meier bands.

**Rule 4 extended — level governance.** A single multiplier is transparent but cannot hold an
error that trends: anchored monthly A/E in the test window drifts 1.386 → 0.771 and the last month
is already out of band. Declare monthly A/E monitoring with thresholds and a recalibration rule
(the validator's proposal: six-month rolling A/E outside 0.90–1.10 for three consecutive months).

**Rule 7 — the give-up order.** Restore it or rewrite it, and say plainly that admitting the finer
bands first *did* reorder it: they are first on the give-up list, i.e. least priority, and the
text's claim that admitting them first "reorders nothing" is the inverse of what the order means.

**Rule 10 amended** — split in two rather than forbidding both cases. The payment state stays out
of the **origination** model (the behavioural-score argument holds and is measured: 28.45% monthly
for two payments behind). A module conditioned on the state **at the valuation date** is permitted
and declared out of current scope, because conditioning does not require projecting. Declare the
scope explicitly: performing pools, **not** the ECL of past-due credits.

**New rule — the age baseline may be flexible.** A basis in log(age) in the scale block,
admissible because age is already on the calendar side. Two conditions declared before the fit:
one basis column is dropped for the collinearity with `rho·log t`, and `g(log t)` is checked for
monotonicity before publication — nothing imposes it today.

**Rule 2 amended** — the Aalen-Johansen comparison is owed under the bands (the 0.1027 vs 0.1498
was measured on the linear form), and the note rate is to be re-priced on nine quarters before any
rebuild is considered.

**Fix C6** in `docs/validation.md`: seventeen findings, five high, not thirteen and three.

---

## Branch 2 — `docs/withdraw-stale` · zero compute · parallel to branch 1

A reader currently finds three different models. This must land before the third validation round,
or a reviewer spends it on artefacts instead of the model.

**Withdraw, do not regenerate.** The validator offers both; regenerating costs a full fit
(`creditsurv report` creates its own cache entry, `cli.py:841-850`) that branch 4 will invalidate.
Eight artefacts are stale, not three: `docs/reports/{backtesting,calibration,methodology,
moratorium}.md`, `coefficients.csv`, `incomplete_cases.csv`, all 13 figures under
`docs/reports/figures/`, and `portfolio_summary.json` with `monthly_default_rate.csv` and
`calendar_check.csv`. Remove the three from the `mkdocs.yml` nav and leave a notice naming the
commit they describe and when they are regenerated — once, after branch 4.

**Fix the stale prose**, each at its own line: `README.md:37` (63.6M → 91.6M cells);
`docs/data.md:37-38,88,93-95` (HARP in the model, estimation to 2021-12);
`docs/decisions.md:40`; `docs/model.md:8,22,26` (`1−S`, not a cumulative incidence — until
branch 5) and `:58-59` (`financial_conditions` was eliminated at step 10);
`docs/methodology.md:10` (steps 5–**10**) and `:169` (59.7M → 72.7M training half);
`docs/reproduce.md:24` **and** `:107`, which state two different costs for `select` on one page
while `:76` already places the real one — delete both literals;
`src/creditsurv/features.py:8-13`, a docstring saying the mortgage rate is not lagged where
`TRANSMISSION_LAGGED` fifty lines below lists it first.

**C4** — `docs/variable_selection.md` argues against three covariates the model carries. Put the
first run's history under a notice at the top and regenerate the results section from
`selection.json`.

**C3** — one out-of-sample backtest figure, not three presented as one. Name the window and the
anchoring beside every number: 1.166 (from 2022, unanchored), 0.958 (from 2025, anchored), 0.920
(the September model).

**The freshness test**, C1's remedy. Follow
`tests/test_procedure.py:318-352` `test_the_configuration_is_what_the_last_selection_chose` — the
`config`-vs-committed-record pattern — and use the word-boundary parsing of
`tests/test_aggregate.py:478` so a substring does not read as a fitted name. Assert that no
`docs/reports/*.md` carries a formula differing from `config.default_formula()`, and that the views
manifest's fit matches. The single-fit check already exists at `site/hooks.py:88-97`; the manifest
already stamps `generated` per view and **nothing compares it to anything**.

---

## Branch 3 — `perf/shape-in-kernel` · enabler · no model change

A shape fit costs **44–87 minutes** on autograd against **2 minutes** compiled. Branch 4 needs
several (the re-selection, the three cuts, the adoption), so this pays for itself immediately.

**The finding that makes it affordable: the jet does not widen.** With the shape reading only
origination covariates, `rho = exp(C[i])` and a row still depends on two scalars, so `_Jet`
(`kernel/likelihood.py:100-231`), the Rust `Jet` (`crates/creditsurv-kernel/src/lib.rs:103-220`)
and every family function are unchanged. Only the accumulators change: three scalars become
vectors and matrices indexed by `i`.

- `kernel/likelihood.py:390,416` — `shape: float` → `np.ndarray`. Signature only; the `_Jet` seed
  already works elementwise.
- `kernel/terms.py:76,132,149,177,239,254-267` — the real work. A third design table (3,001 × ~20);
  `shape_index` becomes an array; `shape_first`/`shape_second` become `np.bincount`-fed vectors;
  the shape×calendar Hessian block is structurally the existing `crossed` (233-236); five scalar
  Hessian writes become `np.ix_` blocks.
- `kernel/factorisation.py:387-459,599,625` — a third table from `expand`. `_only` already
  generalises; `Expanded` gains `shape`/`shape_positions`; **assert no shape column lands on the
  calendar side**, which would reintroduce the age dependence the rules forbid.
- `crates/creditsurv-kernel/src/lib.rs:404-460,481,503,580,742,752,786-796` — two more arrays at
  the boundary (19 args, 13 arrays, 3 design tables), `Totals`' two scalars become `Vec<f64>`, and
  `rate_of` moves **into** the row loop. `predictor`, `contract` and `contract_square` are reused
  unchanged.
- Replace the guards at `engine/scan.py:508-524` with a loan-side check. `engine/fit.py:438-443`
  already threads `ancillary`; `models/fits.py:371-378` simply never forwards it.

**Declared before writing**: the two backends agree element by element to 1e-12 relative at four
points including two outside the data, both families, as rule 13 requires; the whole-half objective
reproduces bit for bit against autograd. A **5–10% per-row regression is accepted and published**,
because un-hoisting `rate_of` is what buys two minutes against eighty. Rule 13's segregation is
unchanged: no new dtype, no struct crossing.

---

## Branch 4 — `feat/macro-block` · the highest-leverage modelling work

On the critical path to **two** fitness verdicts. P2 alone explains ~80% of the 2021–2024 level
drift and is the difference between failing and passing on the 2022 cut (1.420 → 0.959 with its
contribution removed).

**The moratorium indicator.** One ~3-line builder in `features.add_macro_family`
(`src/creditsurv/features.py:417-593`, beside the tuple at 509-520) — it needs no macro series at
all, just `(observation >= FIRST) & (observation <= LAST)`, and `observation` is already computed
at `data/panel.py:682`. Name it in `features.MACRO_DERIVED` (393-416) and in
`procedure.BASE_CATEGORICAL` as protected. **Zero cells.** Verify with `Factorisation.keys()` that
the calendar-key count does not grow — adding a column can only split combinations, and the
level-form candidates already pin the month, so it should be zero but is not provably so. One
cached reading is remade: 3.58 minutes.

**The time-cut split in step 9.** `Fits.fit`/`_estimate`/`_once` (`models/fits.py:134-141,310-405`)
take `months=` beside `parity=`; re-key `Fits.held` (`fits.py:126`) from `int | None` to
`(parity, months)` — the one invasive line, with `reading(parity)` at 195 and
`procedure.py:218-228` as callers. `rules.stability_table` (`models/rules.py:201-222`) takes
`*halves`; its set-of-signs test already generalises to N frames. `not_identified`'s reason string
(248-250) stops saying "origination years". `SelectionRecord.stability` gains a `split` column, with
`reporting/selection.py:37,64,114-117` following. `CellBlocks.months` and the `months` fingerprint
(`engine/cache.py:72,103`) already exist, so a time-cut reading is already cacheable.

**Then one selection and one adoption.** `creditsurv select` with the shape in the specification
(branch 1's rule) and the four signs declared; the three regime cuts; the scenario; rule 2's
Aalen-Johansen comparison under the bands. Judged by rule 5 as extended. Expect the covariate set
to change — `tests/test_procedure.py` pins `config` to the record, so the re-run is mandatory, not
optional.

**Then regenerate** everything branch 2 withdrew, once.

---

## → Third validation round, here

A round earns its cost when the thing it judges has changed enough that a verdict could change.
After branches 1–4, **two verdicts can move**: stage-1 12-month PD from "not yet", and stress
testing from "not fit". Both are blocked by P1/P2/P3/P4, all of which land here.

And the sequencing matters more than the saving: branches 5 and 6 build **on top of** the macro
block. Validating it first is what stops the age baseline and the lifetime PD from being designed
against a specification that turns out wrong — which is exactly what happened to the three
September reports. A fourth round follows branches 5, 6 and 8, for the stage-2 verdict.

---

## Branch 5 — `feat/lifetime-incidence` · L1 · cheap, and moves a verdict

The published "lifetime PD" is `1 − S` over 36 months on the default hazard alone. Observed, with
no model: Kaplan-Meier against Aalen-Johansen on the development half overstates cumulative
default by **31% at 3 years, 57% at 5, 122% at 10, 201% at 15**.

- `views/model.py:351` `projection_views` takes a mapping of cause → `(FitResult, covariates)`,
  the shape `views/streamed.py:67-70` `accumulate` already uses. `views/build.py:86-195` gains
  `cause` vocabulary; **`models/families.py:89-131` is the template that already does this** —
  `selected_fit(cause="prepayment")` plus `_covariates_of` reading the record back.
- Swap `survival_along_path`/`conditional_pd` for `competing_paths`/`lifetime_incidence` at
  `views/model.py:367,373` **and at 386-399**, where the arithmetic is inlined rather than calling
  `pd_term_structure` — so `1 − S` lives in two places, not one. Same at
  `reporting/calibration.py:233-234,253-260`, or the report and the site will disagree.
  `scenario_lifetime_pd` (`models/lifetime_pd.py:526-550`) needs the second fit **inside** its
  per-scenario loop, because the prepayment hazard also moves under the adverse path.
- Column names: `incidence_term_structure` emits `cumulative_incidence`/`marginal_incidence`
  where `site/figures.py:567` and `site/content.py:126-129` read `cumulative_pd`, `pd_12m`,
  `lifetime_pd_baseline`, `lifetime_pd_adverse`. Rename at the view boundary; update
  `tests/test_views.py:284-287` and `tests/test_site.py:98-104`.
- **The horizon becomes residual life.** `term_years * 12 - age` is derivable per cell — both are
  in the key, and `features.py:542-543` already reads `term_years` from the expanded frame — and
  **nothing computes it**. `project_panel` (`lifetime_pd.py:471-523`, extension at 518-522)
  extends every loan a flat `horizon_months`, so a 15-year loan at age 170 is projected 60 months
  past its own maturity. Truncate at contractual maturity, replace the hardcoded
  `min(horizon, 36)` at `views/model.py:371` and `reporting/calibration.py:259` with a declared
  extrapolation policy, and fix the label at `reporting/calibration.py:254`, which says 60m while
  the scenarios are 36.

---

## Branch 6 — `feat/age-baseline` · P5 · 2 of 7 age bands out of sample

For a lifetime PD the shape along age *is* the product. The model under-predicts the 12–35 month
hump by up to 44% and over-predicts the seasoned book by a third; one shape is a compromise across
vintages (1999–2008 read 1.33–1.34 at 12–35 months, 2009–2018 read 0.90–0.95, and the 2022 vintage
out of sample reads 1.75).

- **Expose age as a covariate.** Both `_scan` (`engine/scan.py:131`) and `encode_blocks`
  (`scan.py:307`) drop `special`, which contains `entry_col = AGE_START`, so a formula cannot name
  it. Add the basis columns in `features.add_macro_family` and name them on the calendar side so
  `encode_blocks`'s unclassified-covariate check (308-315) passes.
- **Piecewise first**, reusing rule 12's exact mechanism: `C(age_band, Treatment(ref))` with the
  reference the band carrying the most loan-months, following `BANDED_REFERENCE`
  (`config.py:519-525`) and `default_formula()` (594-611). It is a mechanism the project already
  has and whose reference discipline is tested. A restricted cubic basis is the fallback if bands
  do not reach 7 of 7.
- Drop one basis column for the collinearity with `rho·log t`; check `g(log t)` for monotonicity
  before publication, as branch 1's rule requires.
- Judged by rule 5's new out-of-sample age-band criterion.

---

## Branch 7 — `measure/note-rate` · ~30 minutes, no rebuild

`creditsurv profile --extensions` on nine quarters for the origination spread, anchored on the
**current published key** rather than the base. Replaces a projection with a count; the project has
already measured that a nine-quarter projection runs **14% light**, so the 1.14 margin decides the
answer and the projection alone does not. The rebuild is reconsidered only with this number in
hand, and only if branch 8's prepayment model still fails out of sample without the incentive.

---

## Branch 8 — `feat/prepayment-rebuild` · L2 · blocks branch 5's credibility

In-sample A/E 1.000; out of sample from 2022, A/E **0.767** and 0.465 in 2022, deciles **inverted**
(1.51 in the lowest predicted decile to 0.49 in the highest), Gini **0.18**.

- **Rule 12 was never applied to it.** Its formula is linear on `credit_score + original_ltv +
  debt_to_income` because `Specification.formula` (`models/specification.py:26-30`) has no band
  path and `procedure.LOAN_CONTINUOUS` (`procedure.py:97`) feeds them as continuous. That is the
  same defect rule 12 fixed for default.
- Respect rule 6's signs: `unemployment_change` sits at −0.042 against a declared **+**, kept only
  under rule 11 because removing it left an unfittable model. In an adverse scenario that sign
  accelerates prepayment and so *lowers* cumulative PD. With the bands it may no longer be needed.
- Until it holds out of sample, publish a **declared per-segment CPR** as the assumption — more
  defensible than a model whose deciles invert, and branch 5's incidence needs something credible.

---

## Branch 9 — `feat/geography` and `fix/hurricane-residual` · D7, D8

- **D8** is the one assumption in the validator's table that is **not declared**: national house
  prices and unemployment as proxies for local conditions, when 2008 was a regional crisis. Declare
  it on `docs/data.md`, measure a few-level region's cell cost under the 250M ceiling, and consider
  `ltv_change` on a regional price index.
- **D7**: December 2017 still reads 6.73 bp against a 3.31 baseline, with ~8,000 excess defaults
  from November 2017 to March 2018 — half the first round's 16,706, and 0.56% of in-sample defaults.
  `docs/data_preparation.md` declares it resolved by reading **annual** rates, which cannot see a
  monthly peak. Either treat Sep 2017–Mar 2018 90-days-past-due in TX, FL, PR and VI as moratorium,
  or document the residual with its measure.

---

## Deferred, with the scope declared

**L3, the payment-state module.** Rule 10 is amended in branch 1 to distinguish the origination
model from a valuation-date-conditioned one, and the scope is declared: performing pools, not the
ECL of past-due credits. The module is built after stage 1 and stage 2 hold on performing credit,
which is where the bulk of the book is.

---

## Verification

**Per commit, unchanged:** `uv run ruff check . && uv run ruff format --check . && uv run mypy &&
uv run pytest -m "not network"` plus `uv run mkdocs build --strict`, **and `mypy` on the 3.11 leg**
(`UV_PYTHON=3.11 uv sync -q --group docs --extra kernel && UV_PYTHON=3.11 uv run mypy`), syncing
back with the same extras. Start from **669 tests** and do not go down.

**Branch 2 and 3 carry the strongest property: no published number may change.** `docs/tables/`
and the generated reports must be byte-identical before and after — branch 3 is an implementation
of the same estimator, and the published model's cached optimum must re-certify warm through the
new kernel path at under a thousandth of a standard error, as the compiled kernel already does
(0.30 minutes, one evaluation, 7.73e-05 standard errors, log-likelihood difference exactly 0.0).

**Branch 3 additionally:** the equivalence tests of `tests/test_kernel.py` parametrised over the
shape-with-covariates case, element by element at four points including two outside the data, both
families, 1e-12 relative; and a fit of the fixture three times in three processes compared **bit for
bit**, which is the test that caught the unordered-summation defect.

**Branches 4, 5, 6, 8:** judged by rule 5 as branch 1 extends it, in sample first. The test window
is read **once**, at the end of each, against criteria already recorded. Every result is published
whether it passes or not, and a prediction about the window is written before it is read — the one
made this round was wrong, and recording it is what made that knowable.

**Out of scope, declared:** no change to the cell table (branch 7 measures, it does not rebuild);
no change to the moratorium policy; no new covariate in the key; and no threshold tuned on a test
window.
