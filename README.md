# credit-game

Lifetime PD (probability of default) modelling with **parametric multivariate
survival models**, **time-varying covariates** and **interval censoring**, built on
[lifelines](https://lifelines.readthedocs.io).

Fitted on the **Freddie Mac Single-Family Loan-Level Dataset**: 48.8 million loans,
2.88 billion loan-months, 1999 to 2026. Macroeconomic covariates from FRED. No
sampling — the whole population.

---

## The idea

`lifelines` presents three things as mutually exclusive. The documentation steers
time-varying covariates towards `CoxTimeVaryingFitter`, which is semi-parametric, and
presents `fit_interval_censoring` as a one-row-per-subject method.

They combine. On an episode-split panel, passing the interval bounds together with
`entry_col` produces **exactly** the discrete-time likelihood with time-varying
covariates, in a fully parametric model. For an episode covering loan age
`(a, b]`:

| Case | `entry` | `lower` | `upper` | Contribution |
|---|---|---|---|---|
| Survived the interval | `a` | `b` | `inf` | `log S(b) + H(a) = log[S(b)/S(a)]` |
| Defaulted in it | `a` | `a` | `b` | `log[S(a) − S(b)] + H(a) = log[1 − S(b)/S(a)]` |

The left-truncation term contributes `+H(a) = −log S(a)`, which is what makes each
contribution **conditional on surviving to `a`** — survival at `b` given survival at
`a`. The product over episodes telescopes into the discrete-observation likelihood.

Both rows carry `event = False`: lifelines uses that flag to mean the event time is
known *exactly*, and monthly reporting gives the month, never the day. Genuine
interval knowledge is a censored row with finite bounds; right censoring is the same
construct with an infinite upper bound.

**Why parametric matters here.** Lifetime PD needs to extrapolate past the observation
window, respond to macroeconomic scenarios, and produce a smooth term structure. A Cox
model gives none of the three.

---

## The scale problem, and how it is solved

| | |
|---|---|
| Archives | 40 GB, 28 vintage years |
| Loan-months | **2,876,284,955** |
| After ingest | 17 GB of parquet |
| After aggregation | **15.8 M weighted cells** |
| Compression | **159×** |

Episodes that agree on every covariate and on their position in time are
exchangeable, so they collapse into one row carrying a count, and the likelihood
treats that count as a frequency weight. At this scale that is not an optimisation
but the only thing that makes the problem tractable.

**Episodes are monthly**, set by how often the covariates move rather than by how much
they compress: the time-varying covariates come from monthly series, so an episode
spanning more than a month asks the model to hold constant something the data says
changed. See [`docs/data_preparation.md`](docs/data_preparation.md) for the
measurements behind that.

---

## Reading order

| Document | What it covers |
|---|---|
| [**The portfolio**](docs/portfolio.md) | What the book looks like: outstanding, new lending, mix, drift, macro |
| [Data dictionary](docs/data_dictionary.md) | The record layout, layer by layer |
| [Data preparation](docs/data_preparation.md) | 40 GB of archives to a fittable table |
| [Variable selection](docs/variable_selection.md) | Which covariates survive, and why |

Notebooks: [`01_portfolio.ipynb`](notebooks/01_portfolio.ipynb) carries the evidence;
the statistics themselves live in the package, tested, so a notebook reads like a
report rather than an implementation.

---

## Quickstart

```bash
uv sync
uv run creditsurv fetch-macro    # real FRED data, no API key
uv run creditsurv ingest         # 40 GB of archives to parquet, ~30 min, idempotent
uv run creditsurv portfolio      # describe the book
uv run creditsurv profile        # screen the covariates
uv run creditsurv aggregate      # collapse to weighted cells
uv run creditsurv fit
uv run creditsurv backtest --as-of 2024-12
```

The dataset is **not downloadable programmatically** — free but manual registration
at [Clarity](https://claritydownload.fmapps.freddiemac.com/CRT/). Nothing in this
repository touches the network for it.

---

## Order of operations

Screening comes **before** aggregation, following `nmds`:

```
ingest → screen → decide the specification → aggregate → select → fit
```

This pipeline had it backwards at first — binning and grouping from a specification
chosen in advance, screening afterwards — and the cost was concrete: by the time the
screening ran, the bands were baked into millions of cells, and a mis-binned covariate
could only be found by noticing its coefficient had the wrong sign.

---

## Things the data taught us

Recorded because a repository that only shows what worked is not much use.

**`999` is a missing-value sentinel in the performance file too.** The median
estimated LTV of the 2006 vintage is literally 999, so mark-to-market leverage came
out as `999 − 75` for most of the panel. Caught by the default-rate-by-band table:
credit score and LTV ordered their own risk cleanly and this one did not.

**`channel` cannot be used at four levels.** Until 2008 about half of originations are
coded `T`; from 2009 it vanishes and broker and correspondent absorb it exactly. A
coding change, not a market one — and only the *time series* shows it. The pooled
frequencies look unremarkable.

**Quarterly episodes once looked no better than monthly.** They compressed identically,
which made no sense until the cause was clear: a monthly-varying covariate was still
in the grouping key, and nothing can collapse on age while a covariate moves
underneath it.

**DuckDB wrote 20 GB of spill into the working tree** before anyone noticed, from
grouping all quarters at once. Every loan lives in exactly one quarter's files —
verified, not assumed — so quarters are aggregated one at a time.

**Two fields have exactly one value** across the whole dataset (`amortization_type`,
`interest_only_indicator`). They are listed rather than quietly dropped.

---

## Limitations

- **Prepayment is treated as independent censoring.** It is really a competing risk,
  which biases lifetime PD **upward**. Single-risk is a scope decision.
- **FRED serves revisions, not vintages**, so macro covariates carry mild look-ahead.
  Point-in-time data would need ALFRED; the publication lags partially compensate.
- **ELTV is not used**, despite being the better measure: coverage runs from 0.8% of
  the 1999 vintage to 94% of 2021, so a model built on it would estimate a different
  quantity in every decade. The house-price-indexed drift covers every vintage evenly.
- **No LGD or EAD**, so no expected loss. PD only.

---

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy                        # strict
uv run pytest -m "not network"
```

CI runs lint, format, strict type checking, tests and a packaging build on Python 3.11
and 3.12, with nightly jobs for the live FRED check and the end-to-end pipeline. ruff
does not type-check — it enforces that annotations exist; mypy verifies they are
correct.

## Licence

MIT
