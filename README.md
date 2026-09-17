# credit-game

Lifetime PD (probability of default) modelling with **parametric survival models**,
**time-varying covariates** and **interval censoring**, built on
[lifelines](https://lifelines.readthedocs.io).

Fitted on the **Freddie Mac Single-Family Loan-Level Dataset** -- 49.2 million loans,
2.88 billion loan-months, 1999 to 2026 -- with macroeconomic covariates from FRED. No
sampling: the whole population.

**Documentation: <https://lorenzoperrella.github.io/credit-game/>** -- interactive views of
the portfolio, the calibration and the backtest, opened by segment.

---

## The idea

On an episode-split panel, lifelines' interval-censored likelihood with left truncation is
**exactly** the discrete-time likelihood with time-varying covariates, in a fully parametric
model. For an episode covering loan age `(a, b]`:

| Case | `entry` | `lower` | `upper` | Contribution |
|---|---|---|---|---|
| Survived the interval | `a` | `b` | `inf` | `log S(b) + H(a) = log[S(b)/S(a)]` |
| Defaulted in it | `a` | `a` | `b` | `log[S(a) − S(b)] + H(a) = log[1 − S(b)/S(a)]` |

The left-truncation term `+H(a) = −log S(a)` makes each contribution conditional on surviving
to `a`, and the product over episodes telescopes into the discrete-observation likelihood.
Both rows carry `event = False`: lifelines reads that flag as an exactly known event time, and
monthly reporting gives the month, never the day.

A parametric model is what lifetime PD needs: it extrapolates past the observation window,
responds to a macroeconomic scenario and gives a smooth term structure. A Cox model gives none
of the three.

**At this scale the problem is memory.** Episodes identical on every covariate and position in
time collapse into 63.6 million weighted cells, and fits run block by block on lifelines' own
likelihood, polished to the optimum its optimiser stops short of.

## Where to read

| | |
|---|---|
| [Data](docs/data.md) | Source, event definition, what is out of scope |
| [Portfolio](docs/portfolio.md) | What was lent, when, to whom, and how it performed |
| [Methodology](docs/methodology.md) | Episodes, the selection procedure, the family, the engine |
| [Model](docs/model.md) | Specification, coefficients, term structure, scenarios |
| [Calibration & backtest](docs/calibration.md) | Kaplan-Meier against the model, actual against expected, the acceptance criteria |
| [Validation response](docs/validation.md) | The independent validation's findings and what changed |
| [Decision log](docs/decisions.md) | Choices, rejected alternatives, retractions |
| [Reproduce](docs/reproduce.md) | Commands, measured costs, constraints |

The figures and numbers on those pages are placed at build time from the tables in
`docs/tables`, so on GitHub the pages show text without them; the site shows both.
Notebooks [`01_portfolio.ipynb`](notebooks/01_portfolio.ipynb) and
[`02_lifetime_pd.ipynb`](notebooks/02_lifetime_pd.ipynb) carry the evidence; the statistics
live in the package, tested.

## Quickstart

```bash
uv sync --group docs
uv run creditsurv fetch-macro    # real FRED data, no API key
uv run creditsurv ingest         # 40 GB of archives to parquet, ~30 min, idempotent
uv run creditsurv portfolio      # describe the book before modelling it
uv run creditsurv profile        # screen the covariates before aggregating
uv run creditsurv aggregate --moratorium exclude  # collapse to weighted cells, ~40 min
uv run creditsurv select         # the variable selection on the training half; resumes
uv run creditsurv report --extra-fits  # one fit; writes all three reports
uv run creditsurv views          # the aggregate tables behind the site; never fits
uv run mkdocs serve              # the site, locally
```

The dataset is **not downloadable programmatically** -- free but manual registration at
[Clarity](https://claritydownload.fmapps.freddiemac.com/CRT/). Nothing in this repository
touches the network for it, and only aggregates are published.

`uv run creditsurv prune-archives` reclaims the 40 GB of downloads after verifying the
parquet. It is the only irreversible step, and deliberately a separate command.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy                        # strict
uv run pytest -m "not network"
uv run mkdocs build --strict
```

CI runs lint, format, strict type checking, tests and a packaging build on Python 3.11 and
3.12, builds the site on every pull request and publishes it on every merge to `main`.
[`CLAUDE.md`](CLAUDE.md) holds the rules that are not obvious from the code.

## Licence

MIT
