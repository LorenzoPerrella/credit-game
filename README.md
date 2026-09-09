# credit-game

Lifetime PD (probability of default) modelling with **parametric multivariate survival models**,
**time-varying covariates** and **interval censoring**, built on [lifelines](https://lifelines.readthedocs.io).

## Why this repository exists

These three requirements are usually presented as mutually exclusive in `lifelines`. The
documentation steers time-varying covariates towards `CoxTimeVaryingFitter`, which is
semi-parametric, and presents `fit_interval_censoring` as a one-row-per-subject method.

They can in fact be combined. On an episode-split (counting-process) panel, passing the interval
bounds together with `entry_col` produces exactly the discrete-time likelihood with time-varying
covariates, while keeping a fully parametric model. That matters for lifetime PD, because a
parametric model can extrapolate beyond the observation window, respond to macroeconomic scenarios,
and yield a smooth PD term structure — none of which a Cox model gives you.

## Status

Under construction. Roadmap:

- [x] Tooling, CI and quality gate
- [ ] FRED macroeconomic connector
- [ ] Synthetic loan panel driven by real macro series
- [ ] Episode splitting and interval-censoring encoding
- [ ] Feature engineering and coarse classing
- [ ] Parametric AFT models with time-varying covariates
- [ ] Grouped estimation via frequency weights
- [ ] Model selection and non-parametric benchmarks
- [ ] Lifetime PD, term structure and macro scenarios
- [ ] Backtesting infrastructure
- [ ] CLI and generated reports
- [ ] Documentation and narrative notebook

## Data honesty

The macroeconomic series are **real**, pulled live from FRED with no API key.

The loan panel is **simulated**. This is a deliberate choice, not a shortcut: no loan-level survival
panel is publicly available without registration. Freddie Mac's Single-Family Loan-Level Dataset is
the right dataset and is supported here as an optional path, but it sits behind a Clarity
registration. The open alternatives that come up in searches do not work — the Zenodo Lending Club
dataset is a *granting* dataset with a binary flag and no event time, and OpenIntro's is a single
quarterly snapshot. Neither can support a lifetime PD model.

Simulating the panel also buys something real: because the data-generating process is known, the
repository can assert that the estimator **recovers the true parameters**.

## Licence

MIT
