---
paths:
  - "src/creditsurv/models/engine/**"
  - "src/creditsurv/models/kernel/**"
  - "src/creditsurv/models/blocks.py"
  - "src/creditsurv/models/aft.py"
  - "src/creditsurv/models/fits.py"
  - "crates/**"
  - "tests/test_blocks.py"
  - "tests/test_kernel.py"
  - "tests/test_streaming.py"
---

# The fit engine

The measurements behind every line here are in `docs/development.md` and `docs/reports/engine.md`.

## What it is and what holds it

- **Every interval-censored fit goes through `creditsurv.models.engine`.** A stock lifelines fit
  holds ~680 bytes a row of tape and design copies, which would be 45-50 GB on this table. The engine
  evaluates **lifelines' own likelihood** a block at a time and mirrors private lifelines code.
  `tests/test_blocks.py` holds it to lifelines' coefficients, errors and log-likelihood: rerun it
  before trusting a new lifelines.
- **`eta = A[i] + B[j]`.** Every design column is a function of the loan combination (about 3,000)
  or of the calendar key (macro covariates **plus the age**: 152,565 for the published model,
  286,387 for a selection's covering set), never both. `Factorisation.expand` classifies by
  behaviour and refuses a column that moves with both. A function of the age alone is therefore
  calendar-side and costs no cells.
- **The shape** (rule 14) is `rho = exp(C[i])`, a function of the loan combination. It runs on the
  NumPy path, because the compiled crate takes the shape as one `f64`. `shape=None` is the old
  code literally, so cached fits keep their last digits.
- **The compiled kernel** (`crates/creditsurv-kernel`) is optional at import and runs on
  `config.KERNEL_THREADS = 4`, a declared constant and not `os.cpu_count()`. Its partial sums are
  added in the parts' own order with Neumaier compensation. Bounds checks stay on: unchecked
  indexing segfaulted inside a fit. `-C target-cpu=native` is a regression. `cargo test` runs in
  CI. The 2.80x it reaches against a declared 3x is kept under rule 13's exception.

## Optimisation: the polish decides

- lifelines' SLSQP stops on a change in the **mean** log-likelihood, up to 5.9 standard errors short.
  So **every fit is polished by damped Newton until under 1e-3 standard errors remain**
  (`POLISH_TOLERANCE_SE`). Newton goes first from any start, with the optimiser chain behind it.
  `polish=False` exists only to reproduce lifelines in the equivalence tests.
- **The polish's verdict is binding, not the optimiser's flag.** A method is accepted when the polish
  can finish from its point and certify the optimum. The fallbacks are `L-BFGS-B` and then
  `trust-constr`.
- **The clipped region is not a likelihood.** lifelines clips the interval probability and adds
  the truncation term unclipped, so far from the data the objective goes negative and flat. It is
  returned as `inf` with a zero gradient: a wall.
- **Bounds:** the shape at **3 on the log scale** (it sits in an exponent), coefficients at 100.
  `_check_interior` **refuses** a fit that ends on a bound as not identified, so a bound is never a
  constraint.
- **The nested-model floor:** a child cannot beat its parent's optimum. That bound sits inside the
  objective, at one log-likelihood unit and no wider, because the clipped region starts directly
  below the maximum. `Pinned` against the floor is final for every method and start. A fit circling
  the wall, three quarters of forty evaluations refused with no improvement, is ended.
- **Concavity:** the rank tolerance is numpy's, `max_eig * n * eps`. Flat directions are logged,
  not refused.

## Readings, caches and pools

- **A reading is written once per table** (`engine.cache`) and mapped back in 0.4 s. Its name covers
  the table identity, both covariate lists, the cause, the parity, the window, the batch size and
  the macro panel clipped to `cut - lag`. **Changing the declared calendar list retires every
  reading**: plan 3.6 minutes and ~1 GB a window.
- **A rebuilt table invalidates every cached fit**, since `cells_identity` is name:size:mtime.
  `Fits._elsewhere` hands a foreign optimum over as a **start, never a result**. A `Pinned` from
  such a start is retried cold.
- A successful fit is saved the moment it succeeds. **A cached pickle names its module**, so moving
  a class it holds must leave the old name importable (`models/blocks.py` is that shim).
- **Pooled fits add the parts' shares in the parts' order**, never arrival order. Floating-point
  addition is not associative, and arrival order made two identical fits diverge.
  `tests/test_streaming.py` compares bit for bit. In a pool the local objective is a **share**:
  anything compared with the whole objective, such as the floor or the progress line, belongs to
  the pooled evaluator.
- **Memory is the batch, not the blocks:** 0.97 GB a reader at 250,000 cells. Measure it as
  phys_footprint, never `maxrss`, which understates by about 2.5x.
- **The machine drifts 13-18% within a session.** Quote a speed ratio only against a baseline
  measured in the same run.
