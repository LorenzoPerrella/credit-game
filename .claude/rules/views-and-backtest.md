---
paths:
  - "src/creditsurv/views/**"
  - "src/creditsurv/backtest/**"
  - "src/creditsurv/reporting/**"
  - "src/creditsurv/models/anchoring.py"
  - "src/creditsurv/models/lifetime_pd.py"
  - "tests/test_views.py"
  - "tests/test_backtest.py"
---

# Views, backtests and reports

- **The views are sums.** Every calibration table is loan-months at risk, defaults and expected
  defaults, grouped. Sums add over batches, so one streamed pass fills every table. A **decile** is
  the exception: its boundaries come off a weighted histogram of the log hazard in a pass of its
  own.
- **`creditsurv views` never fits.** It scores the fit `report` saved and stops (`NoCachedFit`) when
  that fit is not cached. It asks `find_fits` for `purpose=None`, meaning the key must be
  *absent*, so a selection fit is deliberately not a report fit. Scoring the training half takes
  the footprint near 15 GB, so nothing heavy runs beside it.
- **Every command that fits must be handed the shape** (`config.default_shape()`). `windows` once
  was not handed it: four campaign fits were scalar-shape and every published out-of-sample number
  was of them. The campaign fits through `read_cells` + `fit_encoding`, because `fit_streamed`
  stores a design. `reporting.builder.specification()` writes `Formula:`, `Shape:` and
  `Generated:` on every report, and `tests/test_site.py` holds them.
- **A table named for a criterion has to evaluate that criterion.** The anchoring window
  (`anchoring.ANCHOR_WINDOW`, 2022-01..2024-12) sets the level and nothing else; the test window
  is the months after it. One decile was reported three ways in a day: 1.676 in sample, 1.456 on
  the anchoring and test windows together unanchored, and 1.150 anchored on the declared window.
  **A number is not interpretable without the model *and* the window it is of.**
- **The anchor reaches only two of the eight hazard chokepoints** (`campaign.py`, `views/build.py`).
  Projections, lifetime PD and the calibration report are unanchored. Any declared adjustment
  must be threaded through every chokepoint, not bolted onto the two that already carry one.

## Checks that go mute at this scale

A diagnostic that was right on thousands of loans says nothing on 48 million:
- the Kaplan-Meier band is a hundredth of a point wide, so report the **magnitude** of a
  deviation, not in-or-out;
- the crossing test needs an exposure floor and a materiality floor;
- **every p-value is 0.0000**, so the p-value arms of the screen and the elimination are inert;
- an out-of-time A/E cannot be read alone, so publish its in-sample dispersion beside it;
- **the Jeffreys interval on a grade is withdrawn** and replaced by rule 5's 25% relative
  tolerance. The bounds stay as columns with `relative_miss` beside them, because a withdrawn
  criterion that stops being reported cannot be checked. A criterion lives in **two places**,
  `docs/rules.md` and `backtest/metrics.py`, and both have to move together.
