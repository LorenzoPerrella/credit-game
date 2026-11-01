---
name: measure-a-fit
description: Compute an acceptance criterion or a table for a fit that is already in the credit-game cache -- out-of-sample A/E by age band, decile or year, anchored on the declared window -- without refitting. Use when a criterion is owed on a cached fit, and before quoting any number that is said to be "of the model".
---

# Measuring a cached fit

## 1. Find the fit, and make sure it is the one you mean

```python
from creditsurv.data.store import find_fits, load_fit

for fingerprint, described in find_fits():
    ...  # read described["formula"], ["ancillary"], ["sample"], ["purpose"], ["cells"], ["as_of"]
```

- **`purpose`**: a selection fit carries `purpose: "selection"`, a report fit does not. `views`
  asks for `purpose=None`, meaning the key must be **absent**. So `views` will not score a
  selection fit, and that is deliberate.
- **`sample`**: a selection writes several fits of the same formula, differing only here:
  `training half`, `even/odd origination years`, `to 2010-06`, `after 2010-06`. **Only `training
  half` is the model.** A band coefficient table was quoted from `after 2010-06` once, and it
  understated a violation two boundaries to four.
- **`cells`**: the table identity, name:size:mtime. A fit on a previous mtime belongs to another
  table, whatever its formula.
- The rows a training-half selection fit uses are the estimation window to `as_of`: the same
  72,671,500 cells `report` fits. So scoring that fit is legitimate when no report fit exists yet.

## 2. Score it without holding the panel

Until `creditsurv score` exists, follow the streamed pattern:
- `creditsurv.data.panel.CellBlocks(path, macro, covariates, rows=250_000, months=(first, last),
  model_only=False).prepared()` yields episodes a batch at a time. `model_only=False` keeps the
  `age` column the segmenters need, and `months` filters on the **observation** month.
- per batch, `backtest.runner.predicted_hazard(fitted, frame, covariates)`, then
  `views.calibration.exposure_totals(frame, hazard, {"age_band": views.segments.age_bands(frame)})`;
- sum the batch tables. Expected defaults are linear in the hazard, so the anchoring multiplier
  can be applied to the summed expectation at the end.

`CellBlocks` scans every batch of the file and filters each one, so a window costs a full scan,
~2.5-3 h through `episode_hazards`. Validators scored all 91.6 M cells directly from the
coefficients in DuckDB in about 30 s: that is the fast path to build, and it has to be validated
against `episode_hazards`, the normative implementation.

## 3. The windows, which are declared and not chosen

- **Anchoring**: `models.anchoring.ANCHOR_WINDOW = ("2022-01", "2024-12")`. It sets the level and
  nothing else. The multiplier is actual over expected there, bounded to [1/4, 4].
- **Test**: the months after the anchoring window, which neither the fit nor the anchor saw.
- A table "named for a criterion has to evaluate that criterion". `acceptance_by_segment` once
  scored from 2022-01, anchoring and test windows together and unanchored, and the same decile
  was reported three ways in one day: 1.676 in sample, 1.456 on that window, 1.150 on the declared
  one.

## 4. Report it with its provenance

Every figure goes out with the fit's hash, its `sample`, the window, the anchoring multiplier and
the batch count, and is compared against the prediction registered **before** it was read (see
the `rule-change` skill).
