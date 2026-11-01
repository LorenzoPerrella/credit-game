---
paths:
  - "src/creditsurv/data/**"
  - "src/creditsurv/features.py"
  - "src/creditsurv/profiling.py"
  - "tests/fixtures.py"
  - "tests/test_aggregate.py"
  - "tests/test_ingest.py"
  - "tests/test_panel.py"
  - "tests/test_features.py"
---

# The data path: rules that are silent when broken

Each of these failed once without an error. The full stories, with counts, are in
`docs/development.md`.

- **A loan without a debt-to-income is a HARP refinance, and nothing else.** The cells keep the
  ratio missing; the model fills it with a constant that the `harp` level absorbs. That is the
  dummy-variable adjustment, not an imputation. `features.absorb_not_reported` raises rather than
  fit the filled ratio without the level (rule 8).
- **The payment state is the month before, never the month itself**: at three missed payments the
  loan *has* defaulted. Lag the **raw code**, because `RA` would otherwise cast to the same NULL as
  "first month".
- **What ended a loan-month has one spelling: the three-state `outcome`**, with default
  precedence. Two booleans used to disagree on 52,357 loan-months. A caller that wants a boolean
  goes through `ended_in`.
- **A moratorium is not a default.** The event definition is a `MoratoriumPolicy`. Each policy
  writes its own `cells_<policy>.parquet` and a fit's fingerprint names its policy. `exclude` was
  chosen by a rule written before either fit.
- **Every macro series is lagged three months**, market quotes included.
- **The weight is a count of loan-months, never an amount.** PD is per obligor.
- **Categorical mappings have no `ELSE` branch.** An unmapped code becomes NULL and the loan is
  dropped. Decide a mapping from a distinct-and-count across vintages, never from the layout
  spreadsheet: `9`/`99` are "not available", not categories.
- **Truncation orders by calendar period, not by loan age**, because `loan_age` restarts at a
  modification. Ordering by age once lost 641 real defaults.
- **Missing-value sentinels are real numbers**: 9999 for credit score, 999 for LTV, DTI, ELTV and
  the MI percentage. Left in, the 2006 vintage's median ELTV is 999.
- **What the cell key holds is decided by a measured cost against a declared ceiling** (rule 7,
  `creditsurv profile --extensions`). A nine-quarter projection ran 14% light. The key stops where
  it is: finer bands and the origination spread are both over the ceiling before the margin.
- **A macro covariate costs zero cells; a loan covariate multiplies the table.** Measure it before
  adding it.
- **A window is read, not filtered.** `load_cells_window` evaluates the observation month in
  DuckDB, because `origination_month + age` is not a column parquet can be asked for. By contrast,
  `CellBlocks(months=...)` scans every batch of the file and filters each one.
- **The projection book is a declared window** (`BOOK_WINDOW_MONTHS`): every origination of the
  year, at its weight. **Dating a book takes three columns**: `origination_book` once set `period`
  and left `origination_period`, and the book showed five vintage eras in one month.
- **The exact row merge belongs to a rebuild**, which can sort by quarter: 1.36x sorted against 1.11x
  unsorted. It is not worth a rebuild of its own.
- **Aggregation peaks at ~11.3 GB.** Nothing else heavy beside it.
- **`creditsurv prune-archives` is irreversible** (40 GB). The hook refuses it, and only the user
  runs it. `creditsurv prune-encodings` only sweeps orphaned readings.
