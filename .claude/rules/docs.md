---
paths:
  - "docs/**"
  - "mkdocs.yml"
  - "src/creditsurv/site/**"
  - "tests/test_site.py"
---

# The documentation site

- **Figures, tables and numbers are placed, never typed.** A page writes a placeholder comment
  (`<!-- value: name -->`, or `figure`, or `table`). `creditsurv.site.hooks` fills it at build time
  from `docs/tables`, which are aggregates `creditsurv views` computes locally and which are
  committed, because CI cannot read the data. The build fails on a placeholder naming nothing, on
  a view the manifest lacks, and on model views from more than one fit.
- **A figure a view cannot place is recorded with its date**, between `<!-- recorded: YYYY-MM-DD -->`
  and `<!-- /recorded -->`, and the site shows the date. `tests/test_docs_provenance.py` fails on
  any other precise number on a section page, and a PostToolUse hook says so when one is written.
  Before recording a number, check that it is still of the configured model: a dated stale number
  is still stale.
- **A generated report states the formula it was run on, or says it is superseded.** It carries
  `Formula:`, `Shape:` and `Generated:`. While a new specification is unscored, the honest state is
  the notice `tests/test_site.py` looks for (`SUPERSEDED_MARKER`) in the report's first lines.
  Regenerating is the fix once the specification is settled.
- **`docs/rules_register.md` is append-only** (the policy check refuses a changed line).
  Corrections are new dated entries that say what they supersede.
- Section pages (`index`, `data`, `portfolio`, `methodology`, `model`, `calibration`,
  `validation`, `decisions`, `reproduce`) summarise the long documents and the generated reports
  under `docs/reports/`.
