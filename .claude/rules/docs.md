---
paths:
  - "docs/**"
  - "mkdocs.yml"
  - "src/creditsurv/site/**"
  - "tests/test_site.py"
---

# The documentation site

- **Figures, tables and numbers are placed, never typed.** A page writes a placeholder comment
  (`figure`, `table` or `value`, then a name). `creditsurv.site.hooks` fills it at build time from
  `docs/tables`, which are aggregates `creditsurv views` computes locally and which are committed,
  because CI cannot read the data. The build fails on a placeholder naming nothing, on a view the
  manifest lacks, and on model views from more than one fit.
- **The hook fills placeholders even inside inline code** (it skips only fenced blocks). To
  mention one in prose, elide the name, for example with `…`, which the pattern does not match.
- **A generated report states the formula it was run on, or says it is superseded.** It carries
  `Formula:`, `Shape:` and `Generated:`. While a new specification is unscored, the honest state is
  the notice `tests/test_site.py` looks for (`SUPERSEDED_MARKER`) in the report's first lines.
  Regenerating is the fix once the specification is settled.
- **The nav in `mkdocs.yml` is explicit.** A new page needs an entry.
- **`docs/rules_register.md` is append-only in spirit.** Corrections go in as new dated entries
  that say what they supersede. A withdrawn rule or prediction stays visible under a notice.
- Section pages (`index`, `data`, `portfolio`, `methodology`, `model`, `calibration`,
  `validation`, `decisions`, `reproduce`) summarise the long documents and the generated reports
  under `docs/reports/`.
