# Working on credit-game

Lifetime PD with parametric survival models, fitted on the whole Freddie Mac book. Read
[README.md](README.md) for what it does.

This file holds only what applies to **every** task. The rest is placed where it is needed:

| what | where | loads |
|---|---|---|
| rules for one part of the code | `.claude/rules/` (`engine`, `data`, `views-and-backtest`, `selection`, `docs`) | when Claude reads a file that rule's `paths` match |
| procedures | skills `verify`, `heavy-run`, `rule-change`, `measure-a-fit`, `ship` | when invoked, or when a description matches the task |
| rules that were broken in prose | hooks in `.claude/hooks/`, wired in `.claude/settings.json` | on every Bash call and at session start |
| the evidence behind all of it | `docs/development.md`, `docs/reports/engine.md`, `docs/rules_register.md` | when read |

## The scale

**91.6 million cells over 2.774 billion loan-months**, 1,671,207 defaults and 33,797,300
prepayments under the `exclude` policy, on a 16 GB laptop. Memory decides what can be fitted, and
time decides what is worth fitting.

- **One heavy job at a time, on mains power.** The hook enforces both. Two at once, or the panel
  held in memory, is how a `report` thrashed for 31 hours.
- **Nothing holds the panel.** Every command reads the cell file a batch at a time. Before writing
  anything that expands rows, read the `heavy-run` skill.

## How a result is allowed to count

- **Rules are declared before the fit they judge.** The form and the criterion go in
  `docs/rules.md`, and the prediction goes in `docs/rules_register.md` **before** the run. The test
  window is read **once**, and whatever comes out is published. The `rule-change` skill has the
  procedure, and the test for when an amendment made after a result is admissible.
- **Nothing is tuned on a test window.** "The criteria were fixed before the run, so the model is
  not to be tuned to them on the test window."
- **Only aggregates leave the machine.** Loan-level data is never published or committed.
- **The specification is an output.** `creditsurv select` writes `docs/reports/selection.json`,
  and `config` follows it; `tests/test_procedure.py` fails when they part.
- **A number is not interpretable without the model *and* the window it is of.** Quote a fit's
  hash, its `sample` and its window beside every figure.
- **Verify a claim against the code that computes it, not against the algebra.** A coherent fit was
  refused on an algebraic argument that one call to `episode_hazards` would have refuted.
- **`nmds` is the methodological reference.** A departure from it is stated and argued, never made
  silently.

## Identification

`period = cohort + age` holds **identically**, so no two of calendar time, vintage and loan age can
be held fixed while the third moves. The baseline hazard's shape is identified only because
calendar time enters through a few macro covariates rather than as a free period effect. So a
covariate-free diagnostic of the hazard shape answers a different question, and a covariate
recomputed every loan-month moves with the age.

## Checks that go mute at this scale

A diagnostic that was right on thousands of loans can say nothing on 48 million: confidence bands a
hundredth of a point wide, every p-value 0.0000, an interval on a grade that tests the arithmetic of
the average. Report the **magnitude** of a deviation, and give out-of-time ratios with their
in-sample dispersion beside them. A marginal relationship shows that a covariate correlates with
the outcome, never that it is identified in a model.

## Irreversible

`creditsurv prune-archives` deletes the 40 GB of downloads. It is a separate command, never
appended to the ingest, and never run without the user asking. The hook refuses it, and the user
runs it themselves if they mean it.

## Conventions

- **Verification before a push or a merge is the `verify` skill**:
  - ruff and the formatter;
  - mypy on 3.12, and on 3.11 in the isolated `.venv-py311`, **never** by re-syncing `.venv`;
  - `pytest -m "not network"`;
  - `mkdocs build --strict`.

  The commit hook runs ruff, the formatter and mypy, and refuses a red commit.
- **Comments say why, with the measurement that settled it.** Commit messages carry the number that
  justified the change.
- **Notebooks carry evidence, not logic.** Every statistic is a tested function in the package. A
  scratch script that computes something worth keeping is a function that has not been written yet.
- **Tests use fixtures in Freddie Mac's own pipe-delimited format**, so every test that needs data
  also exercises the loader. There is no synthetic panel.
- **Stage named paths.** Agents and scratch work leave files in the tree.
- **One CI workflow** (`.github/workflows/ci.yml`), with publishing jobs that wait on the tests.
  **No new top-level folder** for one file: look for where it naturally belongs.
- Repository docs, code comments, commit messages and PR descriptions are in English.

## Where the state is

- **The open findings and the current verdicts** are in `docs/validation.md`.
- **The latest decisions and measurements** are the newest entries of `docs/rules_register.md`.
- The previous `CLAUDE.md` kept a section called "Open", which became stale state. It is preserved,
  dated, in `docs/development.md`.
