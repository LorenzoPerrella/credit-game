---
name: verify
description: Run the full pre-merge checklist for credit-game -- ruff, format, mypy on 3.12 and on 3.11 in an isolated environment, the non-network suite and the strict docs build -- in the order that fails fastest. Use before pushing a branch or opening or merging a PR, and after any change that touches src/, tests/ or docs/.
allowed-tools: Bash(${CLAUDE_SKILL_DIR}/scripts/verify.sh *), Bash(bash ${CLAUDE_SKILL_DIR}/scripts/verify.sh *)
---

# Verify

Run the bundled script and report each step's outcome:

```
bash ${CLAUDE_SKILL_DIR}/scripts/verify.sh            # everything, ~7 minutes
bash ${CLAUDE_SKILL_DIR}/scripts/verify.sh --no-tests # everything but pytest, ~1 minute
```

It stops at the first red step. Fix that, then rerun. When everything is green on a clean tree
it writes `.claude/state/verified` with HEAD, which the hook requires before `gh pr create` or a
push to a branch with an open PR -- so the suite is not optional before publishing.

## What it checks, and why each step is there

| step | cost | why it is not optional |
|---|---|---|
| `ruff check`, `ruff format --check` | seconds | CI runs both; the commit hook refuses a commit with either red |
| `mypy` on 3.12 | ~30 s | `tool.mypy` targets 3.12 |
| `mypy` on 3.11, in `.venv-py311` | ~1 min cold | numpy's stubs infer differently under 3.11: **twice** a branch went green locally and red in CI on `no-any-return` alone |
| `pytest -m "not network"` | ~5.5 min | about 700 tests; two runs at once read each other's temporary files, so the script refuses to start beside one |
| `mkdocs build --strict` | ~6 s | the site fails on a placeholder naming nothing or a view the manifest lacks |

## Rules the script encodes

- **Never re-sync `.venv` to another Python to check 3.11.** That swap underneath a running suite
  produced 13 false failures. The isolated environment is `UV_PROJECT_ENVIRONMENT=.venv-py311`, and
  the PreToolUse hook refuses the old `UV_PYTHON=3.11 uv sync`.
- **`uv sync --group docs --extra kernel`** is what the main environment needs: a bare `uv sync`
  drops the docs group and the build then fails on `pymdownx`, which looks like a broken site and
  is a missing dependency.

## Tests that are red for a reason, not a defect

- `tests/test_site.py::test_the_views_manifest_and_the_configured_specification_name_one_model` is
  **local only** (it skips where the fit cache is absent, which is CI). Red means the committed
  views are of a fit other than the configured formula's: run `creditsurv views` after
  `creditsurv report`, or accept it on a branch whose specification has not been scored yet.
- `test_a_generated_report_states_the_formula_it_was_run_on_or_says_it_is_superseded` is red when
  a report names a formula the configuration has replaced. Regenerate it, or -- while the new
  specification is unscored -- give it the superseded notice the test looks for.

## CI is a superset

CI runs every check here (`tests/test_agent_tooling.py` fails if this script gains one CI lacks),
plus the `policy` job and the end-to-end pipeline on every PR.
