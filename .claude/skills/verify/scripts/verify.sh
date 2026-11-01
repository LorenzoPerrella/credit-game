#!/usr/bin/env bash
# The per-commit checklist, in the order that fails fastest, stopping at the first red step.
#
# The 3.11 leg runs in its own environment (.venv-py311) so the project's .venv is never re-synced
# underneath a suite or a job: that swap once produced 13 false failures mid-run.
set -euo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"

if pgrep -f "pytest" >/dev/null 2>&1; then
  echo "A pytest run is already going; two at once read each other's temporary files. Wait for it."
  exit 1
fi

step() {
  local name=$1
  shift
  printf '\n== %s\n' "$name"
  local started=$SECONDS
  "$@"
  printf -- '-- %s: ok in %ss\n' "$name" "$((SECONDS - started))"
}

step "ruff check" uv run --no-sync ruff check .
step "ruff format --check" uv run --no-sync ruff format --check .
step "mypy, Python 3.12" uv run --no-sync mypy
step "mypy, Python 3.11 (isolated environment)" \
  env UV_PROJECT_ENVIRONMENT=.venv-py311 uv run -q --python 3.11 --extra kernel --group docs mypy
if [[ "${1:-}" != "--no-tests" ]]; then
  step "pytest" uv run --no-sync pytest -m "not network" -q
fi
step "mkdocs build --strict" uv run --no-sync mkdocs build --strict
printf '\n== all green\n'
