---
name: ship
description: Commit, push, open and merge pull requests in credit-game the way this repository writes them -- a commit message that carries the number that justified the change, a PR description for a reviewer, attribution lines, merge on green CI. Use when committing, opening or updating a PR, or merging.
---

# Shipping

## Commits

- **Stage named paths.** `git add -A`, `git add .` and `git commit -a` are refused by the hook,
  because agents and scratch work leave files in the tree. Check `git status --short` first.
- **The commit gate** (PreToolUse) runs ruff, the formatter and mypy, and refuses a red commit. The
  full suite and the 3.11 leg are the `verify` skill's job, before pushing.
- **The message carries the number that justified the change**: the measurement, the before and
  after, the count of what was wrong. "fix the probe" is not a message; "the probe read the
  first year 1.536 high where maximum likelihood reads 0.657 low" is.
- Write it in English, with the subject as the finding rather than the action where possible.
- End it with the attribution line the session's system reminder gives (`Co-Authored-By: ...`).
- Pass long messages through a heredoc (`git commit -q -F - <<'EOF'`). The hook reads only the
  command outside heredocs and quotes, so a message describing a refused command is fine.

## Pull requests

- The description is for a reviewer who was not in the session. Say what changed and the
  measurement behind it, what is deliberately not in it, and how it was verified. Put a table
  where the content is a table.
- End it with the line the system reminder gives (`🤖 Generated with [Claude Code](...)`).
- **A branch whose specification is not yet scored is pushed, not merged.** Configuration on `main`
  must name a model that has met its declared criteria, or carry superseded notices that say it has
  not.
- When a PR's premise changes mid-flight, rewrite its description to carry the correction. Keep
  both commits, so the correction is readable.

## Merging

- Merge when CI is green on the PR: quality on 3.11 and 3.12, and the site build. The end-to-end
  pipeline does not run on PRs (schedule and `workflow_dispatch` only), so dispatch it for a branch
  that touches the CLI: `gh workflow run ci.yml --ref <branch>`.
- The user's standing instruction is to merge PRs as they become ready.
- Squash-merge, keep the branch until its follow-up is done, then delete it.
