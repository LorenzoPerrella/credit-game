#!/usr/bin/env python3
"""PreToolUse guard for Bash: the rules this project used to state and then break.

Each check below is a rule that `CLAUDE.md` wrote in prose and that was broken anyway, with what
it cost. Prose is read; a hook is enforced. A refused command gets a reason that names the safe
alternative, because a refusal Claude cannot act on is just a stall.

Reads the PreToolUse JSON on stdin and, to refuse, prints a `permissionDecision: "deny"` and exits
0. Anything it cannot parse it lets through: a hook that fails closed on its own bug would block
every command in the session.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _jobs import is_heavy, on_battery, running_jobs

PROJECT = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2]))


def verdict(command: str) -> str | None:
    """The reason to refuse `command`, or None to let it run."""
    code = _code(command)
    for check in (_irreversible, _interpreter_swap, _blanket_add, _heavy, _commit_gate):
        reason = check(code)
        if reason:
            return reason
    return None


def _code(command: str) -> str:
    """The command with heredoc bodies and quoted strings removed: what the shell will run.

    Without this a commit whose message *mentions* `git add -A` or `prune-archives` is refused,
    and this project's commit messages describe exactly those things. A heredoc body is data;
    a quoted string is an argument. The residue is what can name a program.
    """
    text = re.sub(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2\b", "<<HEREDOC", command, flags=re.S)
    text = re.sub(r"'[^']*'", "''", text)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', text)


def _irreversible(command: str) -> str | None:
    if re.search(r"\bcreditsurv\s+prune-archives\b", command):
        return (
            "`creditsurv prune-archives` deletes the 40 GB of downloads and cannot be undone. "
            "It is never run by Claude: if it is wanted, the user runs it themselves with `!`."
        )
    return None


def _interpreter_swap(command: str) -> str | None:
    """Re-syncing `.venv` to another Python while a suite ran produced 13 false failures."""
    syncs = re.search(r"\buv\s+(sync|venv)\b", command)
    other = re.search(r"\bUV_PYTHON=|--python\b", command)
    isolated = "UV_PROJECT_ENVIRONMENT=" in command
    if syncs and other and not isolated:
        return (
            "This re-syncs the project's `.venv` to another interpreter, which swaps it under any "
            "suite or job already running. For the 3.11 mypy leg use the isolated environment: "
            "`UV_PROJECT_ENVIRONMENT=.venv-py311 uv run -q --python 3.11 --extra kernel "
            "--group docs mypy` (or the `verify` skill)."
        )
    return None


def _blanket_add(command: str) -> str | None:
    """`git add -A` once committed a subagent's scratch test file."""
    if re.search(r"\bgit\s+add\s+(-A|--all|\.)(\s|$)", command) or re.search(
        r"\bgit\s+commit\s+(-[a-zA-Z]*a[a-zA-Z]*|--all)\b", command
    ):
        return (
            "Stage named paths, not everything: agents and scratch work leave files in the tree, "
            "and `git add -A` / `git commit -a` committed one of them. Run `git status --short` "
            "and `git add <path> ...`."
        )
    return None


def _heavy(command: str) -> str | None:
    if not is_heavy(command):
        return None
    jobs = running_jobs()
    if jobs:
        listed = "; ".join(job.describe() for job in jobs)
        return (
            "A heavy creditsurv job is already running, and two at once on 16 GB is how a "
            f"`report` thrashed for 31 hours. Running: {listed}. Wait for it, or verify and kill "
            "it first (see the `heavy-run` skill)."
        )
    if on_battery():
        return (
            "The machine is on battery, where a long run progresses at 1-2% duty cycle even "
            "under `caffeinate`. Plug in before launching it."
        )
    return None


def _commit_gate(command: str) -> str | None:
    """Lint, format and types before a commit: seconds, where CI found them in minutes."""
    if not re.search(r"\bgit\s+commit\b", command) or os.environ.get("CREDITSURV_HOOK_SKIP_GATE"):
        return None
    tools = PROJECT / ".venv" / "bin"
    checks = [
        ([str(tools / "ruff"), "check", "."], "ruff check"),
        ([str(tools / "ruff"), "format", "--check", "."], "ruff format --check"),
        ([str(tools / "mypy")], "mypy"),
    ]
    failed = []
    for argv, name in checks:
        if not Path(argv[0]).exists():
            continue
        try:
            result = subprocess.run(
                argv, cwd=PROJECT, capture_output=True, text=True, timeout=240, check=False
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode != 0:
            tail = (result.stdout + result.stderr).strip().splitlines()[-6:]
            failed.append(f"{name}:\n" + "\n".join(tail))
    if failed:
        return (
            "The commit gate failed, so CI would too. Fix and commit again "
            "(`uv run ruff format . && uv run ruff check --fix .` handles most of it).\n\n"
            + "\n\n".join(failed)
        )
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        command = str(payload.get("tool_input", {}).get("command", ""))
    except (ValueError, AttributeError):
        return 0
    reason = verdict(command)
    if reason:
        json.dump(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            },
            sys.stdout,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
