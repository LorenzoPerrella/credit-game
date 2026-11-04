#!/usr/bin/env python3
"""SessionStart: the state of the machine and the repository, as facts, before the first prompt.

Every resumed session here used to begin by checking these by hand -- which branch, what is
uncommitted, whether yesterday's run is still alive -- and once did not, and a `report` started
the previous morning was found 31 hours in. Stated as facts rather than instructions, as the
hook documentation advises, and kept to a few lines.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _jobs import on_battery, running_jobs

PROJECT = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2]))


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT, capture_output=True, text=True, timeout=3, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _swap() -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=2, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip() or None


def _newest_register_entry() -> str | None:
    """The last `###` heading of the register: where the method last stood."""
    try:
        headings = [
            line[4:].strip()
            for line in (PROJECT / "docs" / "rules_register.md").read_text().splitlines()
            if line.startswith("### ")
        ]
    except OSError:
        return None
    return headings[-1] if headings else None


def _scheduled_ci() -> str | None:
    """The last scheduled CI run on main, cached for a day.

    The end-to-end job failed every day from 29 October to 1 November on a removed option, and
    nothing in a session said so. One `gh` call a day, two seconds at most, and silence if it
    cannot be made.
    """
    cache = PROJECT / ".claude" / "state" / "scheduled-ci.json"
    today = time.strftime("%Y-%m-%d")
    try:
        cached = json.loads(cache.read_text())
        if cached.get("day") == today:
            return str(cached["line"]) if cached.get("line") else None
    except (OSError, ValueError, KeyError):
        pass
    if shutil.which("gh") is None or os.environ.get("CREDITSURV_HOOK_OFFLINE"):
        return None
    try:
        out = subprocess.run(
            [
                "gh",
                "run",
                "list",
                "--workflow",
                "ci.yml",
                "--event",
                "schedule",
                "--limit",
                "1",
                "--json",
                "conclusion,createdAt,url",
            ],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout
        runs = json.loads(out or "[]")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    line = None
    if runs:
        run = runs[0]
        verdict = run.get("conclusion") or "in progress"
        line = f"Last scheduled CI on main ({run.get('createdAt', '')[:10]}): {verdict}."
        if verdict == "failure":
            line += f" It is failing: {run.get('url', '')}."
    with contextlib.suppress(OSError):
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"day": today, "line": line}))
    return line


def status() -> str:
    lines = []
    branch = _git("branch", "--show-current") or "(detached)"
    counts = _git("rev-list", "--left-right", "--count", "main...HEAD").split()
    behind_ahead = f", {counts[1]} ahead and {counts[0]} behind main" if len(counts) == 2 else ""
    dirty = [line for line in _git("status", "--short").splitlines() if line.strip()]
    lines.append(f"Branch {branch}{behind_ahead}; {len(dirty)} uncommitted path(s).")

    jobs = running_jobs()
    if jobs:
        lines.append(
            "Heavy creditsurv jobs alive: " + "; ".join(job.describe() for job in jobs) + "."
        )
    else:
        lines.append("No heavy creditsurv job is running.")

    recent = _git("log", "--oneline", "-3")
    if recent:
        lines.append("Last commits: " + " | ".join(recent.splitlines()) + ".")
    entry = _newest_register_entry()
    if entry:
        lines.append(f"Newest register entry: {entry}.")
    scheduled = _scheduled_ci()
    if scheduled:
        lines.append(scheduled)

    battery = on_battery()
    if battery is not None:
        lines.append("Power: battery." if battery else "Power: mains.")
    swap = _swap()
    if swap:
        lines.append(f"Swap: {swap}.")
    return " ".join(lines)


def main() -> int:
    json.dump(
        {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": status()}},
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
