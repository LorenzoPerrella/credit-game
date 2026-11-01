#!/usr/bin/env python3
"""SessionStart: the state of the machine and the repository, as facts, before the first prompt.

Every resumed session here used to begin by checking these by hand -- which branch, what is
uncommitted, whether yesterday's run is still alive -- and once did not, and a `report` started
the previous morning was found 31 hours in. Stated as facts rather than instructions, as the
hook documentation advises, and kept to a few lines.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
