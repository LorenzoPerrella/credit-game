#!/usr/bin/env python3
"""PostToolUse on Bash: what the machine and the session did, said while it can still be acted on.

Four things the skills used to ask for by hand, each tied to a measured loss:

- **paging**: a heavy job that pages is not computing. `creditsurv report` ran 31 hours at 202
  million swap-ins, about 108,000 a minute, and nothing said so for a day;
- **a kill that left survivors**: a queued runner whose child was killed started the next job, and
  two selections shared the cores for three and a half hours;
- **the test window read twice**: rule 1 reads it once, and nothing recorded whether it had been;
- **a selection of a new specification with no prediction registered** first.

Each is reported as context, never as a refusal. The command has already run, and these are
facts to act on rather than errors. The state lives in `.claude/state/`, per machine and never
committed. Standard library only, and every probe degrades to saying nothing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _jobs import code, is_heavy, running_jobs

PROJECT = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2]))
STATE = PROJECT / ".claude" / "state"

#: Swap-ins a minute above which a heavy job is paging rather than computing. The 31-hour run
#: averaged about 108,000; a healthy fit stays near zero.
PAGING_PER_MINUTE: int = 20_000

#: The commands that read the declared test window (the months after the anchoring window).
_READS_TEST_WINDOW = re.compile(r"\bcreditsurv\s+(views|windows|score)\b")


def _swap_ins() -> int | None:
    forced = os.environ.get("CREDITSURV_HOOK_SWAPINS")
    if forced is not None:
        return int(forced)
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"Swapins:\s+(\d+)", out.stdout)
    return int(match.group(1)) if match else None


def _load(name: str) -> dict[str, object]:
    try:
        return dict(json.loads((STATE / name).read_text()))
    except (OSError, ValueError, TypeError):
        return {}


def _save(name: str, value: dict[str, object]) -> None:
    with contextlib.suppress(OSError):
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / name).write_text(json.dumps(value))


def paging(now: float) -> str | None:
    swapins = _swap_ins()
    if swapins is None:
        return None
    previous = _load("swap.json")
    _save("swap.json", {"time": now, "swapins": swapins})
    if "time" not in previous:
        return None
    minutes = (now - float(previous["time"])) / 60.0  # type: ignore[arg-type]
    if minutes < 0.5:
        return None
    rate = (swapins - int(previous["swapins"])) / minutes  # type: ignore[call-overload]
    # The process table only when the rate already says something is wrong: `ps` is most of
    # this hook's cost, and it runs after every Bash call.
    if rate < PAGING_PER_MINUTE:
        return None
    jobs = running_jobs()
    if not jobs:
        return None
    listed = "; ".join(job.describe() for job in jobs)
    return (
        f"The machine is paging: {rate:,.0f} swap-ins a minute over the last {minutes:.0f} min "
        f"while a heavy job runs ({listed}). A job that pages is not computing, and waiting does "
        "not fix it: the 31-hour report averaged about 108,000 a minute. Diagnose with `sample` "
        "and `vm_stat`, then kill it (the heavy-run skill)."
    )


def survivors(command: str) -> str | None:
    if not re.search(r"\b(kill|pkill)\b", command):
        return None
    jobs = running_jobs()
    if not jobs:
        return None
    return (
        "After the kill, heavy creditsurv processes are still alive: "
        + "; ".join(job.describe() for job in jobs)
        + ". A queued runner starts the next command when its child dies; verify with "
        "`pgrep -f creditsurv` until it is empty."
    )


def _specification_hash() -> str:
    digest = hashlib.sha256()
    for path in ("src/creditsurv/config.py", "docs/reports/selection.json"):
        try:
            digest.update((PROJECT / path).read_bytes())
        except OSError:
            continue
    return digest.hexdigest()[:12]


def test_window(command: str, now: float) -> str | None:
    match = _READS_TEST_WINDOW.search(command)
    if not match or "--help" in command:
        return None
    specification = _specification_hash()
    log = STATE / "test-window-reads.log"
    earlier: list[str] = []
    with contextlib.suppress(OSError):
        earlier = [
            line for line in log.read_text().splitlines() if f" {specification} " in f" {line} "
        ]
    with contextlib.suppress(OSError):
        STATE.mkdir(parents=True, exist_ok=True)
        with log.open("a") as handle:
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(now))
            handle.write(f"{stamp} {specification} {match.group(1)}\n")
    if not earlier:
        return None
    return (
        f"The test window has already been read for this specification ({specification}): "
        + "; ".join(earlier[-3:])
        + ". Rule 1 reads it once. If this is a re-read of the same result, say so; if the "
        "specification changed, the register should record why before the window is read again."
    )


def unpredicted_selection(command: str) -> str | None:
    """A selection launched while config names a formula the record does not, with no new entry.

    A warning only: config changes for reasons other than a new form. But a new specification
    selected before its prediction is in the register is the order rule 16's process forbids.
    """
    if not re.search(r"\bcreditsurv\s+select\b", command) or "--help" in command:
        return None
    try:
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "src/creditsurv/config.py",
                "docs/rules_register.md",
            ],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    config_changed = "config.py" in status
    register_changed = "rules_register.md" in status
    if config_changed and not register_changed:
        return (
            "A selection was launched with src/creditsurv/config.py changed and the register "
            "untouched. If this is a new specification, its prediction goes in "
            "docs/rules_register.md before the run (the rule-change skill)."
        )
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        # What the shell ran, not what it was handed as data: a heredoc or a quoted message that
        # mentions a command is not that command.
        command = code(str(payload.get("tool_input", {}).get("command", "")))
    except (ValueError, AttributeError):
        return 0
    now = float(os.environ.get("CREDITSURV_HOOK_NOW", time.time()))
    notes = [
        note
        for note in (
            paging(now),
            survivors(command),
            test_window(command, now) if is_heavy(command) or "score" in command else None,
            unpredicted_selection(command),
        )
        if note
    ]
    if notes:
        json.dump(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": "\n\n".join(notes),
                }
            },
            sys.stdout,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
