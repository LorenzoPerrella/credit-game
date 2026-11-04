"""What the hooks need to know about the machine: heavy jobs running, and the power source.

Shared by `guard_bash.py` and `session_status.py` so that "a heavy job is running" has one
definition. Standard library only, and every probe degrades to "unknown" rather than raising,
because a hook that crashes blocks every tool call in the session.

Two environment variables are test seams, not configuration: `CREDITSURV_HOOK_PS` names a file
holding `ps -eo pid=,etime=,rss=,args=` output to read instead of the live table, and
`CREDITSURV_HOOK_POWER` set to `battery` or `ac` stands in for `pmset`. The tests in
`tests/test_agent_tooling.py` use them so a verdict does not depend on what the machine is doing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: The `creditsurv` subcommands that hold the machine. Each reads the cell file or the archives
#: and runs for tens of minutes to hours at several GB -- see the `heavy-run` skill for the
#: measured envelope. Two at once on 16 GB is how a `report` came to thrash for 31 hours.
HEAVY: tuple[str, ...] = (
    "aggregate",
    "family",
    "fit",
    "ingest",
    "moratorium",
    "report",
    "select",
    "views",
    "windows",
)

HEAVY_COMMAND = re.compile(r"\bcreditsurv\s+(" + "|".join(HEAVY) + r")\b")

#: Programs whose command line can mention a heavy command without being one: someone looking
#: for the job is not the job.
_LOOKERS = {"grep", "pgrep", "pkill", "ps", "rg", "sed", "awk", "tail", "head", "less", "cat"}


@dataclass(frozen=True)
class Job:
    pid: int
    elapsed: str
    rss_gb: float
    command: str

    def describe(self) -> str:
        """`etime` is DD-HH:MM:SS, which has been read as minutes when it was hours."""
        running = _spelled(self.elapsed)
        return f"pid {self.pid}, running {running}, {self.rss_gb:.2f} GB: {self.command}"


def code(command: str) -> str:
    """The command with heredoc bodies and quoted strings removed: what the shell will run.

    Shared by both Bash hooks. The PostToolUse one read the raw command until 4 November 2026 and
    logged two reads of the test window on a day none was made: a heredoc that *edited a page*
    mentioning `creditsurv windows` counted as running it.

    Without this a commit whose message *mentions* `git add -A` or `prune-archives` is refused,
    and this project's commit messages describe exactly those things. A heredoc body is data;
    a quoted string is an argument. The residue is what can name a program.
    """
    text = re.sub(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2\b", "<<HEREDOC", command, flags=re.S)
    text = re.sub(r"'[^']*'", "''", text)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', text)


def is_heavy(command: str) -> bool:
    """Whether a shell command launches a heavy `creditsurv` subcommand (not its `--help`)."""
    return bool(HEAVY_COMMAND.search(command)) and "--help" not in command


def running_jobs() -> list[Job]:
    """Heavy `creditsurv` processes alive now, one per process (a wrapper and its child both)."""
    text = _process_table()
    jobs = []
    for line in text.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        pid, elapsed, rss, args = parts
        if os.path.basename(args.split()[0]) in _LOOKERS or not HEAVY_COMMAND.search(args):
            continue
        if "guard_bash.py" in args or "session_status.py" in args:
            continue
        try:
            rss_gb = int(rss) / 1024**2
        except ValueError:
            rss_gb = 0.0
        jobs.append(Job(int(pid), elapsed, rss_gb, _shortened(args)))
    return jobs


def on_battery() -> bool | None:
    """True on battery, False on mains, None where it cannot be told (Linux CI, no `pmset`)."""
    forced = os.environ.get("CREDITSURV_HOOK_POWER")
    if forced in {"battery", "ac"}:
        return forced == "battery"
    if shutil.which("pmset") is None:
        return None
    try:
        out = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=2, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if "Battery Power" in out:
        return True
    if "AC Power" in out:
        return False
    return None


def _process_table() -> str:
    fixture = os.environ.get("CREDITSURV_HOOK_PS")
    if fixture:
        return Path(fixture).read_text()
    try:
        return subprocess.run(
            ["ps", "-eo", "pid=,etime=,rss=,args="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _spelled(elapsed: str) -> str:
    """`01-07:20:57` as `1 d 7 h 20 min` -- the format that was misread."""
    days, _, clock = elapsed.rpartition("-")
    fields = [int(part) for part in clock.split(":") if part.isdigit()]
    while len(fields) < 3:
        fields.insert(0, 0)
    hours, minutes, _ = fields
    head = f"{int(days)} d " if days else ""
    return f"{head}{hours} h {minutes} min" if (days or hours) else f"{minutes} min"


def _shortened(args: str, width: int = 120) -> str:
    match = HEAVY_COMMAND.search(args)
    start = max(0, match.start() - 10) if match else 0
    text = args[start:]
    return text if len(text) <= width else text[: width - 3] + "..."
