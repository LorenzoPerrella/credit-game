"""The agent tooling under `.claude/`: hooks that refuse, and the configuration that wires them.

A hook is logic, and logic in this project is tested. These run each script the way Claude Code
does -- JSON on stdin, a verdict on stdout -- with the process table and the power source injected
through two test seams, so a verdict does not depend on what the machine happens to be running.
Every refusal below is a rule `CLAUDE.md` once stated in prose and that was broken anyway.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".claude" / "hooks"


def _run(script: str, payload: object, **env: str) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "CREDITSURV_HOOK_SKIP_GATE": "1", **env}
    return subprocess.run(
        [sys.executable, str(HOOKS / script)],
        input=payload if isinstance(payload, str) else json.dumps(payload),
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
        check=False,
    )


def _reason(command: str, tmp_path: Path, ps: str = "", power: str = "ac") -> str | None:
    table = tmp_path / "ps.txt"
    table.write_text(ps)
    result = _run(
        "guard_bash.py",
        {"tool_name": "Bash", "tool_input": {"command": command}},
        CREDITSURV_HOOK_PS=str(table),
        CREDITSURV_HOOK_POWER=power,
    )
    assert result.returncode == 0, result.stderr
    if not result.stdout.strip():
        return None
    decision = json.loads(result.stdout)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    return str(decision["permissionDecisionReason"])


@pytest.mark.parametrize(
    ("command", "names"),
    [
        ("uv run creditsurv prune-archives", "!"),
        ("UV_PYTHON=3.11 uv sync -q", ".venv-py311"),
        ("uv sync --python 3.11", ".venv-py311"),
        ("git add -A", "git add <path>"),
        ("git add .", "git add <path>"),
        ("git commit -am 'wip'", "git add <path>"),
    ],
)
def test_a_rule_that_was_broken_in_prose_is_refused_with_its_alternative(
    command: str, names: str, tmp_path: Path
) -> None:
    reason = _reason(command, tmp_path)
    assert reason is not None, f"{command!r} was let through"
    assert names in reason, f"the refusal of {command!r} does not name the way out"


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "git add src/creditsurv/config.py tests/test_baseline.py",
        "git add ./src/creditsurv/config.py",
        "UV_PROJECT_ENVIRONMENT=.venv-py311 uv run --python 3.11 mypy",
        "uv run creditsurv report --help",
        'pgrep -f "creditsurv report"',
        "caffeinate -dimsu uv run creditsurv report --no-extra-fits > log 2>&1",
        # A commit message describing the rules is data, not a command.
        "git commit -q -F - <<'EOF'\nnever git add -A, never creditsurv prune-archives\nEOF",
        'git commit -m "explain why git add -A is refused"',
    ],
)
def test_ordinary_work_is_let_through(command: str, tmp_path: Path) -> None:
    assert _reason(command, tmp_path) is None, f"{command!r} was refused"


def test_a_second_heavy_job_is_refused_and_the_first_is_described(tmp_path: Path) -> None:
    """`etime` is DD-HH:MM:SS, and it was once read as minutes when it was 31 hours."""
    ps = (
        "  4079  01-07:20:57 7599660 /u/.venv/bin/python /u/.venv/bin/creditsurv report\n"
        "  5000        00:03    1200 grep creditsurv report\n"
    )
    reason = _reason("uv run creditsurv select", tmp_path, ps=ps)
    assert reason is not None
    assert "pid 4079" in reason
    assert "1 d 7 h 20 min" in reason
    assert "7.25 GB" in reason
    assert "pid 5000" not in reason, "someone grepping for the job is not the job"


def test_a_heavy_job_on_battery_is_refused(tmp_path: Path) -> None:
    assert _reason("uv run creditsurv views", tmp_path, power="battery") is not None
    assert _reason("uv run creditsurv views", tmp_path, power="ac") is None


def test_a_hook_that_cannot_parse_its_input_lets_the_command_through() -> None:
    """Failing closed on its own bug would block every command in the session."""
    result = _run("guard_bash.py", "not json")
    assert result.returncode == 0
    assert result.stdout == ""


def test_the_session_starts_with_the_state_of_the_machine() -> None:
    result = _run("session_status.py", {"hook_event_name": "SessionStart"})
    assert result.returncode == 0, result.stderr
    context = json.loads(result.stdout)["hookSpecificOutput"]
    assert context["hookEventName"] == "SessionStart"
    assert "Branch" in context["additionalContext"]
    assert "heavy creditsurv job" in context["additionalContext"]


def test_every_hook_the_settings_name_exists_and_the_logs_are_unbuffered() -> None:
    """A renamed script is a hook that silently stops running."""
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text())
    named = [
        argument.replace("${CLAUDE_PROJECT_DIR}", str(ROOT))
        for groups in settings["hooks"].values()
        for group in groups
        for hook in group["hooks"]
        for argument in hook.get("args", [])
        if argument.endswith(".py")
    ]
    assert named, "no hook script is configured"
    for script in named:
        assert Path(script).is_file(), f"{script} is configured and missing"
    # A buffered log read mid-run looks like a stalled job; it was diagnosed as one.
    assert settings["env"]["PYTHONUNBUFFERED"] == "1"


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text()
    assert text.startswith("---\n"), f"{path} has no frontmatter"
    block = text.split("---\n", 2)[1]
    fields = {}
    for line in block.splitlines():
        key, _, value = line.partition(":")
        if value:
            fields[key.strip()] = value.strip()
    return fields


SKILLS = sorted((ROOT / ".claude" / "skills").glob("*/SKILL.md"))


@pytest.mark.parametrize("skill", SKILLS, ids=lambda path: path.parent.name)
def test_every_skill_says_when_to_use_it_and_its_scripts_exist(skill: Path) -> None:
    """The description is the only part Claude sees before invoking, so it must say when."""
    fields = _frontmatter(skill)
    assert fields.get("name") == skill.parent.name
    assert len(fields.get("description", "")) > 80, f"{skill} needs a description that says when"
    body = skill.read_text()
    for script in {part for part in body.split() if "${CLAUDE_SKILL_DIR}/scripts/" in part}:
        relative = script.split("${CLAUDE_SKILL_DIR}/", 1)[1].rstrip(",)`")
        assert (skill.parent / relative).is_file(), f"{skill} names {relative}, which is missing"


def test_there_are_skills_to_check() -> None:
    assert len(SKILLS) >= 5
