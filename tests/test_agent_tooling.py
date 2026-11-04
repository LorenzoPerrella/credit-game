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
    # Offline too: a test that calls GitHub is a test of the network.
    environment = {
        **os.environ,
        "CREDITSURV_HOOK_SKIP_GATE": "1",
        "CREDITSURV_HOOK_OFFLINE": "1",
        **env,
    }
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
        ("git commit -q -m 'no trailer'", "Co-Authored-By"),
        ("gh pr create --title t --body 'no footer'", "Generated with"),
        (
            "uv run python - <<'EOF'\nfrom creditsurv.data.store import load_cells\n"
            "cells = load_cells('exclude')\nEOF",
            "CellBlocks",
        ),
        ("uv run python -c 'import x; x.split_cells(y)'", "CellBlocks"),
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
        "git commit -q -F - <<'EOF'\nnever git add -A, never creditsurv prune-archives\n\n"
        "Co-Authored-By: Claude <noreply@anthropic.com>\nEOF",
        'git commit -m "explain why git add -A is refused" -m "Co-Authored-By: Claude <x@y>"',
        "git commit --amend --no-edit",
        "uv run python -c 'from creditsurv.data.store import load_cells_window'",
        # Mentioning the function in a string is not calling it: the first version refused this.
        "uv run python - <<'EOF'\nnote = 'never call load_cells(...) on the whole table'\nEOF",
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


RULES = sorted((ROOT / ".claude" / "rules").glob("*.md"))


def _paths(rule: Path) -> list[str]:
    block = rule.read_text().split("---\n", 2)[1]
    return [
        line.strip().removeprefix("- ").strip().strip('"')
        for line in block.splitlines()
        if line.strip().startswith("- ")
    ]


def test_the_root_instructions_stay_short() -> None:
    """Loaded into every session: the documentation's target is under 200 lines, this one's 150.

    It was 822 lines and 63 KB, and long instructions are followed less, not more.
    """
    lines = (ROOT / "CLAUDE.md").read_text().splitlines()
    assert len(lines) <= 150, f"CLAUDE.md is {len(lines)} lines; move what is not global to a rule"


@pytest.mark.parametrize("rule", RULES, ids=lambda path: path.stem)
def test_every_rule_is_scoped_and_every_scope_matches_a_file(rule: Path) -> None:
    """A path-scoped rule loads only when a matching file is read, so a dead glob is a dead rule.

    Renaming `models/baseline.py` to `baseline.py` would have silenced a rule pointing at the old
    path, and nothing would have said so -- the class of check this project calls going mute.
    """
    patterns = _paths(rule)
    assert patterns, f"{rule.name} has no paths, so it would load in every session"
    for pattern in patterns:
        assert any(ROOT.glob(pattern)), f"{rule.name}: `{pattern}` matches no file"


def test_there_are_rules_to_check() -> None:
    assert len(RULES) >= 5


@pytest.mark.parametrize(
    "agent", sorted((ROOT / ".claude" / "agents").glob("*.md")), ids=lambda path: path.stem
)
def test_every_subagent_says_when_to_use_it_and_what_it_may_touch(agent: Path) -> None:
    fields = _frontmatter(agent)
    assert fields.get("name") == agent.stem
    assert len(fields.get("description", "")) > 80
    assert fields.get("tools"), (
        f"{agent.name} should list its tools rather than inherit all of them"
    )


_BASE_REGISTER = "# Register\n\n## 1 November\n\nAn entry.\n"


def _repository(tmp_path: Path) -> Path:
    """A throwaway git repository shaped like this one, for the diff-reading checks."""
    root = tmp_path / "repo"
    for path, text in {
        "docs/rules.md": "# Rules\n",
        "docs/rules_register.md": _BASE_REGISTER,
        "src/creditsurv/config.py": "ECONOMIC_DIMENSION = {'a': 'x'}\nOTHER = 1\n",
        "src/creditsurv/models/old.py": "class Kept:\n    pass\n",
        "README.md": "readme\n",
    }.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    for argv in (
        ["init", "-q", "-b", "main"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "add", "."],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "base"],
    ):
        subprocess.run(["git", *argv], cwd=root, check=True, capture_output=True)
    return root


def _policy(root: Path) -> list[object]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("policy_under_test", HOOKS / "policy.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    module.ROOT = root  # type: ignore[attr-defined]
    return list(module.check(["--cached"]))


def _stage(root: Path, edits: dict[str, str | None]) -> None:
    for path, text in edits.items():
        target = root / path
        if text is None:
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)


@pytest.mark.parametrize(
    ("edits", "rule"),
    [
        ({"docs/rules.md": "# Rules\n\nA new rule.\n"}, "rules-without-register"),
        (
            {"docs/rules_register.md": "# Register\n\n## 1 November\n\nRewritten.\n"},
            "register-rewritten",
        ),
        (
            {"src/creditsurv/config.py": "ECONOMIC_DIMENSION = {'a': 'y'}\nOTHER = 1\n"},
            "declared-constant",
        ),
        ({"data/processed/cells.parquet": "x"}, "data-artefact"),
        ({"src/creditsurv/fits.pickle": "x"}, "data-artefact"),
        ({"scripts/tool.py": "print(1)\n"}, "new-top-level"),
        ({".github/workflows/site.yml": "on: push\n"}, "new-workflow"),
    ],
)
def test_the_policy_refuses_what_governance_forbids(
    edits: dict[str, str | None], rule: str, tmp_path: Path
) -> None:
    root = _repository(tmp_path)
    _stage(root, edits)
    found = [v for v in _policy(root) if v.blocking]  # type: ignore[attr-defined]
    assert rule in {v.rule for v in found}, f"{rule} not raised; got {found}"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "edits",
    [
        {
            "docs/rules.md": "# Rules\n\nNew.\n",
            "docs/rules_register.md": _BASE_REGISTER + "\n## 2 November\n\nNew rule.\n",
        },
        {"docs/rules_register.md": _BASE_REGISTER + "\n## 2 November\n\nMore.\n"},
        {"src/creditsurv/config.py": "ECONOMIC_DIMENSION = {'a': 'x'}\nOTHER = 2\n"},
        {"docs/tables/view.parquet": "x", "README.md": "readme, longer\n"},
    ],
)
def test_the_policy_lets_ordinary_change_through(
    edits: dict[str, str | None], tmp_path: Path
) -> None:
    root = _repository(tmp_path)
    _stage(root, edits)
    assert not [v for v in _policy(root) if v.blocking]  # type: ignore[attr-defined]


def test_a_renamed_model_module_is_a_warning_not_a_refusal(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    subprocess.run(
        ["git", "mv", "src/creditsurv/models/old.py", "src/creditsurv/models/new.py"],
        cwd=root,
        check=True,
    )
    found = _policy(root)
    assert any(v.rule == "module-renamed" and not v.blocking for v in found)  # type: ignore[attr-defined]


def test_a_pull_request_waits_for_a_verified_clean_commit(tmp_path: Path) -> None:
    """The suite and the 3.11 leg are too slow for a hook, so the hook checks the stamp instead."""
    root = _repository(tmp_path)
    command = "gh pr create --title t --body 'x\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)'"
    environment = {"CLAUDE_PROJECT_DIR": str(root), "CREDITSURV_HOOK_SKIP_GATE": ""}

    def decide() -> str:
        result = subprocess.run(
            [sys.executable, str(HOOKS / "guard_bash.py")],
            input=json.dumps({"tool_input": {"command": command}}),
            capture_output=True,
            text=True,
            env={**os.environ, **environment},
            timeout=30,
            check=False,
        )
        return result.stdout

    assert "no verification has been recorded" in decide()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    (root / ".claude" / "state").mkdir(parents=True)
    (root / ".claude" / "state" / "verified").write_text(head + "\n")
    subprocess.run(["git", "update-index", "--assume-unchanged", "README.md"], cwd=root, check=True)
    (root / ".gitignore").write_text(".claude/state/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "ignore"],
        cwd=root,
        check=True,
    )
    assert "the last verification was of" in decide(), "a stamp of an older commit must not count"
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    (root / ".claude" / "state" / "verified").write_text(head + "\n")
    assert decide() == "", "a verified, clean commit opens the pull request"
    (root / "README.md").write_text("dirty\n")
    subprocess.run(
        ["git", "update-index", "--no-assume-unchanged", "README.md"], cwd=root, check=True
    )
    assert "uncommitted changes" in decide()


def _post(command: str, state: Path, **env: str) -> str:
    environment = {"CLAUDE_PROJECT_DIR": str(state), **env}
    result = _run("post_bash.py", {"tool_input": {"command": command}}, **environment)
    assert result.returncode == 0, result.stderr
    if not result.stdout.strip():
        return ""
    return str(json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"])


_BUSY = "  4079  01-07:20:57 7599660 /u/.venv/bin/python /u/.venv/bin/creditsurv report\n"


def test_a_heavy_job_that_pages_is_reported(tmp_path: Path) -> None:
    """The 31-hour report averaged about 108,000 swap-ins a minute, and nothing said so."""
    table = tmp_path / "ps.txt"
    table.write_text(_BUSY)
    seams = {"CREDITSURV_HOOK_PS": str(table)}
    assert (
        _post("ls", tmp_path, CREDITSURV_HOOK_SWAPINS="1000", CREDITSURV_HOOK_NOW="0", **seams)
        == ""
    )
    note = _post(
        "ls", tmp_path, CREDITSURV_HOOK_SWAPINS="1001000", CREDITSURV_HOOK_NOW="600", **seams
    )
    assert "paging" in note
    assert "100,000 swap-ins a minute" in note
    calm = _post(
        "ls", tmp_path, CREDITSURV_HOOK_SWAPINS="1001100", CREDITSURV_HOOK_NOW="1200", **seams
    )
    assert calm == ""


def test_a_kill_that_leaves_a_heavy_job_alive_is_reported(tmp_path: Path) -> None:
    table = tmp_path / "ps.txt"
    table.write_text(_BUSY)
    note = _post("kill -TERM 4078", tmp_path, CREDITSURV_HOOK_PS=str(table))
    assert "still alive" in note
    assert "pid 4079" in note
    table.write_text("")
    assert _post("kill -TERM 4079", tmp_path, CREDITSURV_HOOK_PS=str(table)) == ""


def test_a_second_read_of_the_test_window_for_one_specification_is_flagged(
    tmp_path: Path,
) -> None:
    """Rule 1 reads the test window once; the log makes that checkable."""
    (tmp_path / "src" / "creditsurv").mkdir(parents=True)
    (tmp_path / "src" / "creditsurv" / "config.py").write_text("FORMULA = 'a + b'\n")
    assert _post("uv run creditsurv windows", tmp_path) == ""
    assert "already been read" in _post("uv run creditsurv views", tmp_path)
    (tmp_path / "src" / "creditsurv" / "config.py").write_text("FORMULA = 'a + b + c'\n")
    assert _post("uv run creditsurv windows", tmp_path) == "", "a new specification reads afresh"


def test_a_python_file_is_formatted_the_moment_it_is_written(tmp_path: Path) -> None:
    """Measured live: the next Edit on the reformatted file succeeds without a re-read."""
    if not (ROOT / ".venv" / "bin" / "ruff").exists():
        pytest.skip("ruff is not installed in the project environment")
    (tmp_path / ".venv").symlink_to(ROOT / ".venv")
    written = tmp_path / "probe.py"
    written.write_text("import sys,os\ndef f( x ):\n    return x+1\n")
    result = _run(
        "format_on_edit.py",
        {"tool_input": {"file_path": str(written)}},
        CLAUDE_PROJECT_DIR=str(tmp_path),
    )
    assert result.returncode == 0
    assert written.read_text() == "def f(x):\n    return x + 1\n"
