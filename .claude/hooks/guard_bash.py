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

import ast
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import policy
from _jobs import is_heavy, on_battery, running_jobs

PROJECT = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2]))


def verdict(command: str) -> str | None:
    """The reason to refuse `command`, or None to let it run.

    Most checks read only the code -- what the shell will run -- so that a message describing a
    refused command is not refused for it. The attribution checks read the raw text too, because
    the attribution line lives exactly in the heredoc the code view drops.
    """
    code = _code(command)
    for check in (_irreversible, _interpreter_swap, _blanket_add, _heavy, _whole_panel):
        reason = check(code)
        if reason:
            return reason
    for gate in (_attribution, _commit_gate, _publish_gate):
        reason = gate(code, command)
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


def _attribution(code: str, raw: str) -> str | None:
    """Every commit and PR body Claude writes ends with the line the system reminder gives.

    Kept apart from the gates because it is pure text -- no git, no tests -- and because it is the
    one check that stays in the hook only: the user's own commits do not carry the line, so CI
    would refuse them for it.
    """
    committing = re.search(r"\bgit\s+commit\b", code)
    reused = re.search(r"--no-edit\b|\s-[cC]\s", code)
    if committing and not reused and "Co-Authored-By:" not in raw:
        return (
            "This commit carries no `Co-Authored-By:` line, and every commit Claude writes ends "
            "with the attribution line the session's system reminder gives."
        )
    describing = re.search(r"\bgh\s+pr\s+(create|edit)\b", code) and "--body" in code
    if describing and "Generated with [Claude Code]" not in raw:
        return (
            "This pull request body carries no `🤖 Generated with [Claude Code](...)` line, which "
            "every PR Claude writes ends with."
        )
    return None


def _commit_gate(code: str, raw: str) -> str | None:
    """Before a commit: attribution, the repository policy, lint, types and the tests it touches.

    Seconds, where CI found the same things in minutes. The full suite and the 3.11 leg are the
    `verify` skill's, and the push gate below insists they ran.
    """
    if not re.search(r"\bgit\s+commit\b", code) or os.environ.get("CREDITSURV_HOOK_SKIP_GATE"):
        return None
    blocking = [v.render() for v in policy.check(["--cached"]) if v.blocking]
    if blocking:
        return "The repository policy refuses this commit:\n" + "\n".join(blocking)
    failed = [failure for failure in (_lint_and_types(), _targeted_tests()) if failure]
    if failed:
        return (
            "The commit gate failed, so CI would too. Fix and commit again "
            "(`uv run ruff format . && uv run ruff check --fix .` handles most lint).\n\n"
            + "\n\n".join(failed)
        )
    return None


def _publish_gate(code: str, raw: str) -> str | None:
    """Before a PR exists or grows: the full checklist ran on exactly this commit.

    The suite takes 5.5 minutes and the 3.11 leg another two, too long for a hook. So the `verify`
    skill stamps `.claude/state/verified` with the commit and a clean tree, and this compares two
    hashes. A push to a branch with no pull request is a safety copy and passes freely.
    """
    if os.environ.get("CREDITSURV_HOOK_SKIP_GATE"):
        return None
    opening = bool(re.search(r"\bgh\s+pr\s+(create|ready)\b", code))
    pushing = bool(re.search(r"\bgit\s+push\b", code)) and _has_open_pull_request()
    if not (opening or pushing):
        return None
    head = _git_out("rev-parse", "HEAD")
    dirty = bool(_git_out("status", "--porcelain"))
    stamp = STAMP.read_text().split() if STAMP.is_file() else []
    if dirty or not stamp or stamp[0] != head:
        state = (
            "the tree has uncommitted changes"
            if dirty
            else (
                "no verification has been recorded"
                if not stamp
                else f"the last verification was of {stamp[0][:10]}, not {head[:10]}"
            )
        )
        return (
            f"A pull request is about to see this commit, and {state}. Run the `verify` skill "
            "(`bash .claude/skills/verify/scripts/verify.sh`), which stamps the commit on success."
        )
    return None


STAMP = PROJECT / ".claude" / "state" / "verified"

#: Staged path -> the fast tests it must pass before it is committed. The rest of the suite is the
#: `verify` skill's; these are the ones a change to that path is most likely to break, kept small
#: enough to finish inside the hook's budget.
_TARGETED: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "src/creditsurv/config.py",
        ("tests/test_procedure.py::test_the_configuration_is_what_the_last_selection_chose",),
    ),
    (
        "docs/reports/selection.json",
        ("tests/test_procedure.py::test_the_configuration_is_what_the_last_selection_chose",),
    ),
    ("docs/", ("tests/test_site.py", "tests/test_docs_provenance.py")),
    ("src/creditsurv/site/", ("tests/test_site.py", "tests/test_docs_provenance.py")),
    ("mkdocs.yml", ("tests/test_site.py",)),
    (".claude/", ("tests/test_agent_tooling.py",)),
    (".github/workflows/", ("tests/test_agent_tooling.py",)),
    ("uv.lock", ("tests/test_blocks.py",)),
)


def _targeted_tests() -> str | None:
    staged = _git_out("diff", "--cached", "--name-only").splitlines()
    chosen: list[str] = []
    for path in staged:
        for prefix, tests in _TARGETED:
            if path.startswith(prefix):
                chosen.extend(tests)
        if path.startswith("src/creditsurv/") and path.endswith(".py"):
            mirror = PROJECT / "tests" / f"test_{Path(path).stem}.py"
            if mirror.is_file():
                chosen.append(str(mirror.relative_to(PROJECT)))
    chosen = [test for test in dict.fromkeys(chosen) if (PROJECT / test.split("::")[0]).is_file()]
    pytest = PROJECT / ".venv" / "bin" / "pytest"
    if not chosen or not pytest.exists():
        return None
    try:
        result = subprocess.run(
            [str(pytest), "-q", "-x", "-m", "not network", "-p", "no:cacheprovider", *chosen],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=150,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # Fails open: a slow mirror is not a reason to refuse a commit; `verify` runs everything.
        return None
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode in (0, 5):
        return None
    tail = (result.stdout + result.stderr).strip().splitlines()[-12:]
    return "Targeted tests (" + ", ".join(chosen) + ") failed:\n" + "\n".join(tail)


def _lint_and_types() -> str | None:
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
    return "\n\n".join(failed) or None


def _has_open_pull_request() -> bool:
    if shutil.which("gh") is None:
        return False
    branch = _git_out("branch", "--show-current")
    try:
        out = subprocess.run(
            ["gh", "pr", "list", "--head", branch, "--state", "open", "--json", "number"],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return out.strip() not in ("", "[]")


def _git_out(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT, capture_output=True, text=True, timeout=10, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


_WHOLE_TABLE = {"load_cells", "split_cells"}


def _whole_panel(command: str) -> str | None:
    """An ad-hoc script that loads the whole cell table: the four-hour, 7.25 GB scoring pass.

    Read as Python, not as text: the first version matched `load_cells(` anywhere and refused a
    script whose *test strings* mentioned it -- the false positive this file is meant not to have.
    Only a real call counts, found by `ast` in the heredoc, the `-c` argument or the script file
    the command runs. Source that does not parse is let through.
    """
    if not re.search(r"\bpython\d*\b|\buv\s+run\b", command):
        return None
    if any(_calls(source) & _WHOLE_TABLE for source in _scripts(RAW.get("command", ""))):
        return (
            "This script loads the whole cell table (`load_cells` / `split_cells`), which is how a "
            "scoring pass held 7.25 GB for four hours with swap at 5.4 GB. Stream it: "
            "`CellBlocks(path, macro, covariates, rows=250_000, months=(first, last), "
            "model_only=False).prepared()` and sum the per-batch tables (the `measure-a-fit` "
            "skill). A short window read with `load_cells_window` is fine."
        )
    return None


def _calls(source: str) -> set[str]:
    """The names of the functions a piece of Python calls, or nothing if it does not parse."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                names.add(function.id)
            elif isinstance(function, ast.Attribute):
                names.add(function.attr)
    return names


def _scripts(raw: str) -> list[str]:
    """Each piece of Python the command runs: heredoc bodies, `-c` arguments, named `.py` files."""
    heredoc = r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\s*\1\b"
    sources = [body for _, body in re.findall(heredoc, raw, re.S)]
    # `-c` only outside heredoc bodies: a body is data, and one that held a `-c '...'` test string
    # was read as a command and refused.
    outside = re.sub(heredoc, "<<HEREDOC", raw, flags=re.S)
    sources += [
        match[1] or match[2]
        for match in re.findall(r"""\s-c\s+(['"])((?:(?!\1).)*)\1|\s-c\s+(\S+)""", outside, re.S)
    ]
    for name in re.findall(r"(\S+\.py)\b", _code(raw)):
        script = Path(name) if Path(name).is_absolute() else PROJECT / name
        if script.is_file() and ".claude/hooks" not in str(script):
            with contextlib.suppress(OSError):
                sources.append(script.read_text()[:200_000])
    return sources


RAW: dict[str, str] = {}


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        command = str(payload.get("tool_input", {}).get("command", ""))
    except (ValueError, AttributeError):
        return 0
    RAW["command"] = command
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
