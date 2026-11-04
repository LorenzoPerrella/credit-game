#!/usr/bin/env python3
"""Repository invariants checked on a diff: one implementation, run by the commit hook and by CI.

A rule written in two places drifts apart -- `backtest/metrics.py` and `docs/rules.md` disagreed
about the grade criterion for two days -- so the hook and the CI job import this file rather than
each keeping a copy. The hook passes the staged diff; CI passes the pull request's
(`origin/main...HEAD`). A hook binds only Claude, CI binds every author, and the checks are the
same either way.

Each check is precise, which means a violation is almost never a false alarm: one that cried wolf
would teach everyone to ignore it. Blocking checks guard the repository's governance and its data.
Warnings carry costs and risks that are worth knowing but are not errors.

    python3 .claude/hooks/policy.py --staged
    python3 .claude/hooks/policy.py --range origin/main...HEAD

Exits 1 when a blocking check fails. Standard library only.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: What may sit at the top of the repository. A new entry is a declared change to this tuple, not
#: a folder that appears with one file in it.
TOP_LEVEL: tuple[str, ...] = (
    ".claude",
    ".github",
    ".gitignore",
    ".python-version",
    "CLAUDE.md",
    "LICENSE",
    "README.md",
    "crates",
    "docs",
    "mkdocs.yml",
    "notebooks",
    "pyproject.toml",
    "src",
    "tests",
    "uv.lock",
)

#: The one CI workflow. Publishing jobs wait on the tests inside it, which a second file cannot.
WORKFLOWS: tuple[str, ...] = (".github/workflows/ci.yml",)

#: Constants fixed before any fit they judge. Changed without the register, any of them becomes a
#: way of moving a criterion after seeing the result -- the shape of tuning.
DECLARED: dict[str, tuple[str, ...]] = {
    "src/creditsurv/config.py": (
        "MACRO_ELIMINATION_PRIORITY",
        "ECONOMIC_DIMENSION",
        "BANDED_REFERENCE",
        "AGE_BANDS",
        "AGE_BASIS_KNOTS",
        "STABILITY_CUT",
    ),
    "src/creditsurv/backtest/runner.py": ("Acceptance", "ACCEPTANCE"),
    "src/creditsurv/backtest/metrics.py": ("GRADE_TOLERANCE",),
    "src/creditsurv/models/anchoring.py": ("ANCHOR_WINDOW", "MAX_MULTIPLIER"),
    "src/creditsurv/models/rules.py": ("CORRELATION_THRESHOLD", "MATERIALITY_THRESHOLD"),
    "src/creditsurv/models/selection.py": ("VIF_THRESHOLD", "PVALUE_THRESHOLD"),
}

#: Lists whose change retires every cached reading of the table.
CALENDAR_SIDE: tuple[str, ...] = ("MACRO_CANDIDATES", "PROTECTED_MACRO", "TIME_VARYING_CONTINUOUS")

DATA_SUFFIXES: tuple[str, ...] = (".parquet", ".pickle", ".pkl", ".zip", ".feather", ".npy")
LARGEST_FILE: int = 5 * 1024 * 1024
RULES = "docs/rules.md"
REGISTER = "docs/rules_register.md"

#: Common Italian function words, for the one heuristic check here, which only ever warns.
_ITALIAN = re.compile(
    r"\b(che|della|delle|degli|nella|nelle|perché|quindi|anche|sono|questo|questa|"
    r"essere|stato|fatto|ogni|solo|dopo|prima|ancora|senza|tutto)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Violation:
    rule: str
    message: str
    blocking: bool = True

    def render(self) -> str:
        return f"[{'BLOCK' if self.blocking else 'warn'}] {self.rule}: {self.message}"


@dataclass(frozen=True)
class Change:
    status: str
    path: str
    old: str | None = None


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False
    ).stdout


def changes(scope: list[str]) -> list[Change]:
    """The files a diff touches, renames detected."""
    found = []
    for line in _git("diff", "--name-status", "-M", *scope).splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].startswith("R"):
            found.append(Change("R", parts[2], parts[1]))
        elif len(parts) == 2:
            found.append(Change(parts[0][0], parts[1]))
    return found


def _hunks(scope: list[str], path: str) -> list[tuple[str, list[str], list[str]]]:
    """Each hunk of one file as (context of its header, removed lines, added lines)."""
    hunks: list[tuple[str, list[str], list[str]]] = []
    for line in _git("diff", "-U0", *scope, "--", path).splitlines():
        if line.startswith("@@"):
            hunks.append((line.split("@@")[-1].strip(), [], []))
        elif hunks and line.startswith("-") and not line.startswith("---"):
            hunks[-1][1].append(line[1:])
        elif hunks and line.startswith("+") and not line.startswith("+++"):
            hunks[-1][2].append(line[1:])
    return hunks


def _versions(scope: list[str], path: str) -> tuple[str, str]:
    """The file before and after the diff: HEAD and the index, or the merge base and the head."""
    if scope == ["--cached"]:
        return _git("show", f"HEAD:{path}"), _git("show", f":{path}")
    base, _, head = scope[0].partition("...")
    root = _git("merge-base", base, head or "HEAD").strip()
    return _git("show", f"{root}:{path}"), _git("show", f"{head or 'HEAD'}:{path}")


def _changed_lines(scope: list[str], path: str) -> tuple[set[int], set[int]]:
    """Line numbers removed from the old version and added to the new one."""
    old: set[int] = set()
    new: set[int] = set()
    pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    for line in _git("diff", "-U0", *scope, "--", path).splitlines():
        match = pattern.match(line)
        if match:
            start, count = int(match[1]), int(match[2] or 1)
            old.update(range(start, start + count))
            start, count = int(match[3]), int(match[4] or 1)
            new.update(range(start, start + count))
    return old, new


def _spans(source: str, names: tuple[str, ...]) -> dict[str, range]:
    """Where each named top-level assignment or class is defined, by line."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    spans: dict[str, range] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            found = [node.name]
        elif isinstance(node, ast.Assign):
            found = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found = [node.target.id]
        else:
            continue
        for name in found:
            if name in names:
                spans[name] = range(node.lineno, (node.end_lineno or node.lineno) + 1)
    return spans


def _touched_definitions(scope: list[str], path: str, names: tuple[str, ...]) -> list[str]:
    """Which of `names` the diff changes the definition of -- exactly, by line, not by context.

    git's hunk header names the nearest preceding top-level line, so a change to the constant
    *after* a declared one reads as inside it: the first version of this check refused exactly
    that. Parsing both versions puts each definition on its own lines.
    """
    before, after = _versions(scope, path)
    removed, added = _changed_lines(scope, path)
    hit = {
        name
        for source, lines in ((before, removed), (after, added))
        for name, span in _spans(source, names).items()
        if lines & set(span)
    }
    return sorted(hit)


def _size(scope: list[str], path: str) -> int:
    if scope == ["--cached"]:
        blob = _git("ls-files", "-s", "--", path).split()
        return int(_git("cat-file", "-s", blob[1]).strip() or 0) if len(blob) > 1 else 0
    target = ROOT / path
    return target.stat().st_size if target.is_file() else 0


def check(scope: list[str]) -> list[Violation]:
    """Every violation in the diff `scope` describes (`["--cached"]` or `["A...B"]`)."""
    touched = changes(scope)
    paths = {change.path for change in touched}
    out: list[Violation] = []

    if RULES in paths and REGISTER not in paths:
        out.append(
            Violation(
                "rules-without-register",
                f"{RULES} changes and {REGISTER} does not. Rule 17: an amendment is recorded "
                "in the register in the same change, with what changed and why.",
            )
        )

    if REGISTER in paths:
        rewritten = [line for _, removed, _ in _hunks(scope, REGISTER) for line in removed]
        if rewritten:
            out.append(
                Violation(
                    "register-rewritten",
                    f"{len(rewritten)} line(s) of {REGISTER} are removed or changed, and it is "
                    "append-only: write the correction as a new dated entry that says what it "
                    f"supersedes. First: {rewritten[0][:80]!r}",
                )
            )

    for path, names in DECLARED.items():
        if path not in paths or REGISTER in paths:
            continue
        hit = _touched_definitions(scope, path, names)
        if hit:
            out.append(
                Violation(
                    "declared-constant",
                    f"{path} changes {', '.join(hit)}, fixed before any fit it judges. Record the "
                    f"change and its reason in {REGISTER} in the same commit (the rule-change "
                    "skill), or it is a criterion moved after the result.",
                )
            )

    for change in touched:
        if change.status == "D":
            continue
        path = change.path
        if path.startswith("data/") or (
            path.endswith(DATA_SUFFIXES) and not path.startswith("docs/tables/")
        ):
            out.append(
                Violation(
                    "data-artefact",
                    f"{path} is a data artefact. Only aggregates leave the machine: committed "
                    "tables live in docs/tables, and nothing under data/ is ever committed.",
                )
            )
        elif _size(scope, path) > LARGEST_FILE:
            out.append(
                Violation("large-file", f"{path} is over 5 MB; aggregates are small by design.")
            )
        if change.status == "A" and path.split("/", 1)[0] not in TOP_LEVEL:
            out.append(
                Violation(
                    "new-top-level",
                    f"{path} adds a top-level entry. Put it where it naturally belongs, or add it "
                    "to policy.TOP_LEVEL as a declared change.",
                )
            )
        if change.status == "A" and path.startswith(".github/workflows/") and path not in WORKFLOWS:
            out.append(
                Violation(
                    "new-workflow",
                    f"{path} adds a workflow. CI is one file, so that publishing waits on the "
                    "tests: add a job to .github/workflows/ci.yml instead.",
                )
            )
        if change.status == "R" and (change.old or "").startswith("src/creditsurv/models/"):
            out.append(
                Violation(
                    "module-renamed",
                    f"{change.old} -> {path}: a cached pickle names the module it was written "
                    "from. If any cached fit holds a class from it, leave the old name importable.",
                    blocking=False,
                )
            )

    if "src/creditsurv/config.py" in paths:
        calendar = _touched_definitions(scope, "src/creditsurv/config.py", CALENDAR_SIDE)
        if calendar:
            out.append(
                Violation(
                    "readings-retired",
                    f"config changes {', '.join(calendar)}: the calendar side moves, and every "
                    "cached reading is retired -- about 3.6 minutes and 1 GB a window to rebuild.",
                    blocking=False,
                )
            )

    italian = [
        path
        for path in paths
        if path.endswith((".md", ".py"))
        and any(
            len(_ITALIAN.findall(line)) >= 3
            for _, _, added in _hunks(scope, path)
            for line in added
        )
    ]
    if italian:
        out.append(
            Violation(
                "italian-prose",
                f"added lines in {', '.join(sorted(italian))} read as Italian; the repository is "
                "written in English.",
                blocking=False,
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--staged", action="store_true")
    group.add_argument("--range", dest="span")
    arguments = parser.parse_args()
    found = check(["--cached"] if arguments.staged else [arguments.span])
    for violation in found:
        print(violation.render())
    return 1 if any(violation.blocking for violation in found) else 0


if __name__ == "__main__":
    sys.exit(main())
