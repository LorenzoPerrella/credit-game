#!/usr/bin/env python3
"""PostToolUse on Edit and Write: a number typed by hand on a section page, said the moment it is.

`tests/test_docs_provenance.py` is what enforces the rule, in the suite and in CI. This is the
early half: the row "0.458 / 0.823 / 1.420" was typed once and read stale for weeks, and a page
is cheapest to correct while it is the file just written.

The check is `creditsurv.site.provenance`, loaded by path rather than imported, so the hook needs
neither the package's environment nor its dependencies; that module is standard library only at
import for this reason. Silent on any other file, and it never refuses: the edit has happened.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

PROJECT = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2]))
MODULE = PROJECT / "src" / "creditsurv" / "site" / "provenance.py"


def _provenance() -> ModuleType | None:
    spec = importlib.util.spec_from_file_location("_creditsurv_provenance", MODULE)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # A dataclass looks its module up in sys.modules while it is being defined.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def note(target: Path) -> str | None:
    docs = (PROJECT / "docs").resolve()
    try:
        relative = target.resolve().relative_to(docs)
    except ValueError:
        return None
    if not MODULE.is_file():
        return None
    provenance = _provenance()
    if provenance is None or str(relative) not in provenance.SECTION_PAGES:
        return None
    try:
        found = provenance.free_numbers(target.read_text(encoding="utf-8"))
    except OSError:
        return None
    if not found:
        return None
    listed = "; ".join(item.describe() for item in found[:12])
    more = f" and {len(found) - 12} more" if len(found) > 12 else ""
    return (
        f"docs/{relative} now types {len(found)} precise number(s) by hand: {listed}{more}. "
        "On a section page a run's output is a `value` placeholder, and a dated figure goes "
        "between `<!-- recorded: YYYY-MM-DD -->` and `<!-- /recorded -->`; "
        "tests/test_docs_provenance.py fails otherwise."
    )


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        target = Path(str(payload.get("tool_input", {}).get("file_path", "")))
    except (ValueError, AttributeError):
        return 0
    if target.suffix != ".md" or not target.is_file():
        return 0
    message = note(target)
    if message:
        json.dump(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": message,
                }
            },
            sys.stdout,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
