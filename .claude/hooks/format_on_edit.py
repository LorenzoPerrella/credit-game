#!/usr/bin/env python3
"""PostToolUse on Edit and Write: format a Python file the moment it is written.

The commit gate checks lint and format, and in one week it refused commit after commit on import
order and line length, each a round trip. Formatting the one file just written removes the class
instead of reporting it. Only that file, only ruff's safe fixes, and `.venv/bin/ruff` directly,
because `uv run` would double the cost.

Silent when ruff is missing or the file is not Python inside the project. It never refuses: the
edit has already happened.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2]))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        target = Path(str(payload.get("tool_input", {}).get("file_path", "")))
    except (ValueError, AttributeError):
        return 0
    ruff = PROJECT / ".venv" / "bin" / "ruff"
    if target.suffix != ".py" or not ruff.exists() or not target.is_file():
        return 0
    try:
        target.resolve().relative_to(PROJECT.resolve())
    except ValueError:
        return 0
    # Fix, then format: the fixes remove lines (unused imports), and formatting first would leave
    # the blank lines they stood on.
    for argv in (
        [str(ruff), "check", "-q", "--fix", str(target)],
        [str(ruff), "format", "-q", str(target)],
    ):
        try:
            subprocess.run(argv, cwd=PROJECT, capture_output=True, timeout=20, check=False)
        except (OSError, subprocess.SubprocessError):
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
