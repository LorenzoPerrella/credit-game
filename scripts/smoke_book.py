"""Write a small book in the dataset's own archive layout.

The end-to-end pipeline needs data, and the real Single-Family Loan-Level Dataset
cannot be downloaded programmatically -- registration is free but manual. So CI
generates a book instead, from the same simulator the tests use, written into the
same nested archives the download has: ``historical_data_YYYY.zip`` containing one
inner archive per origination quarter.

The point is that the pipeline then runs for real -- ingest, aggregation, fit,
reports -- rather than a stub of it. What it cannot check is whether the estimates
are any good; the test suite answers that, on a book whose parameters are known.

    uv run python scripts/smoke_book.py

Writes into ``CREDITSURV_DATA_DIR`` if that is set, and the project's ``data/``
otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from creditsurv.data.fred import load_macro_panel
from creditsurv.data.ingest import archive_dir
from fixtures import write_book_archives


def main() -> None:
    destination = archive_dir()
    destination.mkdir(parents=True, exist_ok=True)
    written = write_book_archives(destination, load_macro_panel())
    for path in written:
        print(f"{path.name}: {path.stat().st_size / 1024:.0f} KB")
    print(f"{len(written)} vintage archive(s) in {destination}")


if __name__ == "__main__":
    main()
