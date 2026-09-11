"""Connector for FRED, the St. Louis Fed's economic data service.

Uses the public CSV download endpoint, which needs no API key. Raw observations
are cached to parquet so that repeated runs, and runs without a network, behave
identically to the first one.

Two properties of the source shape this module:

* Only one series can be requested per call. Passing several ids to the endpoint
  returns a ZIP archive rather than CSV, so series are fetched individually.
* FRED serves the *latest revision* of every series, not the vintage that was
  known at the time. Any macro covariate therefore carries mild look-ahead.
  ALFRED would be needed for true point-in-time data; the lag applied in
  :mod:`creditsurv.features` partially compensates. This is documented in
  ``docs/data_dictionary.md`` rather than silently ignored.
"""

from __future__ import annotations

import io
import logging
import warnings
from typing import TYPE_CHECKING, Final, cast

import pandas as pd
import requests

from creditsurv.config import (
    MACRO_SERIES,
    MACRO_START,
    Frequency,
    SeriesSpec,
    raw_dir,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

_LOGGER: Final = logging.getLogger(__name__)

_ENDPOINT: Final = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_TIMEOUT_SECONDS: Final = 30
#: FRED writes a bare full stop where an observation is missing.
_MISSING_MARKER: Final = "."


class FredUnavailableError(RuntimeError):
    """Raised when a series can be served neither from the network nor from cache."""


def _cache_path(series_id: str) -> Path:
    return raw_dir() / "fred" / f"{series_id}.parquet"


def _download(spec: SeriesSpec, start: str, end: str | None) -> pd.Series:
    """Fetch one series as a date-indexed float Series."""
    params = {"id": spec.series_id, "cosd": start}
    if end is not None:
        params["coed"] = end

    response = requests.get(_ENDPOINT, params=params, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()

    frame = pd.read_csv(io.StringIO(response.text), na_values=[_MISSING_MARKER])
    if frame.shape[1] != 2:
        message = (
            f"Expected two columns from FRED for {spec.series_id}, got {list(frame.columns)}. "
            "A multi-series request returns a ZIP archive rather than CSV."
        )
        raise FredUnavailableError(message)

    # The first column has been named both DATE and observation_date over time,
    # so it is taken positionally rather than by name.
    date_column, value_column = frame.columns
    observations = pd.Series(
        pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(frame[date_column])),
        name=spec.column,
    )
    return observations.dropna()


def load_series(
    spec: SeriesSpec, *, start: str = MACRO_START, end: str | None = None, refresh: bool = False
) -> pd.Series:
    """Return one raw series, from cache when possible.

    A network failure is not fatal when the series has been cached before: the
    cached copy is used and a warning is emitted. This keeps the project usable
    offline without ever silently pretending a refresh succeeded.
    """
    cache_file = _cache_path(spec.series_id)

    if not refresh and cache_file.exists():
        cached = pd.read_parquet(cache_file)[spec.column]
        _LOGGER.debug("Loaded %s from cache (%d observations)", spec.series_id, len(cached))
        return cached

    try:
        observations = _download(spec, start, end)
    except (requests.RequestException, FredUnavailableError) as error:
        if cache_file.exists():
            warnings.warn(
                f"Could not refresh {spec.series_id} from FRED ({error}); "
                f"using the cached copy at {cache_file}.",
                stacklevel=2,
            )
            return pd.read_parquet(cache_file)[spec.column]
        message = f"{spec.series_id} is not cached and could not be downloaded: {error}"
        raise FredUnavailableError(message) from error

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    observations.to_frame().to_parquet(cache_file)
    _LOGGER.debug("Cached %s (%d observations)", spec.series_id, len(observations))
    return observations


def to_monthly(observations: pd.Series, frequency: Frequency) -> pd.Series:
    """Collapse a series onto a monthly period index.

    The aggregation depends on what the series measures. Weekly rates are
    averaged, because the monthly figure a borrower experiences is the average of
    the weeks in it. Monthly and quarterly levels are carried forward, because
    they are point-in-time readings that stay in force until the next release.
    """
    if frequency in (Frequency.DAILY, Frequency.WEEKLY):
        # Averaged, not sampled: the rate or spread a borrower lives with over a month
        # is the average of its days, not whichever day happened to fall last. It also
        # makes the series robust to the market holidays that leave gaps in the daily
        # ones -- 226 to 303 missing observations each since 1999.
        monthly = observations.resample("MS").mean()
    else:
        monthly = observations.resample("MS").last()

    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    return monthly


def load_macro_panel(
    specs: Iterable[SeriesSpec] = MACRO_SERIES,
    *,
    start: str = MACRO_START,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return a monthly macro panel, one column per series.

    Interior gaps are forward-filled: FRED has occasional holes, and quarterly
    series only report every third month, so these are expected.

    The trailing edge is truncated instead. Series are published on different
    lags -- the house price index runs two to three months behind the
    unemployment rate -- and forward-filling that ragged edge would invent macro
    observations that did not exist, which then leak into every covariate built
    on them. The panel therefore ends at the last month for which *every* series
    has a real observation.
    """
    columns: dict[str, pd.Series] = {}
    for spec in specs:
        raw = load_series(spec, start=start, end=end, refresh=refresh)
        columns[spec.column] = to_monthly(raw, spec.frequency)

    if not columns:
        message = "No series requested."
        raise ValueError(message)

    first_valid = {
        column: cast("pd.Period", series.first_valid_index()) for column, series in columns.items()
    }
    last_valid = {
        column: cast("pd.Period", series.last_valid_index()) for column, series in columns.items()
    }
    first_period = max(first_valid.values())
    last_period = min(last_valid.values())

    panel = pd.DataFrame(columns).reindex(pd.period_range(first_period, last_period, freq="M"))

    interior_gaps = int(panel.isna().sum().sum())
    if interior_gaps:
        _LOGGER.info("Forward-filled %d interior macro observation(s)", interior_gaps)
    panel = panel.ffill()

    trimmed = {column: str(period) for column, period in last_valid.items() if period > last_period}
    if trimmed:
        _LOGGER.info(
            "Panel ends %s; later observations discarded as incomplete across series: %s",
            last_period,
            trimmed,
        )

    panel.index.name = "period"
    return panel
