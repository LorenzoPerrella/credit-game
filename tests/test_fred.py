"""Tests for the FRED connector.

Everything except the single ``network`` test runs offline against synthetic CSV
payloads, so pull-request CI never depends on an upstream service being healthy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
import requests

from creditsurv.config import MACRO_SERIES, Frequency, SeriesSpec
from creditsurv.data import fred
from creditsurv.data.fred import load_series

if TYPE_CHECKING:
    from pathlib import Path

SPEC = SeriesSpec(
    series_id="TESTSERIES",
    column="test_series",
    frequency=Frequency.MONTHLY,
    description="Synthetic series used in tests.",
)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the cache at a scratch directory so tests never touch real data."""
    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))


def _serve(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    def _get(_url: str, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(text)

    monkeypatch.setattr(requests, "get", _get)


def _fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def _get(_url: str, **_kwargs: Any) -> _FakeResponse:
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", _get)


def test_parses_observation_date_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """FRED names the date column observation_date, not DATE. It is read positionally."""
    _serve(monkeypatch, "observation_date,TESTSERIES\n2020-01-01,1.5\n2020-02-01,2.5\n")

    series = fred.load_series(SPEC)

    assert list(series) == [1.5, 2.5]
    assert isinstance(series.index, pd.DatetimeIndex)


def test_treats_full_stop_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """FRED writes a bare full stop where an observation is missing."""
    _serve(monkeypatch, "observation_date,TESTSERIES\n2020-01-01,1.5\n2020-02-01,.\n")

    series = fred.load_series(SPEC)

    assert list(series) == [1.5]


def test_rejects_multi_series_zip_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting several ids returns a ZIP; the shape check must catch it."""
    _serve(monkeypatch, "a,b,c\n1,2,3\n")

    with pytest.raises(fred.FredUnavailableError, match="ZIP archive"):
        fred.load_series(SPEC)


def test_weekly_series_are_averaged_over_the_month() -> None:
    """A monthly rate is the average of the weeks in it, not the last week."""
    index = pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17", "2020-01-24"])
    weekly = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)

    monthly = fred.to_monthly(weekly, Frequency.WEEKLY)

    assert monthly.iloc[0] == pytest.approx(2.5)


def test_quarterly_series_land_on_the_monthly_grid_with_gaps() -> None:
    """Resampling places quarterly readings on the monthly grid but does not fill.

    Filling is the panel's job, not this function's: only the panel knows which
    months are interior gaps and which are the ragged edge, and the two must be
    treated differently.
    """
    quarterly = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-04-01"]))

    monthly = fred.to_monthly(quarterly, Frequency.QUARTERLY)

    by_month = {str(period): value for period, value in monthly.items()}
    assert by_month["2020-01"] == 1.0
    assert by_month["2020-04"] == 2.0
    assert pd.isna(by_month["2020-02"])


def test_cache_is_reused_when_the_network_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Being offline is not fatal once a series has been cached."""
    _serve(monkeypatch, "observation_date,TESTSERIES\n2020-01-01,1.5\n")
    fred.load_series(SPEC)

    _fail(monkeypatch)
    with pytest.warns(UserWarning, match="using the cached copy"):
        series = fred.load_series(SPEC, refresh=True)

    assert list(series) == [1.5]


def test_uncached_series_without_network_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing series must fail loudly rather than yield an empty frame."""
    _fail(monkeypatch)

    with pytest.raises(fred.FredUnavailableError, match="not cached"):
        fred.load_series(SPEC)


def _spec(series_id: str, column: str) -> SeriesSpec:
    return SeriesSpec(series_id, column, Frequency.MONTHLY, "test")


def test_panel_truncates_to_the_common_last_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Series publish on different lags; the ragged edge must not be forward-filled.

    Filling it would invent macro observations that never existed, which then leak
    into every covariate built on them.
    """
    payloads = {
        "EARLY": "observation_date,EARLY\n2020-01-01,1\n2020-02-01,2\n",
        "LATE": "observation_date,LATE\n2020-01-01,9\n2020-02-01,9\n2020-03-01,9\n",
    }

    def _get(_url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(payloads[kwargs["params"]["id"]])

    monkeypatch.setattr(requests, "get", _get)

    panel = fred.load_macro_panel([_spec("EARLY", "early"), _spec("LATE", "late")])

    assert panel.index.max() == pd.Period("2020-02", freq="M")
    assert len(panel) == 2


def test_panel_fills_interior_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interior holes are expected and are carried forward."""
    _serve(
        monkeypatch,
        "observation_date,GAPPY\n2020-01-01,1\n2020-02-01,.\n2020-03-01,3\n",
    )

    panel = fred.load_macro_panel([_spec("GAPPY", "gappy")])

    assert len(panel) == 3
    assert panel["gappy"].tolist() == [1.0, 1.0, 3.0]
    assert not panel.isna().to_numpy().any()


@pytest.mark.network
def test_live_fred_download_still_works() -> None:
    """Guards against the upstream endpoint or CSV layout changing.

    Runs nightly rather than on pull requests: an upstream outage should never
    fail an unrelated change.
    """
    spec = SeriesSpec("UNRATE", "unemployment_rate", Frequency.MONTHLY, "live check")

    series = fred.load_series(spec, start="2020-01-01", end="2020-06-01", refresh=True)

    assert len(series) == 6
    assert series.max() > 10  # April 2020 unemployment spike


def test_a_cache_fetched_from_a_later_start_is_refetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Widening MACRO_START must not be silently ignored.

    The cache is keyed by series id alone, so a cached series fetched from a later
    start would be returned short. The covariates needing the extra history are then
    not wrong but *missing*, and their rows are dropped -- which removes the opening
    months of the earliest vintages and leaves the rest, a left truncation nothing
    downstream can see.
    """
    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    spec = MACRO_SERIES[0]
    calls: list[str] = []

    def fake_download(_: object, start: str, __: object) -> pd.Series:
        calls.append(start)
        index = pd.date_range(start, periods=48, freq="MS")
        return pd.Series(range(48), index=index, dtype=float, name=spec.column)

    monkeypatch.setattr("creditsurv.data.fred._download", fake_download)

    load_series(spec, start="2005-01-01")
    load_series(spec, start="2005-01-01")
    assert calls == ["2005-01-01"], "a cache wide enough must be reused"

    load_series(spec, start="2000-01-01")
    assert calls == ["2005-01-01", "2000-01-01"], "a cache starting too late must be refetched"

    load_series(spec, start="2001-01-01")
    assert len(calls) == 2, "a cache wider than asked for must be reused"


def test_a_first_observation_after_the_start_is_not_refetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VIXCLS begins 1997-01-02 because 1 January is not a trading day.

    Comparing the requested start against the first *observation* would refetch every
    daily series on every single load, forever. The recorded request is compared
    instead.
    """
    monkeypatch.setenv("CREDITSURV_DATA_DIR", str(tmp_path))
    spec = MACRO_SERIES[0]
    calls: list[str] = []

    def fake_download(_: object, start: str, __: object) -> pd.Series:
        calls.append(start)
        # First observation two days after the start, as a market series has.
        opens = pd.Timestamp(start) + pd.Timedelta(2, unit="D")
        index = pd.date_range(opens, periods=24, freq="D")
        return pd.Series(range(24), index=index, dtype=float, name=spec.column)

    monkeypatch.setattr("creditsurv.data.fred._download", fake_download)

    load_series(spec, start="1997-01-01")
    load_series(spec, start="1997-01-01")

    assert len(calls) == 1


@pytest.mark.network
def test_a_vintage_cannot_be_had_without_an_api_key() -> None:
    """The validation's observation that the macro series are revised, held to the network.

    The public graph endpoint **accepts** a vintage date and ignores it: the response is a
    healthy CSV of the *current* series, running to the latest observation. That is worse
    than a refusal, because code written against it would look like a point-in-time
    backtest and be a revised-data backtest with extra steps.

    This test exists to fail the day that changes. A vintage series stops at its vintage
    date; if either request ever comes back short, the door has opened and the backtest can
    be given the data a model would actually have had.
    """
    import requests

    from creditsurv.data.fred import VINTAGE_ENDPOINT

    vintage = "2019-06-01"
    for params in (
        {"id": "UNRATE", "vintage_date": vintage},
        {"id": f"UNRATE_{vintage.replace('-', '')}"},
    ):
        response = requests.get(VINTAGE_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        last = response.text.strip().splitlines()[-1].split(",")[0]
        assert last > vintage, (
            f"{params} came back stopping at {last}: FRED now honours a vintage, and the "
            "backtest can read point-in-time data"
        )
