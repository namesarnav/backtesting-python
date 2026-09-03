"""Phase 1 checkpoint tests for engine.data_loader.DataLoader.

Network-dependent (hits yfinance on first run); subsequent runs hit the
Parquet cache and should be fast. Skips cleanly if there's no network and
no cache yet.
"""

import time

import pytest

from engine.data_loader import DataLoader

TEST_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
TEST_START = "2019-01-01"
TEST_END = "2024-01-01"


@pytest.fixture(scope="module")
def loader():
    return DataLoader(tickers=TEST_TICKERS, start=TEST_START, end=TEST_END)


def _try_load(loader):
    try:
        return loader.load_panel()
    except Exception as e:
        pytest.skip(f"data fetch unavailable in this environment: {e}")


def test_panel_shape_and_alignment(loader):
    panel = _try_load(loader)
    assert isinstance(panel.columns, __import__("pandas").MultiIndex)
    assert set(panel.columns.get_level_values("field")) == {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    assert set(panel.columns.get_level_values("ticker")) == set(TEST_TICKERS)
    # multiple years of daily data
    assert len(panel) > 1000


def test_no_nans_after_loading(loader):
    panel = _try_load(loader)
    assert not panel.isna().any().any(), "NaNs remain in loaded panel"


def test_cached_load_is_fast(loader):
    _try_load(loader)  # ensure cache is warm
    start = time.perf_counter()
    loader.load_panel()
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"cached load took {elapsed:.2f}s, expected < 5s"


def test_close_prices_are_positive(loader):
    panel = _try_load(loader)
    assert (panel["close"] > 0).all().all()


# -- coverage filter ------------------------------------------------------
#
# These use synthetic frames rather than the cache: the point is the policy,
# and a policy test that depends on which companies happened to IPO when is
# a test of Yahoo, not of us.


def _frame(start, end):
    import numpy as np
    import pandas as pd

    index = pd.bdate_range(start, end)
    return pd.DataFrame(
        {f: np.arange(1.0, len(index) + 1.0) for f in ("open", "high", "low", "close", "volume")},
        index=index,
    )


def test_coverage_filter_drops_late_listings_not_the_window():
    """A late-listing ticker must cost us that ticker, never the date range.

    This is the whole reason the filter exists. `load_panel` resolves a
    ragged panel by trimming to the first fully-populated row, so without
    this filter a single 2021 IPO silently converts a five-year backtest
    into a two-year one -- no error, no warning, just a shorter equity
    curve that still looks entirely plausible.
    """
    loader = DataLoader(tickers=["OLD", "LATE"], start="2019-01-01", end="2024-01-01")
    kept, dropped = loader._apply_coverage_filter(
        {
            "OLD": _frame("2019-01-02", "2023-12-29"),
            "LATE": _frame("2021-06-01", "2023-12-29"),
        }
    )

    assert set(kept) == {"OLD"}
    assert [t for t, _ in dropped] == ["LATE"]
    assert "2021-06-01" in dropped[0][1]


def test_coverage_filter_drops_tickers_that_stop_early():
    """Delisted mid-window is the same problem from the other end."""
    loader = DataLoader(tickers=["OLD", "GONE"], start="2019-01-01", end="2024-01-01")
    kept, dropped = loader._apply_coverage_filter(
        {
            "OLD": _frame("2019-01-02", "2023-12-29"),
            "GONE": _frame("2019-01-02", "2021-08-01"),
        }
    )

    assert set(kept) == {"OLD"}
    assert [t for t, _ in dropped] == ["GONE"]


def test_coverage_filter_tolerates_the_calendar_boundary():
    """2019-01-01 is a holiday; the first bar is the 2nd. That is not a gap.

    Without a few days of grace the filter would drop the entire universe,
    which is the kind of off-by-a-holiday that only shows up in January.
    """
    loader = DataLoader(tickers=["A"], start="2019-01-01", end="2024-01-01")
    kept, dropped = loader._apply_coverage_filter({"A": _frame("2019-01-02", "2023-12-29")})

    assert set(kept) == {"A"}
    assert dropped == []


def test_panel_cache_key_separates_coverage_policies():
    """The two policies produce different panels, so they cannot share a file."""
    args = {"tickers": ["AAPL"], "start": "2019-01-01", "end": "2024-01-01"}
    strict = DataLoader(**args, require_full_history=True)._panel_cache_path()
    loose = DataLoader(**args, require_full_history=False)._panel_cache_path()

    assert strict != loose
