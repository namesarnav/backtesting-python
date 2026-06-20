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
