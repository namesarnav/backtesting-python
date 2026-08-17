"""Phase 7 checkpoint tests: the C++ bar loop must match the Python one.

The whole value of the port rests on one claim -- that it computes the same
thing, only faster. These tests are that claim, checked across every branch
of the loop: whole-share rounding, the no-trade band, extra lag, long/short
books and the degenerate all-flat case.

The entire module skips when the extension is not built, so a clone without
a compiler still gets a green suite. That is deliberate: the extension is
optional, and a skipped test is honest where a failing one would be noise.
"""

import numpy as np
import pandas as pd
import pytest

from engine.event_driven import (
    FILL_TOLERANCE,
    EventDrivenBacktester,
    _fastloop,
    _simulate_python,
    available_backends,
)

pytestmark = pytest.mark.skipif(
    "cpp" not in available_backends(),
    reason="C++ extension not built (python setup.py build_ext --inplace)",
)

BASE = dict(
    initial_capital=1_000_000.0,
    cost_rate=7e-4,
    lag_days=1,
    fractional_shares=True,
    rebalance_threshold=0.0,
)

# Every array both backends return. Checking a subset would let a real
# divergence hide in whichever one was skipped.
STATE_KEYS = (
    "equity", "gross_returns", "net_returns", "costs", "turnover", "cash",
    "holdings", "weights_established",
    "fill_bar", "fill_asset", "fill_shares", "fill_price", "fill_notional",
    "fill_cost",
)


def _panel(n_bars=250, n_assets=8, seed=3, kind="long_short"):
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, (n_bars, n_assets)), axis=0)

    if kind == "flat":
        weights = np.zeros((n_bars, n_assets))
    elif kind == "constant_long":
        weights = np.full((n_bars, n_assets), 1.0 / n_assets)
    else:
        weights = rng.normal(0.0, 1.0, (n_bars, n_assets))
        weights /= np.abs(weights).sum(axis=1, keepdims=True)
    return prices, weights


# Cash is a *residual*: equity minus the value of everything held. In a
# long/short book those are two ~$2M numbers that nearly cancel, leaving a
# balance of a few dollars. Catastrophic cancellation means the answer keeps
# absolute precision at the scale of the inputs (~1e-10 on $2M, about one
# ulp) while its *relative* precision is destroyed -- a $5e-10 error on a
# $7.48 balance is 7e-11 relative, which no sane rtol would pass.
#
# So money is compared in dollars and ratios are compared relatively. Using
# one tolerance for both would either fail on cash or wave through a real
# divergence in the returns.
DOLLAR_KEYS = frozenset(
    {
        "equity", "cash", "holdings",
        "fill_shares", "fill_price", "fill_notional", "fill_cost",
    }
)
INDEX_KEYS = frozenset({"fill_bar", "fill_asset"})


def _assert_same(prices, weights, **overrides):
    """Both backends, every output array, same numbers."""
    kwargs = {**BASE, **overrides}
    py = _simulate_python(prices, weights, **kwargs)
    cc = _fastloop.simulate(prices, weights, **kwargs)

    assert set(py) == set(cc) == set(STATE_KEYS)
    for key in STATE_KEYS:
        if key in INDEX_KEYS:
            # Which bar, which ticker. Integers -- exact, or the two
            # backends disagree about which orders were placed at all.
            np.testing.assert_array_equal(py[key], cc[key], err_msg=f"differ on {key!r}")
            continue

        # Bit-identical is not the bar and never will be: NumPy reduces
        # pairwise, the C++ loop accumulates in order, so they differ in the
        # last bit and that compounds gently along the equity path. What
        # matters is that no reported number could ever move.
        atol, rtol = (1e-6, 1e-9) if key in DOLLAR_KEYS else (1e-13, 1e-11)
        np.testing.assert_allclose(
            py[key], cc[key], rtol=rtol, atol=atol, err_msg=f"backends differ on {key!r}"
        )
    return py, cc


# -- the constant that must not drift ------------------------------------


def test_fill_tolerance_constants_agree():
    """The C++ file hardcodes this; if Python's ever changes, this catches it."""
    assert _fastloop.FILL_TOLERANCE == FILL_TOLERANCE


# -- equivalence across every branch of the loop -------------------------


@pytest.mark.parametrize("kind", ["long_short", "constant_long", "flat"])
def test_backends_agree_on_signal_shape(kind):
    prices, weights = _panel(kind=kind)
    _assert_same(prices, weights)


def test_backends_agree_with_zero_cost():
    prices, weights = _panel()
    _assert_same(prices, weights, cost_rate=0.0)


def test_backends_agree_with_whole_shares():
    """The `trunc` branch -- rounding must go the same way in both."""
    prices, weights = _panel()
    _assert_same(prices, weights, fractional_shares=False, initial_capital=50_000.0)


def test_backends_agree_with_rebalance_threshold():
    """The no-trade band, which skips orders and leaves positions to drift."""
    prices, weights = _panel()
    py, _ = _assert_same(prices, weights, rebalance_threshold=0.02)
    # Guard against the test passing vacuously: the band must actually bite.
    unbanded = _simulate_python(prices, weights, **BASE)
    assert py["fill_bar"].size < unbanded["fill_bar"].size


@pytest.mark.parametrize("lag_days", [1, 2, 5])
def test_backends_agree_across_lags(lag_days):
    prices, weights = _panel()
    _assert_same(prices, weights, lag_days=lag_days)


def test_backends_agree_on_a_wide_book():
    """More assets than bars -- exercises the inner loop bounds."""
    prices, weights = _panel(n_bars=40, n_assets=120)
    _assert_same(prices, weights)


def test_backends_agree_on_single_asset():
    prices, weights = _panel(n_assets=1)
    _assert_same(prices, weights)


def test_backends_agree_on_single_bar():
    prices, weights = _panel(n_bars=1)
    _assert_same(prices, weights)


# -- end to end, through the public API ----------------------------------


def test_full_run_matches_between_backends():
    """The result objects a caller actually receives must agree."""
    prices, weights = _panel(n_bars=300, n_assets=6)
    index = pd.bdate_range("2020-01-01", periods=prices.shape[0])
    tickers = [f"T{i}" for i in range(prices.shape[1])]
    close = pd.DataFrame(prices, index=index, columns=tickers)
    signals = pd.DataFrame(weights, index=index, columns=tickers)

    config = {"transaction_cost_bps": 5, "slippage_bps": 2, "lag_days": 1}
    py = EventDrivenBacktester({**config, "backend": "python"}).run(close, signals)
    cc = EventDrivenBacktester({**config, "backend": "cpp"}).run(close, signals)

    pd.testing.assert_series_equal(py.returns, cc.returns, atol=1e-11, check_freq=False)
    pd.testing.assert_frame_equal(py.positions, cc.positions, atol=1e-11, check_freq=False)
    pd.testing.assert_frame_equal(py.trades, cc.trades, atol=1e-11)


def test_cpp_backend_releases_the_gil():
    """Not a correctness test -- a claim in the C++ file, checked.

    If `py::gil_scoped_release` were removed the loop would still be
    correct, so nothing else here would notice. This runs two simulations on
    two threads and asserts the wall clock beats running them back to back,
    which is only possible if the GIL is actually released.
    """
    import threading
    import time

    prices, weights = _panel(n_bars=60_000, n_assets=40)

    def run_once():
        _fastloop.simulate(prices, weights, **BASE)

    run_once()  # warm up

    start = time.perf_counter()
    run_once()
    run_once()
    serial = time.perf_counter() - start

    threads = [threading.Thread(target=run_once) for _ in range(2)]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    parallel = time.perf_counter() - start

    # Two cores doing two jobs should beat one core doing two. The bar is
    # loose (0.85) so a busy CI box does not produce a flaky failure; with
    # the GIL held, parallel would be >= serial rather than below it.
    assert parallel < serial * 0.85, f"serial {serial:.3f}s vs threaded {parallel:.3f}s"


# -- backend selection ---------------------------------------------------


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        EventDrivenBacktester({"backend": "fortran"})


def test_auto_backend_prefers_cpp():
    from engine.event_driven import _resolve_backend

    assert _resolve_backend("auto") == "cpp"


def test_mismatched_shapes_rejected_by_extension():
    """The C++ accessors are unchecked, so this guard is load-bearing."""
    prices, _ = _panel(n_bars=50, n_assets=4)
    _, weights = _panel(n_bars=50, n_assets=7)

    with pytest.raises(ValueError, match="same shape"):
        _fastloop.simulate(prices, weights, **BASE)
