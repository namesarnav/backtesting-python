"""Phase 6 checkpoint tests for engine.event_driven.EventDrivenBacktester.

The checkpoint asks for results "directionally consistent" with the
vectorized engine, with any differences reconciled and explained. These tests
pin down something stronger and more useful:

  * the two engines agree **exactly** on gross returns (to float noise), which
    proves they hold the same book on the same days -- so the lag convention,
    the allocation and the one-bar offset all line up;
  * they differ **only** on costs, and the tests here demonstrate the specific
    mechanism (weight drift between rebalances) rather than just tolerating a
    gap.

Everything is synthetic and offline, like tests/test_backtest.py.
"""

import numpy as np
import pandas as pd
import pytest

from engine.backtest import VectorizedBacktester
from engine.event_driven import EventDrivenBacktester, reconcile

ZERO_COST = {
    "allocation": "equal_weight",
    "transaction_cost_bps": 0,
    "slippage_bps": 0,
    "lag_days": 1,
}

WITH_COST = {**ZERO_COST, "transaction_cost_bps": 5, "slippage_bps": 2}

TICKERS = ["AAA", "BBB", "CCC"]


def _prices(n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    """Deterministic geometric random walk -- same panel on every run."""
    rng = np.random.default_rng(seed)
    steps = 1.0 + rng.normal(0.0005, 0.015, size=(n_days, len(TICKERS)))
    prices = 100.0 * np.cumprod(steps, axis=0)
    index = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=index, columns=TICKERS)


def _long_short_signals(close: pd.DataFrame) -> pd.DataFrame:
    """A signal that actually changes: rank names by 5-day momentum."""
    momentum = close.pct_change(5).fillna(0.0)
    ranks = momentum.rank(axis=1)
    return pd.DataFrame(
        np.where(ranks == len(TICKERS), 1.0, np.where(ranks == 1, -1.0, 0.0)),
        index=close.index,
        columns=close.columns,
    )


def _toy_lookahead_panel():
    """The hand-computed panel from PHASE2_GUIDE.md §4.

    A jumps on day 1, B on day 2, and the signal chases each jump the day
    after it happens. An engine with look-ahead bias "earns" both jumps; an
    honest one earns nothing, because by the time it can act, the move is
    already in the price.
    """
    index = pd.bdate_range("2021-01-04", periods=3)
    close = pd.DataFrame(
        {"A": [100.0, 110.0, 110.0], "B": [50.0, 50.0, 55.0]}, index=index
    )
    signals = pd.DataFrame(
        {"A": [0.0, 1.0, 0.0], "B": [0.0, 0.0, 1.0]}, index=index
    )
    return close, signals


# -- the core reconciliation ---------------------------------------------


@pytest.mark.parametrize("signal_kind", ["constant_long", "long_short"])
def test_gross_returns_match_vectorized_exactly(signal_kind):
    """Both engines must hold the same book on the same days.

    Gross return is the part of the answer that involves no modelling
    choices at all -- it is just "what we held times what it did". If these
    diverge, the engines disagree about the lag or the allocation, which is
    a bug, not a difference of opinion. The tolerance is float noise, not a
    fudge factor.
    """
    close = _prices()
    signals = (
        pd.DataFrame(1.0, index=close.index, columns=close.columns)
        if signal_kind == "constant_long"
        else _long_short_signals(close)
    )

    vectorized = VectorizedBacktester(ZERO_COST).run(close, signals)
    event_driven = EventDrivenBacktester(ZERO_COST).run(close, signals)

    difference = (vectorized.gross_returns - event_driven.gross_returns).abs()
    assert difference.max() < 1e-12


def test_positions_match_vectorized():
    """The engines are exposed to the same weights on the same bars.

    This is the assertion that would have caught the off-by-one: filling at
    the close of bar t means the position is exposed on bar t+1, so the
    event-driven engine reports weights shifted by one relative to the ones
    it established. Get that wrong and the whole book is a day late.
    """
    close = _prices()
    signals = _long_short_signals(close)

    vectorized = VectorizedBacktester(ZERO_COST).run(close, signals)
    event_driven = EventDrivenBacktester(ZERO_COST).run(close, signals)

    pd.testing.assert_frame_equal(
        vectorized.positions, event_driven.positions, atol=1e-12, check_freq=False
    )


def test_zero_cost_net_returns_match_vectorized():
    """With costs off there is nothing left to disagree about."""
    close = _prices()
    signals = _long_short_signals(close)

    vectorized = VectorizedBacktester(ZERO_COST).run(close, signals)
    event_driven = EventDrivenBacktester(ZERO_COST).run(close, signals)

    assert (vectorized.returns - event_driven.returns).abs().max() < 1e-12


# -- look-ahead ----------------------------------------------------------


def test_no_lookahead_on_toy_panel():
    """The guide §4 numbers, reproduced by the event-driven engine.

    An engine that traded on the same bar its signal was computed would show
    +10% on each of days 1 and 2. Both must be zero.
    """
    close, signals = _toy_lookahead_panel()
    result = EventDrivenBacktester(ZERO_COST).run(close, signals)

    assert result.returns.iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert result.returns.iloc[1] == pytest.approx(0.0, abs=1e-12)
    assert result.returns.iloc[2] == pytest.approx(0.0, abs=1e-12)


def test_lag_days_zero_rejected():
    """Inherited from the vectorized engine's config guard, on purpose."""
    with pytest.raises(ValueError, match="lag_days must be >= 1"):
        EventDrivenBacktester({**ZERO_COST, "lag_days": 0})


def test_extra_lag_delays_the_book():
    """lag_days=2 should hold each target one bar later than lag_days=1."""
    close = _prices()
    signals = _long_short_signals(close)

    fast = EventDrivenBacktester(ZERO_COST).run(close, signals)
    slow = EventDrivenBacktester({**ZERO_COST, "lag_days": 2}).run(close, signals)

    shifted = fast.positions.shift(1).fillna(0.0)
    pd.testing.assert_frame_equal(
        shifted, slow.positions, atol=1e-12, check_freq=False
    )


# -- the accounting the vectorized engine cannot do ----------------------


def test_accounting_identity_holds_every_bar():
    """equity == cash + shares @ price, on every single bar.

    This is the invariant that justifies the whole exercise: the vectorized
    engine has no cash balance to be wrong about, so it cannot check this.
    A leak here would mean money appearing or vanishing.
    """
    close = _prices()
    result = EventDrivenBacktester(WITH_COST).run(close, _long_short_signals(close))

    marked = (result.holdings * close).sum(axis=1) + result.cash
    assert (marked - result.equity).abs().max() < 1e-6


def test_scale_invariance_with_fractional_shares():
    """Starting capital must not change the returns when shares can split."""
    close = _prices()
    signals = _long_short_signals(close)

    small = EventDrivenBacktester({**WITH_COST, "initial_capital": 10_000}).run(close, signals)
    large = EventDrivenBacktester({**WITH_COST, "initial_capital": 50_000_000}).run(close, signals)

    assert (small.returns - large.returns).abs().max() < 1e-12


def test_constant_signal_still_costs_money():
    """The headline difference between the two engines, isolated.

    A permanently-constant signal has *zero* turnover in the vectorized
    engine: the target weights never change, so `|w[t] - w[t-1]|` is zero
    forever and it charges nothing after the initial buy-in.

    But holding a constant *weight* is not the same as holding a constant
    *position*. Prices move overnight, the weights drift, and pulling them
    back to equal-weight is a real trade with a real commission. The
    event-driven engine sees those trades because it tracks shares.
    """
    close = _prices()
    signals = pd.DataFrame(1.0, index=close.index, columns=close.columns)

    vectorized = VectorizedBacktester(WITH_COST).run(close, signals)
    event_driven = EventDrivenBacktester(WITH_COST).run(close, signals)

    # Vectorized: one buy-in (turnover 1.0) and nothing after.
    assert vectorized.turnover.iloc[2:].sum() == pytest.approx(0.0, abs=1e-12)
    # Event-driven: rebalancing trades on essentially every bar.
    assert event_driven.turnover.iloc[2:].sum() > 0.01
    assert event_driven.costs.sum() > vectorized.costs.sum()


def test_rebalance_threshold_reduces_turnover():
    """A no-trade band is the standard fix for drift-driven churn."""
    close = _prices()
    signals = pd.DataFrame(1.0, index=close.index, columns=close.columns)

    churny = EventDrivenBacktester(WITH_COST).run(close, signals)
    banded = EventDrivenBacktester(
        {**WITH_COST, "rebalance_threshold": 0.01}
    ).run(close, signals)

    assert banded.turnover.sum() < churny.turnover.sum()
    assert banded.costs.sum() < churny.costs.sum()


def test_integer_shares_tracks_but_does_not_match():
    """Whole-share orders introduce real, bounded tracking error."""
    close = _prices()
    signals = _long_short_signals(close)

    fractional = EventDrivenBacktester(
        {**ZERO_COST, "initial_capital": 100_000}
    ).run(close, signals)
    whole = EventDrivenBacktester(
        {**ZERO_COST, "initial_capital": 100_000, "fractional_shares": False}
    ).run(close, signals)

    difference = (fractional.returns - whole.returns).abs()
    assert difference.max() > 0.0, "whole-share rounding should change something"
    # ...but only at the margin: a $100k account can express a target weight
    # in $100 stocks to well within a basis point a day.
    assert difference.max() < 1e-3


def test_whole_share_error_shrinks_with_capital():
    """Rounding error is a fixed dollar amount, so it dilutes as size grows."""
    close = _prices()
    signals = _long_short_signals(close)
    reference = EventDrivenBacktester(ZERO_COST).run(close, signals).returns

    def tracking_error(capital: float) -> float:
        result = EventDrivenBacktester(
            {**ZERO_COST, "initial_capital": capital, "fractional_shares": False}
        ).run(close, signals)
        return float((result.returns - reference).abs().mean())

    assert tracking_error(10_000_000) < tracking_error(50_000)


# -- the fill log --------------------------------------------------------


def test_fill_log_reconciles_with_turnover():
    """The trade log must account for every dollar of reported turnover."""
    close = _prices()
    result = EventDrivenBacktester(WITH_COST).run(close, _long_short_signals(close))

    fills = result.trades
    assert not fills.empty
    assert set(fills.columns) == {"date", "ticker", "shares", "price", "notional", "cost"}
    # notional is signed (a sale is negative); turnover counts both sides.
    assert fills["notional"].abs().sum() == pytest.approx(
        (result.turnover * result.equity.shift(1).fillna(
            EventDrivenBacktester(WITH_COST).initial_capital)).sum(),
        rel=1e-9,
    )
    assert (fills["cost"] >= 0).all()


def test_fill_prices_are_the_bar_close():
    """Fills happen at the close, so every price must appear in the panel."""
    close = _prices()
    result = EventDrivenBacktester(WITH_COST).run(close, _long_short_signals(close))

    for _, fill in result.trades.head(50).iterrows():
        assert close.loc[fill["date"], fill["ticker"]] == pytest.approx(fill["price"])


# -- input validation ----------------------------------------------------


def test_rejects_non_positive_prices():
    close = _prices(n_days=10)
    close.iloc[3, 1] = 0.0
    signals = pd.DataFrame(1.0, index=close.index, columns=close.columns)

    with pytest.raises(ValueError, match="must all be positive"):
        EventDrivenBacktester(ZERO_COST).run(close, signals)


def test_rejects_mismatched_signals():
    close = _prices(n_days=10)
    signals = pd.DataFrame(1.0, index=close.index, columns=["AAA", "BBB"])

    with pytest.raises(ValueError, match="identical tickers"):
        EventDrivenBacktester(ZERO_COST).run(close, signals)


def test_rejects_non_positive_capital():
    with pytest.raises(ValueError, match="initial_capital must be positive"):
        EventDrivenBacktester({**ZERO_COST, "initial_capital": 0})


# -- the checkpoint helper -----------------------------------------------


def test_reconcile_reports_agreement_and_cost_gap():
    """`reconcile` is the Phase 6 checkpoint expressed as a function."""
    close = _prices()
    signals = _long_short_signals(close)

    # Costs off: the engines must agree on gross return to float noise.
    assert reconcile(close, signals, ZERO_COST)["max_gross_diff"] < 1e-12

    report = reconcile(close, signals, WITH_COST)
    # Costs on: orders are sized on pre-commission equity, so the books
    # differ by a fraction of the cost rate. Bounded, and far below the
    # 7bp/day cost it comes from.
    assert report["max_gross_diff"] < 1e-4
    assert report["n_fills"] > 0
    assert report["event_driven_total_cost"] > report["vectorized_total_cost"]
    # The engines land in the same place; the gap is the cost model.
    assert abs(
        report["vectorized_total_return"] - report["event_driven_total_return"]
    ) < 0.05
