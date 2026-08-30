"""Phase 2 checkpoint tests for engine.backtest.VectorizedBacktester.

These cover the Phase 2 checkpoint: the buy-and-hold regression test the
spec calls for, plus the explicit look-ahead tests it requires.

PHASE2_GUIDE.md §9 explains what each one actually proves (including the
equal-weight vs buy-and-hold trap that produces confusing near-miss
failures if the engine is ever changed to drift weights instead of
rebalancing daily).

Everything here uses hand-built synthetic prices, not `DataLoader` — so these
run offline and are unaffected by the Phase 1 Yahoo rate-limit TODO.

The look-ahead tests at the bottom are the explicit ones the spec requires.
`toy_lookahead_panel()` builds the data from guide §4 that they check against.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

# Costs off: isolates the core return calculation from the cost model, so a
# failure points at one thing rather than two.
ZERO_COST_CONFIG = {
    "allocation": "equal_weight",
    "transaction_cost_bps": 0,
    "slippage_bps": 0,
    "lag_days": 1,
    "vol_lookback": 20,
}

# Matches configs/backtest.yaml.
DEFAULT_CONFIG = {
    "allocation": "equal_weight",
    "transaction_cost_bps": 5,
    "slippage_bps": 2,
    "lag_days": 1,
    "vol_lookback": 20,
}


def _backtester(config: dict):
    """Build a VectorizedBacktester, importing it lazily.

    The import lives in here rather than at the top of the file on purpose:
    while `engine/backtest.py` is still empty, a top-level import would be a
    *collection* error, which aborts the entire test session -- you would not
    be able to run the Phase 0/1 tests at all. Importing inside the function
    turns it into three ordinary test failures instead, so the rest of the
    suite keeps running while you work.
    """
    from engine.backtest import VectorizedBacktester

    return VectorizedBacktester(config)


def _toy_prices() -> pd.DataFrame:
    """Deterministic 3-ticker, 10-day close-price panel.

    Hardcoded rather than randomly generated so the expected numbers are
    stable and you can check any single value with a calculator.
    """
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    return pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 103.0, 102.0, 105.0, 104.0, 106.0, 108.0, 107.0, 110.0],
            "BBB": [50.0, 50.5, 50.0, 51.0, 51.5, 52.0, 51.0, 52.5, 53.0, 52.0],
            "CCC": [200.0, 198.0, 199.0, 201.0, 200.0, 203.0, 205.0, 204.0, 206.0, 208.0],
        },
        index=dates,
    )


def _always_long_signals(close: pd.DataFrame) -> pd.DataFrame:
    """The 'always fully long, equal weight' signal the spec's checkpoint uses.

    Every ticker, every day, weight 1 -- the most boring possible strategy.
    If the engine can't get this trivial case right, nothing built on top of
    it can be trusted.
    """
    return pd.DataFrame(1.0, index=close.index, columns=close.columns)


def _tradeable_dates(close: pd.DataFrame) -> pd.DatetimeIndex:
    """Dates on which a position could actually be held.

    Day 0 has no prior day, so there is no return and (after the 1-day lag)
    no position either. Whether you keep that row as 0.0 or drop it entirely
    is your design call -- see PHASE2_GUIDE.md §11 question 1 -- so these
    tests only assert on day 1 onward and accept either choice.
    """
    return close.index[1:]


def test_buy_and_hold_matches_manual_calculation():
    """THE regression test. Spec's Phase 2 checkpoint.

    'Always fully long, equal weight', with costs off, must reproduce a
    calculation you can do by hand.

    Because this engine rebalances to target weights daily (PHASE2_GUIDE.md
    §3), the correct hand-calculation is the plain average of each day's
    individual stock returns. If this fails by a *small* amount, re-read
    guide §9 -- you have probably implemented drifting buy-and-hold weights
    instead of daily rebalancing.
    """
    close = _toy_prices()
    signals = _always_long_signals(close)

    result = _backtester(ZERO_COST_CONFIG).run(close, signals)

    # Manual calculation: each day, the average return across the 3 stocks.
    expected = close.pct_change().mean(axis=1)

    dates = _tradeable_dates(close)
    assert set(dates).issubset(set(result.returns.index)), (
        "returns series is missing tradeable dates"
    )

    np.testing.assert_allclose(
        result.returns.loc[dates].to_numpy(),
        expected.loc[dates].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        err_msg="portfolio returns do not match the manual equal-weight calculation",
    )

    assert not result.returns.isna().any(), "returns series contains NaNs"


def test_equity_curve_matches_cumulative_product():
    """The equity curve must be internally consistent with the returns.

    Returns compound -- +10% then -10% leaves you down 1%, not flat -- so the
    curve is a cumulative product, never a running sum.
    """
    close = _toy_prices()
    signals = _always_long_signals(close)

    result = _backtester(ZERO_COST_CONFIG).run(close, signals)

    expected = (1.0 + result.returns).cumprod()

    np.testing.assert_allclose(
        result.equity_curve.to_numpy(),
        expected.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        err_msg="equity curve is not (1 + returns).cumprod()",
    )


def test_costs_reduce_returns():
    """Turning costs on must make you poorer, and only on days you traded.

    With an always-long signal the weights are constant after the initial
    buy, so exactly one day has turnover: the day the portfolio goes from
    holding nothing to fully invested.
    """
    close = _toy_prices()
    signals = _always_long_signals(close)

    free = _backtester(ZERO_COST_CONFIG).run(close, signals)
    costly = _backtester(DEFAULT_CONFIG).run(close, signals)

    dates = _tradeable_dates(close)
    free_r = free.returns.loc[dates]
    costly_r = costly.returns.loc[dates]

    assert (costly_r <= free_r + 1e-12).all(), (
        "costs increased returns on some day -- check the sign, and that you "
        "took absolute values of position changes"
    )
    assert (costly_r < free_r - 1e-12).any(), (
        "costs made no difference on any day -- are they being applied at all?"
    )


def toy_lookahead_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The worked example from PHASE2_GUIDE.md §4, as test data.

    Two stocks, three days, and a signal that chases whatever just moved:

        prices    A      B          signal    A    B
        day0     100     50         day0      0    0
        day1     110     50         day1      1    0   ("A jumped, buy A")
        day2     110     55         day2      0    1   ("B jumped, buy B")

    Traded WITHOUT a lag this returns roughly +21% -- pure fiction, earned by
    acting on prices that were not knowable at the time.

    Traded WITH lag=1 it returns 0% before costs, and day2 pays 1 x 7bp of
    turnover to buy A, so:

        day1 =  0.0
        day2 = -0.0007

    Those two numbers are what the engine must produce on this data, and
    `test_toy_panel_matches_hand_calculation` asserts exactly that. The gap
    between +21% and 0% is enormous and unmistakable by design, so the tests
    built on this data fail loudly if the lag is ever removed. Worth doing
    once by hand: delete the shift in `_lag_positions`, watch these fail,
    put it back.
    """
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    close = pd.DataFrame(
        {"A": [100.0, 110.0, 110.0], "B": [50.0, 50.0, 55.0]},
        index=dates,
    )
    signals = pd.DataFrame(
        {"A": [0.0, 1.0, 0.0], "B": [0.0, 0.0, 1.0]},
        index=dates,
    )
    return close, signals


# ---------------------------------------------------------------------------
# Look-ahead tests. The spec requires an explicit one for Phase 2; these are
# it. PHASE2_GUIDE.md §9 describes the exercise version you should still do
# by hand: delete the shift in `_lag_positions`, watch these fail, put it back.
# ---------------------------------------------------------------------------


def test_toy_panel_matches_hand_calculation():
    """The worked example from PHASE2_GUIDE.md §4, checked to the last digit.

    A signal that chases whatever just moved. Traded honestly it earns
    nothing and pays 7bp to buy in; only look-ahead makes it look good.
    """
    close, signals = toy_lookahead_panel()

    result = _backtester(DEFAULT_CONFIG).run(close, signals)

    np.testing.assert_allclose(
        result.returns.to_numpy(),
        [0.0, 0.0, -0.0007],
        rtol=0,
        atol=1e-12,
        err_msg="engine disagrees with the hand-computed example in PHASE2_GUIDE.md §4",
    )


def test_position_held_today_is_the_signal_from_lag_days_ago():
    """Structural check: positions are literally yesterday's target weights."""
    close, signals = toy_lookahead_panel()

    result = _backtester(DEFAULT_CONFIG).run(close, signals)

    # day0: nothing was known beforehand, so we were flat.
    assert result.positions.iloc[0].abs().sum() == 0.0
    # day1 holds day0's signal (flat); day2 holds day1's signal (all of A).
    assert result.positions.iloc[1].to_dict() == {"A": 0.0, "B": 0.0}
    assert result.positions.iloc[2].to_dict() == {"A": 1.0, "B": 0.0}


def test_signal_on_a_given_day_cannot_change_that_days_return():
    """The general no-look-ahead property, not tied to one hand example.

    Take any two strategies that agree up to day k and disagree from day k
    onward. If the engine is honest, their returns must be identical through
    day k -- a decision made on day k can only show up in the P&L from k+1.
    An engine with look-ahead bias fails this immediately, because day k's
    new signal would earn day k's return.
    """
    close = _toy_prices()
    baseline = _always_long_signals(close)

    k = 5
    tampered = baseline.copy()
    # Rewrite day k's signal into something completely different: short one
    # name, flat the rest. If any of that leaks backwards, we have a bug.
    tampered.iloc[k] = [-1.0, 0.0, 0.0]

    base_result = _backtester(DEFAULT_CONFIG).run(close, baseline)
    tampered_result = _backtester(DEFAULT_CONFIG).run(close, tampered)

    np.testing.assert_allclose(
        tampered_result.returns.iloc[: k + 1].to_numpy(),
        base_result.returns.iloc[: k + 1].to_numpy(),
        rtol=0,
        atol=1e-15,
        err_msg=(
            "changing the signal on day k altered the return on day k or earlier -- "
            "the engine is trading on information it should not have yet"
        ),
    )

    # ...and the change must actually land somewhere, or the test is vacuous.
    assert not np.allclose(
        tampered_result.returns.iloc[k + 1 :].to_numpy(),
        base_result.returns.iloc[k + 1 :].to_numpy(),
    ), "tampering with the signal changed nothing at all -- test proves nothing"


def test_lag_of_zero_is_rejected():
    """A lag of 0 IS look-ahead bias, so the engine must refuse to run it."""
    import pytest

    with pytest.raises(ValueError, match="lag_days"):
        _backtester({**DEFAULT_CONFIG, "lag_days": 0})


# ---------------------------------------------------------------------------
# The non-functional requirement, made testable
# ---------------------------------------------------------------------------


def test_engine_contains_no_python_level_iteration():
    """"Vectorized" is a claim about the code, so assert it against the code.

    The engine must express the whole backtest as whole-table operations --
    no `for`/`while` over dates or tickers, and no comprehension standing in
    for one. Every metric this repo reports would be unchanged by a slow
    Python loop, so nothing else in the suite would notice if one appeared;
    this parses the module and looks.
    """
    source = Path(__file__).resolve().parent.parent / "engine" / "backtest.py"
    tree = ast.parse(source.read_text())

    offenders = sorted(
        f"{type(node).__name__} on line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.While, ast.AsyncFor))
    ) + sorted(
        f"comprehension on line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
    )

    assert not offenders, (
        "engine/backtest.py must stay loop-free -- found: " + ", ".join(offenders)
    )
