"""Phase 3 checkpoint tests: one sanity test per strategy, plus shared
contract tests every strategy must satisfy.

The shared tests are the valuable ones. Any strategy added later gets them
for free by being registered in `configs/strategies.yaml`, which means a new
idea cannot quietly introduce look-ahead bias or a malformed signal panel.

All data here is synthetic and deterministic, so these run offline (the
Phase 1 Yahoo TODO doesn't block them) and never flake.
"""

import numpy as np
import pandas as pd
import pytest

from engine.backtest import VectorizedBacktester
from strategies import available_strategies, load_strategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.pairs import PairsStrategy

BACKTEST_CONFIG = {
    "allocation": "equal_weight",
    "transaction_cost_bps": 5,
    "slippage_bps": 2,
    "lag_days": 1,
    "vol_lookback": 20,
}


# --------------------------------------------------------------------------
# Test data
# --------------------------------------------------------------------------


def _trending_prices() -> pd.DataFrame:
    """4 tickers with unambiguous, hand-designed momentum ordering.

    WINNER rises every day, LOSER falls every day, and two drift mildly in
    between. Deterministic so the "does momentum buy the winner?" assertion
    can't be a coincidence.
    """
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    steps = np.arange(len(dates))
    return pd.DataFrame(
        {
            "WINNER": 100.0 * (1.02**steps),
            "MILD_UP": 100.0 * (1.002**steps),
            "MILD_DOWN": 100.0 * (0.998**steps),
            "LOSER": 100.0 * (0.98**steps),
        },
        index=dates,
    )


def _oscillating_prices() -> pd.DataFrame:
    """Two tickers that swing well above and below their own rolling mean."""
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    steps = np.arange(len(dates))
    return pd.DataFrame(
        {
            "SWING": 100.0 + 8.0 * np.sin(steps / 4.0),
            # Phase-shifted so the two are stretched in opposite directions.
            "SWING2": 100.0 + 8.0 * np.cos(steps / 4.0),
        },
        index=dates,
    )


def _cointegrated_prices() -> pd.DataFrame:
    """One genuinely cointegrated pair hidden among unrelated random walks.

    LEAD is a random walk. FOLLOW tracks it via a fixed linear relationship
    plus stationary noise, so the two wander together and their spread is
    mean-reverting -- exactly what the cointegration test should find.
    NOISE1/NOISE2 are independent walks that should not be selected.
    """
    rng = np.random.default_rng(7)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    lead = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    follow = 50.0 + 1.5 * lead + rng.normal(0, 0.8, n)  # stationary residual
    return pd.DataFrame(
        {
            "LEAD": lead,
            "FOLLOW": follow,
            "NOISE1": 100.0 + np.cumsum(rng.normal(0, 1.0, n)),
            "NOISE2": 100.0 + np.cumsum(rng.normal(0, 1.0, n)),
        },
        index=dates,
    )


# Strategies configured for short synthetic panels, with the price data each
# one needs to show its behaviour.
STRATEGY_CASES = {
    "momentum": (
        lambda: MomentumStrategy(lookback=5, long_decile=0.25, short_decile=0.25, long_only=False),
        _trending_prices,
        5,
    ),
    "mean_reversion": (
        lambda: MeanReversionStrategy(window=10, entry_z=1.0, exit_z=0.25),
        _oscillating_prices,
        15,
    ),
    "pairs": (
        lambda: PairsStrategy(
            lookback=60, entry_z=1.0, exit_z=0.25, pvalue_threshold=0.99, n_candidates=10
        ),
        _cointegrated_prices,
        70,
    ),
}


# --------------------------------------------------------------------------
# Shared contract: every strategy must satisfy these
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(STRATEGY_CASES))
def test_signal_panel_is_well_formed(name):
    """Signals must be the same shape as the close panel, with no NaNs.

    The engine multiplies positions by returns element-wise; a mismatched or
    NaN-riddled panel would silently poison the whole backtest.
    """
    build, prices, _ = STRATEGY_CASES[name]
    close = prices()

    signals = build().generate_signals(close)

    assert isinstance(signals, pd.DataFrame)
    assert signals.index.equals(close.index), "signal dates don't match price dates"
    assert signals.columns.equals(close.columns), "signal tickers don't match price tickers"
    assert not signals.isna().any().any(), "signal panel contains NaNs"
    assert np.isfinite(signals.to_numpy()).all(), "signal panel contains inf"


@pytest.mark.parametrize("name", sorted(STRATEGY_CASES))
def test_no_lookahead_in_signal_generation(name):
    """A strategy's signal on day k must not react to prices after day k.

    This is the Phase 3 checkpoint's "no lookahead" requirement, and it is
    the test that would catch a stray `shift(-1)`, a centered rolling window,
    or -- in the pairs case -- fitting on data the strategy shouldn't have
    seen yet. Rewriting the future and demanding the past stay identical is
    a much stronger check than eyeballing a signal panel.
    """
    build, prices, cutoff = STRATEGY_CASES[name]
    close = prices()

    tampered = close.copy()
    # Violently rewrite everything after the cutoff. If any of it leaks
    # backwards, the assertion below fails.
    tampered.iloc[cutoff + 1 :] *= 3.0

    baseline_signals = build().generate_signals(close)
    tampered_signals = build().generate_signals(tampered)

    pd.testing.assert_frame_equal(
        baseline_signals.iloc[: cutoff + 1],
        tampered_signals.iloc[: cutoff + 1],
        obj=f"{name} signals up to the cutoff",
    )


@pytest.mark.parametrize("name", sorted(STRATEGY_CASES))
def test_strategy_runs_through_the_engine(name):
    """Each strategy must compose with the Phase 2 engine unchanged.

    The point of the shared interface: the engine has no idea which of these
    it is running.
    """
    build, prices, _ = STRATEGY_CASES[name]
    close = prices()

    signals = build().generate_signals(close)
    result = VectorizedBacktester(BACKTEST_CONFIG).run(close, signals)

    assert len(result.returns) == len(close)
    assert not result.returns.isna().any()
    assert np.isfinite(result.equity_curve.to_numpy()).all()


@pytest.mark.parametrize("name", sorted(available_strategies()))
def test_registry_builds_every_configured_strategy(name):
    """Everything in configs/strategies.yaml must actually load and run."""
    strategy = load_strategy(name)
    signals = strategy.generate_signals(_cointegrated_prices())
    assert signals.shape == _cointegrated_prices().shape


# --------------------------------------------------------------------------
# Momentum
# --------------------------------------------------------------------------


def test_momentum_longs_the_strongest_and_shorts_the_weakest():
    """The spec's named sanity check: momentum longs the top performers."""
    close = _trending_prices()

    signals = MomentumStrategy(
        lookback=5, long_decile=0.25, short_decile=0.25, long_only=False
    ).generate_signals(close)

    final = signals.iloc[-1]
    assert final["WINNER"] == 1.0, "did not go long the best performer"
    assert final["LOSER"] == -1.0, "did not short the worst performer"
    assert final["MILD_UP"] == 0.0 and final["MILD_DOWN"] == 0.0, \
        "middle of the pack should be flat"


def test_momentum_long_only_never_shorts():
    close = _trending_prices()

    signals = MomentumStrategy(lookback=5, long_decile=0.25, long_only=True).generate_signals(close)

    assert (signals >= 0).all().all(), "long_only=True still produced short positions"
    assert signals.iloc[-1]["WINNER"] == 1.0


def test_momentum_is_flat_until_the_lookback_window_fills():
    """No trailing return means no view -- not a guess from partial data."""
    close = _trending_prices()

    signals = MomentumStrategy(lookback=5, long_decile=0.25).generate_signals(close)

    assert (signals.iloc[:5] == 0.0).all().all()


# --------------------------------------------------------------------------
# Mean reversion
# --------------------------------------------------------------------------


def test_mean_reversion_fades_extremes_in_the_right_direction():
    """Overbought is a SHORT and oversold is a LONG.

    Getting this sign backwards silently converts the strategy into slow
    momentum, so it's worth pinning down explicitly.
    """
    close = _oscillating_prices()
    strategy = MeanReversionStrategy(window=10, entry_z=1.0, exit_z=0.25)

    signals = strategy.generate_signals(close)

    rolling_mean = close.rolling(10, min_periods=10).mean()
    rolling_std = close.rolling(10, min_periods=10).std()
    z = (close - rolling_mean) / rolling_std

    stretched_high = (z > 1.5) & signals.ne(0)
    stretched_low = (z < -1.5) & signals.ne(0)
    assert stretched_high.to_numpy().any(), "test data never gets overbought"
    assert stretched_low.to_numpy().any(), "test data never gets oversold"

    assert (signals[stretched_high].stack() < 0).all(), "overbought should be short"
    assert (signals[stretched_low].stack() > 0).all(), "oversold should be long"


def test_mean_reversion_holds_position_between_entry_and_exit():
    """Hysteresis: a position opened at |z|>=1.0 survives a dip to |z|=0.5.

    Without it, positions flicker every time the z-score jitters over the
    threshold, and the engine charges cost on every flicker.
    """
    close = _oscillating_prices()

    signals = MeanReversionStrategy(window=10, entry_z=1.0, exit_z=0.25).generate_signals(close)

    z = (close - close.rolling(10, min_periods=10).mean()) / close.rolling(10, min_periods=10).std()
    in_between = (z.abs() > 0.25) & (z.abs() < 1.0) & signals.ne(0)
    assert in_between.to_numpy().any(), (
        "no day where a position was held while the z-score sat between the "
        "exit and entry thresholds -- hysteresis isn't being exercised"
    )


# --------------------------------------------------------------------------
# Pairs
# --------------------------------------------------------------------------


def test_pairs_finds_the_cointegrated_pair_among_random_walks():
    close = _cointegrated_prices()
    strategy = PairsStrategy(lookback=60, entry_z=1.0, exit_z=0.25, pvalue_threshold=0.05)

    strategy.generate_signals(close)

    assert strategy.selected_pair is not None, "failed to find the planted cointegrated pair"
    assert set(strategy.selected_pair) == {"LEAD", "FOLLOW"}, (
        f"selected {strategy.selected_pair} instead of the cointegrated LEAD/FOLLOW pair"
    )
    assert strategy.pvalue < 0.05


def test_pairs_never_trades_inside_the_formation_window():
    """The formation window chose the pair, so trading it would be lookahead."""
    close = _cointegrated_prices()
    strategy = PairsStrategy(lookback=60, entry_z=1.0, exit_z=0.25, pvalue_threshold=0.05)

    signals = strategy.generate_signals(close)

    assert (signals.iloc[:60] == 0.0).all().all(), "traded during the formation window"
    assert signals.iloc[60:].abs().to_numpy().sum() > 0, "never traded at all after formation"


def test_pairs_holds_nothing_when_no_pair_cointegrates():
    """Sitting in cash is a legitimate answer, not an error."""
    close = _cointegrated_prices()
    strategy = PairsStrategy(
        lookback=60, entry_z=1.0, exit_z=0.25, pvalue_threshold=1e-12  # impossible bar
    )

    signals = strategy.generate_signals(close)

    assert strategy.selected_pair is None
    assert (signals == 0.0).all().all()


def test_pairs_positions_are_hedged_in_opposite_directions():
    """Long one leg, short the other -- that's what makes it market-neutral."""
    close = _cointegrated_prices()
    strategy = PairsStrategy(lookback=60, entry_z=1.0, exit_z=0.25, pvalue_threshold=0.05)

    signals = strategy.generate_signals(close)
    a, b = strategy.selected_pair

    active = signals[a] != 0
    assert active.any()
    assert (np.sign(signals.loc[active, a]) == -np.sign(signals.loc[active, b])).all(), (
        "the two legs should always point in opposite directions"
    )
