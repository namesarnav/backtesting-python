"""Phase 4 tests: metric formulas against hand-computed values, plus the
walk-forward harness's structural guarantees.

Metric functions are short and easy to write *plausibly* wrong -- an ddof
slip, an annualization by 252 instead of sqrt(252), a drawdown measured from
the start instead of the running peak. Checking against numbers worked out
by hand is the only way to catch that; asserting "returns a float" would
pass on all of those bugs.
"""

import numpy as np
import pandas as pd
import pytest

from metrics.performance import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    total_return,
    win_rate,
)
from metrics.validation import compare_to_benchmark, make_folds, walk_forward

SQRT_252 = np.sqrt(252)


def _series(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="B"))


# --------------------------------------------------------------------------
# Return metrics
# --------------------------------------------------------------------------


def test_total_return_compounds_rather_than_sums():
    # +10%, -50%, +20% -> 1.1 * 0.5 * 1.2 = 0.66, i.e. down 34%.
    # Summing would give -20%, which is the classic wrong answer.
    assert total_return(_series([0.10, -0.50, 0.20])) == pytest.approx(-0.34)


def test_plus_ten_then_minus_ten_loses_money():
    """The canonical demonstration that returns don't average."""
    assert total_return(_series([0.10, -0.10])) == pytest.approx(-0.01)


def test_annualized_return_matches_compounding_by_hand():
    # 0.1% a day for exactly one trading year: 1.001**252 - 1.
    daily = _series([0.001] * 252)
    assert annualized_return(daily) == pytest.approx(1.001**252 - 1)


def test_annualized_volatility_scales_by_sqrt_of_time():
    values = [0.01, 0.02, 0.03, -0.01]
    expected = pd.Series(values).std(ddof=1) * SQRT_252
    assert annualized_volatility(_series(values)) == pytest.approx(expected)


def test_sharpe_matches_hand_calculation():
    # mean 0.02, sample std 0.01 -> 0.02/0.01 * sqrt(252)
    assert sharpe_ratio(_series([0.01, 0.02, 0.03])) == pytest.approx(2.0 * SQRT_252)


def test_sortino_only_penalizes_downside():
    # excess = [-0.01, 0.02, 0.03], mean 0.0133...
    # downside rms = sqrt(0.0001 / 3) = 0.0057735
    returns = _series([-0.01, 0.02, 0.03])
    expected = (np.mean([-0.01, 0.02, 0.03]) / np.sqrt(0.0001 / 3)) * SQRT_252
    assert sortino_ratio(returns) == pytest.approx(expected)


def test_sortino_exceeds_sharpe_when_downside_is_mild():
    """A series whose volatility is mostly upside should score better on
    Sortino than on Sharpe -- that's the entire point of the measure."""
    returns = _series([0.05, 0.06, -0.01, 0.04, 0.07])
    assert sortino_ratio(returns) > sharpe_ratio(returns)


def test_max_drawdown_measures_from_the_running_peak():
    # equity: 1.1, 0.55, 0.66. Peak 1.1, trough 0.55 -> -50%.
    # Measuring from the START (1.0 -> 0.55) would wrongly give -45%.
    assert max_drawdown(_series([0.10, -0.50, 0.20])) == pytest.approx(-0.50)


def test_max_drawdown_is_zero_when_nothing_falls():
    assert max_drawdown(_series([0.01, 0.02, 0.03])) == pytest.approx(0.0)


def test_calmar_is_return_over_worst_drawdown():
    returns = _series([0.10, -0.50, 0.20])
    assert calmar_ratio(returns) == pytest.approx(annualized_return(returns) / 0.50)


def test_win_rate_ignores_flat_days():
    # 2 up, 1 down, 1 flat -> 2/3, not 2/4. A strategy sitting in cash
    # shouldn't have those days counted as losses.
    assert win_rate(_series([0.01, -0.01, 0.0, 0.02])) == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# The "no infinite Sharpe" requirement from the spec's checkpoint
# --------------------------------------------------------------------------


def test_ratios_are_nan_not_inf_when_nothing_was_traded():
    """A fold where the strategy never traded has zero volatility.

    Dividing by it yields inf, which would silently top any leaderboard and
    poison any average. NaN is the honest answer.
    """
    flat = _series([0.0] * 50)

    for value in (sharpe_ratio(flat), sortino_ratio(flat), calmar_ratio(flat)):
        assert not np.isinf(value), "produced an infinite ratio"
        assert np.isnan(value)


def test_ratios_are_nan_for_constant_nonzero_returns():
    """Constant returns also mean zero measured risk."""
    constant = _series([0.001] * 50)
    assert np.isnan(sharpe_ratio(constant))


def test_summarize_returns_every_metric_and_no_infinities():
    returns = _series(np.random.default_rng(0).normal(0.0005, 0.01, 300))
    turnover = _series(np.full(300, 0.2))

    summary = summarize(returns, turnover=turnover)

    for key in ("total_return", "ann_return", "ann_volatility", "sharpe", "sortino",
                "max_drawdown", "calmar", "win_rate", "avg_turnover", "n_periods"):
        assert key in summary
    assert not any(np.isinf(v) for v in summary.values())


# --------------------------------------------------------------------------
# Walk-forward mechanics
# --------------------------------------------------------------------------


def test_folds_tile_the_test_windows_without_gaps_or_overlap():
    """Test windows must partition the timeline exactly once.

    Overlapping them would double-count days when the out-of-sample series is
    stitched together, inflating the sample and any confidence drawn from it.
    """
    folds = make_folds(n_rows=1258, train_window=504, test_window=126, step=126)

    assert len(folds) == 5
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert later.test_start == earlier.test_end, "test windows are not contiguous"
    assert all(f.test_end - f.test_start == 126 for f in folds)
    assert all(f.test_start - f.train_start == 504 for f in folds)


def test_make_folds_rejects_a_sample_too_short_for_one_fold():
    with pytest.raises(ValueError, match="need at least"):
        make_folds(n_rows=100, train_window=504, test_window=126, step=126)


def test_train_window_always_precedes_its_test_window():
    """The property that makes it out-of-sample at all."""
    for fold in make_folds(2000, 504, 126, 126):
        assert fold.train_start < fold.test_start < fold.test_end


# --------------------------------------------------------------------------
# Benchmark comparison
# --------------------------------------------------------------------------


def test_a_strategy_identical_to_the_benchmark_has_beta_one_and_no_alpha():
    rng = np.random.default_rng(3)
    bench = _series(rng.normal(0.0004, 0.01, 400))

    comparison = compare_to_benchmark(bench.copy(), bench)

    assert comparison["beta"] == pytest.approx(1.0)
    assert comparison["alpha_annual"] == pytest.approx(0.0, abs=1e-9)
    assert comparison["correlation"] == pytest.approx(1.0)
    assert comparison["excess_ann_return"] == pytest.approx(0.0, abs=1e-9)


def test_double_the_benchmark_has_beta_two():
    rng = np.random.default_rng(4)
    bench = _series(rng.normal(0.0004, 0.01, 400))

    comparison = compare_to_benchmark(bench * 2.0, bench)

    assert comparison["beta"] == pytest.approx(2.0)


def test_market_neutral_strategy_has_near_zero_beta():
    rng = np.random.default_rng(5)
    bench = _series(rng.normal(0.0004, 0.01, 600))
    independent = _series(rng.normal(0.0004, 0.01, 600))

    comparison = compare_to_benchmark(independent, bench)

    assert abs(comparison["beta"]) < 0.2


# --------------------------------------------------------------------------
# Walk-forward end to end
# --------------------------------------------------------------------------


def _synthetic_panel(n=900, k=8, seed=11):
    """A (field, ticker) panel shaped like DataLoader's output."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    tickers = [f"T{i}" for i in range(k)]
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (n, k)), axis=0)),
        index=dates, columns=tickers,
    )
    frames = {"close": close, "open": close, "high": close * 1.01,
              "low": close * 0.99, "volume": close * 0 + 1e6}
    panel = pd.concat(frames, axis=1)
    panel.columns.names = ["field", "ticker"]
    return panel


BACKTEST_CONFIG = {"allocation": "equal_weight", "transaction_cost_bps": 5,
                   "slippage_bps": 2, "lag_days": 1, "vol_lookback": 20}


def test_walk_forward_produces_contiguous_non_overlapping_oos_series():
    """The stitched out-of-sample series must contain each date exactly once."""
    panel = _synthetic_panel()

    result = walk_forward(panel, "momentum", BACKTEST_CONFIG,
                          train_window=252, test_window=63)

    oos = result["oos_returns"]
    assert not oos.index.duplicated().any(), "a date appears in two test windows"
    assert oos.index.is_monotonic_increasing
    assert len(oos) == result["n_folds"] * 63
    assert len(result["folds"]) == result["n_folds"]


def test_walk_forward_reports_in_and_out_of_sample_separately():
    """Both must be present -- their gap is the actual finding."""
    panel = _synthetic_panel()

    result = walk_forward(panel, "momentum", BACKTEST_CONFIG,
                          train_window=252, test_window=63)

    assert "is_summary" in result and "oos_summary" in result
    assert set(result["folds"].columns) >= {"fold", "is_sharpe", "oos_sharpe",
                                            "test_start", "test_end"}


def test_out_of_sample_windows_never_precede_their_training_data():
    """The guarantee that makes the OOS number meaningful."""
    panel = _synthetic_panel()

    folds = walk_forward(panel, "momentum", BACKTEST_CONFIG,
                         train_window=252, test_window=63)["folds"]

    assert (folds["train_start"] < folds["test_start"]).all()


def test_param_grid_selects_parameters_and_records_the_choice():
    """With a grid, each fold picks its own parameters on train data only."""
    panel = _synthetic_panel()

    result = walk_forward(panel, "momentum", BACKTEST_CONFIG,
                          train_window=252, test_window=63,
                          param_grid={"lookback": [20, 60]})

    assert "chosen_params" in result["folds"].columns
    assert result["folds"]["chosen_params"].notna().all()


def test_walk_forward_runs_for_every_registered_strategy():
    """Any strategy in the registry must survive the harness unchanged."""
    from strategies import available_strategies

    panel = _synthetic_panel(n=800)
    for name in available_strategies():
        result = walk_forward(panel, name, BACKTEST_CONFIG,
                              train_window=252, test_window=63)
        assert len(result["oos_returns"]) > 0
        assert not np.isinf(result["oos_summary"]["sharpe"] or 0)
