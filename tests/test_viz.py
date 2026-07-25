"""Phase 5 tests: charts render, save, and survive awkward inputs.

Chart *appearance* is checked by looking at the output (the spec's checkpoint
asks for visual confirmation, and that was done). What is worth automating is
that the plotting code doesn't crash -- particularly on the degenerate series
that walk-forward and cash-heavy strategies produce all the time.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

from viz.plots import (
    generate_all,
    plot_comparison_dashboard,
    plot_equity_with_drawdown,
    plot_rolling_sharpe,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2019-01-01", periods=400, freq="B")
    return {
        "momentum": pd.Series(rng.normal(0.0005, 0.011, 400), index=idx),
        "mean_reversion": pd.Series(rng.normal(-0.0002, 0.009, 400), index=idx),
        "pairs": pd.Series(rng.normal(0.0001, 0.004, 400), index=idx),
    }


@pytest.fixture
def benchmark(returns):
    rng = np.random.default_rng(1)
    idx = returns["momentum"].index
    return pd.Series(rng.normal(0.0004, 0.010, len(idx)), index=idx)


def test_every_chart_is_written_to_disk(returns, benchmark, tmp_path):
    paths = generate_all(returns, benchmark=benchmark, output_dir=tmp_path)

    assert len(paths) == len(returns) + 2  # one per strategy, plus sharpe + dashboard
    for path in paths:
        assert path.exists() and path.stat().st_size > 0
        assert path.suffix == ".png"


def test_charts_do_not_leak_open_figures(returns, benchmark, tmp_path):
    """Every figure must be closed, or a long run exhausts memory."""
    generate_all(returns, benchmark=benchmark, output_dir=tmp_path)
    assert not matplotlib.pyplot.get_fignums(), "figures were left open"


def test_handles_a_strategy_that_never_traded(tmp_path):
    """All-zero returns: no volatility, no drawdown, nothing to label.

    Common in practice -- pairs sits in cash whenever nothing is cointegrated,
    and walk-forward folds can contain no trades at all.
    """
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    flat = pd.Series(0.0, index=idx)

    path = plot_equity_with_drawdown(flat, "flat_strategy", output_dir=tmp_path)
    assert path.exists()


def test_rolling_sharpe_survives_a_window_longer_than_the_data(tmp_path):
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    short = {"momentum": pd.Series(np.linspace(-0.01, 0.01, 30), index=idx)}

    path = plot_rolling_sharpe(short, window=60, output_dir=tmp_path)
    assert path.exists()


def test_dashboard_works_without_a_benchmark(returns, tmp_path):
    path = plot_comparison_dashboard(returns, benchmark=None, output_dir=tmp_path)
    assert path.exists()


def test_unknown_strategy_names_still_get_a_colour(tmp_path):
    """A fourth strategy added later must not crash the palette lookup."""
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    extra = {f"strategy_{i}": pd.Series(np.full(100, 0.001), index=idx) for i in range(5)}

    path = plot_comparison_dashboard(extra, output_dir=tmp_path)
    assert path.exists()
