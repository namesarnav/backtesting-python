"""Walk-forward validation and benchmark comparison.

A single backtest over five years tells you how a strategy did on *one*
sample of history. That is weak evidence, because every choice made while
building it -- the lookback, the thresholds, the universe -- was made by
someone who had already seen that history. Even with no explicit fitting,
the strategy has been tuned by the developer's own knowledge of what worked.

**Walk-forward validation** attacks this by repeatedly splitting time:

    |------ train 504d ------|-- test 126d --|
              |------ train 504d ------|-- test 126d --|
                        |------ train 504d ------|-- test 126d --|

Parameters are chosen on each train window and judged only on the test
window that follows it -- data the choice never saw. Stitching the test
windows together gives an out-of-sample track record, which is a far better
estimate of live performance than any in-sample number.

**The gap between in-sample and out-of-sample is the real output.** A
strategy with IS Sharpe 2.5 and OOS Sharpe 0.1 is not a good strategy that
got unlucky; it is a curve-fit. Both are reported side by side for exactly
that comparison.

---

**How each fold avoids leakage.** Per fold, the strategy is handed only
`prices[train_start : test_end]` -- never the full panel. This matters more
than it looks:

  * Strategies need warm-up. Momentum with a 126-day lookback produces
    nothing for its first 126 days. Handing it the train window first means
    it is fully warmed by the time the test window starts, so the test
    window is evaluated properly rather than half-empty.
  * `PairsStrategy` picks its pair from the first `lookback` rows of
    whatever it is given. Slicing per fold means that formation window lands
    inside the *train* period, so the pair selection is genuinely
    out-of-sample with respect to the test period. Handing it the full panel
    would fix the pair at 2019 forever and quietly leak.

Returns are then split at `test_start`: everything before is in-sample,
everything after is out-of-sample.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.backtest import VectorizedBacktester
from metrics.performance import TRADING_DAYS, sharpe_ratio, summarize
from strategies import load_strategy


@dataclass(frozen=True)
class Fold:
    """One train/test split, as positional slices into the date index."""

    index: int
    train_start: int
    test_start: int
    test_end: int

    def describe(self, dates: pd.DatetimeIndex) -> str:
        return (
            f"fold {self.index}: train {dates[self.train_start].date()}"
            f"->{dates[self.test_start - 1].date()}, "
            f"test {dates[self.test_start].date()}->{dates[self.test_end - 1].date()}"
        )


def make_folds(
    n_rows: int, train_window: int, test_window: int, step: int
) -> list[Fold]:
    """Roll a train/test window forward through the sample.

    When `step == test_window` the test windows tile the timeline exactly
    once, which is what makes stitching them into a continuous out-of-sample
    series valid. A smaller step would overlap them and double-count days.
    """
    if train_window < 1 or test_window < 1 or step < 1:
        raise ValueError("train_window, test_window and step must all be >= 1")
    if n_rows < train_window + test_window:
        raise ValueError(
            f"need at least {train_window + test_window} rows for one fold, got {n_rows}"
        )

    folds = []
    train_start = 0
    while train_start + train_window + test_window <= n_rows:
        test_start = train_start + train_window
        folds.append(
            Fold(
                index=len(folds),
                train_start=train_start,
                test_start=test_start,
                test_end=test_start + test_window,
            )
        )
        train_start += step
    return folds


def _param_combinations(grid: dict[str, list] | None) -> list[dict]:
    """Cartesian product of a {param: [values]} grid."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[k] for k in keys))
    ]


def _run(panel: pd.DataFrame, strategy, backtest_config: dict):
    """Generate signals and backtest them over whatever slice is given."""
    signals = strategy.generate_signals(panel)
    return VectorizedBacktester(backtest_config).run(panel["close"], signals)


def walk_forward(
    panel: pd.DataFrame,
    strategy_name: str,
    backtest_config: dict,
    train_window: int = 504,
    test_window: int = 126,
    step: int | None = None,
    param_grid: dict[str, list] | None = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """Run rolling train/test validation for one registered strategy.

    Returns a dict with:
      `folds`       -- per-fold in-sample and out-of-sample metrics
      `oos_returns` -- test windows stitched into one continuous series
      `oos_summary` -- metrics over that stitched series (the headline)
      `is_summary`  -- the same over the concatenated train windows

    `param_grid` turns this into true walk-forward *optimization*: each train
    window picks its own best parameters by Sharpe, and the test window
    reports what that choice actually earned next. Without a grid, the
    strategy's configured parameters are used throughout and the exercise is
    purely about IS/OOS stability across time.
    """
    step = test_window if step is None else step
    dates = panel.index
    folds = make_folds(len(dates), train_window, test_window, step)

    rows, oos_chunks, is_chunks = [], [], []

    for fold in folds:
        # Only this fold's own history is ever visible -- see module docstring.
        window = panel.iloc[fold.train_start : fold.test_end]
        train_dates = window.index[: train_window]
        test_dates = window.index[train_window :]

        best_params, best_score, best_result = {}, -np.inf, None
        for params in _param_combinations(param_grid):
            strategy = load_strategy(strategy_name, **params)
            result = _run(window, strategy, backtest_config)
            # Selection uses the TRAIN window only.
            score = sharpe_ratio(result.returns.loc[train_dates], risk_free_rate)
            score = -np.inf if not np.isfinite(score) else score
            if score > best_score or best_result is None:
                best_params, best_score, best_result = params, score, result

        is_returns = best_result.returns.loc[train_dates]
        oos_returns = best_result.returns.loc[test_dates]
        is_chunks.append(is_returns)
        oos_chunks.append(oos_returns)

        row = {
            "fold": fold.index,
            "train_start": train_dates[0].date(),
            "test_start": test_dates[0].date(),
            "test_end": test_dates[-1].date(),
            "is_sharpe": sharpe_ratio(is_returns, risk_free_rate),
            "oos_sharpe": sharpe_ratio(oos_returns, risk_free_rate),
            "is_return": summarize(is_returns)["ann_return"],
            "oos_return": summarize(oos_returns)["ann_return"],
            "oos_max_dd": summarize(oos_returns)["max_drawdown"],
        }
        if param_grid:
            row["chosen_params"] = str(best_params)
        rows.append(row)

    stitched_oos = pd.concat(oos_chunks)
    return {
        "strategy": strategy_name,
        "folds": pd.DataFrame(rows),
        "oos_returns": stitched_oos,
        "oos_summary": summarize(stitched_oos, risk_free_rate=risk_free_rate),
        "is_summary": summarize(pd.concat(is_chunks), risk_free_rate=risk_free_rate),
        "n_folds": len(folds),
    }


def compare_to_benchmark(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    """Measure a strategy against buy-and-hold on the benchmark (SPY).

    Beating the index on raw return is easy if you take more risk, so the
    interesting figures here are the risk-adjusted ones:

      `beta`              -- market exposure. 1.0 moves with the index; 0.0 is
                             market-neutral. A "clever" strategy with beta 1.0
                             is an index fund with extra steps and fees.
      `alpha_annual`      -- return left over after paying for that market
                             exposure. This is the part that is actually skill.
      `information_ratio` -- consistency of out-performance: mean excess return
                             over its own volatility. Sharpe, but measured
                             against the benchmark instead of cash.
    """
    aligned = pd.concat(
        [returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1
    ).dropna()
    if len(aligned) < 2:
        return {k: np.nan for k in
                ("strategy_sharpe", "benchmark_sharpe", "beta", "alpha_annual",
                 "information_ratio", "correlation", "excess_ann_return")}

    strat, bench = aligned["strategy"], aligned["benchmark"]

    variance = bench.var(ddof=1)
    beta = float(strat.cov(bench) / variance) if variance > 0 else np.nan
    alpha_annual = (
        float((strat.mean() - beta * bench.mean()) * periods_per_year)
        if np.isfinite(beta)
        else np.nan
    )

    active = strat - bench
    tracking_error = active.std(ddof=1)
    information_ratio = (
        float(active.mean() / tracking_error * np.sqrt(periods_per_year))
        if tracking_error > 0
        else np.nan
    )

    strat_summary = summarize(strat, risk_free_rate=risk_free_rate)
    bench_summary = summarize(bench, risk_free_rate=risk_free_rate)

    return {
        "strategy_sharpe": strat_summary["sharpe"],
        "benchmark_sharpe": bench_summary["sharpe"],
        "excess_ann_return": strat_summary["ann_return"] - bench_summary["ann_return"],
        "beta": beta,
        "alpha_annual": alpha_annual,
        "information_ratio": information_ratio,
        "correlation": float(strat.corr(bench)),
    }
