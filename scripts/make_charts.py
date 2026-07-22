"""Regenerate every chart in notebooks/results/ from the cached price panel.

Run from anywhere:  python scripts/make_charts.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `engine`/`strategies`/`viz` import when
                                    # this is run as a script rather than -m

import yaml

from engine.backtest import VectorizedBacktester
from engine.data_loader import DataLoader
from strategies import available_strategies, load_strategy
from viz.plots import generate_all


def main() -> None:
    backtest_config = yaml.safe_load(open(REPO_ROOT / "configs" / "backtest.yaml"))
    universe = yaml.safe_load(open(REPO_ROOT / "configs" / "universe.yaml"))
    benchmark_ticker = universe["benchmark"]

    panel = DataLoader().load_panel()
    close = panel["close"]
    benchmark = (
        DataLoader(tickers=[benchmark_ticker])
        .load_panel()["close"][benchmark_ticker]
        .reindex(close.index)
        .ffill()
        .pct_change()
        .fillna(0.0)
    )

    returns = {}
    for name in available_strategies():
        signals = load_strategy(name).generate_signals(panel)
        returns[name] = VectorizedBacktester(backtest_config).run(close, signals).returns

    for path in generate_all(returns, benchmark=benchmark, benchmark_name=benchmark_ticker):
        print(f"  {path.relative_to(path.parents[2])}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
